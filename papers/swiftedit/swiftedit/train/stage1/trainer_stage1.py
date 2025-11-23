import os
import sys
import math
import time
import argparse
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except Exception as e:
    yaml = None

# Local imports
from swiftedit.train.stage1.dataset_synthetic import SyntheticJourneyDBDataset
from swiftedit.models.clip.text_encoder import CLIPTextEncoder
from swiftedit.models.clip.image_encoder import CLIPImageEncoder
from swiftedit.models.vae.vae_sdxl import VAESDXL
from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
from swiftedit.models.generator.generator_ip import GeneratorIP
from swiftedit.models.ip_adapter.projector import Projector
from swiftedit.models.ip_adapter.ip_adapter_branch import IPAdapterBranch
from swiftedit.models.inversion.inversion_net import InversionNet


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_placeholders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ${paths.key} placeholders in config values."""
    import re
    paths = cfg.get("paths", {})
    
    def resolve_value(value):
        if isinstance(value, str):
            # Match ${paths.key} pattern
            pattern = r'\$\{paths\.(\w+)\}'
            matches = re.findall(pattern, value)
            for match in matches:
                if match in paths:
                    value = value.replace(f'${{paths.{match}}}', str(paths[match]))
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value
    
    return {k: resolve_value(v) for k, v in cfg.items()}


def _load_config(defaults_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("pyyaml is required to load configuration files. Please install pyyaml.")
    with open(defaults_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if override_path is not None and os.path.isfile(override_path):
        with open(override_path, 'r') as f:
            over = yaml.safe_load(f)
        cfg = _deep_update(cfg, over)
    cfg = _resolve_placeholders(cfg)
    return cfg


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class SimpleEMA:
    """Minimal EMA utility for model parameters.
    Maintains shadow params: shadow = decay * shadow + (1 - decay) * param
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.device = next(model.parameters()).device
        # initialize shadow
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone().to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])


def build_models(cfg: Dict[str, Any], device: torch.device):
    # Dtypes
    sys_dtype = str(cfg.get('system', {}).get('dtype', 'float16')).lower()
    use_autocast = bool(cfg.get('system', {}).get('autocast', True))

    # Core components
    text_enc = CLIPTextEncoder.from_config({
        'provider': cfg['models']['clip']['text'].get('provider', 'openclip'),
        'model_name': cfg['models']['clip']['text'].get('model_name', 'ViT-L-14'),
        'pretrained': cfg['models']['clip']['text'].get('pretrained', 'laion2b_s32b_b82k'),
        'freeze': True,
        'dtype': cfg['models']['clip']['text'].get('dtype', 'float32'),
        'max_length': cfg['models']['clip']['text'].get('max_length', 77),
        'openclip_dir': cfg['paths'].get('openclip_dir', None),
        'device': str(device),  # Pass device to avoid CUDA initialization
    }).to(device)

    img_enc = CLIPImageEncoder.from_config({
        'provider': cfg['models']['clip']['image'].get('provider', 'openclip'),
        'model_name': cfg['models']['clip']['image'].get('model_name', 'ViT-L-14'),
        'pretrained': cfg['models']['clip']['image'].get('pretrained', 'laion2b_s32b_b82k'),
        'freeze': True,
        'dtype': cfg['models']['clip']['image'].get('dtype', 'float32'),
        'openclip_dir': cfg['paths'].get('openclip_dir', None),
        'device': str(device),  # Pass device to avoid CUDA initialization
    }).to(device)

    vae = VAESDXL.from_config({
        'repo_dir': cfg['models']['vae'].get('repo_dir', cfg['paths'].get('sdxl_vae_dir', 'assets/sdxl-vae')),
        'scaling_factor': cfg['models']['vae'].get('scaling_factor', 0.18215),
        'image_norm': cfg['models']['vae'].get('image_norm', '[-1,1]'),
        'sample_size': cfg['models']['vae'].get('sample_size', 512),
        'dtype': cfg['models']['vae'].get('dtype', sys_dtype),
    }, device=device).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    base_gen = SwiftBrushV2.from_config({
        'latent_channels': cfg['models']['generator'].get('latent_channels', 4),
        'text_embed_dim': cfg['models']['generator'].get('text_embed_dim', 768),
        'hidden_dim': cfg['models']['generator'].get('hidden_dim', 1536),
        'heads': cfg['models']['generator'].get('heads', 12),
        'dtype': cfg['models']['generator'].get('dtype', sys_dtype),
    }).to(device)
    base_gen.eval()
    for p in base_gen.parameters():
        p.requires_grad = False

    gen_ip = GeneratorIP.from_config(cfg, device=device)

    projector = Projector.from_config({
        'in_dim': cfg['models']['ip_adapter']['projector'].get('in_dim', 768),
        'out_dim': cfg['models']['ip_adapter']['projector'].get('out_dim', 768),
        'num_tokens': cfg['models']['ip_adapter']['projector'].get('num_tokens', 4),
    }).to(device)

    ip_branch = IPAdapterBranch.from_config({
        'token_dim': cfg['models']['ip_adapter']['branch'].get('token_dim', cfg['models']['ip_adapter']['projector'].get('out_dim', 768)),
        'attn_dim': cfg['models']['generator'].get('text_embed_dim', 768),
        'heads': cfg['models']['generator'].get('heads', 12),
        's_x': cfg['models']['ip_adapter']['branch'].get('default_scale_sx', 1.0),
        'trainable': not cfg['training']['stage1']['freeze'].get('ip_adapter_branch', False),
        'dtype': cfg['models']['generator'].get('dtype', sys_dtype),
    }).to(device)

    # Attach projector/branch to generator_ip
    gen_ip.set_projector(projector)
    gen_ip.set_ip_adapter_branch(ip_branch)

    inv_net = InversionNet.from_config(cfg, device=device)

    return {
        'text_encoder': text_enc,
        'image_encoder': img_enc,
        'vae': vae,
        'base_generator': base_gen,
        'generator_ip': gen_ip,
        'projector': projector,
        'ip_branch': ip_branch,
        'inversion_net': inv_net,
    }


def train_stage1(cfg: Dict[str, Any]):
    # Get device from config, with smart fallback
    device_str = cfg.get('system', {}).get('device', None)
    if device_str is not None:
        req_device = str(device_str).lower()
        if req_device == 'cuda' and not torch.cuda.is_available():
            fallback = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
            warnings.warn(f"CUDA requested in config but not available; falling back to {fallback.upper()}.")
            device_str = fallback
        elif req_device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            fallback = 'cuda' if torch.cuda.is_available() else 'cpu'
            warnings.warn(f"MPS requested in config but not available; falling back to {fallback.upper()}.")
            device_str = fallback
    if device_str is None:
        # Auto-detect: prefer MPS > CUDA > CPU
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_str = 'mps'
        elif torch.cuda.is_available():
            device_str = 'cuda'
        else:
            device_str = 'cpu'
    device = torch.device(device_str)
    dtype_str = str(cfg.get('system', {}).get('dtype', 'float16')).lower()
    use_autocast = bool(cfg.get('system', {}).get('autocast', True))

    models = build_models(cfg, device)
    text_enc = models['text_encoder']
    img_enc = models['image_encoder']
    vae = models['vae']
    base_gen = models['base_generator']
    gen_ip = models['generator_ip']
    projector = models['projector']
    ip_branch = models['ip_branch']
    inv_net = models['inversion_net']

    # Freeze components per config
    frz = cfg['training']['stage1']['freeze']
    if frz.get('generator', True):
        base_gen.eval()
        for p in base_gen.parameters():
            p.requires_grad = False
    if frz.get('vae', True):
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
    if frz.get('clip_text', True):
        text_enc.eval()
        for p in text_enc.parameters():
            p.requires_grad = False
    if frz.get('clip_image', True):
        img_enc.eval()
        for p in img_enc.parameters():
            p.requires_grad = False
    if frz.get('projector', False):
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False
    else:
        projector.train()
    if frz.get('ip_adapter_branch', False):
        ip_branch.eval()
        for p in ip_branch.parameters():
            p.requires_grad = False
    else:
        ip_branch.train()
    if frz.get('inversion_net', False):
        inv_net.eval()
        for p in inv_net.parameters():
            p.requires_grad = False
    else:
        inv_net.train()

    # Dataset and dataloader
    captions_path = cfg['training']['stage1']['dataset'].get('captions_path', cfg['paths']['datasets'].get('journeydb_captions'))
    image_resolution = int(cfg['training']['stage1'].get('image_resolution', 512))
    batch_size = int(cfg['training']['stage1'].get('batch_size', 4))

    dataset = SyntheticJourneyDBDataset(
        config=cfg,
        captions_path=captions_path,
        num_samples=None,
        device=device.type,
        dtype=dtype_str,
        seed=cfg.get('system', {}).get('seed', 42),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # Optimizer
    lr = float(cfg['training']['stage1'].get('lr', 1e-5))
    weight_decay = float(cfg['training']['stage1'].get('weight_decay', 1e-4))

    params: List[nn.Parameter] = []
    params += [p for p in inv_net.parameters() if p.requires_grad]
    params += [p for p in projector.parameters() if p.requires_grad]
    params += [p for p in ip_branch.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # EMA for inversion net
    ema_decay = float(cfg['training']['stage1'].get('ema_decay', 0.999))
    ema = SimpleEMA(inv_net, decay=ema_decay)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_autocast and device.type == 'cuda')

    # Training loop
    num_iters = int(cfg['training']['stage1'].get('num_iters', 100000))
    log_every = int(cfg['training']['stage1'].get('log_every', 100))
    save_every = int(cfg['training']['stage1'].get('save_every', 2000))
    lambda_stage1 = float(cfg['training']['stage1']['losses'].get('lambda_stage1', 1.0))

    checkpoints_dir = cfg['paths'].get('checkpoints_dir', 'checkpoints')
    stage_ckpt_dir = os.path.join(checkpoints_dir, 'stage1')
    _ensure_dir(stage_ckpt_dir)
    logs_dir = cfg['paths'].get('logs_dir', 'logs')
    _ensure_dir(logs_dir)

    it = 0
    data_iter = iter(loader)
    start_time = time.time()
    while it < num_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Unpack batch
        epsilon = batch['epsilon'].to(device)
        z = batch['z'].to(device)
        x_hat = batch['x_hat'].to(device)
        captions = batch.get('caption', None)
        if captions is None:
            # fall back to generating generic caption
            captions = ["a photo"] * epsilon.shape[0]

        # Recompute c_y via text encoder (fp32)
        with torch.no_grad():
            c_y = text_enc.forward(captions, normalize=True).to(device)

        # Recompute image tokens via image encoder + projector
        with torch.no_grad():
            img_feat = img_enc.forward(x_hat, normalize=True).to(device)
        img_tokens = projector(img_feat)

        # Forward through inversion net and generator ip
        inv_net.train()  # ensure train mode for stats if any
        gen_ip.train()   # to allow gradients into ip_branch/projector integrations

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_autocast and device.type == 'cuda'):
            eps_hat = inv_net(z, c_y)
            z_hat = gen_ip(epsilon=eps_hat, text_emb=c_y, img_tokens=img_tokens, mask=None, scales=None, return_details=False)
            if isinstance(z_hat, (tuple, list)):
                z_hat = z_hat[0]
            # Losses
            l_rec = F.mse_loss(z_hat, z)
            l_regr = F.mse_loss(eps_hat, epsilon)
            loss = l_rec + lambda_stage1 * l_regr
        scaler.scale(loss).backward()
        # Optional grad clipping
        grad_clip = cfg['training']['stage1'].get('grad_clip_norm', None)
        if grad_clip is not None and grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        # EMA update
        ema.update(inv_net)

        it += 1
        if it % log_every == 0:
            elapsed = time.time() - start_time
            it_s = it / max(elapsed, 1e-6)
            print(f"[Stage1] it={it}/{num_iters} loss={loss.item():.6f} l_rec={l_rec.item():.6f} l_regr={l_regr.item():.6f} ({it_s:.2f} it/s)")

        if it % save_every == 0 or it == num_iters:
            # Save checkpoint
            ckpt = {
                'it': it,
                'model_inv_net': inv_net.state_dict(),
                'model_projector': projector.state_dict(),
                'model_ip_branch': ip_branch.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': {k: v.cpu() for k, v in ema.shadow.items()},
                'config': cfg,
            }
            ckpt_path = os.path.join(stage_ckpt_dir, f"stage1_step_{it}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"[Stage1] Saved checkpoint to {ckpt_path}")

    print("[Stage1] Training completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="SwiftEdit Stage 1 Trainer")
    parser.add_argument('--config', type=str, default=os.path.join('swiftedit', 'configs', 'stage1.yaml'), help='Path to stage1 config YAML')
    parser.add_argument('--defaults', type=str, default=os.path.join('swiftedit', 'configs', 'defaults.yaml'), help='Path to defaults config YAML')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = _load_config(args.defaults, args.config)
    # Set seed
    seed = int(cfg.get('system', {}).get('seed', 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_stage1(cfg)

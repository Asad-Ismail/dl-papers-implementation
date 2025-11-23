import os
import sys
import math
import time
import argparse
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import yaml
except Exception as e:
    yaml = None

# Optional imports for losses and scheduler; trainer will gracefully degrade if missing
try:
    from swiftedit.losses.dists_loss import DISTS
    _HAS_DISTS = True
except Exception:
    DISTS = None
    _HAS_DISTS = False

try:
    from swiftedit.schedulers.noise_scheduler import TeacherDiffusion
    _HAS_TEACHER = True
except Exception:
    TeacherDiffusion = None
    _HAS_TEACHER = False

from swiftedit.train.stage2.dataset_real import RealCommonCanvasDataset

# Model builders
from swiftedit.models.clip.text_encoder import CLIPTextEncoder
from swiftedit.models.clip.image_encoder import CLIPImageEncoder
from swiftedit.models.ip_adapter.projector import Projector
from swiftedit.models.ip_adapter.ip_adapter_branch import IPAdapterBranch
from swiftedit.models.vae.vae_sdxl import VAESDXL
from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
from swiftedit.models.generator.generator_ip import GeneratorIP
from swiftedit.models.inversion.inversion_net import InversionNet


class SimpleEMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow, f"EMA: parameter {name} not in shadow registry"
            new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


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
        raise RuntimeError("PyYAML not available; please install pyyaml to load configs.")
    with open(defaults_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if override_path and os.path.isfile(override_path):
        with open(override_path, 'r') as f:
            override = yaml.safe_load(f)
        cfg = _deep_update(cfg, override)
    cfg = _resolve_placeholders(cfg)
    return cfg


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    return None


def build_models(cfg: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    # Build encoders
    text_enc = CLIPTextEncoder.from_config({
        "provider": cfg["models"]["clip"]["text"].get("provider", "openclip"),
        "model_name": cfg["models"]["clip"]["text"].get("model_name", "ViT-L-14"),
        "pretrained": cfg["models"]["clip"]["text"].get("pretrained", "laion2b_s32b_b82k"),
        "openclip_dir": cfg["paths"].get("openclip_dir", None),
        "freeze": cfg["models"]["clip"]["text"].get("freeze", True),
        "dtype": cfg["models"]["clip"]["text"].get("dtype", "float32"),
        "max_length": cfg["models"]["clip"]["text"].get("max_length", 77),
    })
    text_enc.to(device)
    text_enc.eval()
    img_enc = CLIPImageEncoder.from_config({
        "provider": cfg["models"]["clip"]["image"].get("provider", "openclip"),
        "model_name": cfg["models"]["clip"]["image"].get("model_name", "ViT-L-14"),
        "pretrained": cfg["models"]["clip"]["image"].get("pretrained", "laion2b_s32b_b82k"),
        "openclip_dir": cfg["paths"].get("openclip_dir", None),
        "freeze": cfg["models"]["clip"]["image"].get("freeze", True),
        "dtype": cfg["models"]["clip"]["image"].get("dtype", "float32"),
    })
    img_enc.to(device)
    img_enc.eval()

    projector = Projector.from_config(cfg.get("models", {}).get("ip_adapter", {}).get("projector", {}), device=str(device))
    branch_cfg = cfg.get("models", {}).get("ip_adapter", {}).get("branch", {})
    ip_branch = IPAdapterBranch.from_config(branch_cfg, device=str(device))

    vae = VAESDXL.from_config(cfg.get("models", {}).get("vae", {}), device=str(device))
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    base_gen = SwiftBrushV2.from_config(cfg.get("models", {}).get("generator", {}), device=str(device))
    base_gen.to(device)
    base_gen.eval()
    for p in base_gen.parameters():
        p.requires_grad = False

    gen_ip = GeneratorIP.from_config(cfg, device=str(device))
    gen_ip.set_projector(projector)
    gen_ip.set_ip_adapter_branch(ip_branch)
    gen_ip.to(device)
    # Freeze GeneratorIP and branch if requested
    if cfg.get("training", {}).get("stage2", {}).get("freeze", {}).get("generator_ip", True):
        gen_ip.eval()
        for p in gen_ip.parameters():
            p.requires_grad = False
    if cfg.get("training", {}).get("stage2", {}).get("freeze", {}).get("ip_adapter_branch", True):
        ip_branch.freeze()
    else:
        ip_branch.unfreeze()
    if cfg.get("training", {}).get("stage2", {}).get("freeze", {}).get("projector", True):
        for p in projector.parameters():
            p.requires_grad = False
    else:
        for p in projector.parameters():
            p.requires_grad = True

    inv_net = InversionNet.from_config(cfg, device=str(device))
    inv_net.to(device)
    inv_net.train()

    return {
        "text_encoder": text_enc,
        "image_encoder": img_enc,
        "projector": projector,
        "ip_branch": ip_branch,
        "vae": vae,
        "base_generator": base_gen,
        "generator_ip": gen_ip,
        "inversion_net": inv_net,
    }


def train_stage2(cfg: Dict[str, Any]) -> None:
    device_str = cfg.get("system", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_autocast = cfg.get("system", {}).get("autocast", True)
    dtype_str = cfg.get("system", {}).get("dtype", "float16")
    compute_dtype = _map_dtype_str(dtype_str) or torch.float16

    # Build models
    models = build_models(cfg, device)
    text_enc: CLIPTextEncoder = models["text_encoder"]
    img_enc: CLIPImageEncoder = models["image_encoder"]
    projector: Projector = models["projector"]
    ip_branch: IPAdapterBranch = models["ip_branch"]
    vae: VAESDXL = models["vae"]
    gen_ip: GeneratorIP = models["generator_ip"]
    inv_net: InversionNet = models["inversion_net"]

    # Dataset and DataLoader
    ds_cfg = cfg.get("training", {}).get("stage2", {}).get("dataset", {})
    root = ds_cfg.get("root", cfg.get("paths", {}).get("datasets", {}).get("commoncanvas_root", "data/commoncanvas"))
    prompts_file = ds_cfg.get("prompts_file", "prompts.json")
    image_resolution = cfg.get("training", {}).get("stage2", {}).get("image_resolution", 512)
    dataset = RealCommonCanvasDataset(
        config=cfg, root=root, prompts_file=prompts_file, image_resolution=image_resolution, device=device, sample_posterior=False
    )
    batch_size = cfg.get("training", {}).get("stage2", {}).get("batch_size", 1)
    num_workers = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Optimizer for inversion net only
    lr = cfg.get("training", {}).get("stage2", {}).get("lr", 1e-5)
    weight_decay = cfg.get("training", {}).get("stage2", {}).get("weight_decay", 0.0)
    optim = torch.optim.AdamW(inv_net.parameters(), lr=lr, weight_decay=weight_decay)

    # EMA
    ema_decay = cfg.get("training", {}).get("stage2", {}).get("ema_decay", 0.999)
    ema = SimpleEMA(inv_net, decay=ema_decay)

    # Loss weights
    losses_cfg = cfg.get("training", {}).get("stage2", {}).get("losses", {})
    dists_weight = float(losses_cfg.get("dists_weight", 1.0))
    regu_weight = float(losses_cfg.get("regu_weight", 0.5))
    w_t_schedule = losses_cfg.get("w_t_schedule", "uniform")
    t_range = losses_cfg.get("t_range", [0, 999])

    # Teacher diffusion
    teacher = None
    if _HAS_TEACHER:
        try:
            teacher = TeacherDiffusion.from_config(cfg.get("schedulers", {}).get("teacher", {}), device=device)
        except Exception as e:
            print(f"[Stage2] Warning: Failed to initialize TeacherDiffusion: {e}. Disabling regularization.")
            teacher = None
    else:
        print("[Stage2] TeacherDiffusion not available; proceeding without SDS regularization.")

    # DISTS loss
    dists = None
    if _HAS_DISTS:
        try:
            dists = DISTS(device=device)
        except Exception as e:
            print(f"[Stage2] Warning: Failed to initialize DISTS: {e}. Falling back to L2 loss.")
            dists = None
    else:
        print("[Stage2] DISTS not available; using L2 loss as a fallback.")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and device.type == 'cuda'))

    # Training loop
    num_iters = cfg.get("training", {}).get("stage2", {}).get("num_iters", 180000)
    log_every = cfg.get("training", {}).get("stage2", {}).get("log_every", 100)
    save_every = cfg.get("training", {}).get("stage2", {}).get("save_every", 2000)
    grad_clip_norm = cfg.get("training", {}).get("stage2", {}).get("grad_clip_norm", 1.0)

    checkpoints_dir = cfg.get("paths", {}).get("checkpoints_dir", "checkpoints")
    _ensure_dir(checkpoints_dir)

    global_iter = 0
    start_time = time.time()

    for epoch in range(max(1, (num_iters // max(1, len(loader))) + 1)):
        for batch in loader:
            if global_iter >= num_iters:
                break
            # Batch components
            x = batch["x"]  # (B, 3, H, W), CPU, [0,1]
            z = batch["z"].to(device)  # (B, 4, H/8, W/8)
            c_y = batch["c_y"].to(device)
            c_x_tokens = batch["c_x_tokens"].to(device)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_autocast and device.type == 'cuda'), dtype=compute_dtype):
                # Predict epsilon and reconstruct latent
                eps_hat = inv_net(z, c_y)
                z_hat = gen_ip(eps_hat, c_y, img_tokens=c_x_tokens, mask=None, scales=None, return_details=False)
                # Decode to image space for perceptual loss
                x_hat = vae.decode(z_hat, output_norm="[0,1]")

                # Perceptual loss (DISTS or L2)
                x_dev = x.to(device).float()
                x_hat_dev = x_hat.float()
                if dists is not None:
                    L_perc = dists(x_dev, x_hat_dev)
                else:
                    L_perc = F.mse_loss(x_hat_dev, x_dev)

                # SDS-inspired regularization
                L_regu = torch.tensor(0.0, device=device)
                if teacher is not None:
                    B = z.shape[0]
                    t_min, t_max = int(t_range[0]), int(t_range[1])
                    t = teacher.sample_t(B, t_min=t_min, t_max=t_max)  # (B,)
                    alpha_t, sigma_t = teacher.alpha_sigma(t)  # scalars or (B,)
                    # reshape for broadcasting over latent
                    while alpha_t.dim() < z.dim():
                        alpha_t = alpha_t.view(-1, *([1] * (z.dim() - 1)))
                        sigma_t = sigma_t.view(-1, *([1] * (z.dim() - 1)))
                    z_t = alpha_t * z + sigma_t * eps_hat
                    eps_teacher = teacher.predict_eps(z_t, t, c_y)
                    diff = eps_hat - eps_teacher
                    # weight schedule
                    if w_t_schedule == "uniform":
                        w_t = torch.ones_like(t, dtype=diff.dtype, device=device)
                    elif w_t_schedule == "snr":
                        # approximate SNR weighting: (alpha^2 / sigma^2)
                        snr = (alpha_t.flatten() ** 2) / (sigma_t.flatten() ** 2 + 1e-8)
                        w_t = snr
                    else:
                        w_t = torch.ones_like(t, dtype=diff.dtype, device=device)
                    # reshape w_t for reduction
                    while w_t.dim() < diff.dim():
                        w_t = w_t.view(-1, *([1] * (diff.dim() - 1)))
                    L_regu = 0.5 * (w_t * (diff ** 2)).mean()

                loss = dists_weight * L_perc + regu_weight * L_regu

            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(inv_net.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optim)
            scaler.update()

            # EMA update
            ema.update(inv_net)

            # Logging
            if (global_iter + 1) % log_every == 0:
                elapsed = time.time() - start_time
                print(f"[Stage2] Iter {global_iter+1}/{num_iters} | dists={float(L_perc.detach().cpu()):.4f} | regu={float(L_regu.detach().cpu()):.4f} | loss={float(loss.detach().cpu()):.4f} | time={elapsed:.1f}s")

            # Checkpointing
            if (global_iter + 1) % save_every == 0 or (global_iter + 1) == num_iters:
                ckpt = {
                    "iter": global_iter + 1,
                    "config": cfg,
                    "inversion_net": inv_net.state_dict(),
                    "optimizer": optim.state_dict(),
                    "ema_shadow": ema.shadow,
                }
                # Optionally save projector/ip_branch if unfrozen
                if any(p.requires_grad for p in projector.parameters()):
                    ckpt["projector"] = projector.state_dict()
                if any(p.requires_grad for p in ip_branch.parameters()):
                    ckpt["ip_branch"] = ip_branch.state_dict()
                ckpt_path = os.path.join(checkpoints_dir, f"stage2_iter_{global_iter+1}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"[Stage2] Saved checkpoint to {ckpt_path}")

            global_iter += 1
            if global_iter >= num_iters:
                break

    print("[Stage2] Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwiftEdit Stage 2 Trainer")
    parser.add_argument("--defaults", type=str, default=os.path.join("swiftedit", "configs", "defaults.yaml"), help="Path to defaults.yaml")
    parser.add_argument("--config", type=str, default=os.path.join("swiftedit", "configs", "stage2.yaml"), help="Path to stage2.yaml override")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_config(args.defaults, args.config)
    train_stage2(cfg)


if __name__ == "__main__":
    main()

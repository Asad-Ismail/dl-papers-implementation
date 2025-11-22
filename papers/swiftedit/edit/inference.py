import os
import sys
import time
import argparse
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Local imports from the repo
from swiftedit.models.clip.text_encoder import CLIPTextEncoder
from swiftedit.models.clip.image_encoder import CLIPImageEncoder
from swiftedit.models.vae.vae_sdxl import VAESDXL
from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
from swiftedit.models.generator.generator_ip import GeneratorIP
from swiftedit.models.ip_adapter.projector import Projector
from swiftedit.models.ip_adapter.ip_adapter_branch import IPAdapterBranch
from swiftedit.models.inversion.inversion_net import InversionNet
from swiftedit.edit.mask_extractor import MaskExtractor


TensorOrPILOrPath = Union[torch.Tensor, Image.Image, str]


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in {"float16", "fp16", "half"}:
        return torch.float16
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if s in {"float32", "fp32", "single"}:
        return torch.float32
    return None


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(defaults_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    import yaml
    with open(defaults_path, "r") as f:
        base = yaml.safe_load(f)
    if override_path and os.path.exists(override_path):
        with open(override_path, "r") as f:
            over = yaml.safe_load(f)
        _deep_update(base, over)
    return base


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pil_load(path: str, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if resolution is not None and resolution > 0:
        img = img.resize((resolution, resolution), Image.Resampling.BICUBIC)
    return img


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float().view(img.size[1], img.size[0], 3) / 255.0).numpy())
    # The above fast path is brittle across platforms; fallback to standard conversion if shape mismatch
    if x.ndim != 3 or x.shape[-1] != 3:
        import numpy as np
        x = torch.from_numpy(np.array(img).astype("float32") / 255.0)
    x = x.permute(2, 0, 1).contiguous()  # (3,H,W)
    return x


def _tensor_from_input(image: TensorOrPILOrPath, resolution: int) -> torch.Tensor:
    if isinstance(image, str):
        pil = _pil_load(image, resolution)
        return _pil_to_tensor(pil)
    elif isinstance(image, Image.Image):
        pil = image
        if resolution is not None and resolution > 0 and (pil.size[0] != resolution or pil.size[1] != resolution):
            pil = pil.resize((resolution, resolution), Image.Resampling.BICUBIC)
        return _pil_to_tensor(pil)
    elif isinstance(image, torch.Tensor):
        x = image
        if x.ndim == 3 and x.shape[0] in (1, 3):
            return x.float().clamp(0, 1)
        elif x.ndim == 4:
            return x[0].float().clamp(0, 1)
        else:
            raise ValueError("Unsupported tensor shape for image input: {}".format(x.shape))
    else:
        raise ValueError("Unsupported image input type: {}".format(type(image)))


def _mask_from_input(mask: Optional[TensorOrPILOrPath], target_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, str):
        m = Image.open(mask).convert("L").resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
        import numpy as np
        m = torch.from_numpy(np.array(m).astype("float32") / 255.0)
        m = m.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return m
    if isinstance(mask, Image.Image):
        m = mask.convert("L").resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
        import numpy as np
        m = torch.from_numpy(np.array(m).astype("float32") / 255.0)
        m = m.unsqueeze(0).unsqueeze(0)
        return m
    if isinstance(mask, torch.Tensor):
        m = mask.float()
        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 3:
            if m.shape[0] == 1:
                m = m.unsqueeze(0)
            elif m.shape[0] in (target_hw[0], target_hw[1]):
                # ambiguous, force to (1,1,H,W)
                m = m.unsqueeze(0).unsqueeze(0)
            else:
                m = m.unsqueeze(0)
        elif m.ndim == 4:
            pass
        else:
            raise ValueError("Unsupported mask tensor shape: {}".format(m.shape))
        # resize to target
        m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
        m = m.clamp(0, 1)
        return m
    raise ValueError("Unsupported mask input type: {}".format(type(mask)))


def build_inference_models(cfg: Dict[str, Any], device: Optional[torch.device] = None) -> Dict[str, nn.Module]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate models
    txt_enc = CLIPTextEncoder.from_config(cfg)
    img_enc = CLIPImageEncoder.from_config(cfg)
    vae = VAESDXL.from_config(cfg, device=device)
    base_g = SwiftBrushV2.from_config(cfg, device=device)
    gen_ip = GeneratorIP.from_config(cfg, device=device)
    projector = Projector.from_config(cfg)
    ip_branch = IPAdapterBranch.from_config(cfg)
    gen_ip.set_projector(projector)
    gen_ip.set_ip_adapter_branch(ip_branch)

    # Freeze irrelevant modules (encoders, vae, generator)
    for m in [txt_enc, img_enc, vae, base_g, projector, ip_branch, gen_ip]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        m.to(device)

    invnet = InversionNet.from_config(cfg, device=device)
    invnet.eval()

    return {
        "text_encoder": txt_enc,
        "image_encoder": img_enc,
        "vae": vae,
        "base_generator": base_g,
        "generator_ip": gen_ip,
        "projector": projector,
        "ip_branch": ip_branch,
        "inversion_net": invnet,
    }


def _load_inversion_checkpoint(invnet: nn.Module, ckpt_path: Optional[str], ema: bool = True) -> None:
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Inversion checkpoint not found: {ckpt_path}", file=sys.stderr)
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        if ema and "ema_shadow" in ckpt:
            # EMA state is a flat name->tensor dict; try to load
            try:
                invnet.load_state_dict(ckpt["ema_shadow"], strict=False)
                print("[INFO] Loaded EMA weights into inversion net.")
                return
            except Exception as e:
                print(f"[WARN] Failed to load EMA shadow: {e}")
        if "inversion_net" in ckpt and isinstance(ckpt["inversion_net"], dict):
            state = ckpt["inversion_net"]
        else:
            # maybe the dict is directly the state dict
            keys = list(ckpt.keys())
            if keys and keys[0].startswith("conv_in") or keys[0].startswith("text_proj"):
                state = ckpt
    if state is None:
        print("[WARN] No compatible state dict found in checkpoint.", file=sys.stderr)
        return
    missing, unexpected = invnet.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading inversion net: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading inversion net: {unexpected}")


def edit_image(
    image: TensorOrPILOrPath,
    prompt_src: str,
    prompt_edit: str,
    cfg: Dict[str, Any],
    inversion_ckpt: Optional[str] = None,
    user_mask: Optional[TensorOrPILOrPath] = None,
    scales: Optional[Dict[str, float]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Perform one-step text-guided image editing.

    Returns a dict with keys: x_edit (torch.Tensor in [0,1], 3xHxW), x_src, mask, timings (dict), and possibly intermediates.
    """
    device = torch.device(device) if isinstance(device, str) else (device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    models = build_inference_models(cfg, device=device)
    invnet = models["inversion_net"]
    _load_inversion_checkpoint(invnet, inversion_ckpt, ema=True)

    # Config specifics
    infer_res = int(cfg.get("inference", {}).get("image_resolution", 512))
    vae_norm = cfg.get("models", {}).get("vae", {}).get("image_norm", "[-1,1]")
    dtype_str = cfg.get("system", {}).get("dtype", "float16")
    autocast_enable = bool(cfg.get("system", {}).get("autocast", True)) and device.type == "cuda"
    amp_dtype = _map_dtype_str(dtype_str) or torch.float16

    # Preprocess image
    x_src = _tensor_from_input(image, infer_res).unsqueeze(0)  # (1,3,H,W) in [0,1]

    # Encode for CLIP image tokens
    with torch.no_grad():
        img_emb = models["image_encoder"].forward(x_src.to(device), normalize=True)
        if isinstance(img_emb, dict):
            img_emb = img_emb.get("pooled", None)
        c_x_tokens = models["projector"].forward(img_emb.to(device))  # (1,N,D)

    # Text embeddings
    with torch.no_grad():
        c_src = models["text_encoder"].forward([prompt_src], normalize=True)
        c_edit = models["text_encoder"].forward([prompt_edit], normalize=True)

    # VAE encode
    x_for_vae = x_src.clone()
    if vae_norm == "[-1,1]":
        x_for_vae = x_for_vae * 2.0 - 1.0
    with torch.no_grad():
        enc = models["vae"].encode(x_for_vae.to(device), sample_posterior=False)
        if isinstance(enc, tuple):
            z = enc[0]
        else:
            z = enc
        # Ensure latent scaling is handled by VAESDXL.encode
        z = z.detach()

    # Prepare mask
    H_lat, W_lat = z.shape[-2], z.shape[-1]
    if user_mask is not None and bool(cfg.get("inference", {}).get("use_user_mask_if_provided", True)):
        mask = _mask_from_input(user_mask, (H_lat, W_lat)).to(device)
    else:
        # self-guided mask
        me_cfg = cfg.get("aram", {}).get("mask", {})
        blur_sigma = float(me_cfg.get("blur_sigma", 1.0))
        soft_th = float(me_cfg.get("soft_threshold", 0.0))
        mask_extractor = MaskExtractor(
            inversion_net=invnet,
            text_encoder=models["text_encoder"],
            blur_sigma=blur_sigma,
            soft_threshold=soft_th,
            clamp_min=float(me_cfg.get("clamp_min", 0.0)),
            clamp_max=float(me_cfg.get("clamp_max", 1.0)),
            dtype=dtype_str,
        )
        with torch.no_grad():
            mask = mask_extractor.compute_mask(z, c_src, c_edit, batch_concat=True, return_intermediate=False)
        mask = mask.to(device)

    # ARaM scales
    if scales is None:
        ar = cfg.get("inference", {}).get("aram_scales", {})
        scales = {
            "s_y": float(ar.get("s_y", cfg.get("aram", {}).get("s_y", 1.0))),
            "s_edit": float(ar.get("s_edit", cfg.get("aram", {}).get("s_edit", 0.3))),
            "s_non_edit": float(ar.get("s_non_edit", cfg.get("aram", {}).get("s_non_edit", 1.5))),
        }

    # Inversion to eps_hat under edit prompt
    z_in = z.to(device)
    c_edit_in = c_edit.to(device)

    # Forward GIP to get z_hat
    timings: Dict[str, float] = {}
    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enable, dtype=amp_dtype):
        eps_hat = invnet.forward(z_in, c_edit_in)
    timings["inversion_s"] = time.time() - t0

    t1 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enable, dtype=amp_dtype):
        z_hat = models["generator_ip"].forward(
            epsilon=eps_hat,
            text_emb=c_edit_in,
            img_tokens=c_x_tokens.to(device),
            mask=mask,
            scales=scales,
            return_details=False,
        )
        if isinstance(z_hat, tuple):
            z_hat = z_hat[0]
    timings["generation_s"] = time.time() - t1

    # Decode to image [0,1]
    t2 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enable, dtype=amp_dtype):
        x_edit = models["vae"].decode(z_hat, output_norm="[0,1]")
    timings["decode_s"] = time.time() - t2

    timings["total_s"] = time.time() - t0

    out = {
        "x_src": x_src.squeeze(0).cpu().clamp(0, 1),
        "x_edit": x_edit.squeeze(0).cpu().clamp(0, 1),
        "mask": mask.squeeze(0).cpu(),
        "timings": timings,
        "z": z.cpu(),
        "z_hat": z_hat.cpu(),
    }
    return out


def save_image(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.detach().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        img = tensor[0].cpu().numpy()
        import numpy as np
        arr = (img * 255.0).round().astype("uint8")
        Image.fromarray(arr, mode="L").save(path)
        return
    elif tensor.shape[0] == 3:
        img = tensor.permute(1, 2, 0).cpu().numpy()
        import numpy as np
        arr = (img * 255.0).round().astype("uint8")
        Image.fromarray(arr).save(path)
        return
    else:
        raise ValueError("Unsupported tensor shape for saving image: {}".format(tensor.shape))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SwiftEdit one-step editing inference")
    ap.add_argument("--image", type=str, required=True, help="Path to source image")
    ap.add_argument("--src-prompt", type=str, required=True, help="Source prompt (describes the input)")
    ap.add_argument("--edit-prompt", type=str, required=True, help="Edit prompt (desired output)")
    ap.add_argument("--output", type=str, required=True, help="Output edited image path")
    ap.add_argument("--mask", type=str, default=None, help="Optional user mask (path); white=edit region")
    ap.add_argument("--defaults", type=str, default=os.path.join("swiftedit", "configs", "defaults.yaml"))
    ap.add_argument("--config", type=str, default=None, help="Optional override config (e.g., configs/inference.yaml)")
    ap.add_argument("--invnet-ckpt", type=str, default=None, help="Path to inversion net checkpoint (use EMA if present)")
    ap.add_argument("--resolution", type=int, default=None, help="Override inference resolution (square)")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu")
    ap.add_argument("--sy", type=float, default=None, help="ARaM text scale s_y")
    ap.add_argument("--se", type=float, default=None, help="ARaM edit-region image scale s_edit")
    ap.add_argument("--sne", type=float, default=None, help="ARaM non-edit-region image scale s_non_edit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.defaults, args.config)
    if args.resolution is not None:
        cfg.setdefault("inference", {}).setdefault("image_resolution", args.resolution)
        cfg["inference"]["image_resolution"] = args.resolution

    scales = None
    if args.sy is not None or args.se is not None or args.sne is not None:
        base_scales = cfg.get("inference", {}).get("aram_scales", {})
        scales = {
            "s_y": float(args.sy if args.sy is not None else base_scales.get("s_y", 1.0)),
            "s_edit": float(args.se if args.se is not None else base_scales.get("s_edit", 0.3)),
            "s_non_edit": float(args.sne if args.sne is not None else base_scales.get("s_non_edit", 1.5)),
        }

    out = edit_image(
        image=args.image,
        prompt_src=args.src_prompt,
        prompt_edit=args.edit_prompt,
        cfg=cfg,
        inversion_ckpt=args.invnet_ckpt,
        user_mask=args.mask,
        scales=scales,
        device=args.device,
    )

    _ensure_dir(os.path.dirname(args.output) or ".")
    save_image(out["x_edit"], args.output)

    # Optionally also save the mask next to output
    mask_path = os.path.splitext(args.output)[0] + "_mask.png"
    try:
        save_image(out["mask"], mask_path)
    except Exception as e:
        print(f"[WARN] Failed to save mask: {e}")

    t = out.get("timings", {})
    print("Timings (s):", {k: round(float(v), 4) for k, v in t.items()})


if __name__ == "__main__":
    main()

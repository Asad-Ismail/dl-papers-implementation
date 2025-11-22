"""
SwiftEdit edit package initializer.

This module aggregates and re-exports the editing utilities:
- Self-guided mask extraction (MaskExtractor, self_guided_mask)
- ARaM (Attention Rescaling and Masking) helpers (ARaM, ARaMScales, aram_combine)
- End-to-end inference pipeline (build_inference_models, edit_image, save_image)

Additionally, it provides lightweight configuration loading/merging helpers and
a convenience function run_edit(...) for programmatic editing with config files.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union
import os

# Defensive imports of submodules
try:
    from .mask_extractor import MaskExtractor, build_mask_extractor, self_guided_mask  # type: ignore
except Exception:  # pragma: no cover
    MaskExtractor = None  # type: ignore
    build_mask_extractor = None  # type: ignore
    self_guided_mask = None  # type: ignore

try:
    from .aram import ARaM, ARaMScales, ensure_mask_shape, resize_mask, broadcast_mask_to_attn, aram_combine  # type: ignore
except Exception:  # pragma: no cover
    ARaM = None  # type: ignore
    ARaMScales = None  # type: ignore
    ensure_mask_shape = None  # type: ignore
    resize_mask = None  # type: ignore
    broadcast_mask_to_attn = None  # type: ignore
    aram_combine = None  # type: ignore

try:
    from .inference import build_inference_models, edit_image, save_image  # type: ignore
except Exception:  # pragma: no cover
    build_inference_models = None  # type: ignore
    edit_image = None  # type: ignore
    save_image = None  # type: ignore

# Optional YAML/JSON config loaders
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

import json


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge upd into base (in-place) and return base.

    - If both base[k] and upd[k] are dicts, merge recursively
    - Otherwise, base[k] is replaced by upd[k]
    """
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file into a dict.

    Supports .yml/.yaml (requires PyYAML) and .json; raises FileNotFoundError
    if path does not exist and ValueError if YAML parsing is requested but
    PyYAML is not installed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if ext in (".yml", ".yaml"):
        if yaml is None:
            raise ValueError("PyYAML is required to parse YAML configs.")
        return yaml.safe_load(text)  # type: ignore
    if ext == ".json":
        return json.loads(text)
    # Best-effort: try YAML then JSON
    if yaml is not None:
        try:
            return yaml.safe_load(text)  # type: ignore
        except Exception:
            pass
    return json.loads(text)


def run_edit(
    defaults_path: str,
    override_path: Optional[str],
    image: Union[str, "torch.Tensor", "PIL.Image.Image"],
    src_prompt: str,
    edit_prompt: str,
    out_path: Optional[str] = None,
    mask: Optional[Union[str, "torch.Tensor", "PIL.Image.Image"]] = None,
    device: Optional[str] = None,
    inversion_ckpt: Optional[str] = None,
    scales: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Convenience function to run the SwiftEdit pipeline programmatically.

    - Loads defaults config and optionally merges an override config
    - Executes edit_image(image, src_prompt, edit_prompt, cfg, ...)
    - Saves output image if out_path is provided

    Returns the dict produced by edit_image (contains edited image tensor,
    mask, timings, and other metadata).
    """
    if edit_image is None:
        raise RuntimeError("edit_image is not available. Ensure swiftedit.edit.inference is importable.")

    cfg = load_config(defaults_path)
    if override_path and os.path.isfile(override_path):
        override = load_config(override_path)
        cfg = deep_update(cfg, override)

    result = edit_image(
        image=image,
        prompt_src=src_prompt,
        prompt_edit=edit_prompt,
        cfg=cfg,
        inversion_ckpt=inversion_ckpt,
        user_mask=mask,
        scales=scales,
        device=device,
    )

    if out_path is not None:
        if save_image is None:
            raise RuntimeError("save_image is not available. Ensure swiftedit.edit.inference is importable.")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_image(result.get("x_edit"), out_path)  # type: ignore
        result["out_path"] = out_path
    return result


__all__ = [
    # Mask extractor
    "MaskExtractor",
    "build_mask_extractor",
    "self_guided_mask",
    # ARaM
    "ARaM",
    "ARaMScales",
    "ensure_mask_shape",
    "resize_mask",
    "broadcast_mask_to_attn",
    "aram_combine",
    # Inference
    "build_inference_models",
    "edit_image",
    "save_image",
    # Config helpers and convenience runner
    "deep_update",
    "load_config",
    "run_edit",
]

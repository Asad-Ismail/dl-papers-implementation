"""
SwiftEdit Models Package

Provides unified builders and public interfaces for core model components:
- VAE (SDXL-compatible): encode/decode between image and latent space
- One-step generator (SwiftBrushV2 placeholder): maps epsilon + text to latent
- CLIP encoders: text and image embeddings via OpenCLIP
- IP-Adapter components: projector and image-conditioning branch
- Optional inversion network and EMA (imported lazily when available)

This package exposes convenience functions to build commonly used models from a
configuration dictionary, following the repository's defaults.yaml/stage*.yaml schemas.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# VAE
from .vae import VAESDXL, build_vae as build_vae, load_vae as load_vae

# Generator (one-step)
from .generator.swiftbrushv2 import SwiftBrushV2, build_generator as build_generator

# CLIP encoders
from .clip.text_encoder import CLIPTextEncoder, build_text_encoder
from .clip.image_encoder import CLIPImageEncoder, build_image_encoder

# IP-Adapter components
from .ip_adapter.projector import Projector, build_projector
from .ip_adapter.ip_adapter_branch import IPAdapterBranch, build_ip_adapter_branch

# Optional modules (available after implementation)
try:
    from .generator.generator_ip import GeneratorIP  # type: ignore
except Exception:
    GeneratorIP = None  # type: ignore

try:
    from .inversion.inversion_net import InversionNet  # type: ignore
except Exception:
    InversionNet = None  # type: ignore

try:
    from .inversion.ema import EMA  # type: ignore
except Exception:
    EMA = None  # type: ignore

__all__ = [
    # VAE
    "VAESDXL",
    "build_vae",
    "load_vae",
    # Generator
    "SwiftBrushV2",
    "build_generator",
    # CLIP
    "CLIPTextEncoder",
    "build_text_encoder",
    "CLIPImageEncoder",
    "build_image_encoder",
    # IP-Adapter
    "Projector",
    "build_projector",
    "IPAdapterBranch",
    "build_ip_adapter_branch",
    # Optional exports
    "GeneratorIP",
    "InversionNet",
    "EMA",
    # Helpers
    "build_models_from_config",
]


def build_models_from_config(config: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    """
    Build common SwiftEdit model components from a repository-level configuration dict.

    Parameters
    - config: dict-like, typically loaded from configs/defaults.yaml or stage*.yaml
    - device: optional device override (e.g., "cuda" or "cpu")

    Returns
    - models: dict containing constructed model instances with keys:
        - text_encoder: CLIPTextEncoder
        - image_encoder: CLIPImageEncoder
        - vae: VAESDXL
        - generator: SwiftBrushV2 (one-step G)
        - projector: Projector (IP-Adapter projector)
        - ip_adapter_branch: IPAdapterBranch (image-conditioning branch)
      Optional keys if modules are available in the environment:
        - generator_ip: GeneratorIP
        - inversion_net: InversionNet
        - ema: EMA (if enabled in config)
    """
    models: Dict[str, Any] = {}

    # CLIP encoders
    try:
        models["text_encoder"] = build_text_encoder(config, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build CLIP text encoder: {e}")

    try:
        models["image_encoder"] = build_image_encoder(config, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build CLIP image encoder: {e}")

    # VAE
    try:
        models["vae"] = build_vae(config, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build VAE: {e}")

    # One-step generator
    try:
        models["generator"] = build_generator(config, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build one-step generator: {e}")

    # IP-Adapter projector and branch
    try:
        models["projector"] = build_projector(config, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build IP-Adapter projector: {e}")

    try:
        models["ip_adapter_branch"] = build_ip_adapter_branch(config, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to build IP-Adapter branch: {e}")

    # Optional components
    # GeneratorIP wrapper (decoupled attention + ARaM), if implemented
    if GeneratorIP is not None:
        try:
            models["generator_ip"] = GeneratorIP.from_config(config, device=device)  # type: ignore
        except Exception:
            # Non-fatal if not yet implemented or misconfigured
            pass

    # InversionNet (F_theta), if implemented
    if InversionNet is not None:
        try:
            models["inversion_net"] = InversionNet.from_config(config, device=device)  # type: ignore
        except Exception:
            pass

    # EMA helper, if requested
    ema_cfg = (config.get("models", {}).get("inversion_net", {}).get("ema", {}))
    if EMA is not None and isinstance(ema_cfg, dict) and ema_cfg.get("enabled", False):
        try:
            decay = float(ema_cfg.get("decay", 0.999))
            models["ema"] = EMA(decay)  # type: ignore
        except Exception:
            pass

    return models

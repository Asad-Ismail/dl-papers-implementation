"""
SwiftEdit Generator Package

Aggregates generator-related modules and provides convenience builders.

Exports:
- SwiftBrushV2: one-step latent generator G(ε, c_y) -> ẑ
- GeneratorIP: IP-Adapter and ARaM-enabled wrapper around a base generator
- DecoupledCrossAttention: multi-head attention that decouples text and image KV streams

Convenience functions:
- build_generator(config, device=None): construct base one-step generator
- load_generator(config, device=None): alias to build_generator
- build_generator_ip(config, device=None): construct GeneratorIP wrapper
- load_generator_ip(config, device=None): alias to build_generator_ip
- build_decoupled_cross_attention(config, device=None): construct decoupled attention module
- build_generator_stack(config, device=None, projector=None, ip_branch=None):
    build a base generator and optional GeneratorIP, wiring projector/ip_branch if provided.

This package initializer enables stable imports such as:
    from swiftedit.models.generator import SwiftBrushV2, GeneratorIP, build_generator_ip

and is used by trainers, inference, and evaluation utilities.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Base one-step generator
from .swiftbrushv2 import SwiftBrushV2, build_generator as _build_generator

# Decoupled attention (optional for downstream usage)
from .decoupled_cross_attention import (
    DecoupledCrossAttention,
    build_decoupled_cross_attention,
)

# IP-conditioned generator wrapper (optional)
try:
    from .generator_ip import GeneratorIP  # type: ignore
    _HAS_GEN_IP = True
except Exception:
    GeneratorIP = None  # type: ignore
    _HAS_GEN_IP = False

try:
    # Optional builder alias if defined in module
    from .generator_ip import build_generator_ip as _build_generator_ip  # type: ignore
except Exception:
    _build_generator_ip = None  # type: ignore


def build_generator(config: Dict[str, Any], device: Optional[str] = None) -> SwiftBrushV2:
    """Construct the base one-step generator from a configuration dict.

    Parameters:
    - config: repository-level configuration dictionary
    - device: optional device string (e.g., "cuda", "cpu")

    Returns:
    - SwiftBrushV2 instance
    """
    return _build_generator(config, device=device)


def load_generator(config: Dict[str, Any], device: Optional[str] = None) -> SwiftBrushV2:
    """Alias to build_generator for symmetry with other loaders."""
    return build_generator(config, device=device)


def build_generator_ip(config: Dict[str, Any], device: Optional[str] = None) -> GeneratorIP:
    """Construct the IP-conditioned generator wrapper if available.

    Raises RuntimeError if GeneratorIP is unavailable in the current build.
    """
    if not _HAS_GEN_IP or _build_generator_ip is None:
        raise RuntimeError("GeneratorIP is not available. Ensure swiftedit.models.generator.generator_ip exists and imports correctly.")
    return _build_generator_ip(config, device=device)  # type: ignore


def load_generator_ip(config: Dict[str, Any], device: Optional[str] = None) -> GeneratorIP:
    """Alias to build_generator_ip for API symmetry."""
    return build_generator_ip(config, device=device)


def build_generator_stack(
    config: Dict[str, Any],
    device: Optional[str] = None,
    projector: Optional[Any] = None,
    ip_branch: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a base generator and, if available, an IP-conditioned wrapper.

    If a projector and ip_branch are provided, they are wired into the GeneratorIP instance.

    Returns a dict with keys:
    - "base_generator": SwiftBrushV2
    - "generator_ip": GeneratorIP or None
    - "decoupled_attn": DecoupledCrossAttention (constructed via build_decoupled_cross_attention)
    """
    gen_base = build_generator(config, device=device)
    models: Dict[str, Any] = {"base_generator": gen_base}

    # Build decoupled attention helper (optional downstream use)
    try:
        dec_attn = build_decoupled_cross_attention(config, device=device)
    except Exception:
        dec_attn = None
    models["decoupled_attn"] = dec_attn

    # Build and wire GeneratorIP if available
    gen_ip = None
    if _HAS_GEN_IP and _build_generator_ip is not None:
        try:
            gen_ip = build_generator_ip(config, device=device)
            if projector is not None and hasattr(gen_ip, "set_projector"):
                gen_ip.set_projector(projector)  # type: ignore[attr-defined]
            if ip_branch is not None and hasattr(gen_ip, "set_ip_adapter_branch"):
                gen_ip.set_ip_adapter_branch(ip_branch)  # type: ignore[attr-defined]
        except Exception:
            gen_ip = None
    models["generator_ip"] = gen_ip

    return models


__all__ = [
    "SwiftBrushV2",
    "DecoupledCrossAttention",
    "GeneratorIP",
    "build_generator",
    "load_generator",
    "build_generator_ip",
    "load_generator_ip",
    "build_decoupled_cross_attention",
    "build_generator_stack",
]

"""SwiftEdit VAE package

Exports SDXL-compatible VAE wrappers and builders.

Usage examples:
- from swiftedit.models.vae import VAESDXL, build_vae
- vae = build_vae(config)
"""
from __future__ import annotations

from typing import Any, Optional

from .vae_sdxl import VAESDXL, build_vae

__all__ = [
    "VAESDXL",
    "build_vae",
]


def load_vae(config: dict, device: Optional[str] = None) -> VAESDXL:
    """Alias to build_vae for symmetry with other model loaders.

    Args:
        config: Repository-level configuration dict containing a models.vae section.
        device: Optional device override (e.g., "cuda" or "cpu").

    Returns:
        An instantiated VAESDXL object.
    """
    return build_vae(config, device=device)

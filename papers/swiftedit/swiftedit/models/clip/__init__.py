"""
SwiftEdit CLIP encoders package

This module aggregates and re-exports the text and image encoder wrappers that are
used across training, inference, and evaluation. It also provides convenience
builders for constructing encoders from a unified configuration dict.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Defensive imports to keep package importable even if one submodule fails
try:  # Text encoder
    from .text_encoder import (
        CLIPTextEncoder,
        build_text_encoder as _build_text_encoder,
    )
except Exception:  # pragma: no cover
    CLIPTextEncoder = None  # type: ignore

    def _build_text_encoder(config: Dict[str, Any], device: Optional[str] = None):  # type: ignore
        raise RuntimeError(
            "CLIP text encoder implementation is unavailable. Ensure open-clip-torch is installed "
            "and swiftedit/models/clip/text_encoder.py imports correctly."
        )

try:  # Image encoder
    from .image_encoder import (
        CLIPImageEncoder,
        build_image_encoder as _build_image_encoder,
    )
except Exception:  # pragma: no cover
    CLIPImageEncoder = None  # type: ignore

    def _build_image_encoder(config: Dict[str, Any], device: Optional[str] = None):  # type: ignore
        raise RuntimeError(
            "CLIP image encoder implementation is unavailable. Ensure open-clip-torch is installed "
            "and swiftedit/models/clip/image_encoder.py imports correctly."
        )


def build_text_encoder(config: Dict[str, Any], device: Optional[str] = None) -> "CLIPTextEncoder":
    """Convenience builder for the text encoder from a repo-level config.

    Args:
        config: Repository configuration dict (expects models.clip.text keys).
        device: Optional device override (e.g., "cuda", "cpu", or torch.device).

    Returns:
        An initialized CLIPTextEncoder instance.
    """
    return _build_text_encoder(config, device=device)


def build_image_encoder(config: Dict[str, Any], device: Optional[str] = None) -> "CLIPImageEncoder":
    """Convenience builder for the image encoder from a repo-level config.

    Args:
        config: Repository configuration dict (expects models.clip.image keys).
        device: Optional device override (e.g., "cuda", "cpu", or torch.device).

    Returns:
        An initialized CLIPImageEncoder instance.
    """
    return _build_image_encoder(config, device=device)


def build_clip_encoders(
    config: Dict[str, Any], device: Optional[str] = None
) -> Tuple["CLIPTextEncoder", "CLIPImageEncoder"]:
    """Build both text and image CLIP encoders from a configuration dict.

    This helper ensures consistent construction and device placement across
    the two encoders.

    Args:
        config: Repository configuration dict.
        device: Optional device override for both encoders.

    Returns:
        A tuple of (text_encoder, image_encoder).
    """
    txt = build_text_encoder(config, device=device)
    img = build_image_encoder(config, device=device)
    return txt, img


__all__ = [
    "CLIPTextEncoder",
    "CLIPImageEncoder",
    "build_text_encoder",
    "build_image_encoder",
    "build_clip_encoders",
]

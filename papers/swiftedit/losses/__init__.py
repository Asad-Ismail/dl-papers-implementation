"""
SwiftEdit Losses and Metrics Package

This module aggregates and re-exports perceptual losses (DISTS), pixel fidelity metrics
(PSNR/MSE with optional masking), and CLIP-based semantic scores. It also provides
convenience builders to construct commonly used loss/metric components from a
repository configuration dictionary.

Public exports:
- DISTS: Perceptual loss wrapper with graceful fallback to MSE
- normalize_to_01, apply_mask: Image/mask utilities
- mse, psnr: Pixel-wise fidelity metrics
- masked_mse, masked_psnr, masked_psnr_mse: Region-specific (foreground/background) metrics
- CLIPScorer: OpenCLIP-based text-image similarity scorer
- build_scorer_from_config: Factory for CLIPScorer from config
- clip_score_whole: CLIP-Whole score convenience
- clip_score_edited: CLIP-Edited score convenience (composite/crop modes)

Additional helpers:
- build_perceptual_loss_from_config(cfg, device=None)
- build_metrics_suite(cfg, device=None)

Note: All builders accept a plain Python dict configuration (as loaded from YAML/JSON).
Keys follow the repository config schema (configs/defaults.yaml, configs/stage*.yaml).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Perceptual loss (DISTS) wrapper
try:
    from .dists_loss import DISTS  # noqa: F401
except Exception as e:  # pragma: no cover
    DISTS = None  # type: ignore

# Pixel fidelity metrics and utilities
try:
    from .psnr_mse import (
        normalize_to_01,
        apply_mask,
        mse,
        psnr,
        masked_mse,
        masked_psnr,
        masked_psnr_mse,
    )  # noqa: F401
except Exception:
    normalize_to_01 = None  # type: ignore
    apply_mask = None  # type: ignore
    mse = None  # type: ignore
    psnr = None  # type: ignore
    masked_mse = None  # type: ignore
    masked_psnr = None  # type: ignore
    masked_psnr_mse = None  # type: ignore

# CLIP-based semantic scoring
try:
    from .clip_scores import (
        CLIPScorer,
        build_scorer_from_config,
        clip_score_whole,
        clip_score_edited,
    )  # noqa: F401
except Exception:
    CLIPScorer = None  # type: ignore
    build_scorer_from_config = None  # type: ignore
    clip_score_whole = None  # type: ignore
    clip_score_edited = None  # type: ignore


def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safe nested dict get using dot-separated path.

    Example: _get(cfg, "training.stage2.image_resolution", 512)
    """
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_perceptual_loss_from_config(cfg: Dict[str, Any], device: Optional[str] = None) -> Optional[DISTS]:
    """Construct a DISTS perceptual loss instance from config.

    If DISTS is unavailable, returns None (caller should handle fallback).

    Args:
        cfg: Repository configuration dictionary.
        device: Optional device string (e.g., "cuda" or "cpu"). If None, inferred by DISTS.

    Returns:
        DISTS instance or None if backend unavailable.
    """
    if DISTS is None:
        return None
    # Prefer training.stage2.image_resolution for resizing; fall back to inference.image_resolution
    resize_to = _get(cfg, "training.stage2.image_resolution", None)
    if resize_to is None:
        resize_to = _get(cfg, "inference.image_resolution", None)
    # Reduction default: mean over batch
    reduction = "mean"
    # DISTS backbone runs in float32; device managed internally
    return DISTS(reduction=reduction, resize_to=resize_to, clamp=True, require_grad=False, device=device)


def build_metrics_suite(cfg: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    """Build a standard metrics suite used across training/evaluation.

    Returns a dict containing available components:
      - "dists": DISTS instance or None
      - "clip_scorer": CLIPScorer instance or None
      - Utility callables re-exported from psnr_mse and clip_scores

    Args:
        cfg: Configuration dict containing model and eval settings.
        device: Optional device override for CLIP scorer and DISTS.

    Returns:
        Dict[str, Any]: Metrics components.
    """
    suite: Dict[str, Any] = {}
    suite["dists"] = build_perceptual_loss_from_config(cfg, device=device)

    # CLIP scorer (evaluation)
    scorer = None
    if CLIPScorer is not None:
        try:
            # Prefer eval.piebench.clip_model, then models.clip.image
            if build_scorer_from_config is not None:
                scorer = build_scorer_from_config(cfg, device=device)
        except Exception:
            scorer = None
    suite["clip_scorer"] = scorer

    # Re-export utility functions (may be None if unavailable)
    suite.update({
        "normalize_to_01": normalize_to_01,
        "apply_mask": apply_mask,
        "mse": mse,
        "psnr": psnr,
        "masked_mse": masked_mse,
        "masked_psnr": masked_psnr,
        "masked_psnr_mse": masked_psnr_mse,
        "clip_score_whole": clip_score_whole,
        "clip_score_edited": clip_score_edited,
    })

    return suite


__all__ = [
    # DISTS perceptual loss
    "DISTS",
    "build_perceptual_loss_from_config",
    # Pixel fidelity metrics and utilities
    "normalize_to_01",
    "apply_mask",
    "mse",
    "psnr",
    "masked_mse",
    "masked_psnr",
    "masked_psnr_mse",
    # CLIP scoring
    "CLIPScorer",
    "build_scorer_from_config",
    "clip_score_whole",
    "clip_score_edited",
    # Metrics suite builder
    "build_metrics_suite",
]

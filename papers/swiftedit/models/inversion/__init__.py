"""
SwiftEdit Inversion module package.

Public API:
- InversionNet: Fθ mapping latents z and text embeddings c_y to noise ε̂
- build_inversion_net: convenience builder from a configuration dict
- EMA: Exponential Moving Average utility for tracking inversion network shadow weights
- build_ema_from_config: factory to construct EMA from config and register a model

This package aggregates the inversion components for convenient imports:
    from swiftedit.models.inversion import InversionNet, EMA, build_inversion_net, build_ema_from_config
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .inversion_net import InversionNet, build_inversion_net
from .ema import EMA, build_ema_from_config

__all__ = [
    "InversionNet",
    "build_inversion_net",
    "EMA",
    "build_ema_from_config",
]


def load_inversion_net(config: Dict[str, Any], device: Optional[str] = None) -> InversionNet:
    """Alias to build_inversion_net for API symmetry.

    Parameters
    ----------
    config : Dict[str, Any]
        Repository-level configuration dict.
    device : Optional[str]
        Device override (e.g., "cuda" or "cpu"). If None, builder defaults apply.

    Returns
    -------
    InversionNet
        Constructed inversion network instance.
    """
    return build_inversion_net(config, device=device)

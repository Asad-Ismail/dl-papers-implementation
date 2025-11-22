"""
SwiftEdit IP-Adapter package initializer.

This module aggregates and re-exports the IP-Adapter components used for image
conditioning within decoupled cross-attention, namely:
  - Projector: maps a global CLIP image embedding vector to a short sequence of
    conditioning tokens (typically N=4) with feature dimension matching text tokens.
  - IPAdapterBranch: learns key/value projections (W_Kx, W_Vx) for the image token
    stream and stores a global scaling factor s_x for image-conditioned attention.

It also provides convenience builder functions that construct these components
from a unified configuration dictionary, and a helper to build both together.

Public API:
  - Projector, IPAdapterBranch classes
  - build_projector(config, device=None)
  - build_ip_adapter_branch(config, device=None)
  - build_ip_adapter(config, device=None) -> Dict[str, nn.Module]

These builders expect a repository-level config dict structured similarly to
swiftedit/configs/*.yaml, and optionally accept a device override.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Defensive imports to keep the package importable even if submodules fail.
try:
    from .projector import Projector, build_projector as _build_projector
except Exception as e:  # pragma: no cover
    Projector = None  # type: ignore
    _build_projector = None  # type: ignore

try:
    from .ip_adapter_branch import (
        IPAdapterBranch,
        build_ip_adapter_branch as _build_ip_adapter_branch,
    )
except Exception as e:  # pragma: no cover
    IPAdapterBranch = None  # type: ignore
    _build_ip_adapter_branch = None  # type: ignore


def build_projector(config: Dict[str, Any], device: Optional[str] = None) -> Projector:
    """Construct an IP-Adapter Projector from a unified config dict.

    Parameters:
      - config: repository-level configuration dict
      - device: optional device override (e.g., "cuda", "cpu", torch.device)

    Returns:
      - Projector instance
    """
    if _build_projector is None:
        raise RuntimeError(
            "IP-Adapter projector implementation unavailable. Ensure swiftedit.models.ip_adapter.projector is importable."
        )
    return _build_projector(config, device=device)


def build_ip_adapter_branch(
    config: Dict[str, Any], device: Optional[str] = None
) -> IPAdapterBranch:
    """Construct an IP-Adapter image-conditioning branch from a config dict.

    Parameters:
      - config: repository-level configuration dict
      - device: optional device override (e.g., "cuda", "cpu", torch.device)

    Returns:
      - IPAdapterBranch instance
    """
    if _build_ip_adapter_branch is None:
        raise RuntimeError(
            "IP-Adapter branch implementation unavailable. Ensure swiftedit.models.ip_adapter.ip_adapter_branch is importable."
        )
    return _build_ip_adapter_branch(config, device=device)


def build_ip_adapter(
    config: Dict[str, Any], device: Optional[str] = None
) -> Dict[str, Any]:
    """Build both IP-Adapter components (projector and branch) from config.

    Returns a dictionary with keys:
      - "projector": Projector instance
      - "branch": IPAdapterBranch instance

    This helper simplifies constructing and wiring IP-Adapter parts downstream.
    """
    proj = build_projector(config, device=device)
    branch = build_ip_adapter_branch(config, device=device)
    return {"projector": proj, "branch": branch}


__all__ = [
    "Projector",
    "IPAdapterBranch",
    "build_projector",
    "build_ip_adapter_branch",
    "build_ip_adapter",
]

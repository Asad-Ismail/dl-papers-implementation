# Copyright (c) 2025 SwiftEdit
# Perceptual DISTS loss wrapper with graceful fallback.
from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Map image tensor to [0,1] range if it appears to be in [-1,1].
    Expects shape (N,C,H,W). Returns float32 tensor in [0,1].
    """
    x = x.float()
    # Heuristic: if min < 0, assume [-1,1]
    if torch.is_floating_point(x) and torch.min(x) < 0.0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _reduce(loss_per_sample: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "none":
        return loss_per_sample
    if reduction == "sum":
        return loss_per_sample.sum()
    # default mean
    return loss_per_sample.mean()


class DISTS(nn.Module):
    """DISTS perceptual loss.

    Attempts to use DISTS-pytorch if installed; otherwise falls back to per-sample MSE in image space.

    Args:
        reduction: 'mean' | 'sum' | 'none'
        resize_to: Optional[int] square side to resize inputs before computing loss
        clamp: Whether to clamp inputs to [0,1] after normalization
        require_grad: Whether to allow gradient flow through the DISTS backbone (default False)
    """

    def __init__(
        self,
        reduction: str = "mean",
        resize_to: Optional[int] = None,
        clamp: bool = True,
        require_grad: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.resize_to = resize_to
        self.clamp = clamp
        self.require_grad = require_grad

        self._has_dists = False
        self._dists_model: Optional[nn.Module] = None

        # Try to import DISTS implementation
        dists_mod = None
        try:
            import DISTS_pytorch as dmod  # type: ignore
            dists_mod = dmod
        except Exception:
            try:
                import dists_pytorch as dmod  # alternate name
                dists_mod = dmod
            except Exception:
                dists_mod = None

        if dists_mod is not None:
            try:
                model = dists_mod.DISTS()
                # Place on device/dtype
                if device is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device=device)
                # DISTS reference runs in float32
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(self.require_grad)
                self._dists_model = model
                self._has_dists = True
            except Exception:
                self._dists_model = None
                self._has_dists = False
        # register buffer for device tracking if no model
        if not self._has_dists:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # store device handle
        self._device = device
        self._dtype = dtype if dtype is not None else torch.float32

    @property
    def has_dists(self) -> bool:
        return self._has_dists and (self._dists_model is not None)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}, resize_to={self.resize_to}, clamp={self.clamp}, has_dists={self.has_dists}"

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize_to is None:
            return x
        h, w = x.shape[-2:]
        if h == self.resize_to and w == self.resize_to:
            return x
        return F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bicubic", align_corners=False, antialias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure correct shape and type
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"DISTS expects 4D tensors (N,C,H,W); got {x.shape} and {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Batch sizes must match")
        if x.shape[1] != 3 or y.shape[1] != 3:
            # Attempt to adapt: if latent (C=4) or grayscale, try to convert to 3-channels by simple mapping
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            if y.shape[1] == 1:
                y = y.repeat(1, 3, 1, 1)
            # If 4 channels (e.g., latents), this is not supported; raise
            if x.shape[1] != 3 or y.shape[1] != 3:
                raise ValueError("DISTS expects 3-channel images in RGB space; provide decoded images")

        x = _to_01(x)
        y = _to_01(y)
        if self.clamp:
            x = x.clamp(0.0, 1.0)
            y = y.clamp(0.0, 1.0)

        x = self._maybe_resize(x)
        y = self._maybe_resize(y)

        if self.has_dists:
            # DISTS typically returns per-sample values when size preserves batch
            with torch.set_grad_enabled(self.require_grad):
                out = self._dists_model(x, y)
            # Handle different return types (scalar vs vector)
            if isinstance(out, torch.Tensor):
                if out.ndim == 0:
                    # scalar for the batch; make per-sample by repeating
                    loss_per = out.repeat(x.shape[0]).to(device=x.device)
                else:
                    loss_per = out.view(-1)
            else:
                # Unknown type, fallback to MSE per-sample
                loss_per = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
        else:
            # Fallback: per-sample MSE on images
            loss_per = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)

        return _reduce(loss_per, self.reduction)


__all__ = ["DISTS"]

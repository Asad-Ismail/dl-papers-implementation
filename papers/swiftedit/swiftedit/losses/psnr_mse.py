"""
PSNR and MSE utilities with optional masking.

Implements:
- normalize_to_01: maps inputs in [-1,1] or arbitrary to [0,1]
- mse: mean squared error with optional soft mask
- psnr: PSNR (dB) computed from MSE and data_range
- masked_mse / masked_psnr: helpers to apply a mask (optionally background = 1 - mask)
- masked_psnr_mse: returns both PSNR and MSE for a given (possibly inverted) mask

Notes:
- Inputs are expected to be tensors of shape (B, C, H, W) or (C, H, W).
- Masks accepted as (B, 1, H, W), (B, H, W), (H, W) and are broadcast when needed.
- Mask values can be soft (continuous in [0,1]); computation uses weighted MSE.
- For PieBench, background region uses (1 - GT_mask).
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

__all__ = [
    "normalize_to_01",
    "apply_mask",
    "mse",
    "psnr",
    "masked_mse",
    "masked_psnr",
    "masked_psnr_mse",
]


def normalize_to_01(x: Tensor, clamp: bool = True) -> Tensor:
    """Normalize input image tensor to [0,1].

    If any element < 0, assume input in [-1,1] and remap.
    Otherwise assumes already in [0,1]. Optionally clamps to [0,1].
    """
    if x.dtype.is_floating_point:
        if torch.any(x < 0):
            x = (x + 1.0) * 0.5
        if clamp:
            x = x.clamp(0.0, 1.0)
    return x


def _ensure_bchw(x: Tensor) -> Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected tensor with 3 or 4 dims (C,H,W) or (B,C,H,W), got shape {tuple(x.shape)}")
    return x


def _prepare_mask(mask: Optional[Tensor], x: Tensor) -> Optional[Tensor]:
    """Prepare/broadcast mask to shape (B, 1, H, W) matching x (B,C,H,W)."""
    if mask is None:
        return None
    mask = mask.to(dtype=x.dtype, device=x.device)
    # Accept (H,W), (B,H,W), (B,1,H,W), (1,H,W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        # could be (B,H,W) or (1,H,W)
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4:
        if mask.size(1) != 1 and mask.size(1) != x.size(1):
            # If channel not 1 and not matching image channels, try squeeze
            mask = mask.mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")

    # Resize to match spatial resolution if needed
    if (mask.size(-2) != x.size(-2)) or (mask.size(-1) != x.size(-1)):
        mask = F.interpolate(mask, size=(x.size(-2), x.size(-1)), mode="bilinear", align_corners=False)

    # Broadcast batch if needed
    if mask.size(0) == 1 and x.size(0) > 1:
        mask = mask.expand(x.size(0), -1, -1, -1)
    elif mask.size(0) != x.size(0):
        raise ValueError(f"Mask batch {mask.size(0)} != input batch {x.size(0)}")

    # Clamp to [0,1]
    mask = mask.clamp(0.0, 1.0)
    return mask


def apply_mask(x: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Apply mask to tensor x (B,C,H,W) by multiplying along spatial dims.

    If mask has shape (B,1,H,W), it will be broadcast across channels.
    If mask is None, returns x unchanged.
    """
    x = _ensure_bchw(x)
    if mask is None:
        return x
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(-2) != x.size(-2) or mask.size(-1) != x.size(-1):
        mask = F.interpolate(mask, size=(x.size(-2), x.size(-1)), mode="bilinear", align_corners=False)
    if mask.size(0) != x.size(0):
        if mask.size(0) == 1:
            mask = mask.expand(x.size(0), -1, -1, -1)
        else:
            raise ValueError("Mask batch mismatch")
    return x * mask


def mse(
    x: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Compute (masked) Mean Squared Error.

    - x, y shapes: (B,C,H,W) or (C,H,W)
    - mask optional soft mask; broadcastable to (B,1,H,W)
    - normalize: if True, map inputs to [0,1]
    - reduction: "mean" (default), "sum", or "none" for per-sample
    """
    x = _ensure_bchw(x).to(dtype=torch.float32)
    y = _ensure_bchw(y).to(dtype=torch.float32)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    if normalize:
        x = normalize_to_01(x)
        y = normalize_to_01(y)

    diff2 = (x - y) ** 2

    if mask is None:
        # per-sample mean across C,H,W
        per_sample = diff2.flatten(1).mean(dim=1)
    else:
        m = _prepare_mask(mask, x)  # (B,1,H,W)
        # weight per pixel; broadcast to channels
        w = m
        # sum across channels then average over C by weighting per-pixel equally across channels
        # Implement weighted mean across spatial (and channels equally):
        # Compute per-pixel mean across channels of squared error, then weight by mask
        per_pixel = diff2.mean(dim=1, keepdim=True)  # (B,1,H,W)
        weighted_sum = (per_pixel * w).sum(dim=(2, 3)).squeeze(1)  # (B,)
        weight_total = w.sum(dim=(2, 3)).squeeze(1)  # (B,)
        # Avoid div by zero: if all weights zero, fall back to unmasked mean for that sample
        unmasked = diff2.flatten(1).mean(dim=1)
        per_sample = torch.where(weight_total > eps, weighted_sum / (weight_total + eps), unmasked)

    if reduction == "none":
        return per_sample
    elif reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def psnr(
    x: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
    data_range: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Compute PSNR (dB) optionally with a soft mask.

    PSNR = 10 * log10( data_range^2 / MSE ) where MSE computed with the same mask.
    """
    mse_val = mse(x, y, mask=mask, reduction="none", normalize=normalize, eps=eps)
    # Avoid division by zero -> clamp MSE to eps
    mse_val = torch.clamp(mse_val, min=eps)
    psnr_per = 10.0 * torch.log10((data_range ** 2) / mse_val)
    if reduction == "none":
        return psnr_per
    elif reduction == "mean":
        return psnr_per.mean()
    elif reduction == "sum":
        return psnr_per.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def masked_mse(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    use_background: bool = True,
    reduction: str = "mean",
    normalize: bool = True,
) -> Tensor:
    """Convenience wrapper to compute MSE on a region.

    - If use_background=True, region = 1 - mask (background)
    - Else, region = mask (edited/foreground)
    """
    m = _prepare_mask(mask, _ensure_bchw(x))
    if use_background:
        m = 1.0 - m
    return mse(x, y, mask=m, reduction=reduction, normalize=normalize)


def masked_psnr(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    use_background: bool = True,
    reduction: str = "mean",
    data_range: float = 1.0,
    normalize: bool = True,
) -> Tensor:
    """Convenience wrapper to compute PSNR on a region.

    - If use_background=True, region = 1 - mask (background)
    - Else, region = mask (edited/foreground)
    """
    m = _prepare_mask(mask, _ensure_bchw(x))
    if use_background:
        m = 1.0 - m
    return psnr(x, y, mask=m, reduction=reduction, data_range=data_range, normalize=normalize)


def masked_psnr_mse(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    use_background: bool = True,
    reduction: str = "mean",
    data_range: float = 1.0,
    normalize: bool = True,
) -> Dict[str, Tensor]:
    """Compute both PSNR and MSE on the masked region.

    Returns dict with keys: {"psnr": ..., "mse": ...}
    """
    m = _prepare_mask(mask, _ensure_bchw(x))
    if use_background:
        m = 1.0 - m
    out_mse = mse(x, y, mask=m, reduction=reduction, normalize=normalize)
    out_psnr = psnr(x, y, mask=m, reduction=reduction, data_range=data_range, normalize=normalize)
    return {"psnr": out_psnr, "mse": out_mse}

# Copyright (c) 2025 SwiftEdit Authors
# ARaM: Attention Rescaling and Masking utilities
# Implements helpers to resize and broadcast masks to attention feature shapes
# and combines decoupled text/image attention outputs with region-aware scales
# according to Eq. (9) in the reproduction plan.

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = dtype_str.lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    return None


def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def ensure_mask_shape(mask: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    """Ensure mask has shape (B, 1, H, W). Accepts (B,H,W), (1,H,W), (H,W)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        # Could be (B,H,W) or (1,H,W)
        if mask.shape[0] == b:
            mask = mask.unsqueeze(1)
        else:
            mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        # (B,1,H,W) expected
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    # Broadcast to batch if needed
    if mask.shape[0] == 1 and b > 1:
        mask = mask.expand(b, -1, -1, -1)
    # Resize if spatial size mismatch
    if mask.shape[-2] != h or mask.shape[-1] != w:
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    return clamp_01(mask)


def resize_mask(mask: torch.Tensor, target_hw: Tuple[int, int], mode: str = "bilinear",
                align_corners: bool = False) -> torch.Tensor:
    """Resize a mask to target (H, W). Accepts (B,1,Hm,Wm) or (B,Hm,Wm) or (Hm,Wm)."""
    h, w = target_hw
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    mask = F.interpolate(mask, size=(h, w), mode=mode, align_corners=align_corners)
    return clamp_01(mask)


def broadcast_mask_to_attn(mask: torch.Tensor, attn_shape: Tuple[int, ...],
                           feat_hw: Optional[Tuple[int, int]] = None,
                           num_heads: Optional[int] = None) -> torch.Tensor:
    """
    Broadcast mask to be compatible with an attention tensor.

    Expected attention shapes (examples):
    - (B, Hh, L, D) for attention outputs per head/token
    - (B, L, D) when heads are folded or single-head
    - (B, Hh, L, S) for attention maps (query length L, source length S)

    We construct a mask over the query length L by resizing the base mask to (H_feat, W_feat)
    and flattening to L = H_feat * W_feat, then broadcasting across heads and channels.

    Args:
        mask: Base mask, shape (B,1,Hm,Wm) or similar
        attn_shape: Shape tuple of the target attention tensor
        feat_hw: Optional (H_feat, W_feat); if None and attn_shape has a length dimension L, we try to infer
        num_heads: Optional number of heads; if None, inferred from attn_shape when possible

    Returns:
        A mask tensor broadcastable to the attention tensor: typically (B, Hh, L, 1) or (B, L, 1)
    """
    if len(attn_shape) < 3:
        raise ValueError(f"Unexpected attention shape: {attn_shape}")
    b = attn_shape[0]
    if len(attn_shape) == 4:
        # (B, Hh, L, D) or (B, Hh, L, S)
        hheads = attn_shape[1]
        L = attn_shape[2]
    elif len(attn_shape) == 3:
        # (B, L, D) or (B, L, S)
        hheads = 1
        L = attn_shape[1]
    else:
        # Fallback: assume second dim is heads and third is L
        hheads = attn_shape[1]
        L = attn_shape[2]
    if num_heads is not None:
        hheads = num_heads

    # Infer feature H,W if not given. Try to find a factorization close to square.
    if feat_hw is None:
        # Try perfect square first
        s = int(L**0.5)
        if s * s == L:
            feat_h, feat_w = s, s
        else:
            # Find factors near square
            best = (1, L)
            diff_best = L - 1
            for f in range(1, s + 1):
                if L % f == 0:
                    g = L // f
                    diff = abs(f - g)
                    if diff < diff_best:
                        diff_best = diff
                        best = (f, g)
            feat_h, feat_w = best
    else:
        feat_h, feat_w = feat_hw

    m = ensure_mask_shape(mask, b, feat_h, feat_w)  # (B,1,H_feat,W_feat)
    m = m.view(b, 1, feat_h * feat_w, 1)  # (B,1,L,1)
    if hheads > 1:
        m = m.expand(b, hheads, feat_h * feat_w, 1)
    return clamp_01(m)


class ARaMScales:
    """Container for ARaM scaling coefficients."""
    def __init__(self, s_y: float = 1.0, s_edit: float = 0.3, s_non_edit: float = 1.5, s_x: Optional[float] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.s_y = torch.tensor(float(s_y), device=device, dtype=dtype or torch.float32)
        self.s_edit = torch.tensor(float(s_edit), device=device, dtype=dtype or torch.float32)
        self.s_non_edit = torch.tensor(float(s_non_edit), device=device, dtype=dtype or torch.float32)
        # Optional global image scale fallback (used when mask is None)
        self.s_x = torch.tensor(float(s_x if s_x is not None else 1.0), device=device, dtype=dtype or torch.float32)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "ARaMScales":
        if device is not None or dtype is not None:
            self.s_y = self.s_y.to(device=device, dtype=dtype or self.s_y.dtype)
            self.s_edit = self.s_edit.to(device=device, dtype=dtype or self.s_edit.dtype)
            self.s_non_edit = self.s_non_edit.to(device=device, dtype=dtype or self.s_non_edit.dtype)
            self.s_x = self.s_x.to(device=device, dtype=dtype or self.s_x.dtype)
        return self

    def as_dict(self) -> Dict[str, float]:
        return {
            "s_y": float(self.s_y.item()),
            "s_edit": float(self.s_edit.item()),
            "s_non_edit": float(self.s_non_edit.item()),
            "s_x": float(self.s_x.item()),
        }


class ARaM:
    """
    Manages ARaM mask preparation and application across different feature resolutions.

    This utility caches resized masks keyed by (H, W) to reduce repeated interpolation
    overhead during multi-layer attention application.
    """

    def __init__(self, base_mask: torch.Tensor, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        if base_mask.dim() == 2:
            base_mask = base_mask.unsqueeze(0).unsqueeze(0)
        elif base_mask.dim() == 3:
            base_mask = base_mask.unsqueeze(1)
        elif base_mask.dim() != 4:
            raise ValueError(f"Unsupported base mask shape: {tuple(base_mask.shape)}")
        self.base_mask = clamp_01(base_mask.float())
        if device is not None:
            self.base_mask = self.base_mask.to(device)
        self.cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.device = device or self.base_mask.device
        self.dtype = dtype or self.base_mask.dtype

    def get_mask(self, target_hw: Tuple[int, int]) -> torch.Tensor:
        key = (int(target_hw[0]), int(target_hw[1]))
        if key in self.cache:
            return self.cache[key]
        m = F.interpolate(self.base_mask, size=key, mode="bilinear", align_corners=False)
        m = clamp_01(m)
        self.cache[key] = m
        return m

    def clear_cache(self) -> None:
        self.cache.clear()

    def broadcast_to_attn(self, attn_shape: Tuple[int, ...], feat_hw: Optional[Tuple[int, int]] = None,
                          num_heads: Optional[int] = None) -> torch.Tensor:
        # If feat_hw given, use cached resize; otherwise infer from L
        b = attn_shape[0]
        if feat_hw is not None:
            m = self.get_mask(feat_hw)  # (B,1,H,W)
            if m.shape[0] == 1 and b > 1:
                m = m.expand(b, -1, -1, -1)
            return broadcast_mask_to_attn(m, attn_shape, feat_hw=feat_hw, num_heads=num_heads)
        else:
            # Fall back to dynamic path using ensure_mask_shape
            return broadcast_mask_to_attn(self.base_mask, attn_shape, feat_hw=None, num_heads=num_heads)


def aram_combine(text_attn: torch.Tensor, img_attn: torch.Tensor,
                 mask: Optional[torch.Tensor], scales: Optional[Dict[str, float]] = None,
                 global_img_scale: Optional[float] = None) -> torch.Tensor:
    """
    Combine text and image attention outputs under ARaM.

    Implements:
      h = s_y * M * Attn_text + s_edit * M * Attn_img + s_non_edit * (1 - M) * Attn_img
    Fallback (mask is None):
      h = Attn_text + s_x * Attn_img

    Args:
        text_attn: Attention output for text conditioning (same shape as img_attn)
        img_attn: Attention output for image conditioning
        mask: Optional soft mask in [0,1]; broadcastable to attention output
        scales: Optional dict with keys {"s_y", "s_edit", "s_non_edit", optional "s_x"}
        global_img_scale: Optional fallback scale for image attention when mask is None (overrides scales["s_x"]) if provided

    Returns:
        Combined attention output tensor
    """
    if text_attn.shape != img_attn.shape:
        raise ValueError(f"text_attn and img_attn shapes must match, got {text_attn.shape} vs {img_attn.shape}")

    s_y = 1.0
    s_edit = 0.3
    s_non = 1.5
    s_x = 1.0
    if scales is not None:
        s_y = float(scales.get("s_y", s_y))
        s_edit = float(scales.get("s_edit", s_edit))
        s_non = float(scales.get("s_non_edit", s_non))
        s_x = float(scales.get("s_x", s_x))
    if global_img_scale is not None:
        s_x = float(global_img_scale)

    if mask is None:
        return text_attn + s_x * img_attn

    # Prepare mask broadcast
    # Try to broadcast along last dims; if fails, reshape via broadcast_mask_to_attn using inferred feat size
    try:
        m = mask
        # If mask lacks channel, add
        if m.dim() == text_attn.dim() - 1:
            m = m.unsqueeze(-1)
        out = s_y * (m * text_attn) + s_edit * (m * img_attn) + s_non * ((1.0 - m) * img_attn)
        return out
    except Exception:
        # Fallback: infer L from attention shape and build mask accordingly
        attn_shape = text_attn.shape
        if len(attn_shape) == 4:
            b, hheads, L, D = attn_shape
        elif len(attn_shape) == 3:
            b, L, D = attn_shape
            hheads = 1
        else:
            raise ValueError(f"Unsupported attention tensor shape: {attn_shape}")
        m = broadcast_mask_to_attn(mask, attn_shape, feat_hw=None, num_heads=hheads)
        # Broadcast across feature/channel dims
        if len(attn_shape) == 4:
            m = m.expand(b, hheads, L, D)
        else:
            m = m.expand(b, L, D)
        out = s_y * (m * text_attn) + s_edit * (m * img_attn) + s_non * ((1.0 - m) * img_attn)
        return out


__all__ = [
    "ARaM",
    "ARaMScales",
    "ensure_mask_shape",
    "resize_mask",
    "broadcast_mask_to_attn",
    "aram_combine",
]

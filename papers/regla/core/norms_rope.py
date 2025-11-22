"""
Norms and RoPE (Rotary Positional Embeddings) utilities for REGLA.

Implements:
- RMSNorm (stable normalization, no bias)
- LayerNorm wrapper (delegates to torch.nn.LayerNorm)
- StableNorm: configurable norm type used post-attention projection
- RoPE: precompute cosine/sine tables and apply rotary transforms to q,k

Shape conventions used across the project:
- We primarily use [batch, heads, seq_len, head_dim] for attention projections
- Streaming/incremental decoding may use seq_len=1 with a positions tensor

RoPE references:
- Su et al., RoFormer: https://arxiv.org/abs/2104.09864
- Common implementation practice in many libraries (e.g., GPT-NeoX, Llama)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs by their RMS (root mean square) over the last dimension.
    It uses a learnable scale parameter (weight) and no bias by default.

    Args:
        dim: dimension of the last axis to normalize
        eps: epsilon added to denominator for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS over the last dimension
        # rms = sqrt(mean(x^2))
        # Normalize: x / (rms + eps)
        # Scale by learnable weight
        # Shape invariant across batch/head/seq dims.
        orig_dtype = x.dtype
        # Use float32 for stability when in fp16/bf16
        if x.dtype in (torch.float16, torch.bfloat16):
            x_float = x.to(torch.float32)
            rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
            x_norm = (x_float / rms).to(orig_dtype)
        else:
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            x_norm = x / rms
        return x_norm * self.weight


class LayerNorm(nn.Module):
    """LayerNorm wrapper with configurable epsilon and no bias option.

    For stability in attention output, bias is often unnecessary. We keep parity
    with RMSNorm API.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        if not bias:
            # Torch's LayerNorm does not expose bias directly when elementwise_affine=True
            # To simulate no-bias, we set bias parameter to zeros and freeze it.
            with torch.no_grad():
                self.ln.bias.zero_()
            self.ln.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class StableNorm(nn.Module):
    """Configurable normalization used post-attention output projection.

    Args:
        dim: feature dimension
        norm_type: 'rmsnorm' (default) or 'layernorm'
        eps: epsilon for numerical stability
        ln_bias: whether to keep bias when using LayerNorm
    """

    def __init__(self, dim: int, norm_type: str = "rmsnorm", eps: float = 1e-5, ln_bias: bool = True):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(dim, eps=eps)
        elif norm_type == "layernorm":
            self.norm = LayerNorm(dim, eps=eps, bias=ln_bias)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


@dataclass
class RoPECache:
    """Container for precomputed RoPE cos/sin tables.

    Shapes:
        cos: [seq_len, head_dim] or [seq_len, head_dim/2] depending on implementation
        sin: same shape as cos

    We implement the two-by-two rotation (pairwise) formulation where cos/sin are
    defined across head_dim/2 frequencies and broadcast to construct rotations over
    the full head_dim.
    """
    cos: torch.Tensor
    sin: torch.Tensor


def _compute_rope_angles(head_dim: int, base: float = 10000.0, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute RoPE frequency angles (theta) for each pair in the head dimension.

    Returns a 1D tensor of shape [head_dim//2] containing the inverse frequencies.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    half_dim = head_dim // 2
    # Standard RoPE inv_freq
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=dtype, device=device) / half_dim))
    return inv_freq


def precompute_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> RoPECache:
    """Precompute cos/sin tables for RoPE.

    Args:
        seq_len: maximum sequence length for which to precompute tables
        head_dim: per-head dimensionality (must be even)
        base: RoPE base
        device: device for tensors
        dtype: dtype for tables (float32 recommended even if model uses bfloat16)

    Returns:
        RoPECache with cos and sin tensors of shape [seq_len, head_dim//2]
    """
    if dtype is None:
        dtype = torch.float32
    inv_freq = _compute_rope_angles(head_dim, base=base, dtype=dtype, device=device)
    # Positions 0..seq_len-1
    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(-1)  # [seq_len, 1]
    # Compute angles: outer product positions * inv_freq
    angles = positions * inv_freq.unsqueeze(0)  # [seq_len, half_dim]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return RoPECache(cos=cos, sin=sin)


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dim by splitting into pairs (even, odd) and applying 90-degree rotation.

    Given x[..., 2i] and x[..., 2i+1] pairs, returns [-x[..., 2i+1], x[..., 2i]].
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape(x.shape)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cache: RoPECache, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to q and k.

    Args:
        q: query tensor, shape [..., head_dim]
        k: key tensor, shape [..., head_dim]
        cache: RoPECache containing cos/sin tables of shape [seq_len, head_dim//2]
        positions: optional Long/Float tensor specifying position indices for each token.
            If None, we assume the sequence dimension is present and aligned with cache.

    Returns:
        (q_rot, k_rot): rotated q and k tensors with the same shapes as input.

    Broadcasting rules:
        - We construct cos/sin expanded to match q/k shape over the last dimension using pairwise mapping.
        - If positions is provided (e.g., in streaming mode with seq_len=1), we index cos/sin with positions.
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {q.shape[-1]}")
    half_dim = q.shape[-1] // 2

    # Build cos/sin for the current positions
    if positions is not None:
        # positions shape may be [batch, heads, seq] or [seq]
        # We'll flatten positions to 1D indices and then reshape cos/sin accordingly.
        # Ensure positions is integer type for indexing
        if positions.dtype not in (torch.int32, torch.int64):
            positions = positions.to(torch.int64)
        # Index cos/sin: result [*positions_shape, half_dim]
        cos = cache.cos.index_select(0, positions.reshape(-1)).reshape(*positions.shape, half_dim)
        sin = cache.sin.index_select(0, positions.reshape(-1)).reshape(*positions.shape, half_dim)
    else:
        # Assume q/k have explicit seq dimension matching cache length.
        # We try to infer seq_len from either third from last dim or create broadcast.
        # Common shapes: [B, H, S, D]
        # We'll expand cos/sin to [1, 1, S, half_dim] for broadcasting.
        seq_len = q.shape[-2] if q.dim() >= 3 else cache.cos.shape[0]
        cos = cache.cos[:seq_len].view(*([1] * (q.dim() - 2)), seq_len, half_dim)
        sin = cache.sin[:seq_len].view(*([1] * (q.dim() - 2)), seq_len, half_dim)

    # Expand cos/sin to pairwise last-dim: interleave to match head_dim via [half_dim] -> [head_dim] with even/odd positions.
    # We construct cos_pair and sin_pair such that:
    #   cos_pair[..., 2i] = cos[..., i]
    #   cos_pair[..., 2i+1] = cos[..., i]
    # same for sin.
    cos_pair = torch.repeat_interleave(cos, repeats=2, dim=-1)
    sin_pair = torch.repeat_interleave(sin, repeats=2, dim=-1)

    # Apply rotation: [x * cos + rotate_half(x) * sin]
    q_rot = (q * cos_pair) + (_rope_rotate_half(q) * sin_pair)
    k_rot = (k * cos_pair) + (_rope_rotate_half(k) * sin_pair)
    return q_rot, k_rot


def infer_positions(seq_len: int, start_pos: int = 0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Utility to create positions tensor [seq_len] for non-streaming forward passes."""
    return torch.arange(start_pos, start_pos + seq_len, device=device)


__all__ = [
    "RMSNorm",
    "LayerNorm",
    "StableNorm",
    "RoPECache",
    "precompute_rope_cache",
    "apply_rope",
    "infer_positions",
]

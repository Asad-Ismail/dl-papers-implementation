# Copyright (c) 2025
# REGLA: Refining Gated Linear Attention - Feature Maps and Variance Scaling
#
# This module implements: 
#  - Safe exponential feature maps for queries and keys with running-max stabilization
#  - Variance reduction scaling (alpha) as in the REGLA paper reproduction plan
#  - Baseline nonnegative feature maps (ELU+1, ReLU)
#
# Expected tensor layout for q_raw/k_raw throughout this module is (B, T, H, M), where:
#  - B: batch size
#  - T: sequence length (use T=1 for single-step inference)
#  - H: number of heads
#  - M: feature dimension m used by linear attention features
# The functions are written to be permissive with shapes; if T is missing, it will be
# broadcast/unsqueezed where sensible, but the canonical layout should be respected by callers.

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch

__all__ = [
    "safe_exp_query",
    "safe_exp_key",
    "apply_variance_scale",
    "variance_alpha",
    "elu1_feature_map",
    "relu_feature_map",
    "compute_rebase_scale",
]


def _to_float(x: torch.Tensor) -> torch.Tensor:
    """Do computations in float32 for improved numerical stability where needed."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def _ensure_bthm(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has shape (B, T, H, M).
    Accepts (B, H, M) or (B, 1, H, M) or (B, T, H, M). Returns a view with T dim.
    """
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        # (B, H, M) -> (B, 1, H, M)
        return x.unsqueeze(1)
    raise ValueError(f"Expected tensor of shape (B, T, H, M) or (B, H, M); got {tuple(x.shape)}")


def safe_exp_query(q_raw: torch.Tensor) -> torch.Tensor:
    """
    Safe exponential feature map for queries.
    φ_q(q_raw) = exp(q_raw − max_dim(q_raw)) computed per token and head, along the feature dim M.

    Args:
        q_raw: Tensor of shape (B, T, H, M) or (B, H, M).
    Returns:
        phi_q: Tensor of same shape as q_raw, nonnegative and elementwise bounded by 1.
    """
    q = _ensure_bthm(q_raw)
    q32 = _to_float(q)
    q_max = q32.amax(dim=-1, keepdim=True)
    phi_q32 = torch.exp(q32 - q_max)
    phi_q = phi_q32.to(q.dtype)
    # Return in the same rank as input
    return phi_q if q_raw.dim() == 4 else phi_q.squeeze(1)




def safe_exp_key(
    k_raw: torch.Tensor,
    running_max: Optional[torch.Tensor] = None,
    training: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe exponential feature map for keys with global (training) or running (inference) max.

    Training/global variant (default when running_max is None):
      φ_k = exp(k_raw − max_{t<=T, dim}(k_raw)) with a single per-(B,H) scalar baseline shared across T.

    Inference/running variant (when running_max is provided):
      Given running_max (per-(B,H) scalar baseline possibly with singleton dims), update
      new_max = max(running_max, max_dim(k_t)) and compute φ_k = exp(k_raw − new_max).

    Args:
        k_raw: Tensor of shape (B, T, H, M) or (B, H, M) or (B, 1, H, M).
        running_max: Optional tensor baseline for inference; shape broadcastable to (B, T, H, 1).
        training: Optional flag to force training/global behavior when True.
    Returns:
        (phi_k, new_max):
            phi_k: Same shape as k_raw, nonnegative and elementwise bounded by 1.
            new_max: Tensor baseline with shape (B, T, H, 1) for training=False streaming, or (B, 1, H, 1) for training/global.
    """
    k = _ensure_bthm(k_raw)
    k32 = _to_float(k)
    # Per time-step max across feature dim
    k_dim_max = k32.amax(dim=-1, keepdim=True)  # (B, T, H, 1)

    if running_max is None or (training is True):
        # Global per-sequence, per-head max across time and dim
        global_max = k_dim_max.amax(dim=1, keepdim=True)  # (B, 1, H, 1)
        phi_k32 = torch.exp(k32 - global_max)
        phi_k = phi_k32.to(k.dtype)
        return (phi_k if k_raw.dim() == 4 else phi_k.squeeze(1), global_max)
    else:
        # Inference streaming: update running max pointwise with current step(s)
        # running_max is expected broadcastable to (B, T, H, 1)
        new_max = torch.maximum(k_dim_max, running_max)
        phi_k32 = torch.exp(k32 - new_max)
        phi_k = phi_k32.to(k.dtype)
        return (phi_k if k_raw.dim() == 4 else phi_k.squeeze(1), new_max)


def compute_rebase_scale(prev_max: torch.Tensor, new_max: torch.Tensor) -> torch.Tensor:
    """
    Compute multiplicative scale to rebase accumulated states when the running max increases.
    scale = exp(prev_max - new_max). Should be broadcastable to state shapes (B, H, M) or (B, 1, H, 1).
    """
    prev32 = _to_float(prev_max)
    new32 = _to_float(new_max)
    return torch.exp(prev32 - new32).to(new_max.dtype)


def variance_alpha(m: int | torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Compute variance-reduction scaling constant alpha as:
      α = 1 / (e * sqrt(m * (e^2 − 1)))
    Args:
      m: feature dimension m (int or tensor)
      device, dtype: optional torch device and dtype for returned tensor
    Returns:
      alpha: 0-D or 1-D tensor broadcastable to feature tensors
    """
    e2_minus_1 = math.e ** 2 - 1.0
    if isinstance(m, torch.Tensor):
        m_t = m.to(dtype=torch.float32)
        alpha_val = 1.0 / (math.e * torch.sqrt(m_t * e2_minus_1))
        return alpha_val.to(device=device if device is not None else m.device, dtype=dtype or torch.float32)
    else:
        alpha_val = 1.0 / (math.e * math.sqrt(float(m) * e2_minus_1))
        t = torch.tensor(alpha_val, device=device, dtype=dtype or torch.float32)
        return t


def apply_variance_scale(phi: torch.Tensor, m: int) -> torch.Tensor:
    """
    Apply variance scaling alpha to a feature tensor phi along the last dimension (feature dim M).
    α = 1 / (e * sqrt(m * (e^2 − 1)))
    Args:
      phi: Tensor (..., M)
      m: feature dimension M
    Returns:
      phi_scaled: Tensor of same shape and dtype as phi
    """
    alpha = variance_alpha(m, device=phi.device, dtype=torch.float32)
    # Multiply in higher precision when mixed precision for stability
    phi32 = _to_float(phi)
    out = (phi32 * alpha).to(phi.dtype)
    return out


def elu1_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Baseline feature map: ELU(x) + 1. Ensures nonnegativity.
    Args:
      x: Tensor (..., M)
    Returns:
      phi: Tensor of same shape ≥ 0
    """
    x32 = _to_float(x)
    phi32 = torch.nn.functional.elu(x32) + 1.0
    return phi32.to(x.dtype)


def relu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Baseline feature map: ReLU(x). Ensures nonnegativity.
    Args:
      x: Tensor (..., M)
    Returns:
      phi: Tensor of same shape ≥ 0
    """
    return torch.nn.functional.relu(x)

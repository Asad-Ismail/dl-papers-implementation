# Copyright (c) 2025 REGLA
# Core gating mechanism (Refined Gate) and fast-weight recurrence utilities
# Implements per-head, per-channel gating and REGLA state updates
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "RefinedGate",
    "ScalarGate",
    "regla_step",
    "init_state",
]


def _ensure_4d(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure input has shape (B, T, H, D) by inserting T dim if missing.
    Returns the possibly reshaped tensor and a flag whether T was originally present.
    """
    if x.dim() == 4:
        return x, True
    if x.dim() == 3:
        # (B, H, D) -> (B, 1, H, D)
        return x.unsqueeze(1), False
    raise ValueError(f"Expected tensor rank 3 or 4, got {x.shape}")


class RefinedGate(nn.Module):
    """Refined gating mechanism for REGLA.

    For each head and channel (d_head), computes two sigmoid activations g and r:
        g = sigmoid(W_g v + b_g)
        r = sigmoid(W_r v + b_r)
    and mixes a lower/upper envelope to produce a forget factor per channel:
        lower = g^2
        upper = 1 - (1 - g)^2
        mix = (1 - r) * lower + r * upper   in (0, 1)

    The resulting per-channel mix (B, [T,] H, d_head) is broadcast over the feature
    dimension m when updating the fast-weight state S in the recurrence.

    Two modes are supported:
      - per-head parameters (default): separate (W, b) for each head
      - shared across heads: single Linear shared among heads
    """

    def __init__(
        self,
        d_head: int,
        n_heads: int,
        share_across_heads: bool = False,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_head = int(d_head)
        self.n_heads = int(n_heads)
        self.share = bool(share_across_heads)

        if self.share:
            self.lin_g = nn.Linear(d_head, d_head, bias=True)
            self.lin_r = nn.Linear(d_head, d_head, bias=True)
            nn.init.xavier_uniform_(self.lin_g.weight)
            nn.init.xavier_uniform_(self.lin_r.weight)
            nn.init.constant_(self.lin_g.bias, bias_init)
            nn.init.constant_(self.lin_r.bias, bias_init)
        else:
            # Per-head parameter tensors
            self.weight_g = nn.Parameter(torch.empty(n_heads, d_head, d_head))
            self.bias_g = nn.Parameter(torch.full((n_heads, d_head), float(bias_init)))
            self.weight_r = nn.Parameter(torch.empty(n_heads, d_head, d_head))
            self.bias_r = nn.Parameter(torch.full((n_heads, d_head), float(bias_init)))
            # Xavier init per head
            nn.init.xavier_uniform_(self.weight_g)
            nn.init.xavier_uniform_(self.weight_r)

    def forward(
        self,
        v: torch.Tensor,
        return_gr: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute refined forget mix from value vectors v.

        Args:
            v: Tensor shaped (B, H, d_head) or (B, T, H, d_head)
            return_gr: If True, also return g and r for logging
        Returns:
            mix: (B, H, d_head) or (B, T, H, d_head) aligned with input rank
            g, r: same shape as mix if return_gr else (None, None)
        """
        v4, had_T = _ensure_4d(v)
        B, T, H, D = v4.shape
        assert H == self.n_heads and D == self.d_head, (
            f"Expected v last dims (H={self.n_heads}, D={self.d_head}), got {v4.shape}"
        )

        if self.share:
            # Apply shared linear per position and head independently
            v_flat = v4.reshape(B * T * H, D)
            g = torch.sigmoid(self.lin_g(v_flat))
            r = torch.sigmoid(self.lin_r(v_flat))
            g = g.view(B, T, H, D)
            r = r.view(B, T, H, D)
        else:
            # Batched per-head matmul: (B,T,H,D) x (H,D,D) -> (B,T,H,D)
            # Using einsum for clarity and good perf on small D
            g_preact = torch.einsum("bthd,hde->bthe", v4, self.weight_g) + self.bias_g.view(1, 1, H, D)
            r_preact = torch.einsum("bthd,hde->bthe", v4, self.weight_r) + self.bias_r.view(1, 1, H, D)
            g = torch.sigmoid(g_preact)
            r = torch.sigmoid(r_preact)

        lower = g * g
        one_minus_g = 1.0 - g
        upper = 1.0 - (one_minus_g * one_minus_g)
        mix = (1.0 - r) * lower + r * upper

        if not had_T:
            mix = mix.squeeze(1)
            g = g.squeeze(1)
            r = r.squeeze(1)
        if return_gr:
            return mix, g, r
        return mix, None, None


class ScalarGate(nn.Module):
    """A baseline per-channel sigmoid gate that can be used in ablations.

    Computes g = sigmoid(W v + b) per head and channel and returns g to be
    broadcast along the feature dimension m for state decay.
    """

    def __init__(self, d_head: int, n_heads: int, share_across_heads: bool = False, bias_init: float = 0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.share = share_across_heads
        if self.share:
            self.lin = nn.Linear(d_head, d_head, bias=True)
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.constant_(self.lin.bias, bias_init)
        else:
            self.weight = nn.Parameter(torch.empty(n_heads, d_head, d_head))
            self.bias = nn.Parameter(torch.full((n_heads, d_head), float(bias_init)))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v4, had_T = _ensure_4d(v)
        B, T, H, D = v4.shape
        if self.share:
            v_flat = v4.reshape(B * T * H, D)
            g = torch.sigmoid(self.lin(v_flat)).view(B, T, H, D)
        else:
            g = torch.sigmoid(
                torch.einsum("bthd,hde->bthe", v4, self.weight) + self.bias.view(1, 1, H, D)
            )
        if not had_T:
            g = g.squeeze(1)
        return g


@torch.no_grad()
def init_state(
    batch_size: int,
    n_heads: int,
    d_head: int,
    m: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    init_max: float = float("-inf"),
) -> Dict[str, torch.Tensor]:
    """Initialize fast-weight state and auxiliary buffers.

    Returns a dictionary with:
      - S: (B, H, d_head, m) zero state
      - c: (B, H, m) zero normalization accumulator (for sum-norm mode)
      - k_running_max: (B, H, 1) initialized to init_max (for safe-exp keys)
    """
    dev = device
    dt = dtype if dtype is not None else torch.float32
    S = torch.zeros(batch_size, n_heads, d_head, m, device=dev, dtype=dt)
    c = torch.zeros(batch_size, n_heads, m, device=dev, dtype=dt)
    k_running_max = torch.full((batch_size, n_heads, 1), init_max, device=dev, dtype=dt)
    return {"S": S, "c": c, "k_running_max": k_running_max}


def regla_step(
    S_prev: torch.Tensor,
    v_t: torch.Tensor,
    phi_k_t: torch.Tensor,
    phi_q_t: torch.Tensor,
    mix_t: torch.Tensor,
    *,
    sum_norm: bool = False,
    c_prev: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    rebase_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """One recurrent step update for REGLA fast weights.

    Args:
      S_prev: (B, H, d_head, m) previous state
      v_t: (B, H, d_head) value for current token
      phi_k_t: (B, H, m) key feature map for current token
      phi_q_t: (B, H, m) query feature map for current token (already variance-scaled if desired)
      mix_t: (B, H, d_head) refined per-channel forget factor in (0,1)
      sum_norm: whether to maintain and apply sum normalization (off by default)
      c_prev: (B, H, m) previous normalizer if sum_norm is True
      eps: numerical epsilon for division stability
      rebase_scale: optional scale factor for rebasing when running-max increases; shape (B, H, 1) or (B, H, 1, 1)

    Returns:
      y_t: (B, H, d_head) head outputs for current token
      S_new: (B, H, d_head, m) updated state
      c_new: (B, H, m) updated normalizer if sum_norm else None
    """
    B, H, D, M = S_prev.shape

    # Optional rebase when the key running max increased: scale S and c accordingly
    if rebase_scale is not None:
        # Shapes: rebase_scale (B,H,1) -> (B,H,1,1) and (B,H,1)
        if rebase_scale.dim() == 3:
            scale_S = rebase_scale.unsqueeze(-1)
            scale_c = rebase_scale
        elif rebase_scale.dim() == 4:
            scale_S = rebase_scale
            scale_c = rebase_scale.squeeze(-1)
        else:
            raise ValueError("rebase_scale must have dim 3 or 4")
        S_prev = S_prev * scale_S
        if sum_norm and c_prev is not None:
            c_prev = c_prev * scale_c

    # Decay previous state with per-channel mix (broadcast along feature dim m)
    S_decay = S_prev * mix_t.unsqueeze(-1)

    # Outer product v_t ⊗ phi_k_t^T -> (B,H,D,M)
    outer = torch.einsum("bhd,bhm->bhdm", v_t, phi_k_t)

    S_new = S_decay + outer

    # y_t = S_new @ phi_q_t
    y_t = torch.einsum("bhdm,bhm->bhd", S_new, phi_q_t)

    if sum_norm:
        if c_prev is None:
            raise ValueError("c_prev must be provided when sum_norm=True")
        c_new = c_prev + phi_k_t
        denom = torch.einsum("bhm,bhm->bh", c_new, phi_q_t).unsqueeze(-1)
        y_t = y_t / (denom + eps)
        return y_t, S_new, c_new

    return y_t, S_new, None

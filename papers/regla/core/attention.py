from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_maps import (
    safe_exp_query,
    safe_exp_key,
    apply_variance_scale,
    elu1_feature_map,
    relu_feature_map,
    variance_alpha,
    compute_rebase_scale,
)
from .gating_and_recurrence import (
    RefinedGate,
    ScalarGate,
    init_state as init_fw_state,
    regla_step,
)
from .norms_rope import apply_rope, RoPECache, StableNorm


def _shape_projection(x: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    # x: (B, T, d_model)
    B, T, _ = x.shape
    return x.view(B, T, n_heads, d_head)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, D)
    B, T, H, D = x.shape
    return x.contiguous().view(B, T, H * D)


class ReGLA(nn.Module):
    """
    REGLA multi-head attention module.

    - Linear-time attention via safe exp feature maps and refined gating with fast-weight state.
    - Drops sum normalization by default and stabilizes with StableNorm; optionally enable sum norm.
    - Supports RoPE on q,k; variance reduction scaling alpha.

    State dict keys:
      - S: (B, H, d_head, m)
      - c: (B, H, m) if sum_norm True
      - k_running_max: (B, H, 1)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        m: int = 64,
        rope: bool = True,
        stable_norm: str = "rmsnorm",
        stable_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        use_sum_norm: bool = False,
        alpha_scaling: bool = True,
        gate_share_across_heads: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.m = m
        self.rope_enabled = rope
        self.use_sum_norm = use_sum_norm
        self.alpha_scaling = alpha_scaling

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.gate = RefinedGate(d_head=d_head, n_heads=n_heads, share_across_heads=gate_share_across_heads)

        # StableNorm after output projection
        self.out_norm = StableNorm(d_model, norm_type=stable_norm, eps=stable_norm_eps)
        self.out_drop = nn.Dropout(dropout)

        # alpha for variance scaling applied to phi_q only (mathematically equivalent to symmetric split)
        alpha = variance_alpha(m)
        self.register_buffer("alpha", alpha.to(dtype=torch.float32), persistent=False)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        return init_fw_state(batch_size, self.n_heads, self.d_head, self.m, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        rope_cache: Optional[RoPECache] = None,
        positions: Optional[torch.Tensor] = None,
        return_state: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Projections
        q = _shape_projection(self.W_q(x), self.n_heads, self.d_head)  # (B,T,H,D)
        k = _shape_projection(self.W_k(x), self.n_heads, self.d_head)
        v = _shape_projection(self.W_v(x), self.n_heads, self.d_head)

        # RoPE on q,k if enabled
        if self.rope_enabled and rope_cache is not None:
            q, k = apply_rope(q, k, rope_cache, positions=positions)

        # Prepare initial state
        if state is None:
            state = self.init_state(B, device=device, dtype=torch.float32)
        S = state["S"]  # (B,H,D,m) float32 accumulator preferred
        c = state.get("c", None) if self.use_sum_norm else None
        k_running_max = state.get("k_running_max", torch.full((B, self.n_heads, 1), float("-inf"), device=device, dtype=torch.float32))

        y_list = []
        new_S = S
        new_c = c
        new_running_max = k_running_max

        for t in range(T):
            q_t = q[:, t]  # (B,H,D)
            k_t = k[:, t]  # (B,H,D)
            v_t = v[:, t]  # (B,H,D)

            # Feature maps
            phi_q_t = safe_exp_query(q_t)
            # apply alpha scaling to phi_q
            if self.alpha_scaling:
                phi_q_t = apply_variance_scale(phi_q_t, self.m)
            # Keys with running max (streaming-style); during training this still works stepwise
            phi_k_t, step_max = safe_exp_key(k_t, running_max=new_running_max)
            # compute rebase scale if running max increased
            rebase = compute_rebase_scale(new_running_max, step_max)  # (B,H,1)
            new_running_max = step_max  # (B,H,1)

            # Gate from v_t
            mix_t = self.gate(v_t)  # (B,H,D)

            # Recurrence step
            y_t, new_S, new_c = regla_step(
                new_S,
                v_t,
                phi_k_t,
                phi_q_t,
                mix_t,
                sum_norm=self.use_sum_norm,
                c_prev=new_c,
                eps=eps,
                rebase_scale=rebase.unsqueeze(-1),  # (B,H,1,1)
            )

            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)  # (B,T,H,D)
        y = _merge_heads(y)  # (B,T,HD)
        y = self.W_o(y)
        y = self.out_drop(y)
        y = self.out_norm(y)

        new_state = {
            "S": new_S,
            "k_running_max": new_running_max,
        }
        if self.use_sum_norm:
            new_state["c"] = new_c

        if return_state:
            return y, new_state
        else:
            return y, None


class FastDecayGLA(nn.Module):
    """
    Fast Decay Gated Linear Attention baseline.

    S_t = G_t ⊙ S_{t-1} + v_t ⊗ phi_k_t^T
    with G_t = sigma(W_z v_t) outer sigma(W_f v_t)^T producing (D x m) gate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        m: int = 64,
        rope: bool = True,
        stable_norm: str = "rmsnorm",
        stable_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.m = m
        self.rope_enabled = rope

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        # Gating projections per head
        self.W_z = nn.Linear(d_head, d_head, bias=True)
        self.W_f = nn.Linear(d_head, m, bias=True)

        self.out_norm = StableNorm(d_model, norm_type=stable_norm, eps=stable_norm_eps)
        self.out_drop = nn.Dropout(dropout)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        return init_fw_state(batch_size, self.n_heads, self.d_head, self.m, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        rope_cache: Optional[RoPECache] = None,
        positions: Optional[torch.Tensor] = None,
        return_state: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, _ = x.shape
        device = x.device

        q = _shape_projection(self.W_q(x), self.n_heads, self.d_head)
        k = _shape_projection(self.W_k(x), self.n_heads, self.d_head)
        v = _shape_projection(self.W_v(x), self.n_heads, self.d_head)

        if self.rope_enabled and rope_cache is not None:
            q, k = apply_rope(q, k, rope_cache, positions=positions)

        if state is None:
            state = self.init_state(B, device=device, dtype=torch.float32)
        S = state["S"]
        k_running_max = state.get("k_running_max", torch.full((B, self.n_heads, 1), float("-inf"), device=device, dtype=torch.float32))

        y_list = []
        new_S = S
        new_running_max = k_running_max

        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]

            phi_q_t = safe_exp_query(q_t)
            phi_k_t, step_max = safe_exp_key(k_t, running_max=new_running_max)
            rebase = compute_rebase_scale(new_running_max, step_max)
            new_running_max = step_max

            # G_t = sigma(W_z v) outer sigma(W_f v)^T -> (B,H,D,m)
            z = torch.sigmoid(self.W_z(v_t))  # (B,H,D)
            f = torch.sigmoid(self.W_f(v_t))  # (B,H,m)
            G = z.unsqueeze(-1) * f.unsqueeze(-2)  # (B,H,D,m)

            if rebase is not None:
                new_S = new_S * rebase.unsqueeze(-1)  # (B,H,1) -> (B,H,1,1)

            # Update state
            outer = torch.einsum("bhd,bhm->bhdm", v_t, phi_k_t)
            new_S = G * new_S + outer
            # Readout
            y_t = torch.einsum("bhdm,bhm->bhd", new_S, phi_q_t)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = _merge_heads(y)
        y = self.W_o(y)
        y = self.out_drop(y)
        y = self.out_norm(y)

        new_state = {
            "S": new_S,
            "k_running_max": new_running_max,
        }

        if return_state:
            return y, new_state
        else:
            return y, None


class _LABase(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        m: int = 64,
        rope: bool = True,
        dropout: float = 0.0,
        sum_norm: bool = True,
        feature_map: str = "elu1",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.m = m
        self.rope_enabled = rope
        self.sum_norm = sum_norm
        assert feature_map in {"elu1", "relu"}
        self.feature_map = feature_map

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.out_drop = nn.Dropout(dropout)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        # Use same layout as fast-weight state for compatibility
        S = torch.zeros(batch_size, self.n_heads, self.d_head, self.m, device=device, dtype=torch.float32)
        c = torch.zeros(batch_size, self.n_heads, self.m, device=device, dtype=torch.float32)
        return {"S": S, "c": c}

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu1":
            return elu1_feature_map(x)
        else:
            return relu_feature_map(x)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        rope_cache: Optional[RoPECache] = None,
        positions: Optional[torch.Tensor] = None,
        return_state: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, _ = x.shape
        device = x.device

        q = _shape_projection(self.W_q(x), self.n_heads, self.d_head)
        k = _shape_projection(self.W_k(x), self.n_heads, self.d_head)
        v = _shape_projection(self.W_v(x), self.n_heads, self.d_head)

        if self.rope_enabled and rope_cache is not None:
            q, k = apply_rope(q, k, rope_cache, positions=positions)

        if state is None:
            state = self.init_state(B, device=device, dtype=torch.float32)
        S = state["S"]
        c = state["c"]

        y_list = []
        for t in range(T):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            phi_q_t = self._phi(q_t)
            phi_k_t = self._phi(k_t)

            # Update sums
            S = S + torch.einsum("bhd,bhm->bhdm", v_t, phi_k_t)
            c = c + phi_k_t

            # Readout with sum norm
            denom = torch.einsum("bhm,bhm->bh", c, phi_q_t).unsqueeze(-1)  # (B,H,1)
            num = torch.einsum("bhdm,bhm->bhd", S, phi_q_t)
            y_t = num / (denom + eps)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = _merge_heads(y)
        y = self.W_o(y)
        y = self.out_drop(y)
        new_state = {"S": S, "c": c}
        if return_state:
            return y, new_state
        else:
            return y, None


class LAELU1Attention(_LABase):
    def __init__(self, d_model: int, n_heads: int, d_head: int, m: int = 64, rope: bool = True, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, d_head, m=m, rope=rope, dropout=dropout, sum_norm=True, feature_map="elu1")


class LAReLUAttention(_LABase):
    def __init__(self, d_model: int, n_heads: int, d_head: int, m: int = 64, rope: bool = True, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, d_head, m=m, rope=rope, dropout=dropout, sum_norm=True, feature_map="relu")


class SoftmaxMHA(nn.Module):
    """Standard scaled dot-product multi-head attention with KV cache for streaming."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        rope: bool = True,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        self.rope_enabled = rope

        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        # KV cache tensors with zero length initially
        return {"k_cache": None, "v_cache": None, "length": torch.zeros(batch_size, dtype=torch.long, device=device)}

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        rope_cache: Optional[RoPECache] = None,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, _ = x.shape
        device = x.device

        q = _shape_projection(self.W_q(x), self.n_heads, self.d_head)
        k = _shape_projection(self.W_k(x), self.n_heads, self.d_head)
        v = _shape_projection(self.W_v(x), self.n_heads, self.d_head)

        if self.rope_enabled and rope_cache is not None:
            q, k = apply_rope(q, k, rope_cache, positions=positions)

        if state is not None and state.get("k_cache") is not None:
            # Append to cache
            k_cache = state["k_cache"]  # (B,H,L,D)
            v_cache = state["v_cache"]  # (B,H,L,D)
            k = torch.cat([k_cache, k.transpose(1, 2)], dim=2)  # to (B,H,L+T,D)
            v = torch.cat([v_cache, v.transpose(1, 2)], dim=2)
        else:
            k = k.transpose(1, 2)  # (B,H,T,D)
            v = v.transpose(1, 2)

        q = q.transpose(1, 2)  # (B,H,T,D)
        L = k.size(2)

        # Causal mask
        causal = torch.tril(torch.ones(T, L, device=device, dtype=torch.bool))
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        y = torch.einsum("bhij,bhjd->bhid", attn_probs, v)  # (B,H,T,D)
        y = y.transpose(1, 2)
        y = _merge_heads(y)
        y = self.W_o(y)
        y = self.resid_drop(y)

        new_state = None
        if return_state:
            new_state = {
                "k_cache": k,
                "v_cache": v,
            }
        return y, new_state


__all__ = [
    "ReGLA",
    "FastDecayGLA",
    "LAELU1Attention",
    "LAReLUAttention",
    "SoftmaxMHA",
]

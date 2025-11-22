from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .norms_rope import RMSNorm, LayerNorm as _LayerNorm
from .attention import ReGLA, FastDecayGLA, LAELU1Attention, LAReLUAttention, SoftmaxMHA
from .norms_rope import RoPECache

__all__ = ["MLP", "DecoderBlock", "build_attention"]


class MLP(nn.Module):
    """Feed-forward block: Linear -> GELU -> Dropout -> Linear.

    Args:
        d_model: model dimension (input and output)
        hidden_dim: expansion dimension (e.g., 4 * d_model)
        dropout: dropout probability applied after activation and final projection
        activation: activation function name (gelu or silu). Defaults to gelu.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        act = activation.lower()
        if act == "gelu":
            self.act = F.gelu
        elif act == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def build_attention(
    attn_type: str,
    d_model: int,
    n_heads: int,
    d_head: int,
    *,
    m: int = 64,
    rope: bool = True,
    dropout: float = 0.0,
    stable_norm: str = "rmsnorm",
    stable_norm_eps: float = 1e-5,
    use_sum_norm: bool = False,
    alpha_scaling: bool = True,
    gate_share_across_heads: bool = False,
) -> nn.Module:
    """Factory to build attention module according to attn_type.

    Supported attn_type values: "regla", "fast_decay", "la_elu1", "la_relu", "softmax".
    """
    attn_type = attn_type.lower()
    if attn_type == "regla":
        return ReGLA(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            m=m,
            rope=rope,
            stable_norm=stable_norm,
            stable_norm_eps=stable_norm_eps,
            dropout=dropout,
            use_sum_norm=use_sum_norm,
            alpha_scaling=alpha_scaling,
            gate_share_across_heads=gate_share_across_heads,
        )
    elif attn_type == "fast_decay":
        return FastDecayGLA(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            m=m,
            rope=rope,
            stable_norm=stable_norm,
            stable_norm_eps=stable_norm_eps,
            dropout=dropout,
        )
    elif attn_type == "la_elu1":
        return LAELU1Attention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            m=m,
            rope=rope,
            dropout=dropout,
        )
    elif attn_type == "la_relu":
        return LAReLUAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            m=m,
            rope=rope,
            dropout=dropout,
        )
    elif attn_type == "softmax":
        return SoftmaxMHA(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            rope=rope,
            attn_dropout=dropout,
            resid_dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported attention type: {attn_type}")


class DecoderBlock(nn.Module):
    """PreNorm decoder block with configurable attention module and MLP.

    Structure:
      x -> Norm1 -> Attention -> Dropout -> Residual -> Norm2 -> MLP -> Dropout -> Residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        mlp_hidden_dim: int,
        *,
        attn_type: str = "regla",
        m: int = 64,
        rope: bool = True,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        stable_norm: str = "rmsnorm",
        stable_norm_eps: float = 1e-5,
        use_sum_norm: bool = False,
        alpha_scaling: bool = True,
        gate_share_across_heads: bool = False,
        mlp_activation: str = "gelu",
    ):
        super().__init__()
        self.attn_type = attn_type.lower()
        # PreNorms
        if norm_type.lower() == "rmsnorm":
            self.norm1 = RMSNorm(d_model, eps=norm_eps)
            self.norm2 = RMSNorm(d_model, eps=norm_eps)
        elif norm_type.lower() == "layernorm":
            self.norm1 = _LayerNorm(d_model, eps=norm_eps, bias=True)
            self.norm2 = _LayerNorm(d_model, eps=norm_eps, bias=True)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Attention
        self.attn = build_attention(
            attn_type=self.attn_type,
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            m=m,
            rope=rope,
            dropout=dropout,
            stable_norm=stable_norm,
            stable_norm_eps=stable_norm_eps,
            use_sum_norm=use_sum_norm,
            alpha_scaling=alpha_scaling,
            gate_share_across_heads=gate_share_across_heads,
        )
        self.attn_drop = nn.Dropout(dropout)

        # MLP
        self.mlp = MLP(d_model, mlp_hidden_dim, dropout=dropout, activation=mlp_activation)
        self.mlp_drop = nn.Dropout(dropout)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Optional[Dict[str, torch.Tensor]]:
        """Initialize and return the attention submodule state for streaming.

        Some attention types (e.g., SoftmaxMHA) will return KV cache dict; REGLA/GLA return fast-weight state dict.
        """
        if hasattr(self.attn, "init_state"):
            return self.attn.init_state(batch_size=batch_size, device=device, dtype=dtype)
        return None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        rope_cache: Optional[RoPECache] = None,
        positions: Optional[torch.Tensor] = None,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # PreNorm -> Attention
        h = self.norm1(x)
        if hasattr(self.attn, "forward"):
            attn_kwargs: Dict[str, Any] = {
                "state": state,
                "rope_cache": rope_cache,
                "positions": positions,
                "return_state": return_state,
            }
            y, new_state = self.attn(h, **attn_kwargs)
        else:
            raise RuntimeError("Attention module lacks a forward method")
        y = self.attn_drop(y)
        x = x + y

        # PreNorm -> MLP
        h2 = self.norm2(x)
        y2 = self.mlp(h2)
        y2 = self.mlp_drop(y2)
        x = x + y2
        return (x, new_state) if return_state else (x, None)

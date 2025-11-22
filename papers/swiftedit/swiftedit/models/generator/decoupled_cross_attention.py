# Copyright (c) 2025 SwiftEdit Authors
# Decoupled cross-attention for text and image conditioning
# Implements a multi-head scaled dot-product attention that produces
# separate outputs for text tokens and image tokens given shared queries.

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in {"float16", "fp16"}:
        return torch.float16
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if s in {"float32", "fp32"}:
        return torch.float32
    return None


class DecoupledCrossAttention(nn.Module):
    """
    Multi-head scaled dot-product attention that computes decoupled cross-attention
    for text and image conditioning streams using the same query tensor.

    Inputs:
      - q: (B, L, D) queries or (B, Hh, L, Dh) if already split into heads
      - k_y, v_y: text keys/values of shape (B, Ny, D) or (B, Hh, Ny, Dh)
      - k_x, v_x: image keys/values of shape (B, Nx, D) or (B, Hh, Nx, Dh)

    Outputs:
      - dict with keys:
        * "text": attended features from text branch (B, L, D)
        * "image": attended features from image branch (B, L, D) (zeros if k_x/v_x is None)
        * optionally, attention weights if need_weights=True: "attn_text", "attn_image" of shape (B, Hh, L, N)

    Notes:
      - If input tensors are not split into heads, internal linear projections map to per-head tensors.
      - If image tokens are missing (None or Nx == 0), the image output is zeros.
      - An optional attention mask can be provided to bias logits before softmax.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        project_inputs: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.project_inputs = project_inputs

        # Projections for queries and per-branch KV when inputs are (B, *, D)
        factory_kwargs = {"device": device, "dtype": dtype}
        if project_inputs:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.k_proj_y = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.v_proj_y = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.k_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.v_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        else:
            self.q_proj = None
            self.k_proj_y = None
            self.v_proj_y = None
            self.k_proj_x = None
            self.v_proj_x = None

        # Output projections to merge heads back to D
        self.out_proj_text = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj_image = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Dropout layers
        self.attn_drop = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    @staticmethod
    def _is_head_split(t: torch.Tensor, num_heads: int) -> bool:
        # Accept shapes (B, Hh, L, Dh) or (B, Hh, N, Dh)
        return t.dim() == 4 and t.shape[1] == num_heads

    def _reshape_to_heads(self, x: torch.Tensor, is_kv: bool = False) -> torch.Tensor:
        """Reshape (B, T, D) -> (B, Hh, T, Dh) or pass-through if already (B, Hh, T, Dh)."""
        if self._is_head_split(x, self.num_heads):
            return x
        B, T, D = x.shape
        assert D == self.embed_dim, f"Expected last dim {self.embed_dim}, got {D}"
        x = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
        self,
        q: torch.Tensor,
        k_y: torch.Tensor,
        v_y: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute decoupled cross-attention outputs.

        Args:
            q: Queries (B, L, D) or (B, Hh, L, Dh)
            k_y, v_y: Text keys/values (B, Ny, D) or (B, Hh, Ny, Dh)
            k_x, v_x: Image keys/values (B, Nx, D) or (B, Hh, Nx, Dh); can be None
            attn_mask: Optional mask added to logits before softmax. Shapes supported:
                - (B, 1, L, 1) broadcastable to (B, Hh, L, N)
                - (B, 1, 1, N) broadcastable per-token
                - (B, Hh, L, N) exact
            need_weights: If True, include attention weights in output dict.

        Returns:
            dict with keys "text", "image" and optionally "attn_text", "attn_image".
        """
        B = q.shape[0]
        # Project if needed and normalize shapes to (B, Hh, T, Dh)
        if self.q_proj is not None and not self._is_head_split(q, self.num_heads):
            q = self.q_proj(q)
        q_h = self._reshape_to_heads(q)
        scale = 1.0 / math.sqrt(self.head_dim)

        # Text branch projections/reshape
        if self.k_proj_y is not None and not self._is_head_split(k_y, self.num_heads):
            k_y = self.k_proj_y(k_y)
        if self.v_proj_y is not None and not self._is_head_split(v_y, self.num_heads):
            v_y = self.v_proj_y(v_y)
        ky_h = self._reshape_to_heads(k_y)
        vy_h = self._reshape_to_heads(v_y)

        # Image branch projections/reshape (if available)
        img_available = (k_x is not None) and (v_x is not None)
        if img_available:
            if self.k_proj_x is not None and not self._is_head_split(k_x, self.num_heads):
                k_x = self.k_proj_x(k_x)
            if self.v_proj_x is not None and not self._is_head_split(v_x, self.num_heads):
                v_x = self.v_proj_x(v_x)
            kx_h = self._reshape_to_heads(k_x)
            vx_h = self._reshape_to_heads(v_x)
        else:
            kx_h = None
            vx_h = None

        # Attention logits
        # q_h: (B, Hh, L, Dh); ky_h: (B, Hh, Ny, Dh)
        attn_logits_text = torch.matmul(q_h, ky_h.transpose(-1, -2)) * scale  # (B, Hh, L, Ny)
        if attn_mask is not None:
            # Attempt to broadcast mask to logits
            try:
                attn_logits_text = attn_logits_text + attn_mask
            except RuntimeError:
                # Try to adjust mask shape: if mask is spatial (B,1,L,1), keep as is; otherwise broadcast last dim
                attn_logits_text = attn_logits_text + attn_mask.expand_as(attn_logits_text)
        attn_text = F.softmax(attn_logits_text, dim=-1)
        attn_text = self.attn_drop(attn_text)
        out_text_h = torch.matmul(attn_text, vy_h)  # (B, Hh, L, Dh)

        # Image attention if available
        if img_available and kx_h is not None and vx_h is not None:
            attn_logits_image = torch.matmul(q_h, kx_h.transpose(-1, -2)) * scale  # (B, Hh, L, Nx)
            if attn_mask is not None:
                try:
                    attn_logits_image = attn_logits_image + attn_mask
                except RuntimeError:
                    attn_logits_image = attn_logits_image + attn_mask.expand_as(attn_logits_image)
            attn_image = F.softmax(attn_logits_image, dim=-1)
            attn_image = self.attn_drop(attn_image)
            out_image_h = torch.matmul(attn_image, vx_h)  # (B, Hh, L, Dh)
        else:
            attn_image = None
            out_image_h = torch.zeros_like(out_text_h)

        # Merge heads back to (B, L, D)
        out_text = out_text_h.permute(0, 2, 1, 3).contiguous().view(B, -1, self.embed_dim)
        out_image = out_image_h.permute(0, 2, 1, 3).contiguous().view(B, -1, self.embed_dim)

        # Output projections and dropout
        out_text = self.out_proj_text(out_text)
        out_text = self.proj_drop(out_text)
        out_image = self.out_proj_image(out_image)
        out_image = self.proj_drop(out_image)

        out: Dict[str, torch.Tensor] = {"text": out_text, "image": out_image}
        if need_weights:
            out["attn_text"] = attn_text
            if attn_image is not None:
                out["attn_image"] = attn_image
            else:
                out["attn_image"] = torch.zeros(B, self.num_heads, q_h.shape[2], 1, device=out_text.device, dtype=out_text.dtype)
        return out

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[torch.device] = None) -> "DecoupledCrossAttention":
        models_cfg = cfg.get("models", {})
        gen_cfg = models_cfg.get("generator", {})
        embed_dim = int(gen_cfg.get("text_embed_dim", 768))
        num_heads = int(gen_cfg.get("heads", 12))
        dropout = float(gen_cfg.get("attn_dropout", 0.0))
        dtype_str = gen_cfg.get("dtype", cfg.get("system", {}).get("dtype", "float16"))
        dtype = _map_dtype_str(dtype_str)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, device=device, dtype=dtype)


def build_decoupled_cross_attention(config: Dict[str, Any], device: Optional[torch.device] = None) -> DecoupledCrossAttention:
    return DecoupledCrossAttention.from_config(config, device=device)


__all__ = [
    "DecoupledCrossAttention",
    "build_decoupled_cross_attention",
]

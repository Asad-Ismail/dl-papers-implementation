from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .blocks import DecoderBlock
from .norms_rope import RoPECache, precompute_rope_cache, infer_positions


@dataclass
class TransformerConfig:
    # Model dims
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    mlp_hidden_dim: int = 3072

    # Attention selection
    attn_type: str = "regla"  # "regla", "softmax", "fast_decay", "la_elu1", "la_relu", or "hybrid"
    # Hybrid options
    hybrid_sa_ratio: float = 0.5  # fraction of layers using softmax when attn_type=="hybrid"
    hybrid_pattern: str = "alternate"  # "alternate" | "first_sa" | "last_sa"
    hybrid_map: Optional[Dict[int, str]] = None  # explicit per-layer overrides

    # Linear attention feature dim
    m: int = 64

    # Positional encoding
    rope: bool = True
    max_seq_len: int = 2048

    # Dropout
    dropout: float = 0.1

    # Norms
    norm_type: str = "rmsnorm"  # prenorm type inside blocks
    norm_eps: float = 1e-5
    stable_norm: str = "rmsnorm"  # post-attention inside attention modules
    stable_norm_eps: float = 1e-5

    # REGLA specifics
    use_sum_norm: bool = False
    alpha_scaling: bool = True
    gate_share_across_heads: bool = False

    # Head tying and final norm
    tie_embeddings: bool = True
    use_final_norm: bool = True


class TransformerLM(nn.Module):
    """
    Transformer language model with configurable attention per layer, RoPE handling,
    and streaming state support. LM head is tied to token embeddings by default.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model

        # Embeddings and head (tied weights)
        self.tok_emb = nn.Embedding(cfg.vocab_size, d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Build layer attention type list
        attn_types = self._build_layer_types()

        # Decoder blocks
        blocks: List[DecoderBlock] = []
        for layer_idx in range(cfg.n_layers):
            attn_type = attn_types[layer_idx]
            block = DecoderBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_head=cfg.d_head,
                mlp_hidden_dim=cfg.mlp_hidden_dim,
                attn_type=attn_type,
                m=cfg.m,
                rope=cfg.rope,
                dropout=cfg.dropout,
                norm_type=cfg.norm_type,
                norm_eps=cfg.norm_eps,
                stable_norm=cfg.stable_norm,
                stable_norm_eps=cfg.stable_norm_eps,
                use_sum_norm=cfg.use_sum_norm,
                alpha_scaling=cfg.alpha_scaling,
                gate_share_across_heads=cfg.gate_share_across_heads,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # Optional final norm (PreNorm architecture benefits from this for logits stability)
        self.final_norm = None
        if cfg.use_final_norm:
            # Lazy import to avoid circular, but norms are lightweight
            from .norms_rope import RMSNorm

            self.final_norm = RMSNorm(d_model, eps=cfg.norm_eps)

        # LM head tied to embeddings by default
        self.lm_head = nn.Linear(d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE cache placeholder (computed on first forward for device/dtype)
        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)
        self._rope_max_len: int = 0

        # Init parameters with a standard scheme
        self.apply(self._init_weights)

    # ------------------------- Initialization helpers -------------------------
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------- Hybrid mapping helpers -------------------------
    def _build_layer_types(self) -> List[str]:
        cfg = self.cfg
        # Start with default type for all
        types = [cfg.attn_type] * cfg.n_layers
        if cfg.attn_type != "hybrid":
            return types

        # Build a hybrid mapping of SA and REGLA by ratio and pattern
        num_sa = int(round(cfg.n_layers * cfg.hybrid_sa_ratio))
        indices = list(range(cfg.n_layers))
        sa_indices: List[int] = []
        if cfg.hybrid_pattern == "alternate":
            # pick every other starting at 0 till reaching num_sa
            for idx in indices:
                if len(sa_indices) >= num_sa:
                    break
                if idx % 2 == 0:
                    sa_indices.append(idx)
            # If not enough (odd layers), fill next indices
            i = 1
            while len(sa_indices) < num_sa and i < cfg.n_layers:
                if i not in sa_indices:
                    sa_indices.append(i)
                i += 2
        elif cfg.hybrid_pattern == "first_sa":
            sa_indices = indices[:num_sa]
        elif cfg.hybrid_pattern == "last_sa":
            sa_indices = indices[-num_sa:]
        else:
            sa_indices = indices[:num_sa]

        # Default is REGLA; set SA where needed
        types = ["regla"] * cfg.n_layers
        for i in sa_indices:
            types[i] = "softmax"

        # Apply explicit overrides if provided
        if cfg.hybrid_map:
            for k, v in cfg.hybrid_map.items():
                if 0 <= k < cfg.n_layers:
                    types[k] = v
        return types

    # ------------------------- RoPE cache handling ----------------------------
    def _ensure_rope_cache(self, needed_len: int, device: torch.device, dtype: torch.dtype) -> Optional[RoPECache]:
        if not self.cfg.rope:
            return None
        if self._rope_cos is not None and self._rope_sin is not None and self._rope_max_len >= needed_len:
            return RoPECache(cos=self._rope_cos[:needed_len], sin=self._rope_sin[:needed_len])
        # (re)compute cache on correct device/dtype
        cache = precompute_rope_cache(needed_len, self.cfg.d_head, device=device, dtype=dtype)
        self._rope_cos = cache.cos
        self._rope_sin = cache.sin
        self._rope_max_len = needed_len
        return cache

    # ------------------------- State initialization ---------------------------
    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> List[Optional[Dict[str, torch.Tensor]]]:
        states: List[Optional[Dict[str, torch.Tensor]]] = []
        for block in self.blocks:
            if hasattr(block, "init_state") and callable(block.init_state):
                states.append(block.init_state(batch_size, device=device, dtype=dtype))
            else:
                states.append(None)
        return states

    # ------------------------- Forward pass -----------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)
        state: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        start_pos: int = 0,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Optional[Dict[str, torch.Tensor]]]]]:
        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.tok_emb.weight.dtype

        # Embedding
        x = self.tok_emb(input_ids)  # (B, T, d_model)
        x = self.drop(x)

        # RoPE cache and positions
        rope_cache = self._ensure_rope_cache(start_pos + T, device=device, dtype=torch.float32)
        positions = infer_positions(T, start_pos=start_pos, device=device) if self.cfg.rope else None

        # Prepare states list
        if state is None:
            state = [None] * len(self.blocks)
        new_states: List[Optional[Dict[str, torch.Tensor]]] = []

        # Pass through blocks
        h = x
        for i, block in enumerate(self.blocks):
            h, st = block(
                h,
                state=state[i],
                rope_cache=rope_cache,
                positions=positions,
                return_state=return_state,
            )
            new_states.append(st if return_state else None)

        # Optional final norm
        if self.final_norm is not None:
            h = self.final_norm(h)

        # LM head (tied by default)
        logits = self.lm_head(h)

        return logits, (new_states if return_state else None)

    # ------------------------- Utility methods --------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def to_dtype(self, dtype: torch.dtype) -> "TransformerLM":
        return self.to(dtype=dtype)


__all__ = ["TransformerConfig", "TransformerLM"]

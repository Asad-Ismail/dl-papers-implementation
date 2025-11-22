"""
Inversion Network (Fθ)
Maps SDXL latent z and pooled text embedding c_y to predicted noise ε̂.

This module mirrors the simplified SwiftBrushV2 one-step generator architecture
but reverses the I/O: instead of ε -> z, we predict ε from z with the same
conditioning pathways, enabling the Stage 1/2 training objectives described in
SwiftEdit.

Public interface:
- class InversionNet: nn.Module implementing forward(z, text_emb) -> eps_hat
- class ResidualConvBlock: lightweight residual conv block
- classmethod InversionNet.from_config(cfg, device=None)
- function build_inversion_net(config, device=None)
- method init_from_generator(base_generator): copies compatible weights

Notes:
- Shapes: z, eps_hat: (B, C=4, H, W), text_emb: (B, D=768)
- Dtypes: configurable via string ("float16"|"bfloat16"|"float32") or torch.dtype
- Initialization from generator attempts to copy conv_in/conv_out and text_proj
  weights when available and shape-compatible.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _map_dtype_str(dtype_str: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    s = str(dtype_str).lower()
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32"}:
        return torch.float32
    return None


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True, device=device, dtype=dtype)

        # init
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class InversionNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        hidden_dim: int = 1536,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        d = _map_dtype_str(dtype)
        dev = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # z -> hidden
        self.conv_in = nn.Conv2d(latent_channels, hidden_dim, kernel_size=3, padding=1, bias=True, device=dev, dtype=d)
        # text -> hidden (broadcast spatial)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim, bias=True, device=dev, dtype=torch.float32 if d is None else torch.float32)
        # keep text proj in fp32 for stability; output will be cast on add

        self.gn = nn.GroupNorm(num_groups=max(1, hidden_dim // 64), num_channels=hidden_dim)
        self.act = nn.GELU()
        self.res1 = ResidualConvBlock(hidden_dim, kernel_size=3, dtype=d, device=dev)
        self.res2 = ResidualConvBlock(hidden_dim, kernel_size=3, dtype=d, device=dev)
        # hidden -> eps
        self.conv_out = nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, padding=1, bias=True, device=dev, dtype=d)

        # init
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
        nn.init.xavier_uniform_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

        self.latent_channels = latent_channels
        self.text_embed_dim = text_embed_dim
        self.hidden_dim = hidden_dim
        self._dtype = d
        self._device = dev

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> "InversionNet":
        # read nested config at cfg["models"]["inversion_net"] if present, else fallback to defaults
        inv_cfg = cfg.get("models", {}).get("inversion_net", {}) if isinstance(cfg, dict) else {}
        gen_cfg = cfg.get("models", {}).get("generator", {}) if isinstance(cfg, dict) else {}
        latent_channels = int(inv_cfg.get("latent_channels", gen_cfg.get("latent_channels", 4)))
        text_embed_dim = int(inv_cfg.get("text_embed_dim", gen_cfg.get("text_embed_dim", 768)))
        hidden_dim = int(inv_cfg.get("hidden_dim", gen_cfg.get("hidden_dim", 1536)))
        dtype = inv_cfg.get("dtype", cfg.get("system", {}).get("dtype", None))
        model = cls(latent_channels=latent_channels, text_embed_dim=text_embed_dim, hidden_dim=hidden_dim, dtype=dtype, device=device)

        # Optional weight init from base generator
        base_gen = None
        # Heuristic: check cfg for already constructed generator instance
        if isinstance(cfg, dict) and "_models" in cfg:
            base_gen = cfg["_models"].get("generator")
        # If not provided, attempt lazy import and construct from config
        if base_gen is None:
            try:
                from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
                base_gen = SwiftBrushV2.from_config(cfg, device=device)
            except Exception:
                base_gen = None
        if base_gen is not None:
            try:
                model.init_from_generator(base_gen)
            except Exception:
                pass
        return model

    def init_from_generator(self, base_generator: nn.Module) -> None:
        """Copy weights from a compatible base generator.
        We look for attributes conv_in, conv_out, text_proj with matching shapes
        and copy their parameters into the inversion net.
        """
        # conv_in
        if hasattr(base_generator, "conv_in"):
            src = getattr(base_generator, "conv_in")
            if isinstance(src, nn.Conv2d) and src.weight.shape[:2] == self.conv_in.weight.shape[:2]:
                with torch.no_grad():
                    self.conv_in.weight.copy_(src.weight)
                    if src.bias is not None and self.conv_in.bias is not None and src.bias.shape == self.conv_in.bias.shape:
                        self.conv_in.bias.copy_(src.bias)
        # conv_out
        if hasattr(base_generator, "conv_out"):
            src = getattr(base_generator, "conv_out")
            if isinstance(src, nn.Conv2d) and src.weight.shape[:2] == self.conv_out.weight.shape[:2]:
                with torch.no_grad():
                    self.conv_out.weight.copy_(src.weight)
                    if src.bias is not None and self.conv_out.bias is not None and src.bias.shape == self.conv_out.bias.shape:
                        self.conv_out.bias.copy_(src.bias)
        # text_proj
        if hasattr(base_generator, "text_proj"):
            src = getattr(base_generator, "text_proj")
            if isinstance(src, nn.Linear) and src.weight.shape == self.text_proj.weight.shape:
                with torch.no_grad():
                    self.text_proj.weight.copy_(src.weight)
                    if src.bias is not None and self.text_proj.bias is not None and src.bias.shape == self.text_proj.bias.shape:
                        self.text_proj.bias.copy_(src.bias)

    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict epsilon given latent z and pooled text embedding.

        Args:
            z: (B, C=4, H, W)
            text_emb: (B, D=768) pooled CLIP text embeddings (L2-normalized recommended)
        Returns:
            eps_hat: (B, C=4, H, W)
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        B, C, H, W = z.shape
        assert C == self.latent_channels, f"Expected latent channels {self.latent_channels}, got {C}"
        assert text_emb.shape[0] == B, f"Batch mismatch: z({B}) vs text_emb({text_emb.shape[0]})"
        assert text_emb.shape[1] == self.text_embed_dim, f"Expected text dim {self.text_embed_dim}, got {text_emb.shape[1]}"

        h = self.conv_in(z)
        # text conditioning
        t = self.text_proj(text_emb)  # (B, hidden)
        # cast t to module dtype if needed
        if self._dtype is not None and t.dtype != self._dtype:
            t = t.to(self._dtype)
        t = t.view(B, self.hidden_dim, 1, 1)
        h = h + t
        h = self.gn(h)
        h = self.act(h)
        h = self.res1(h)
        h = self.res2(h)
        eps_hat = self.conv_out(h)
        return eps_hat


def build_inversion_net(config: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> InversionNet:
    return InversionNet.from_config(config, device=device)


__all__ = ["InversionNet", "ResidualConvBlock", "build_inversion_net"]

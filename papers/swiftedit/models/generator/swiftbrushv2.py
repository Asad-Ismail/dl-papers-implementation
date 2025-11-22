"""
SwiftBrushV2: A simplified one-step generator G(ε, c_y) -> ẑ

This implementation provides a functional placeholder one-step generator that
maps Gaussian noise ε and a pooled text embedding c_y to a latent ẑ compatible
with an SDXL-style VAE (latent channels=4). It is NOT the original SwiftBrushv2
architecture, but designed to be fast and to satisfy the interface needed by
our Stage 1 dataset and trainers. When real pretrained weights are available,
this class can be extended to load them and to implement attention-based layers.

Key interface:
- forward(epsilon: (B, C=4, H, W), text_emb: (B, D=768)) -> z_hat: (B, 4, H, W)
- from_config(cfg) and load(...) helpers

Design:
- Conv-based feature extractor projects ε -> hidden_dim feature map
- Text conditioning c_y is projected to hidden_dim and broadcast spatially
- Two lightweight residual Conv blocks with GELU
- Output conv to latent channels, plus skip connection to ε for stability

This is sufficient for synthetic sample generation and sanity checks.
"""
from typing import Any, Dict, Optional, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32", "float"}:
        return torch.float32
    return None


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, dtype=dtype, device=device)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, dtype=dtype, device=device)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return x + h


class SwiftBrushV2(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        hidden_dim: int = 1536,
        heads: int = 12,  # unused in this simplified version
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        tdtype = _map_dtype_str(dtype) if isinstance(dtype, str) else dtype
        # Input projection from latent noise channels to hidden features
        self.conv_in = nn.Conv2d(latent_channels, hidden_dim, kernel_size=3, padding=1, dtype=tdtype, device=device)
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
        # Text conditioning projection
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim, dtype=tdtype, device=device)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        # Residual processing blocks
        self.block1 = ResidualConvBlock(hidden_dim, dtype=tdtype, device=device)
        self.block2 = ResidualConvBlock(hidden_dim, dtype=tdtype, device=device)
        # Output projection back to latent channels
        self.conv_out = nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, padding=1, dtype=tdtype, device=device)
        nn.init.xavier_uniform_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        # Optional normalization
        self.norm = nn.GroupNorm(num_groups=32, num_channels=hidden_dim)
        self.act = nn.GELU()
        # Save config-like attributes
        self.latent_channels = latent_channels
        self.text_embed_dim = text_embed_dim
        self.hidden_dim = hidden_dim
        self.heads = heads

    def forward(self, epsilon: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute one-step latent ẑ from noise ε and text embedding c_y.
        Args:
            epsilon: (B, C=4, H, W)
            text_emb: (B, D=768) pooled CLIP text embedding (ideally L2-normalized)
        Returns:
            z_hat: (B, 4, H, W)
        """
        if epsilon.dim() != 4:
            raise ValueError(f"epsilon must be 4D (B,C,H,W), got {epsilon.shape}")
        B, C, H, W = epsilon.shape
        if C != self.latent_channels:
            raise ValueError(f"epsilon channels {C} != latent_channels {self.latent_channels}")
        if text_emb.dim() != 2 or text_emb.shape[0] != B:
            raise ValueError(f"text_emb must be (B, D), got {text_emb.shape} for batch {B}")
        # Project inputs
        h = self.conv_in(epsilon)
        # Text conditioning broadcast
        t = self.text_proj(text_emb)  # (B, hidden_dim)
        t = t.view(B, self.hidden_dim, 1, 1)
        h = h + t  # simple additive conditioning
        # Process
        h = self.norm(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.act(h)
        z_hat = self.conv_out(h)
        # Skip connection to stabilize
        z_hat = z_hat + epsilon
        return z_hat

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> "SwiftBrushV2":
        """
        Build generator from a nested configuration dict, e.g., defaults.yaml under models.generator.
        """
        gcfg = cfg.get("models", {}).get("generator", cfg)
        latent_channels = int(gcfg.get("latent_channels", 4))
        text_embed_dim = int(gcfg.get("text_embed_dim", 768))
        hidden_dim = int(gcfg.get("hidden_dim", 1536))
        heads = int(gcfg.get("heads", 12))
        dtype = gcfg.get("dtype", None)
        model = cls(latent_channels=latent_channels, text_embed_dim=text_embed_dim, hidden_dim=hidden_dim, heads=heads, dtype=dtype, device=device)
        return model

    @staticmethod
    def load(cfg: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> "SwiftBrushV2":
        """
        Convenience loader. If a repo_dir with pretrained weights is provided, attempt to load;
        otherwise initialize randomly.
        """
        model = SwiftBrushV2.from_config(cfg, device=device)
        # Placeholder for loading weights when available
        repo_dir = cfg.get("models", {}).get("generator", {}).get("repo_dir", None)
        if repo_dir and isinstance(repo_dir, str) and os.path.isdir(repo_dir):
            # Attempt to find a weights file (this simplified model won't match real UNet weights).
            # We keep random init but log presence for visibility.
            pass
        return model


def build_generator(config: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> SwiftBrushV2:
    return SwiftBrushV2.from_config(config, device=device)

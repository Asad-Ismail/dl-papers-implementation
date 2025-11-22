from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

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


class GeneratorIP(nn.Module):
    """
    IP-conditioned one-step generator wrapper with decoupled cross-attention and ARaM.

    This module wraps a base one-step generator G (e.g., SwiftBrushV2) to incorporate
    - Text-guided decoupled cross-attention using pooled text embedding as tokens
    - Image-guided decoupled cross-attention via IP-Adapter image tokens
    - ARaM (mask-aware attention rescaling) per Eq. (9) of the reproduction plan

    It computes attention features from epsilon queries and text/image key-value pairs,
    rescales them with provided mask and scales, and fuses the resulting conditioning
    feature map into the base generator output to produce the final latent ẑ.
    """

    def __init__(
        self,
        base_generator: nn.Module,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        attn_dim: int = 768,
        num_text_tokens: int = 4,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        aram_default_scales: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.base_gen = base_generator
        self.latent_channels = latent_channels
        self.text_embed_dim = text_embed_dim
        self.attn_dim = attn_dim
        self.num_text_tokens = num_text_tokens
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.float16

        # Query projection from epsilon latent map -> per-location query vector
        self.q_proj = nn.Conv2d(latent_channels, attn_dim, kernel_size=1, bias=True, device=self.device, dtype=self.dtype)

        # Text K/V projection from pooled text embedding
        self.w_k_text = nn.Linear(text_embed_dim, attn_dim, bias=True, device=self.device, dtype=self.dtype)
        self.w_v_text = nn.Linear(text_embed_dim, attn_dim, bias=True, device=self.device, dtype=self.dtype)

        # Fallback image K/V projection if IP-Adapter branch is not provided
        self.w_k_img_fallback = nn.Linear(attn_dim, attn_dim, bias=True, device=self.device, dtype=self.dtype)
        self.w_v_img_fallback = nn.Linear(attn_dim, attn_dim, bias=True, device=self.device, dtype=self.dtype)

        # Conditioning map projection to latent channels
        self.cond_out = nn.Conv2d(attn_dim, latent_channels, kernel_size=1, bias=True, device=self.device, dtype=self.dtype)

        # Optional references to external components
        self.ip_adapter_branch: Optional[nn.Module] = None  # expects forward(img_tokens)->(Kx,Vx) and get_scale()
        self.projector: Optional[nn.Module] = None  # to be used outside for producing img_tokens from CLIP image embedding

        # Default ARaM scales
        default_scales = {"s_y": 1.0, "s_edit": 0.3, "s_non_edit": 1.5}
        if aram_default_scales is not None:
            default_scales.update(aram_default_scales)
        self.register_buffer("s_y_default", torch.tensor(float(default_scales["s_y"])) , persistent=False)
        self.register_buffer("s_edit_default", torch.tensor(float(default_scales["s_edit"])) , persistent=False)
        self.register_buffer("s_non_edit_default", torch.tensor(float(default_scales["s_non_edit"])) , persistent=False)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[str] = None) -> "GeneratorIP":
        from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
        # Build base generator from config
        base = SwiftBrushV2.from_config(cfg, device=device)
        mcfg = cfg.get("models", {}).get("generator", {})
        latent_channels = int(mcfg.get("latent_channels", 4))
        text_embed_dim = int(mcfg.get("text_embed_dim", 768))
        # Attention dim is aligned to text embed dim for KV compatibility
        attn_dim = int(mcfg.get("text_embed_dim", 768))
        num_text_tokens = 4
        # Dtype
        dtype = _map_dtype_str(mcfg.get("dtype", None))
        # ARaM scales
        acfg = cfg.get("aram", {})
        aram_scales = {
            "s_y": float(acfg.get("s_y", 1.0)),
            "s_edit": float(acfg.get("s_edit", 0.3)),
            "s_non_edit": float(acfg.get("s_non_edit", 1.5)),
        }
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_ip = cls(
            base_generator=base,
            latent_channels=latent_channels,
            text_embed_dim=text_embed_dim,
            attn_dim=attn_dim,
            num_text_tokens=num_text_tokens,
            dtype=dtype,
            device=dev,
            aram_default_scales=aram_scales,
        )
        return gen_ip

    def set_ip_adapter_branch(self, branch: nn.Module) -> None:
        self.ip_adapter_branch = branch

    def set_projector(self, projector: nn.Module) -> None:
        self.projector = projector

    def _compute_text_kv(self, text_emb: torch.Tensor, num_tokens: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce text K,V from pooled text embedding; replicate to a small token count.
        Inputs:
          - text_emb: (B, D_text)
        Returns:
          - K_y, V_y: (B, N_text, attn_dim)
        """
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        B = text_emb.shape[0]
        num_tokens = num_tokens or self.num_text_tokens
        k_y = self.w_k_text(text_emb.to(dtype=self.dtype))  # (B, attn_dim)
        v_y = self.w_v_text(text_emb.to(dtype=self.dtype))  # (B, attn_dim)
        k_y = k_y.unsqueeze(1).expand(B, num_tokens, self.attn_dim)
        v_y = v_y.unsqueeze(1).expand(B, num_tokens, self.attn_dim)
        return k_y, v_y

    def _compute_img_kv(self, img_tokens: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce image K,V from tokens using IP-Adapter branch if provided, else linear fallback.
        Inputs:
          - img_tokens: (B, N_img, C_tok) expected
        Returns:
          - K_x, V_x: (B, N_img, attn_dim)
        """
        if img_tokens is None:
            # If no image tokens provided, create zeros to neutralize image branch
            return torch.zeros(1, 1, self.attn_dim, device=self.device, dtype=self.dtype), torch.zeros(1, 1, self.attn_dim, device=self.device, dtype=self.dtype)
        if img_tokens.dim() == 2:
            img_tokens = img_tokens.unsqueeze(0)
        if self.ip_adapter_branch is not None:
            k_x, v_x = self.ip_adapter_branch(img_tokens.to(self.device))
            # Ensure dtype
            k_x = k_x.to(self.dtype)
            v_x = v_x.to(self.dtype)
            return k_x, v_x
        # Fallback: assume tokens already in attn_dim, pass through small linear layers
        t = img_tokens.to(dtype=self.dtype)
        k_x = self.w_k_img_fallback(t)
        v_x = self.w_v_img_fallback(t)
        return k_x, v_x

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        Inputs:
          - q: (B, L, d)
          - k: (B, N, d)
          - v: (B, N, d)
        Returns:
          - out: (B, L, d)
        """
        d = q.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)  # (B, L, N)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)  # (B, L, d)
        return out

    def _prepare_queries(self, epsilon: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Project epsilon to query vectors per spatial location.
        Returns (q_flat, H, W) where q_flat is (B, L=H*W, d)
        """
        if epsilon.dim() == 3:
            epsilon = epsilon.unsqueeze(0)
        B, C, H, W = epsilon.shape
        q_map = self.q_proj(epsilon.to(self.dtype))  # (B, d, H, W)
        q_flat = q_map.flatten(2).transpose(1, 2)  # (B, L, d)
        return q_flat, H, W

    def _reshape_to_map(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, L, d) -> (B, d, H, W)
        return x.transpose(1, 2).reshape(x.shape[0], self.attn_dim, H, W)

    def _resize_mask(self, mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # mask expected shape (B, 1, h, w) or (1, h, w)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        m = F.interpolate(mask.to(self.dtype), size=(H, W), mode="bilinear", align_corners=False)
        m = m.clamp(0.0, 1.0)
        return m

    def forward(
        self,
        epsilon: torch.Tensor,
        text_emb: torch.Tensor,
        img_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        scales: Optional[Dict[str, float]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute ẑ = GIP(ε, c_y, c_x) with optional ARaM.

        Inputs:
          - epsilon: (B, C=latent_channels, H, W)
          - text_emb: (B, D_text)
          - img_tokens: (B, N_img, C_tok) produced by IP-Adapter projector
          - mask: (B, 1, H_mask, W_mask) optional user or self-guided mask; continuous [0,1]
          - scales: dict with keys s_y, s_edit, s_non_edit; if None uses defaults
          - return_details: if True, returns a dict of intermediate attn maps and used mask

        Returns:
          - z_hat: (B, C, H, W)
          - details (optional): dict with 'text_attn_map', 'img_attn_map', 'mask'
        """
        use_scales = {
            "s_y": self.s_y_default.item(),
            "s_edit": self.s_edit_default.item(),
            "s_non_edit": self.s_non_edit_default.item(),
        }
        if scales is not None:
            use_scales.update({k: float(v) for k, v in scales.items() if k in use_scales})

        # Prepare queries
        q_flat, H, W = self._prepare_queries(epsilon)
        # Prepare text/image K,V
        k_y, v_y = self._compute_text_kv(text_emb, num_tokens=self.num_text_tokens)
        k_x, v_x = self._compute_img_kv(img_tokens)

        # Compute attentions
        text_attn_flat = self._attention(q_flat, k_y, v_y)  # (B, L, d)
        img_attn_flat = self._attention(q_flat, k_x, v_x)    # (B, L, d)
        text_attn_map = self._reshape_to_map(text_attn_flat, H, W)
        img_attn_map = self._reshape_to_map(img_attn_flat, H, W)

        # Apply ARaM scaling
        if mask is not None:
            m = self._resize_mask(mask, H, W)  # (B,1,H,W)
            s_y = use_scales["s_y"]
            s_edit = use_scales["s_edit"]
            s_non = use_scales["s_non_edit"]
            # Region-aware combination
            h_map = (
                s_y * m * text_attn_map +
                s_edit * m * img_attn_map +
                s_non * (1.0 - m) * img_attn_map
            )
        else:
            # Global decoupled attention: h = Attn_text + s_x Attn_img
            if self.ip_adapter_branch is not None and hasattr(self.ip_adapter_branch, "get_scale"):
                s_x = float(self.ip_adapter_branch.get_scale())
            else:
                # fallback image scale 1.0
                s_x = 1.0
            h_map = text_attn_map + s_x * img_attn_map
            m = None

        # Fuse conditioning into base generator output
        cond = self.cond_out(h_map)  # (B, C, H, W)
        base = self.base_gen(epsilon, text_emb)  # (B, C, H, W)
        z_hat = base + cond

        details = None
        if return_details:
            details = {
                "text_attn_map": text_attn_map.detach(),
                "img_attn_map": img_attn_map.detach(),
                "mask": m.detach() if m is not None else None,
            }
        return z_hat, details


def build_generator_ip(config: Dict[str, Any], device: Optional[str] = None) -> GeneratorIP:
    return GeneratorIP.from_config(config, device=device)

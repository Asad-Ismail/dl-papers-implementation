"""
IP-Adapter image-conditioning branch.

This module implements the learnable projection matrices W_Kx and W_Vx that map
projected image tokens (from a global CLIP image embedding via Projector) into
key/value representations compatible with the generator's cross-attention.

Equation reference (Eq. 3):
  h_l = Attn(Q_l, K_y, V_y) + s_x Attn(Q_l, K_x, V_x)

During Stage 1 training, only the image-conditioning branch (this module) and
the projector are trained. In Stage 2, this branch is frozen.

Shapes:
- Input image tokens: (B, N_img, token_dim)
- Output K_x, V_x:    (B, N_img, attn_dim)

Note:
- The actual multi-head splitting is handled by the generator's attention module.
- The global scale s_x is stored as a buffer (non-trainable by default) but can
  be made a parameter if desired.
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    ds = str(dtype_str).lower()
    if ds in {"float16", "fp16", "half"}:
        return torch.float16
    if ds in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if ds in {"float32", "fp32"}:
        return torch.float32
    return None


class IPAdapterBranch(nn.Module):
    """Image-conditioning branch for decoupled cross-attention.

    Parameters:
    - token_dim: Dimension of incoming image tokens (from Projector), typically 768
    - attn_dim:  Cross-attention model dimension expected by the generator (e.g., text embed dim)
    - heads:     Number of attention heads in the generator (informational; not used directly here)
    - bias:      Include bias in projection layers
    - s_x:       Global scale for image attention contribution (Eq. 3)
    - trainable: If False, parameters are frozen at construction
    - dtype:     Torch dtype for parameters (e.g., float16 for speed)
    """

    def __init__(
        self,
        token_dim: int = 768,
        attn_dim: int = 768,
        heads: int = 12,
        bias: bool = True,
        s_x: float = 1.0,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device

        self.token_dim = int(token_dim)
        self.attn_dim = int(attn_dim)
        self.heads = int(heads)
        self.use_bias = bool(bias)

        # Projections for image tokens -> K_x, V_x
        self.W_Kx = nn.Linear(self.token_dim, self.attn_dim, bias=self.use_bias, **factory_kwargs)
        self.W_Vx = nn.Linear(self.token_dim, self.attn_dim, bias=self.use_bias, **factory_kwargs)

        # Initialize with Xavier uniform; biases to zero
        nn.init.xavier_uniform_(self.W_Kx.weight)
        nn.init.xavier_uniform_(self.W_Vx.weight)
        if self.W_Kx.bias is not None:
            nn.init.zeros_(self.W_Kx.bias)
        if self.W_Vx.bias is not None:
            nn.init.zeros_(self.W_Vx.bias)

        # Global scale s_x
        self.register_buffer("s_x", torch.tensor(float(s_x), **({"dtype": torch.float32, "device": device} if device is not None else {})))

        # Freeze on construction if not trainable
        if not trainable:
            self.freeze()

    def forward(self, img_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (K_x, V_x) from image tokens.

        Args:
            img_tokens: Tensor of shape (B, N_img, token_dim) or (N_img, token_dim)

        Returns:
            K_x, V_x: Tensors of shape (B, N_img, attn_dim)
        """
        if img_tokens.dim() == 2:
            img_tokens = img_tokens.unsqueeze(0)
        assert img_tokens.dim() == 3, f"img_tokens must be (B, N, C), got {tuple(img_tokens.shape)}"
        B, N, C = img_tokens.shape
        assert C == self.token_dim, f"token_dim mismatch: expected {self.token_dim}, got {C}"

        Kx = self.W_Kx(img_tokens)
        Vx = self.W_Vx(img_tokens)
        return Kx, Vx

    @torch.no_grad()
    def set_scale(self, s_x: float) -> None:
        self.s_x.data.fill_(float(s_x))

    def get_scale(self) -> float:
        return float(self.s_x.item())

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[str] = None) -> "IPAdapterBranch":
        """Build an IPAdapterBranch from repository-level configuration.

        Expected keys:
        - cfg["models"]["ip_adapter"]["branch"]["trainable"], ["default_scale_sx"] (optional)
        - cfg["models"]["ip_adapter"]["projector"]["out_dim"] -> token_dim
        - cfg["models"]["generator"]["text_embed_dim"] -> attn_dim (fallback)
        - cfg["models"]["generator"]["heads"] -> heads (informational)
        - cfg["models"]["ip_adapter"]["branch"]["dtype"] (optional string)
        """
        m_ip = cfg.get("models", {}).get("ip_adapter", {})
        m_branch = m_ip.get("branch", {})
        m_proj = m_ip.get("projector", {})
        m_gen = cfg.get("models", {}).get("generator", {})

        token_dim = int(m_proj.get("out_dim", m_gen.get("text_embed_dim", 768)))
        attn_dim = int(m_gen.get("text_embed_dim", token_dim))
        heads = int(m_gen.get("heads", 12))
        trainable = bool(m_branch.get("trainable", True))
        s_x = float(m_branch.get("default_scale_sx", 1.0))
        dtype = _map_dtype_str(m_branch.get("dtype", None))
        dev = torch.device(device) if device is not None else None

        return cls(
            token_dim=token_dim,
            attn_dim=attn_dim,
            heads=heads,
            s_x=s_x,
            trainable=trainable,
            dtype=dtype,
            device=dev,
        )


def build_ip_adapter_branch(config: Dict[str, Any], device: Optional[str] = None) -> IPAdapterBranch:
    """Convenience builder for IPAdapterBranch from config dict."""
    return IPAdapterBranch.from_config(config, device=device)

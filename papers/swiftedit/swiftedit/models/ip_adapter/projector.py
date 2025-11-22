import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    IP-Adapter style projector: maps a global image embedding vector to a small
    sequence of conditioning tokens for cross-attention injection.

    Input:  image embedding e_x of shape (B, in_dim)
    Output: token sequence tokens_x of shape (B, num_tokens, out_dim)

    Default: two-layer MLP with GELU non-linearity producing num_tokens*out_dim
    values which are then reshaped into tokens. Optional LayerNorm on tokens.
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 768,
        num_tokens: int = 4,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
        normalize_tokens: bool = True,
        token_layernorm: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.normalize_tokens = normalize_tokens

        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim)

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias, **factory_kwargs),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * out_dim, bias=bias, **factory_kwargs),
        )

        self.ln = nn.LayerNorm(out_dim, elementwise_affine=True, **{k: v for k, v in factory_kwargs.items() if k == "device"}) if token_layernorm else None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Xavier initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: Tensor of shape (B, in_dim) or (in_dim,) for single sample.
        Returns:
            tokens: Tensor of shape (B, num_tokens, out_dim)
        """
        if img_emb.dim() == 1:
            img_emb = img_emb.unsqueeze(0)
        assert img_emb.dim() == 2 and img_emb.size(-1) == self.in_dim, (
            f"Expected img_emb shape (B,{self.in_dim}), got {tuple(img_emb.shape)}"
        )
        B = img_emb.size(0)
        y = self.proj(img_emb)  # (B, num_tokens*out_dim)
        y = y.view(B, self.num_tokens, self.out_dim)
        if self.ln is not None:
            # apply LayerNorm to each token independently
            y = self.ln(y)
        if self.normalize_tokens:
            y = F.normalize(y, p=2, dim=-1)
        return y

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[str] = None) -> "Projector":
        """
        Build from a config dictionary. Expected schema (examples):
        cfg = {
            'models': {
                'ip_adapter': {
                    'projector': {
                        'in_dim': 768,
                        'out_dim': 768,
                        'num_tokens': 4,
                        'hidden_dim': 1024,
                        'normalize_tokens': True,
                    }
                }
            }
        }
        or directly pass the projector sub-config.
        """
        sub = cfg
        # descend into nested keys if present
        if 'models' in cfg and 'ip_adapter' in cfg['models'] and 'projector' in cfg['models']['ip_adapter']:
            sub = cfg['models']['ip_adapter']['projector']
        in_dim = int(sub.get('in_dim', 768))
        out_dim = int(sub.get('out_dim', 768))
        num_tokens = int(sub.get('num_tokens', 4))
        hidden_dim = sub.get('hidden_dim', None)
        normalize_tokens = bool(sub.get('normalize_tokens', True))
        token_layernorm = bool(sub.get('token_layernorm', True))

        torch_device = torch.device(device) if device is not None else None
        # dtype handling: default to float32 for stability
        dtype_str = sub.get('dtype', None)
        dtype = None
        if isinstance(dtype_str, str):
            if dtype_str.lower() in ("fp16", "float16", "half"):
                dtype = torch.float16
            elif dtype_str.lower() in ("bf16", "bfloat16"):
                dtype = torch.bfloat16
            elif dtype_str.lower() in ("fp32", "float32"):
                dtype = torch.float32

        return cls(
            in_dim=in_dim,
            out_dim=out_dim,
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            normalize_tokens=normalize_tokens,
            token_layernorm=token_layernorm,
            device=torch_device,
            dtype=dtype,
        )


def build_projector(config: Dict[str, Any], device: Optional[str] = None) -> Projector:
    """Convenience builder that extracts the projector configuration block
    from the repository-level config.
    """
    return Projector.from_config(config, device=device)

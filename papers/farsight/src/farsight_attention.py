import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .causal_mask import build_causal_mask, build_attention_register


class FarSightAttention(nn.Module):
    """
    FarSightAttention module:
      - Applies FarSight attention with Upper-Triangular Attention Register and Positional Awareness Encoding.
    """
    def __init__(self, hid_dim: int, n_heads: int, seq_len: int,
                 decay_base: float = 1024.0, p: float = 1.0, device: str = None):
        super().__init__()
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = hid_dim // n_heads
        # compute sigma from decay_base and seq_len
        self.sigma = math.log(decay_base) / seq_len
        # device handling
        dev = torch.device(device) if device is not None else torch.device('cpu')
        # build masks
        C = build_causal_mask(seq_len, device=dev, dtype=torch.float32)          # [T, T]
        P = build_attention_register(seq_len, self.sigma, device=dev, dtype=torch.float32)  # [T, T]
        # register buffers
        self.register_buffer('C', C)           # causal mask
        self.register_buffer('P', P)           # attention register
        # linear projections
        self.q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(hid_dim, hid_dim,bias=False)
        self.v_proj = nn.Linear(hid_dim, hid_dim,bias=False)
        self.out_proj = nn.Linear(hid_dim, hid_dim,bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        B, T, D = x.size()
        assert D == self.hid_dim, f"Expected input dim {self.hid_dim}, got {D}"
        assert T <= self.seq_len, f"Sequence length {T} exceeds maximum {self.seq_len}"
        # project
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)  # [B, T, H, Dh]
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        # compute raw attention scores
        # Q: [B, T, H, Dh], K: [B, T, H, Dh] -> scores: [B, H, T, T]
        scores = torch.einsum('bthd,bThd->bh tT'.replace(' ', ''), Q, K)  # [B, H, T, T]
        scores = scores / math.sqrt(self.head_dim)
        # slice masks to T
        C = self.C[:T, :T]               # [T, T]
        P = self.P[:T, :T]
        # apply masks and register
        # broadcast masks to [B, H, T, T]
        scores = scores * C.unsqueeze(0).unsqueeze(1) + P.unsqueeze(0).unsqueeze(1)

        # softmax and reapply causal mask
        attn = F.softmax(scores, dim=-1) * C.unsqueeze(0).unsqueeze(1)
        print(attn)
        # attention output
        out = torch.einsum('bhtT,bThd->bthd'.replace(' ', ''), attn, V)  # [B, T, H, Dh]
        out = out.reshape(B, T, D)
        return self.out_proj(out)

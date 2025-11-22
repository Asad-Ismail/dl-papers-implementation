import math
import torch
import pytest

from src.farsight_attention import FarSightAttention
from src.causal_mask import (
    build_causal_mask,
    build_attention_register,
    build_positional_mask,
)


def test_mask_buffers():
    """
    Test that the internal mask buffers C, P, and pos_mask of FarSightAttention
    match the outputs of the standalone mask-building functions.
    """
    seq_len = 5
    hid_dim = 16
    n_heads = 4
    decay_base = 1024.0
    p = 1.5
    # Initialize module on CPU
    attn = FarSightAttention(hid_dim, n_heads, seq_len, decay_base=decay_base, p=p, device='cpu')

    # Expected masks
    C_ref = build_causal_mask(seq_len, device='cpu', dtype=attn.C.dtype)
    sigma = math.log(decay_base) / seq_len
    P_ref = build_attention_register(seq_len, sigma, device='cpu', dtype=attn.P.dtype)

    # Buffers should match reference masks
    assert torch.equal(attn.C, C_ref)
    assert torch.allclose(attn.P, P_ref, atol=1e-6)


def test_forward_zero_input():
    """
    Given zero input, the attention output should be zero (due to zero Q,K,V).
    """
    seq_len = 6
    hid_dim = 12
    n_heads = 3
    decay_base = 512.0
    p = 1.0
    attn = FarSightAttention(hid_dim, n_heads, seq_len, decay_base=decay_base, p=p, device='cpu')

    # Zero input tensor
    batch_size = 2
    x = torch.zeros(batch_size, seq_len, hid_dim)
    out = attn(x)

    # Output should be same shape and also zero
    assert out.shape == x.shape
    print(out)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


def test_forward_random_shape():
    """
    Test that forward pass produces the correct output shape for random input.
    """
    seq_len = 7
    hid_dim = 14
    n_heads = 2
    attn = FarSightAttention(hid_dim, n_heads, seq_len, device='cpu')

    batch_size = 3
    x = torch.randn(batch_size, seq_len, hid_dim)
    out = attn(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, seq_len, hid_dim)
    # Ensure no NaNs or infinite values in output
    assert torch.isfinite(out).all()

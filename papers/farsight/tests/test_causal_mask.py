import os
import sys
import torch
import pytest


from src.causal_mask import build_causal_mask, build_attention_register, build_positional_mask


def test_build_causal_mask():
    # Test lower-triangular binary mask
    T = 4
    C = build_causal_mask(T)
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=C.dtype)
    assert torch.equal(C, expected), f"Causal mask mismatch: {C} vs {expected}"


def test_build_attention_register():
    # Test attention register values
    T = 4
    sigma = 0.5
    P = build_attention_register(T, sigma)
    assert P.shape == (T, T)
    P_np = P.cpu().numpy()
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert P_np[i, j] == 0.0, f"Expected 0 at P[{i},{j}], got {P_np[i,j]}"
            else:
                expected = -sigma * (j - i)
                assert pytest.approx(P_np[i, j], rel=1e-6) == expected, \
                    f"P[{i},{j}] expected {expected}, got {P_np[i,j]}"


def test_build_positional_mask():
    # Test positional decay mask
    T = 4
    p = 1.0
    pos_mask = build_positional_mask(T, p)
    assert pos_mask.shape == (T, T)
    pos_np = pos_mask.cpu().numpy()
    for i in range(T):
        alpha_i = 1.0 - (i / T) ** p
        for j in range(T):
            if j <= i:
                assert pos_np[i, j] == pytest.approx(1.0, rel=1e-6), \
                    f"Expected 1 at pos_mask[{i},{j}], got {pos_np[i,j]}"
            else:
                expected = alpha_i * ((j - i) / T)
                assert pytest.approx(pos_np[i, j], rel=1e-6) == expected, \
                    f"pos_mask[{i},{j}] expected {expected}, got {pos_np[i,j]}"
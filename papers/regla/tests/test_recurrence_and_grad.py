import math
import pytest
import torch

from regla.core.feature_maps import safe_exp_query, safe_exp_key
from regla.core.gating_and_recurrence import regla_step


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize("B,H,D,M,T", [
    (1, 2, 4, 6, 8),
    (2, 3, 8, 16, 12),
])
@torch.no_grad()
def test_regla_recurrence_matches_manual(B: int, H: int, D: int, M: int, T: int):
    """
    Validate that the regla_step recurrence matches a manual implementation on short sequences.
    We construct random v, q_raw, k_raw, compute safe φ_q/φ_k (training/global max), and apply
    refined forget factors generated from a fixed synthetic formula (here using a simple sigmoid on a
    random proxy) to avoid dependency on gate module internals.
    """
    set_seed(0)
    device = torch.device("cpu")
    # Random raw inputs
    v = torch.randn(B, T, H, D, device=device)
    q_raw = torch.randn(B, T, H, M, device=device)
    k_raw = torch.randn(B, T, H, M, device=device)

    # Safe feature maps (training/global max for keys)
    phi_q, _ = safe_exp_query(q_raw), None
    phi_k, gmax = safe_exp_key(k_raw, running_max=None, training=True)
    assert phi_q.shape == (B, T, H, M)
    assert phi_k.shape == (B, T, H, M)
    assert gmax.shape[0] == B

    # Synthetic refined forget factors per-step (B, T, H, D)
    # Use a proxy to emulate gate outputs: r ~ sigmoid(noise), g ~ sigmoid(noise)
    proxy_r = torch.randn(B, T, H, D, device=device)
    proxy_g = torch.randn(B, T, H, D, device=device)
    r = torch.sigmoid(proxy_r)
    g = torch.sigmoid(proxy_g)
    lower = g ** 2
    upper = 1.0 - (1.0 - g) ** 2
    mix = (1.0 - r) * lower + r * upper  # (B, T, H, D)

    # Initialize states
    S_prev = torch.zeros(B, H, D, M, device=device)
    c_prev = torch.zeros(B, H, M, device=device)

    # Manual loop and regla_step loop
    ys_regla = []
    ys_manual = []
    S_manual = S_prev.clone()
    c_manual = c_prev.clone()

    for t in range(T):
        v_t = v[:, t]
        phi_k_t = phi_k[:, t]
        phi_q_t = phi_q[:, t]
        mix_t = mix[:, t]

        # regla_step computation without sum norm
        y_t, S_prev, _ = regla_step(S_prev, v_t, phi_k_t, phi_q_t, mix_t, sum_norm=False, c_prev=None)
        ys_regla.append(y_t)

        # Manual update: S_t = mix_t ⊙ S_{t-1} + v_t ⊗ φ_k_t^T; y_t = S_t φ_q_t
        S_manual = S_manual * mix_t.unsqueeze(-1) + torch.einsum("bhd,bhm->bhdm", v_t, phi_k_t)
        y_m = torch.einsum("bhdm,bhm->bhd", S_manual, phi_q_t)
        ys_manual.append(y_m)

    y_regla = torch.stack(ys_regla, dim=1)  # (B, T, H, D)
    y_manual = torch.stack(ys_manual, dim=1)

    max_abs_err = (y_regla - y_manual).abs().max().item()
    assert max_abs_err < 1e-6, f"Max abs error too high: {max_abs_err}"


@pytest.mark.parametrize("B,H,D,M,T", [
    (1, 2, 4, 6, 8),
    (2, 2, 4, 8, 10),
])
@torch.no_grad()
def test_sum_norm_matches_manual(B: int, H: int, D: int, M: int, T: int):
    """
    Validate that sum normalization in regla_step matches manual normalization by c_t accumulation.
    """
    set_seed(1)
    device = torch.device("cpu")
    v = torch.randn(B, T, H, D, device=device)
    q_raw = torch.randn(B, T, H, M, device=device)
    k_raw = torch.randn(B, T, H, M, device=device)

    phi_q = safe_exp_query(q_raw)
    phi_k, _ = safe_exp_key(k_raw, running_max=None, training=True)

    # Use a simple mix=ones (no forget) to align with standard linear attention + sum norm
    mix = torch.ones(B, T, H, D, device=device)

    S_prev = torch.zeros(B, H, D, M, device=device)
    c_prev = torch.zeros(B, H, M, device=device)

    ys_regla = []
    ys_manual = []
    S_manual = S_prev.clone()
    c_manual = c_prev.clone()

    eps = 1e-6
    for t in range(T):
        v_t = v[:, t]
        phi_k_t = phi_k[:, t]
        phi_q_t = phi_q[:, t]
        mix_t = mix[:, t]

        # regla_step with sum norm
        y_t, S_prev, c_prev = regla_step(S_prev, v_t, phi_k_t, phi_q_t, mix_t, sum_norm=True, c_prev=c_prev, eps=eps)
        ys_regla.append(y_t)

        # Manual update
        S_manual = S_manual * mix_t.unsqueeze(-1) + torch.einsum("bhd,bhm->bhdm", v_t, phi_k_t)
        c_manual = c_manual + phi_k_t
        denom = torch.einsum("bhm,bhm->bh", c_manual, phi_q_t) + eps
        y_m = torch.einsum("bhdm,bhm->bhd", S_manual, phi_q_t) / denom.unsqueeze(-1)
        ys_manual.append(y_m)

    y_regla = torch.stack(ys_regla, dim=1)
    y_manual = torch.stack(ys_manual, dim=1)

    max_abs_err = (y_regla - y_manual).abs().max().item()
    assert max_abs_err < 1e-6, f"Sum-norm mismatch: {max_abs_err}"


def _gate_grad_near_extreme(extreme: str = "zero"):
    """
    Compare gradient magnitude of refined gate formula vs scalar sigmoid near extremes of g.
    We do not depend on gate module internals; instead, we construct the refined mixing function
    from scalar parameters and compare dL/db_g under a simple loss.
    extreme: "zero" to set g near 0 with r near 1 (upper envelope), "one" to set g near 1 with r near 0 (lower envelope).
    Returns tuple (grad_refined, grad_scalar).
    """
    device = torch.device("cpu")
    # Learnable biases representing pre-sigmoid activations for g and r
    b_g = torch.tensor(-8.0 if extreme == "zero" else 8.0, requires_grad=True, device=device)
    # Select r to emphasize the envelope that improves gradients in the given extreme
    b_r = torch.tensor(8.0 if extreme == "zero" else -8.0, requires_grad=True, device=device)

    # Target mix value away from exact extremes to induce gradient signal
    target = torch.tensor(0.3 if extreme == "zero" else 0.7, device=device)

    # Refined gate mix
    g = torch.sigmoid(b_g)
    r = torch.sigmoid(b_r)
    lower = g ** 2
    upper = 1.0 - (1.0 - g) ** 2  # = 2g - g^2
    mix = (1.0 - r) * lower + r * upper
    loss_refined = (mix - target) ** 2
    loss_refined.backward(retain_graph=True)
    grad_refined = b_g.grad.detach().abs().item()

    # Reset grad on b_g for scalar case
    b_g.grad.zero_()

    # Scalar gate
    g_s = torch.sigmoid(b_g)
    loss_scalar = (g_s - target) ** 2
    loss_scalar.backward()
    grad_scalar = b_g.grad.detach().abs().item()

    return grad_refined, grad_scalar


def test_gate_gradient_behavior_near_zero():
    set_seed(123)
    grad_refined, grad_scalar = _gate_grad_near_extreme("zero")
    # Expect refined gate gradient to be larger than scalar near zero (relaxed threshold)
    assert grad_refined > grad_scalar * 1.5, f"Refined grad {grad_refined} not sufficiently larger than scalar {grad_scalar} near zero"


def test_gate_gradient_behavior_near_one():
    set_seed(123)
    grad_refined, grad_scalar = _gate_grad_near_extreme("one")
    # Expect refined gate gradient to be larger than scalar near one (relaxed threshold)
    assert grad_refined > grad_scalar * 1.5, f"Refined grad {grad_refined} not sufficiently larger than scalar {grad_scalar} near one"


if __name__ == "__main__":
    pytest.main([__file__])

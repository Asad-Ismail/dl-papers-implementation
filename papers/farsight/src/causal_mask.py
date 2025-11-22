import torch

def build_causal_mask(T: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Build a lower-triangular causal mask matrix C of shape (T, T).
    C[i,j] = 1 if j <= i else 0.
    """
    # Create a T x T matrix of ones and take lower triangle
    return torch.tril(torch.ones((T, T), dtype=dtype, device=device))


def build_attention_register(T: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Build an attention register matrix P of shape (T, T).
    P[i,j] = 0 for j <= i; P[i,j] = -sigma * (j - i) for j > i.
    """
    # idx: column indices, jdx: row indices for broadcasting
    idx = torch.arange(T, device=device).view(-1, 1)
    jdx = torch.arange(T, device=device).view(1, -1)
    diff = (jdx - idx).clamp(min=0).to(dtype)
    return diff * (-sigma)


def build_positional_mask(T: int, p: float = 1.0, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Build a positional decay mask of shape (T, T).
    For positions j <= i: mask[i,j] = 1.
    For j > i: mask[i,j] = alpha_i * ((j - i) / T),
      where alpha_i = 1 - (i / T)^p.
    """
    # Prepare index vectors
    idx = torch.arange(T, device=device).float()
    jdx = torch.arange(T, device=device).float()
    # Broadcast to matrices
    i_mat = idx.view(-1, 1)  # shape (T,1)
    j_mat = jdx.view(1, -1)  # shape (1,T)
    # Causal indicator matrix j <= i
    causal = j_mat <= i_mat
    # Compute alpha for each row i
    alpha = (1 - (idx / T) ** p).to(dtype)
    # Compute distance (j - i)
    delta = (j_mat - i_mat)
    # Build mask: 1 for j<=i, else alpha_i * (delta / T)
    mask = torch.where(causal,
                       torch.ones_like(delta, dtype=dtype),
                       alpha.view(-1, 1) * (delta / T))
    return mask.to(dtype)

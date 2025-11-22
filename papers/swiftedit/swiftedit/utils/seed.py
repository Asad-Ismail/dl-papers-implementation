"""
Reproducibility and seeding utilities for SwiftEdit.

This module provides helper functions to set global RNG seeds across
Python's random, NumPy, and PyTorch (CPU/CUDA), as well as convenient
helpers for DataLoader workers and torch.Generator creation.

Public API:
- set_seed(seed: int, deterministic: bool = False) -> Dict[str, Any]
- set_torch_deterministic(deterministic: bool = False) -> None
- seed_worker(worker_id: int) -> None
- get_torch_generator(seed: int) -> torch.Generator

Usage:
>>> from swiftedit.utils.seed import set_seed, seed_worker, get_torch_generator
>>> info = set_seed(42, deterministic=True)
>>> g = get_torch_generator(42)
>>> loader = DataLoader(dataset, worker_init_fn=seed_worker, generator=g)
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict


def _has_torch() -> bool:
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


def _has_numpy() -> bool:
    try:
        import numpy as np  # noqa
        return True
    except Exception:
        return False


def set_torch_deterministic(deterministic: bool = False) -> None:
    """Configure PyTorch's deterministic/cuDNN settings.

    Args:
        deterministic: If True, enforce deterministic algorithms at the
            potential cost of performance.
    """
    if not _has_torch():
        return
    import torch
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        # Older PyTorch versions may not support this fully; best-effort only
        pass

    # cuDNN settings
    try:
        import torch.backends.cudnn as cudnn
        if deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            # Allow autotuner for speed when determinism is not required
            cudnn.deterministic = False
            cudnn.benchmark = True
    except Exception:
        pass


def set_seed(seed: int, deterministic: bool = False) -> Dict[str, Any]:
    """Set global RNG seeds for Python, NumPy, and PyTorch.

    Args:
        seed: The base integer seed to set.
        deterministic: If True, toggles PyTorch/cuDNN deterministic behavior.

    Returns:
        A dictionary with environment info about the seeding state.
    """
    info: Dict[str, Any] = {"seed": int(seed), "deterministic": bool(deterministic)}

    # Python RNG
    random.seed(seed)
    info["python_random"] = seed

    # OS hash seed for deterministic hashing in Python
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        info["PYTHONHASHSEED"] = os.environ["PYTHONHASHSEED"]
    except Exception:
        pass

    # NumPy RNG
    if _has_numpy():
        import numpy as np
        np.random.seed(seed)
        info["numpy_random"] = seed
    else:
        info["numpy_random"] = None

    # Torch RNGs
    if _has_torch():
        import torch
        torch.manual_seed(seed)
        info["torch_manual_seed"] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            info["torch_cuda_manual_seed_all"] = seed
        else:
            info["torch_cuda_manual_seed_all"] = None

        set_torch_deterministic(deterministic)
        info["torch_cuda_available"] = torch.cuda.is_available()
        try:
            info["torch_version"] = torch.__version__
        except Exception:
            info["torch_version"] = None
    else:
        info.update({
            "torch_manual_seed": None,
            "torch_cuda_manual_seed_all": None,
            "torch_cuda_available": None,
            "torch_version": None,
        })

    return info


def seed_worker(worker_id: int) -> None:
    """Seed function suitable for DataLoader(worker_init_fn=...).

    This uses the initial torch seed for the worker to derive per-worker
    seeds for Python's random and NumPy to ensure distinct, reproducible
    sequences across workers.

    Args:
        worker_id: The worker id provided by PyTorch's DataLoader.
    """
    # Derive a worker-specific seed from PyTorch's initial seed
    if not _has_torch():
        # Fallback: still try to set Python/NumPy seeds deterministically
        base_seed = worker_id
    else:
        import torch
        base_seed = torch.initial_seed() % (2**32)

    random.seed(base_seed)
    if _has_numpy():
        import numpy as np
        np.random.seed(base_seed)


def get_torch_generator(seed: int):
    """Create a torch.Generator seeded with the given seed.

    Useful to pass into DataLoader(generator=...) for reproducible shuffling
    and sampling.
    """
    if not _has_torch():
        raise ImportError("PyTorch is required to create a torch.Generator")
    import torch
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


__all__ = [
    "set_seed",
    "set_torch_deterministic",
    "seed_worker",
    "get_torch_generator",
]

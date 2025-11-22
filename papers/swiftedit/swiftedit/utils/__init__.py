# swiftedit/utils/__init__.py
"""
SwiftEdit utilities package initializer.

This module aggregates and re-exports commonly used utility components:
- Logging (ExperimentLogger, build_logger_from_config, save_yaml, get_env_info)
- Seeding and determinism (set_seed, set_torch_deterministic, seed_worker, get_torch_generator)
- Checkpoint IO (CheckpointManager and helpers)
- Timing/performance profiling (Timer, MultiTimer, scoped_timer, cuda_sync, perf_counter_ms, measure_time)
- Visualization helpers (optional; re-exported if available)

Imports are defensive: if an optional submodule fails to import, the corresponding
symbols are set to None and omitted from __all__.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Logging utilities
try:
    from .logger import (
        ExperimentLogger,
        build_logger_from_config,
        get_env_info,
        save_yaml,
    )
except Exception:  # pragma: no cover
    ExperimentLogger = None  # type: ignore
    build_logger_from_config = None  # type: ignore
    get_env_info = None  # type: ignore
    save_yaml = None  # type: ignore

# Seeding utilities
try:
    from .seed import (
        set_seed,
        set_torch_deterministic,
        seed_worker,
        get_torch_generator,
    )
except Exception:  # pragma: no cover
    set_seed = None  # type: ignore
    set_torch_deterministic = None  # type: ignore
    seed_worker = None  # type: ignore
    get_torch_generator = None  # type: ignore

# Checkpoint utilities
try:
    from .checkpoint import (
        CheckpointManager,
        ensure_dir,
        save_checkpoint,
        load_checkpoint,
        state_dict_from_model,
        load_model_state,
        maybe_load_ema_shadow,
        build_checkpoint_payload,
        save_models_bundle,
        load_models_from_checkpoint,
        save_config_snapshot,
        LoadReport,
    )
except Exception:  # pragma: no cover
    CheckpointManager = None  # type: ignore
    ensure_dir = None  # type: ignore
    save_checkpoint = None  # type: ignore
    load_checkpoint = None  # type: ignore
    state_dict_from_model = None  # type: ignore
    load_model_state = None  # type: ignore
    maybe_load_ema_shadow = None  # type: ignore
    build_checkpoint_payload = None  # type: ignore
    save_models_bundle = None  # type: ignore
    load_models_from_checkpoint = None  # type: ignore
    save_config_snapshot = None  # type: ignore
    LoadReport = None  # type: ignore

# Timing utilities
try:
    from .timer import (
        Timer,
        MultiTimer,
        scoped_timer,
        cuda_sync,
        perf_counter_ms,
        measure_time,
    )
except Exception:  # pragma: no cover
    Timer = None  # type: ignore
    MultiTimer = None  # type: ignore
    scoped_timer = None  # type: ignore
    cuda_sync = None  # type: ignore
    perf_counter_ms = None  # type: ignore
    measure_time = None  # type: ignore

# Visualization utilities (optional; may not exist yet)
try:
    from .viz import (
        save_image_grid,
        tensor_to_pil,
        pil_to_tensor,
        draw_mask_overlay,
    )
except Exception:  # pragma: no cover
    save_image_grid = None  # type: ignore
    tensor_to_pil = None  # type: ignore
    pil_to_tensor = None  # type: ignore
    draw_mask_overlay = None  # type: ignore


def available_utils() -> Dict[str, bool]:
    """Return availability map for utility groups.

    Keys:
      - logging, seed, checkpoint, timer, viz
    Values:
      - bool indicating whether the group is available (at least one primary symbol)
    """
    return {
        "logging": ExperimentLogger is not None and build_logger_from_config is not None,
        "seed": set_seed is not None and set_torch_deterministic is not None,
        "checkpoint": CheckpointManager is not None and save_checkpoint is not None,
        "timer": Timer is not None and measure_time is not None,
        "viz": save_image_grid is not None or tensor_to_pil is not None,
    }


__all__ = [
    # Logging
    "ExperimentLogger",
    "build_logger_from_config",
    "get_env_info",
    "save_yaml",
    # Seeding
    "set_seed",
    "set_torch_deterministic",
    "seed_worker",
    "get_torch_generator",
    # Checkpoint IO
    "CheckpointManager",
    "ensure_dir",
    "save_checkpoint",
    "load_checkpoint",
    "state_dict_from_model",
    "load_model_state",
    "maybe_load_ema_shadow",
    "build_checkpoint_payload",
    "save_models_bundle",
    "load_models_from_checkpoint",
    "save_config_snapshot",
    "LoadReport",
    # Timing
    "Timer",
    "MultiTimer",
    "scoped_timer",
    "cuda_sync",
    "perf_counter_ms",
    "measure_time",
    # Visualization (optional)
    "save_image_grid",
    "tensor_to_pil",
    "pil_to_tensor",
    "draw_mask_overlay",
    # Meta
    "available_utils",
]

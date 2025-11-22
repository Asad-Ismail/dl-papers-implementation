"""
SwiftEdit Stage 2 package initializer

Provides:
- deep_update: recursive dict merge
- load_config: YAML/JSON config loader
- run_stage2: convenience entry to load+merge configs and start Stage 2 training
- train_stage2: thin wrapper around the Stage 2 trainer (real-image training)
- build_stage2_dataset: factory wrapper for the Stage 2 real dataset

Stage 2 optimizes the inversion network F_theta on real images using a perceptual
loss (DISTS) and an SDS-style regularization term provided by a diffusion teacher.
All encoders and the IP-Adapter branch are frozen by default; only F_theta is trained.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

try:
    from .trainer_stage2 import train_stage2 as _train_stage2  # type: ignore
except Exception:
    _train_stage2 = None  # type: ignore

try:
    from .dataset_real import build_stage2_dataset as _build_stage2_dataset  # type: ignore
except Exception:
    _build_stage2_dataset = None  # type: ignore


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge upd into base (mutates base) and return base."""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML (.yml/.yaml) or JSON (.json) file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise ValueError("PyYAML is required to load YAML configs but is not installed.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)  # type: ignore
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Unknown extension: try YAML then JSON
    if _HAS_YAML:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)  # type: ignore
        except Exception:
            pass
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_stage2(cfg: Dict[str, Any]) -> None:
    """Run Stage 2 training with an already merged configuration dictionary."""
    if _train_stage2 is None:
        raise RuntimeError("Stage 2 trainer is unavailable (import failed).")
    _train_stage2(cfg)


def build_stage2_dataset(config: Dict[str, Any], **kwargs: Any):  # -> Dataset
    """Construct the Stage 2 real dataset from a configuration dictionary.

    Pass-through wrapper to swiftedit.train.stage2.dataset_real.build_stage2_dataset.
    """
    if _build_stage2_dataset is None:
        raise RuntimeError("Stage 2 dataset builder is unavailable (import failed).")
    return _build_stage2_dataset(config, **kwargs)


def run_stage2(defaults_path: str, override_path: Optional[str] = None) -> None:
    """Load defaults and optional override configs, deep-merge, and run Stage 2."""
    base = load_config(defaults_path)
    if override_path and os.path.isfile(override_path):
        over = load_config(override_path)
        deep_update(base, over)
    train_stage2(base)


__all__ = [
    "deep_update",
    "load_config",
    "run_stage2",
    "train_stage2",
    "build_stage2_dataset",
]

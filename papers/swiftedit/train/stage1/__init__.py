"""
SwiftEdit Stage 1 training package

This module aggregates Stage 1 training entry points and dataset builders, and
provides light-weight configuration helpers to load and merge YAML/JSON config
files before dispatching to the trainer.

Public API
- train_stage1(cfg): Run Stage 1 training with a provided config dict
- build_stage1_dataset(cfg, **kwargs): Construct the synthetic Stage 1 dataset
- run_stage1(defaults_path, override_path=None): Load/merge configs and train
- load_config(path): Load YAML/JSON config file to a dict
- deep_update(base, upd): Recursively merge two dicts (in-place on base)
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json

# Optional YAML support
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None  # type: ignore
    _HAS_YAML = False

# Defensive imports for trainer and dataset
try:
    from .trainer_stage1 import train_stage1 as _train_stage1  # type: ignore
except Exception:
    _train_stage1 = None  # type: ignore

try:
    from .dataset_synthetic import build_stage1_dataset as _build_stage1_dataset  # type: ignore
except Exception:
    _build_stage1_dataset = None  # type: ignore


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge upd into base (mutates base) and return base."""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a dict.

    Raises FileNotFoundError if missing. Raises ValueError if YAML parsing is
    required but PyYAML is not installed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if ext in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise ValueError("PyYAML is required to parse YAML configs. Install pyyaml.")
        return yaml.safe_load(content)  # type: ignore
    if ext == ".json":
        return json.loads(content)
    # Fallback: try YAML then JSON
    if _HAS_YAML:
        try:
            return yaml.safe_load(content)  # type: ignore
        except Exception:
            pass
    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"Unsupported config format for {path}: {e}")


def run_stage1(defaults_path: str, override_path: Optional[str] = None) -> None:
    """Load defaults and optional override config, merge, and run Stage 1 training."""
    if _train_stage1 is None:
        raise RuntimeError("Stage 1 trainer is unavailable. Ensure swiftedit/train/stage1/trainer_stage1.py imports correctly.")
    cfg = load_config(defaults_path)
    if override_path is not None and os.path.isfile(override_path):
        over = load_config(override_path)
        deep_update(cfg, over)
    _train_stage1(cfg)


# Public aliases

def train_stage1(cfg: Dict[str, Any]) -> None:  # pragma: no cover - thin wrapper
    if _train_stage1 is None:
        raise RuntimeError("Stage 1 trainer is unavailable. Ensure dependencies are installed.")
    return _train_stage1(cfg)  # type: ignore


def build_stage1_dataset(config: Dict[str, Any], **kwargs: Any):  # pragma: no cover - thin wrapper
    if _build_stage1_dataset is None:
        raise RuntimeError("Stage 1 synthetic dataset builder unavailable.")
    return _build_stage1_dataset(config, **kwargs)  # type: ignore


__all__ = [
    "train_stage1",
    "build_stage1_dataset",
    "run_stage1",
    "load_config",
    "deep_update",
]

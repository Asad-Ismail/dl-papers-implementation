from __future__ import annotations

"""
SwiftEdit Training Package

This package aggregates Stage 1 (synthetic) and Stage 2 (real) training entry points
and dataset builders. It also offers convenience helpers to load/merge configs and
run each stage programmatically.

Public API:
- train_stage1(cfg: Dict[str, Any]) -> None
- train_stage2(cfg: Dict[str, Any]) -> None
- run_stage1(defaults_path: str, override_path: Optional[str] = None) -> None
- run_stage2(defaults_path: str, override_path: Optional[str] = None) -> None
- build_stage1_dataset(config: Dict[str, Any], **kwargs) -> Dataset
- build_stage2_dataset(config: Dict[str, Any], **kwargs) -> Dataset
"""

from typing import Any, Dict, Optional

# Re-export trainers (with defensive imports)
try:
    from .stage1.trainer_stage1 import train_stage1 as _train_stage1
except Exception:  # pragma: no cover
    _train_stage1 = None  # type: ignore

try:
    from .stage2.trainer_stage2 import train_stage2 as _train_stage2
except Exception:  # pragma: no cover
    _train_stage2 = None  # type: ignore

# Re-export dataset builders
try:
    from .stage1.dataset_synthetic import build_stage1_dataset as _build_stage1_dataset
except Exception:  # pragma: no cover
    _build_stage1_dataset = None  # type: ignore

try:
    from .stage2.dataset_real import build_stage2_dataset as _build_stage2_dataset
except Exception:  # pragma: no cover
    _build_stage2_dataset = None  # type: ignore

# Optional YAML for config loading
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

import json
import os


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yml", ".yaml"):
            if yaml is None:
                raise ImportError("PyYAML is required to load YAML configs. Install pyyaml.")
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            # Try YAML first then JSON
            if yaml is not None:
                try:
                    f.seek(0)
                    return yaml.safe_load(f)
                except Exception:
                    pass
            f.seek(0)
            return json.load(f)


def run_stage1(defaults_path: str, override_path: Optional[str] = None) -> None:
    """Load configs and run Stage 1 training.

    Args:
        defaults_path: Path to defaults.yaml
        override_path: Optional path to stage1.yaml (or json) to override defaults
    """
    if _train_stage1 is None:
        raise RuntimeError("Stage 1 trainer not available (import failed)")
    cfg = _load_config(defaults_path)
    if override_path is not None and os.path.exists(override_path):
        override = _load_config(override_path)
        _deep_update(cfg, override)
    return _train_stage1(cfg)


def run_stage2(defaults_path: str, override_path: Optional[str] = None) -> None:
    """Load configs and run Stage 2 training.

    Args:
        defaults_path: Path to defaults.yaml
        override_path: Optional path to stage2.yaml (or json) to override defaults
    """
    if _train_stage2 is None:
        raise RuntimeError("Stage 2 trainer not available (import failed)")
    cfg = _load_config(defaults_path)
    if override_path is not None and os.path.exists(override_path):
        override = _load_config(override_path)
        _deep_update(cfg, override)
    return _train_stage2(cfg)


# Re-export symbols with friendly names
train_stage1 = _train_stage1  # type: ignore
train_stage2 = _train_stage2  # type: ignore
build_stage1_dataset = _build_stage1_dataset  # type: ignore
build_stage2_dataset = _build_stage2_dataset  # type: ignore


__all__ = [
    "train_stage1",
    "train_stage2",
    "run_stage1",
    "run_stage2",
    "build_stage1_dataset",
    "build_stage2_dataset",
]

"""
SwiftEdit Evaluation Package

This module aggregates evaluation utilities and datasets, providing a stable
public API for running PieBench evaluations and programmatic metric computation.

Exports:
- PieBenchDataset, build_piebench_dataset: dataset utilities
- evaluate_piebench, evaluate_sample: evaluation orchestrator and per-sample eval
- run_piebench_evaluation: convenience function to run evaluation given config paths
- load_config, deep_update: lightweight helpers for config IO and merging
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json

# Defensive imports: dataset and evaluator modules
try:
    from .piebench_loader import PieBenchDataset, build_piebench_dataset  # type: ignore
except Exception:  # pragma: no cover
    PieBenchDataset = None  # type: ignore
    build_piebench_dataset = None  # type: ignore

try:
    from .evaluate_piebench import evaluate_piebench, evaluate_sample  # type: ignore
except Exception:  # pragma: no cover
    evaluate_piebench = None  # type: ignore
    evaluate_sample = None  # type: ignore

# Optional YAML support
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_YAML = False


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionary upd into base (in place), returning base.
    Non-dict values overwrite; dict values are merged recursively.
    """
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration file from YAML or JSON into a dict.

    Raises FileNotFoundError if path does not exist.
    Raises ValueError if unable to parse due to missing parser or bad format.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yml", ".yaml"):
        if not _HAS_YAML:
            raise ValueError("PyYAML not available to parse YAML config.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)  # type: ignore
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Try YAML first if available, else JSON
    if _HAS_YAML:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)  # type: ignore
        except Exception:
            pass
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_piebench_evaluation(
    defaults_path: str,
    override_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run PieBench evaluation end-to-end.

    Parameters:
    - defaults_path: path to the base defaults YAML/JSON config
    - override_path: optional path to an override YAML/JSON config

    Returns a summary dict produced by evaluate_piebench.
    """
    if evaluate_piebench is None:
        raise RuntimeError("evaluate_piebench is not available. Import failed.")
    cfg = load_config(defaults_path)
    if override_path and os.path.isfile(override_path):
        override_cfg = load_config(override_path)
        deep_update(cfg, override_cfg)
    return evaluate_piebench(cfg)


__all__ = [
    "PieBenchDataset",
    "build_piebench_dataset",
    "evaluate_piebench",
    "evaluate_sample",
    "run_piebench_evaluation",
    "load_config",
    "deep_update",
]

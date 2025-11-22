"""
SwiftEdit Checkpoint Utilities

Provides helpers to save and load checkpoints, manage a directory of checkpoints
with retention, and load model/EMA states with reporting.

Intended usage across training stages and inference.
"""
from __future__ import annotations

import os
import re
import sys
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class LoadReport:
    missing_keys: List[str]
    unexpected_keys: List[str]
    error_msgs: List[str]

    def ok(self) -> bool:
        return len(self.error_msgs) == 0

    def summary(self) -> str:
        return (
            f"LoadReport(missing={len(self.missing_keys)}, unexpected={len(self.unexpected_keys)}, "
            f"errors={len(self.error_msgs)})"
        )


# -----------------------------
# Filesystem helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _is_checkpoint_file(path: str) -> bool:
    return path.endswith(".pt") or path.endswith(".pth") or path.endswith(".ckpt")


# -----------------------------
# Basic save/load utilities
# -----------------------------

def save_checkpoint(file_path: str, payload: Dict[str, Any], create_dirs: bool = True) -> None:
    """Save a Python dict checkpoint via torch.save.

    Args:
        file_path: output path ending with .pt/.pth/.ckpt
        payload: dictionary to serialize (models/optimizers/configs/etc)
        create_dirs: create parent dirs if they don't exist
    """
    if not _HAS_TORCH:
        raise ImportError("torch is required to save checkpoints")
    parent = os.path.dirname(os.path.abspath(file_path))
    if create_dirs:
        ensure_dir(parent)
    torch.save(payload, file_path)


def load_checkpoint(file_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint dict via torch.load.

    Returns the loaded dictionary, or raises FileNotFoundError/ImportError.
    """
    if not _HAS_TORCH:
        raise ImportError("torch is required to load checkpoints")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")
    return torch.load(file_path, map_location=map_location or "cpu")


# -----------------------------
# Model state helpers
# -----------------------------

def state_dict_from_model(model: Any) -> Dict[str, Any]:
    """Return model.state_dict() if available; raises otherwise."""
    if not hasattr(model, "state_dict"):
        raise AttributeError("Model has no state_dict method")
    return model.state_dict()


def load_model_state(model: Any, state_dict: Dict[str, Any], strict: bool = False) -> LoadReport:
    """Load state_dict into model and return a detailed report."""
    if not hasattr(model, "load_state_dict"):
        raise AttributeError("Model has no load_state_dict method")
    report = LoadReport(missing_keys=[], unexpected_keys=[], error_msgs=[])
    try:
        res = model.load_state_dict(state_dict, strict=strict)
        # res is a NamedTuple MissingKeys/UnexpectedKeys for torch
        missing = getattr(res, "missing_keys", [])
        unexpected = getattr(res, "unexpected_keys", [])
        report.missing_keys = list(missing)
        report.unexpected_keys = list(unexpected)
    except Exception as e:
        report.error_msgs.append(str(e))
    return report


def maybe_load_ema_shadow(model: Any, checkpoint: Dict[str, Any], key: str = "ema_shadow", strict: bool = False) -> Optional[LoadReport]:
    """Load EMA shadow parameters into model if present, otherwise return None."""
    shadow = checkpoint.get(key)
    if shadow is None:
        return None
    # Shadow might be a flat tensor list or a state_dict; prefer state_dict
    if isinstance(shadow, dict):
        return load_model_state(model, shadow, strict=strict)
    # If it's a list of tensors, apply by matching parameter order
    if hasattr(model, "parameters"):
        with torch.no_grad():
            for p, s in zip(model.parameters(), shadow):
                try:
                    p.copy_(s)
                except Exception:
                    pass
        return LoadReport(missing_keys=[], unexpected_keys=[], error_msgs=[])
    return None


# -----------------------------
# Bundle save/load
# -----------------------------

def build_checkpoint_payload(
    models: Dict[str, Any],
    optimizers: Optional[Dict[str, Any]] = None,
    schedulers: Optional[Dict[str, Any]] = None,
    ema_shadow: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized checkpoint payload dict.

    - models: mapping name -> nn.Module
    - optimizers/schedulers: name -> torch.optim / torch.optim.lr_scheduler objects
    - ema_shadow: state_dict of EMA weights for primary trainable model(s)
    - cfg: configuration dict to snapshot
    - step: training iteration/step
    - extra: any additional metadata
    """
    payload: Dict[str, Any] = {
        "version": 1,
        "timestamp": time.time(),
        "step": step,
        "models": {},
    }
    if cfg is not None:
        payload["config"] = cfg
    if optimizers is not None:
        payload["optimizers"] = {k: v.state_dict() for k, v in optimizers.items()}
    if schedulers is not None:
        payload["schedulers"] = {k: v.state_dict() for k, v in schedulers.items()}
    if ema_shadow is not None:
        payload["ema_shadow"] = ema_shadow
    if extra is not None:
        payload["extra"] = extra

    for name, model in models.items():
        try:
            sd = state_dict_from_model(model)
        except Exception as e:
            sd = {"_error": str(e)}
        payload["models"][name] = sd
    return payload


def save_models_bundle(
    out_path: str,
    models: Dict[str, Any],
    optimizers: Optional[Dict[str, Any]] = None,
    schedulers: Optional[Dict[str, Any]] = None,
    ema_model: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a bundle of models/optimizers/schedulers and optional EMA shadow to a checkpoint path."""
    ema_shadow = None
    if ema_model is not None:
        try:
            ema_shadow = state_dict_from_model(ema_model)
        except Exception:
            ema_shadow = None
    payload = build_checkpoint_payload(models, optimizers, schedulers, ema_shadow, cfg, step, extra)
    save_checkpoint(out_path, payload, create_dirs=True)


def load_models_from_checkpoint(
    ckpt_path: str,
    models: Dict[str, Any],
    map_location: Optional[str] = None,
    use_ema_if_available: bool = False,
    strict: bool = False,
) -> Tuple[Dict[str, LoadReport], Dict[str, Any]]:
    """Load models from a checkpoint into the provided model instances.

    Returns:
        (reports, checkpoint_dict)
    where reports maps model name -> LoadReport.
    """
    checkpoint = load_checkpoint(ckpt_path, map_location=map_location)
    model_states: Dict[str, Any] = checkpoint.get("models", {})
    reports: Dict[str, LoadReport] = {}
    for name, model in models.items():
        report: Optional[LoadReport] = None
        # Prefer EMA when requested
        if use_ema_if_available:
            report = maybe_load_ema_shadow(model, checkpoint, key="ema_shadow", strict=strict)
        if report is None:
            state = model_states.get(name)
            if state is None:
                report = LoadReport(missing_keys=[], unexpected_keys=[], error_msgs=[f"missing state for {name}"])
            else:
                report = load_model_state(model, state, strict=strict)
        reports[name] = report
    return reports, checkpoint


# -----------------------------
# Config snapshot utilities
# -----------------------------

def save_config_snapshot(cfg: Dict[str, Any], out_path: str) -> None:
    """Save configuration dict to YAML or JSON file."""
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    if _HAS_YAML:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


# -----------------------------
# Checkpoint manager to handle retention and naming
# -----------------------------
class CheckpointManager:
    """Manage checkpoint files within a directory with a naming scheme and retention policy."""

    def __init__(self, dir_path: str, prefix: str = "ckpt", keep_last: int = 5):
        self.dir_path = dir_path
        self.prefix = prefix
        self.keep_last = max(1, int(keep_last))
        ensure_dir(self.dir_path)
        self._pattern = re.compile(rf"^{re.escape(prefix)}_iter(\d+)(?:\.pt|\.pth|\.ckpt)$")

    def _build_name(self, iteration: int) -> str:
        return f"{self.prefix}_iter{int(iteration):06d}.pt"

    def save(self, iteration: int, payload: Dict[str, Any]) -> str:
        path = os.path.join(self.dir_path, self._build_name(iteration))
        save_checkpoint(path, payload, create_dirs=True)
        self.cleanup()
        return path

    def list(self) -> List[str]:
        files = []
        for fn in os.listdir(self.dir_path):
            if _is_checkpoint_file(fn) and fn.startswith(self.prefix + "_iter"):
                files.append(os.path.join(self.dir_path, fn))
        files.sort()
        return files

    def latest(self) -> Optional[str]:
        files = self.list()
        return files[-1] if files else None

    def cleanup(self) -> None:
        files = self.list()
        if len(files) <= self.keep_last:
            return
        to_remove = files[:-self.keep_last]
        for p in to_remove:
            try:
                os.remove(p)
            except Exception:
                pass

    def load_latest(self, map_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        latest = self.latest()
        if latest is None:
            return None
        return load_checkpoint(latest, map_location=map_location)


__all__ = [
    "LoadReport",
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
    "CheckpointManager",
]

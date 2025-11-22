"""
SwiftEdit package initializer.

This module exposes convenient top-level utilities and re-exports for
building models from configuration, loading configs, and resolving devices.

Primary public utilities:
- __version__: Package version string
- get_version(): Return version string
- load_config(path: str) -> Dict[str, Any]: Load a YAML or JSON config file
- deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]: Deep merge
- merge_configs(defaults_path: str, override_path: Optional[str]) -> Dict[str, Any]
- resolve_device(cfg: Dict[str, Any]) -> torch.device
- build_models_from_config(cfg: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]

Convenience re-exports are provided for subpackages:
- models, train, edit, eval, losses, schedulers, utils
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

# Optional YAML support
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

# Optional torch for device resolution
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

# Re-export models builder
from .models import build_models_from_config as _build_models_from_config

# Convenience imports (do not error if missing optional submodules)
try:
    from . import models as models  # noqa: F401
except Exception:
    models = None  # type: ignore

try:
    from . import train as train  # noqa: F401
except Exception:
    train = None  # type: ignore

try:
    from . import edit as edit  # noqa: F401
except Exception:
    edit = None  # type: ignore

try:
    from . import eval as eval  # noqa: F401
except Exception:
    eval = None  # type: ignore

try:
    from . import losses as losses  # noqa: F401
except Exception:
    losses = None  # type: ignore

try:
    from . import schedulers as schedulers  # noqa: F401
except Exception:
    schedulers = None  # type: ignore

try:
    from . import utils as utils  # noqa: F401
except Exception:
    utils = None  # type: ignore


__version__ = "0.1.0"


def get_version() -> str:
    """Return the SwiftEdit package version string."""
    return __version__


def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration file from YAML or JSON.

    Args:
        path: Path to a YAML or JSON config file.
    Returns:
        A dictionary representing the configuration.
    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file format is unsupported.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yml", ".yaml"):
            if not _HAS_YAML:
                raise ValueError("PyYAML not available; cannot load YAML configs.")
            return yaml.safe_load(f)  # type: ignore
        elif ext == ".json":
            return json.load(f)
        else:
            # Try YAML first, then JSON as fallback
            if _HAS_YAML:
                try:
                    f.seek(0)
                    return yaml.safe_load(f)  # type: ignore
                except Exception:
                    pass
            f.seek(0)
            try:
                return json.load(f)
            except Exception as e:
                raise ValueError(f"Unsupported config format for {path}: {e}")


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Deeply merge two dictionaries (in-place on base) and return base.

    Nested dictionaries in `upd` overwrite or merge into `base`.
    Other types are assigned directly.
    """
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def merge_configs(defaults_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    """Load defaults YAML/JSON and optionally merge an override file.

    Args:
        defaults_path: Path to defaults config.
        override_path: Optional path to overrides.
    Returns:
        Merged config dictionary.
    """
    cfg = load_config(defaults_path)
    if override_path and os.path.isfile(override_path):
        override = load_config(override_path)
        cfg = deep_update(cfg, override)
    return cfg


def resolve_device(cfg: Optional[Dict[str, Any]] = None) -> "torch.device":
    """Resolve device from config or environment.

    Prefers cfg["system"]["device"] if present, otherwise chooses CUDA if available.
    Raises ImportError if torch is not available.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required to resolve device.")
    dev_str = None
    if cfg is not None:
        dev_str = (
            cfg.get("system", {}).get("device")  # type: ignore[union-attr]
            if isinstance(cfg.get("system"), dict)
            else None
        )
    if dev_str is not None:
        return torch.device(dev_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models_from_config(cfg: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    """Public wrapper for the centralized model builder.

    This delegates to swiftedit.models.build_models_from_config and returns
    a dictionary of instantiated models.
    """
    return _build_models_from_config(cfg, device=device)


__all__ = [
    "__version__",
    "get_version",
    "load_config",
    "deep_update",
    "merge_configs",
    "resolve_device",
    "build_models_from_config",
    "models",
    "train",
    "edit",
    "eval",
    "losses",
    "schedulers",
    "utils",
]

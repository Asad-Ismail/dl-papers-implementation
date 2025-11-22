"""
SwiftEdit Logging Utilities

Provides simple experiment logging, version info, config snapshotting, and CSV metric/result writing.

Public API:
- ExperimentLogger: class managing log directory, version logging, config snapshots, and metrics.
- build_logger_from_config(cfg, base_log_dir=None): factory building an ExperimentLogger from a config dict.
- get_env_info(): returns a dictionary of environment and package versions.
- save_yaml(data, path): saves a Python dict to a YAML file.

This module is intentionally lightweight and does not depend on heavy logging frameworks.
"""
from __future__ import annotations

import os
import sys
import time
import json
import csv
from datetime import datetime
from typing import Any, Dict, Optional, Union

# Optional dependencies
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_get(d: Dict[str, Any], keys: Union[str, list], default: Any = None) -> Any:
    """Safely get nested config values using dot path or list of keys."""
    if isinstance(keys, str):
        keys = keys.split(".")
    cur: Any = d
    for k in keys:  # type: ignore
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def get_env_info() -> Dict[str, Any]:
    """Collect environment and package version info.

    Returns a dictionary containing versions of core packages and CUDA/device info.
    """
    info: Dict[str, Any] = {}
    # Python and OS
    info["python"] = sys.version.replace("\n", " ")
    info["platform"] = sys.platform

    # Torch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            try:
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_compute_capability"] = torch.cuda.get_device_capability(0)
            except Exception:
                pass
        # Compile availability
        info["torch_compile_available"] = hasattr(torch, "compile")
    except Exception as e:
        info["torch_error"] = str(e)

    # torchvision
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except Exception:
        pass

    # diffusers
    try:
        import diffusers
        info["diffusers_version"] = getattr(diffusers, "__version__", "unknown")
    except Exception:
        pass

    # transformers
    try:
        import transformers
        info["transformers_version"] = getattr(transformers, "__version__", "unknown")
    except Exception:
        pass

    # open-clip
    try:
        import open_clip
        info["open_clip_version"] = getattr(open_clip, "__version__", "unknown")
    except Exception:
        pass

    # timm
    try:
        import timm
        info["timm_version"] = getattr(timm, "__version__", "unknown")
    except Exception:
        pass

    # numpy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        pass

    # pandas
    try:
        import pandas as _pd
        info["pandas_version"] = _pd.__version__
    except Exception:
        pass

    return info


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save a dict to YAML at the given path. Falls back to JSON if PyYAML missing."""
    _ensure_dir(os.path.dirname(path))
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        # Fallback to JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class ExperimentLogger:
    """Simple experiment logger managing directories, config snapshots, and metrics.

    Typical usage:
      logger = ExperimentLogger(cfg)
      logger.setup()  # saves config snapshot and version info
      logger.log("Starting training...")
      logger.log_metrics(step, {"loss": 0.123})
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        base_log_dir: Optional[str] = None,
        results_csv_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        project_name = _safe_get(cfg, "logging.project_name", default="SwiftEdit")
        base_dir = base_log_dir or _safe_get(cfg, "paths.logs_dir", default="logs")
        ts = _timestamp()
        run_name = run_name or f"{project_name}_{ts}"
        self.run_dir = os.path.join(base_dir, run_name)
        _ensure_dir(self.run_dir)

        # results directory (can be global or under run_dir)
        default_results_dir = _safe_get(cfg, "logging.results_csv_dir", default="results")
        if results_csv_dir is None:
            # Place results under global dir, not per-run
            self.results_dir = default_results_dir
        else:
            self.results_dir = results_csv_dir
        _ensure_dir(self.results_dir)

        # Paths
        self.log_file = os.path.join(self.run_dir, "log.txt")
        self.config_snapshot_path = os.path.join(self.run_dir, "config.yaml")
        self.env_info_path = os.path.join(self.run_dir, "env.json")
        self.metrics_csv_path = os.path.join(self.run_dir, "metrics.csv")

        # Flags
        self.version_log = bool(_safe_get(cfg, "logging.version_log", default=True))
        self.save_config_flag = bool(_safe_get(cfg, "logging.save_config_snapshot", default=True))

        # Internal state
        self._log_fp: Optional[Any] = None

    def setup(self) -> None:
        """Perform initial setup: save config snapshot and environment info, open log file."""
        # Save config snapshot
        if self.save_config_flag:
            try:
                save_yaml(self.cfg, self.config_snapshot_path)
            except Exception:
                # Fallback to JSON
                with open(self.config_snapshot_path.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
                    json.dump(self.cfg, f, indent=2)
        # Save environment info
        if self.version_log:
            env = get_env_info()
            try:
                with open(self.env_info_path, "w", encoding="utf-8") as f:
                    json.dump(env, f, indent=2)
            except Exception:
                pass
        # Open log file
        try:
            self._log_fp = open(self.log_file, "a", encoding="utf-8")
            self.log("Logger initialized.")
        except Exception:
            self._log_fp = None

    def log(self, msg: str, also_print: bool = True) -> None:
        """Write a message to the log file with timestamp; also print to stdout by default."""
        line = f"[{_timestamp()}] {msg}"
        if self._log_fp is not None:
            try:
                self._log_fp.write(line + "\n")
                self._log_fp.flush()
            except Exception:
                pass
        if also_print:
            print(line)

    def log_metrics(self, step: Union[int, float, str], metrics: Dict[str, Union[int, float]], csv_name: Optional[str] = None) -> None:
        """Append metrics to a CSV under the run directory.

        If pandas is available, uses DataFrame append; otherwise, uses the csv module.
        """
        csv_path = os.path.join(self.run_dir, csv_name) if csv_name else self.metrics_csv_path
        _ensure_dir(os.path.dirname(csv_path))
        row: Dict[str, Any] = {"step": step}
        row.update(metrics)
        # Try pandas
        if pd is not None:
            try:
                df = pd.DataFrame([row])
                if os.path.isfile(csv_path):
                    # Append without headers
                    df.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(csv_path, index=False)
                return
            except Exception:
                pass
        # Fallback to csv module
        write_header = not os.path.isfile(csv_path)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            self.log(f"Failed to write metrics CSV: {e}")

    def save_results_row(self, row: Dict[str, Any], csv_filename: str = "results.csv") -> str:
        """Save a single results row to a CSV in the global results directory.

        Returns the path to the CSV file.
        """
        csv_path = os.path.join(self.results_dir, csv_filename)
        _ensure_dir(os.path.dirname(csv_path))
        # Try pandas first
        if pd is not None:
            try:
                df = pd.DataFrame([row])
                if os.path.isfile(csv_path):
                    df.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(csv_path, index=False)
                return csv_path
            except Exception:
                pass
        # Fallback to csv module
        write_header = not os.path.isfile(csv_path)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            self.log(f"Failed to write results CSV: {e}")
        return csv_path

    def close(self) -> None:
        """Close log file if open."""
        if self._log_fp is not None:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None


def build_logger_from_config(cfg: Dict[str, Any], base_log_dir: Optional[str] = None, results_csv_dir: Optional[str] = None) -> ExperimentLogger:
    """Factory to build and setup an ExperimentLogger from a config dict."""
    logger = ExperimentLogger(cfg, base_log_dir=base_log_dir, results_csv_dir=results_csv_dir)
    logger.setup()
    return logger


__all__ = [
    "ExperimentLogger",
    "build_logger_from_config",
    "get_env_info",
    "save_yaml",
]

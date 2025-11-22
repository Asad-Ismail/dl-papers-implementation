"""
EMA utilities for the inversion network (Fθ).

Implements an Exponential Moving Average (EMA) helper that tracks shadow
parameters of a target model and provides methods to update, copy, and
serialize EMA weights. This module is intended to be used during Stage 1 and
Stage 2 training to stabilize the inversion network.

Public API:
- class EMA: Exponential moving average of model parameters
  - __init__(model=None, decay=0.999, use_num_updates=True, device=None)
  - register(model): register parameters to track (optional if provided at init)
  - update(model): update shadow params with current model params
  - copy_to(model): load EMA weights into the given model
  - state_dict(): serialize EMA shadows and config
  - load_state_dict(state): restore EMA shadows and config
  - to(device): move EMA tensors to device
  - param_names(): list of parameter names tracked

- function build_ema_from_config(cfg: Dict[str, Any], model: nn.Module) -> EMA

Notes:
- Only parameters that require gradients at registration time are tracked.
- Buffers are not tracked; EMA focuses on trainable parameters.
- EMA shadows are stored in float32 for numerical stability by default.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EMA:
    """Exponential Moving Average of model parameters.

    Args:
        model: Optional nn.Module to register immediately.
        decay: EMA decay factor in (0, 1). Typical ~0.999.
        use_num_updates: If True, use bias-corrected decay based on num_updates
                         (i.e., dynamic momentum: decay = min(decay, (1 + i) / (10 + i))).
        device: Optional device to store shadow params; defaults to CUDA if available.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        decay: float = 0.999,
        use_num_updates: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay: float = float(decay)
        self.use_num_updates: bool = bool(use_num_updates)
        self.device: torch.device = device if device is not None else _infer_device()
        self._num_updates: int = 0
        self._registered: bool = False
        self._names: List[str] = []
        self._shadow: Dict[str, torch.Tensor] = {}

        if model is not None:
            self.register(model)

    def register(self, model: nn.Module) -> None:
        """Register parameters of the model to track in EMA.

        Only parameters with requires_grad=True at registration are tracked.
        """
        self._names.clear()
        self._shadow.clear()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Store shadow copy in float32 for stability
            self._shadow[name] = param.detach().data.float().to(self.device).clone()
            self._names.append(name)
        self._registered = True
        self._num_updates = 0

    def _compute_decay(self) -> float:
        if not self.use_num_updates:
            return self.decay
        # Bias-corrected or warmup-like dynamic decay. The exact schedule is
        # heuristic; keep it simple and monotonic towards self.decay.
        i = self._num_updates
        dynamic = min(self.decay, (1.0 + i) / (10.0 + i))
        return float(dynamic)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadows using current model parameters.

        Shadow = d * Shadow + (1 - d) * Param
        """
        if not self._registered:
            # Auto-register on first update
            self.register(model)
        d = self._compute_decay()
        for name, param in model.named_parameters():
            if name not in self._shadow:
                # If new parameter appears (unlikely), initialize on the fly
                if param.requires_grad:
                    self._shadow[name] = param.detach().data.float().to(self.device).clone()
                    if name not in self._names:
                        self._names.append(name)
                continue
            if not param.requires_grad:
                # Skip non-trainable params
                continue
            shadow = self._shadow[name]
            # Move param to EMA device and cast to float32
            p = param.detach().data.to(self.device).float()
            shadow.mul_(d).add_(p, alpha=1.0 - d)
        self._num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module, strict: bool = False) -> None:
        """Copy EMA shadows into the target model parameters."""
        missing: List[str] = []
        for name, param in model.named_parameters():
            if name in self._shadow:
                src = self._shadow[name]
                # Cast back to param dtype and device
                param.detach().data.copy_(src.to(param.device).to(param.dtype))
            else:
                missing.append(name)
        if strict and missing:
            raise RuntimeError(f"EMA.copy_to missing parameters: {missing}")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "use_num_updates": self.use_num_updates,
            "num_updates": self._num_updates,
            "device": str(self.device),
            "names": list(self._names),
            # Store CPU tensors to reduce checkpoint size coupling to device
            "shadow": {k: v.detach().cpu() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.use_num_updates = bool(state.get("use_num_updates", self.use_num_updates))
        self._num_updates = int(state.get("num_updates", 0))
        names = state.get("names", [])
        self._names = list(names)
        shadow = state.get("shadow", {})
        # Load to configured device
        self._shadow = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                         for k, v in shadow.items()}
        self._registered = True if len(self._shadow) > 0 else False

    def to(self, device: Optional[torch.device] = None) -> "EMA":
        new_dev = device if device is not None else self.device
        if new_dev == self.device:
            return self
        for k in list(self._shadow.keys()):
            self._shadow[k] = self._shadow[k].to(new_dev)
        self.device = new_dev
        return self

    def param_names(self) -> List[str]:
        return list(self._names)


def build_ema_from_config(cfg: Dict[str, Any], model: nn.Module) -> EMA:
    """Factory to build EMA from a configuration dict and register a model.

    Expected cfg keys:
        cfg["models"]["inversion_net"]["ema"]["enabled"]: bool
        cfg["models"]["inversion_net"]["ema"]["decay"]: float
    If not enabled, this function still returns an EMA instance for convenience,
    but the caller may choose not to use it.
    """
    try:
        ema_cfg = cfg.get("models", {}).get("inversion_net", {}).get("ema", {})
    except Exception:
        ema_cfg = {}
    decay = float(ema_cfg.get("decay", 0.999))
    enabled = bool(ema_cfg.get("enabled", True))
    device = _infer_device()
    ema = EMA(model if enabled else None, decay=decay, use_num_updates=True, device=device)
    if enabled and not ema._registered:
        ema.register(model)
    return ema


__all__ = ["EMA", "build_ema_from_config"]

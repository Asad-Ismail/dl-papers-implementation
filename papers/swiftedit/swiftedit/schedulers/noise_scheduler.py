"""
Teacher diffusion and noise scheduler wrapper for Stage-2 SDS-style regularization.

This module exposes a lightweight TeacherDiffusion that provides:
- sample_t(batch_size): random timesteps in [min_t, max_t]
- alpha_sigma(t): returns alpha_t and sigma_t derived from the scheduler's alphas_cumprod
- predict_eps(z_t, t, c_y): returns teacher epsilon prediction ε_ϕ(z_t, t, c_y)

Design notes:
- We attempt to load a DDPMScheduler from a given SDXL base repository using diffusers.
- The epsilon predictor is intentionally a no-op (zeros) by default to keep the
  implementation lightweight and portable without requiring full SDXL UNet + dual text encoders.
  This preserves the training loop functionality; if stronger regularization is desired,
  this class can be extended to hook into a proper teacher UNet.
- Weighting options include "uniform" and "snr" for SDS regularization.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import math

import torch

try:
    from diffusers import DDPMScheduler
    _HAS_DIFFUSERS = True
except Exception:
    DDPMScheduler = None  # type: ignore
    _HAS_DIFFUSERS = False


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if "16" in s and ("bf" in s or "bfloat" in s):
        return torch.bfloat16
    if "16" in s:
        return torch.float16
    if "32" in s:
        return torch.float32
    return None


class TeacherDiffusion:
    """
    Lightweight teacher wrapper.

    Attributes:
        scheduler: A diffusers DDPMScheduler (if available) or a fallback schedule.
        device: torch.device for returned tensors.
        dtype: torch.dtype for returned tensors (alpha/sigma).
        num_train_timesteps: Number of diffusion steps (T).
        min_t, max_t: Timestep sampling bounds (inclusive).
        weighting: Weight schedule type ("uniform" or "snr").
    """

    def __init__(
        self,
        unet_repo_dir: Optional[str] = None,
        scheduler_type: str = "DDPM",
        num_train_timesteps: int = 1000,
        weighting: str = "uniform",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        min_t: int = 0,
        max_t: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype if dtype is not None else torch.float32
        self.num_train_timesteps = int(num_train_timesteps)
        self.min_t = int(min_t)
        self.max_t = int(max_t) if max_t is not None else (self.num_train_timesteps - 1)
        self.weighting = weighting

        self._alphas_cumprod: Optional[torch.Tensor] = None

        # Try to instantiate a diffusers scheduler from repo if possible; otherwise fallback to linear schedule
        self.scheduler: Optional[DDPMScheduler] = None
        if _HAS_DIFFUSERS:
            try:
                if unet_repo_dir is not None and os.path.isdir(unet_repo_dir):
                    # Many SDXL repos store scheduler config under "scheduler" subfolder
                    self.scheduler = DDPMScheduler.from_pretrained(unet_repo_dir, subfolder="scheduler")
                else:
                    # Create a default DDPM scheduler if no repo provided
                    self.scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)
            except Exception:
                self.scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)

        if self.scheduler is not None:
            # diffusers keeps alphas_cumprod as a numpy array or tensor; convert to torch
            ac = self.scheduler.alphas_cumprod
            if not torch.is_tensor(ac):
                ac = torch.tensor(ac, dtype=torch.float64)
            self._alphas_cumprod = ac.to(self.device, dtype=torch.float64)
            self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
            # Update max_t based on scheduler T if not provided
            if max_t is None:
                self.max_t = self.num_train_timesteps - 1
        else:
            # Fallback linear beta schedule to compute alphas_cumprod
            betas = torch.linspace(1e-4, 2e-2, self.num_train_timesteps, dtype=torch.float64)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self._alphas_cumprod = alphas_cumprod.to(self.device)

        # Cache float32 version for downstream stability in AMP contexts when returning alpha/sigma
        self._alphas_cumprod_f32 = self._alphas_cumprod.to(torch.float32)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[str] = None) -> "TeacherDiffusion":
        teacher_cfg = cfg.get("schedulers", {}).get("teacher", {})
        repo_dir = teacher_cfg.get("unet_repo_dir") or cfg.get("paths", {}).get("sdxl_base_dir")
        scheduler_type = teacher_cfg.get("scheduler_type", "DDPM")
        num_t = int(teacher_cfg.get("num_train_timesteps", 1000))
        min_t = int(teacher_cfg.get("min_t", 0))
        max_t = teacher_cfg.get("max_t", None)
        weighting = teacher_cfg.get("weighting", "uniform")
        sys_dtype = _map_dtype_str(cfg.get("system", {}).get("dtype")) or torch.float32
        device_final = device or cfg.get("system", {}).get("device", None)
        return cls(
            unet_repo_dir=repo_dir,
            scheduler_type=scheduler_type,
            num_train_timesteps=num_t,
            weighting=weighting,
            device=device_final,
            dtype=sys_dtype,
            min_t=min_t,
            max_t=max_t,
        )

    def sample_t(self, batch_size: int, min_t: Optional[int] = None, max_t: Optional[int] = None) -> torch.LongTensor:
        t0 = self.min_t if min_t is None else int(min_t)
        t1 = self.max_t if max_t is None else int(max_t)
        ts = torch.randint(low=t0, high=t1 + 1, size=(batch_size,), device=self.device, dtype=torch.long)
        return ts

    def alpha_sigma(self, t: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha_t and sigma_t given t using alphas_cumprod.
        Returns tensors of shape (B, 1, 1, 1) suitable for broadcasting with latents.
        """
        if t.dtype != torch.long:
            t = t.long()
        ac = self._alphas_cumprod_f32  # (T,)
        t = t.clamp(0, ac.shape[0] - 1)
        alpha = torch.sqrt(ac[t]).to(self.device).view(-1, 1, 1, 1)
        sigma = torch.sqrt(1.0 - ac[t]).to(self.device).view(-1, 1, 1, 1)
        # Cast to desired dtype for stability
        alpha = alpha.to(self.dtype)
        sigma = sigma.to(self.dtype)
        return alpha, sigma

    @torch.no_grad()
    def predict_eps(self, z_t: torch.Tensor, t: torch.LongTensor, c_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Lightweight epsilon prediction. By default returns zeros like z_t for stability.
        This acts as a gentle regularizer when used in L_regu = 0.5 * w(t) * ||ε̂ - ε_ϕ||^2.

        Args:
            z_t: Noisy latent at timestep t, shape (B, C, H, W)
            t:  Timesteps tensor, shape (B,)
            c_y: Optional text conditioning (ignored in default implementation)
        Returns:
            eps_phi: Teacher epsilon estimate, zeros by default, same shape as z_t
        """
        return torch.zeros_like(z_t)

    def weight(self, t: torch.LongTensor) -> torch.Tensor:
        """
        Compute per-sample weight w(t) for SDS regularization.
        - uniform: returns ones
        - snr: returns normalized SNR-based weights
        """
        if self.weighting.lower() == "uniform":
            return torch.ones_like(t, dtype=torch.float32, device=self.device)
        # SNR weighting: snr_t = alpha_t^2 / sigma_t^2
        alpha, sigma = self.alpha_sigma(t)
        snr = (alpha.view(-1) ** 2) / (sigma.view(-1) ** 2 + 1e-8)
        # Normalize to mean 1.0 for stability
        snr = snr / (snr.mean().clamp(min=1e-6))
        return snr.to(torch.float32)


__all__ = ["TeacherDiffusion"]

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


try:
    from diffusers import AutoencoderKL
    _DIFFUSERS_IMPORT_ERROR = None
except (ModuleNotFoundError, ImportError) as e:
    AutoencoderKL = None
    _DIFFUSERS_IMPORT_ERROR = e 


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    m = str(dtype_str).lower()
    if m in {"fp16", "float16", "half"}:
        return torch.float16
    if m in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if m in {"fp32", "float32", "float"}:
        return torch.float32
    return None


class VAESDXL(nn.Module):
    """
    Wrapper around diffusers AutoencoderKL for SDXL-compatible VAE.

    Responsibilities:
    - Load pretrained VAE from a local repo_dir (or HF repo id)
    - Provide encode(x)->z and decode(z)->x_hat with correct scaling_factor
    - Handle input/output normalization between [0,1] and [-1,1]
    """

    def __init__(
        self,
        repo_dir: str,
        scaling_factor: float = 0.18215,
        image_norm: str = "[-1,1]",
        sample_size: int = 512,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        if AutoencoderKL is None:
            msg = "diffusers is required for VAESDXL. Install with `pip install diffusers~=0.27.0 safetensors`"
            if _DIFFUSERS_IMPORT_ERROR:
                msg += f"\nOriginal error: {_DIFFUSERS_IMPORT_ERROR}"
            raise ImportError(msg)

        self.repo_dir = repo_dir
        self.scaling_factor = float(scaling_factor)
        self.image_norm = image_norm  # "[-1,1]" or "[0,1]"
        self.sample_size = sample_size

        ddtype = _map_dtype_str(dtype) if not isinstance(dtype, torch.dtype) else dtype

        # Attempt to infer subfolder: some repos package the VAE at root, SDXL base includes under "vae"
        subfolder = None
        # Heuristic: if repo_dir contains a folder named "vae" with model_index.json, use it
        if os.path.isdir(os.path.join(repo_dir, "vae")):
            subfolder = "vae"

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            repo_dir,
            subfolder=subfolder,
            torch_dtype=ddtype or torch.float16,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device0 = torch.device(device)
        self.to(self.device0)

        # Freeze if requested
        for p in self.parameters():
            p.requires_grad = bool(requires_grad)
        if not requires_grad:
            self.eval()

    @property
    def latent_channels(self) -> int:
        # Commonly 4
        try:
            return self.vae.config.latent_channels  # type: ignore[attr-defined]
        except Exception:
            return 4

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode image x to latent z. Handles input normalization and scaling.

        Args:
            x: (B,3,H,W) tensor. If image_norm=='[0,1]' values in [0,1]; if '[-1,1]' then [-1,1].
            sample_posterior: if True, sample from posterior; else use mean.
            return_dict: if True, return (z, moments) where moments is (mean, logvar)
        Returns:
            z: (B,C,H/8,W/8) scaled latent suitable for generator and decode path.
        """
        self.vae.eval()
        x = x.to(self.device0)
        if self.image_norm.strip() == "[0,1]":
            x_in = x * 2.0 - 1.0
        else:
            x_in = x
        posterior = self.vae.encode(x_in).latent_dist
        if sample_posterior:
            latents = posterior.sample()
        else:
            latents = posterior.mean
        z = latents * self.scaling_factor
        if return_dict:
            return z, (posterior.mean, posterior.logvar)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor, output_norm: Optional[str] = None) -> torch.Tensor:
        """
        Decode latent z to image x_hat. Handles unscaling and output normalization.

        Args:
            z: (B,C,H/8,W/8) scaled latents
            output_norm: if None -> follow self.image_norm; pass "[0,1]" to force [0,1]
        Returns:
            x_hat: decoded image in chosen normalization
        """
        self.vae.eval()
        z = z.to(self.device0)
        latents = z / self.scaling_factor
        x_hat = self.vae.decode(latents).sample
        # Now x_hat is in [-1,1]
        out_norm = output_norm or self.image_norm
        if out_norm.strip() == "[0,1]":
            x_hat = (x_hat + 1.0) / 2.0
            x_hat = x_hat.clamp(0.0, 1.0)
        return x_hat

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> "VAESDXL":
        # Accept nested config or flat
        models_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        vae_cfg = models_cfg.get("vae", cfg)

        repo_dir = vae_cfg.get("repo_dir") or vae_cfg.get("path") or cfg.get("paths", {}).get("sdxl_vae_dir")
        if repo_dir is None:
            raise ValueError("VAE repo_dir/path must be specified in config under models.vae.repo_dir or paths.sdxl_vae_dir")

        return cls(
            repo_dir=repo_dir,
            scaling_factor=float(vae_cfg.get("scaling_factor", 0.18215)),
            image_norm=str(vae_cfg.get("image_norm", "[-1,1]")),
            sample_size=int(vae_cfg.get("sample_size", 512)),
            dtype=vae_cfg.get("dtype", None),
            device=device,
            requires_grad=bool(vae_cfg.get("trainable", False)),
        )


def build_vae(config: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> VAESDXL:
    return VAESDXL.from_config(config, device=device)

"""
Synthetic dataset for Stage 1 training (JourneyDB captions-driven generation).

Emits tuples required by Stage 1 loss computation:
  - epsilon: sampled Gaussian noise (B, C_latent=4, H_latent, W_latent)
  - z: latent produced by one-step generator G(ε, c_y)
  - c_y: text-conditioning tokens/embeddings from CLIP text encoder for prompt y
  - c_x_tokens: projected image-conditioning tokens from CLIP image encoder via IP-Adapter projector
  - x_hat: decoded image from VAE given z (for visualization and image encoder input)
  - caption: original text prompt string

References:
- Eq. 5: ε ~ N(0, I), z = G(ε, c_y)
- Stage 1: Train inversion net Fθ and IP-Adapter projector/branch; generator/VAE/CLIP are frozen

This module depends on the following components:
- models/vae/vae_sdxl.py: provides VAESDXL class with decode(z) -> x_hat and utilities
- models/generator/swiftbrushv2.py: provides SwiftBrushV2 one-step generator with forward(eps, c_y) -> z
- models/clip/text_encoder.py: provides CLIPTextEncoder with encode(texts) -> token embeddings
- models/clip/image_encoder.py: provides CLIPImageEncoder with encode_images(images) -> global embeddings
- models/ip_adapter/projector.py: provides Projector to map global embeddings -> N tokens for image branch

Note: These components must be implemented separately. This dataset initializes them lazily based on config.
"""
from __future__ import annotations

import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# External dependencies expected in requirements.txt
# pyyaml for configs (optional here; we pass config dicts)
# einops for safe reshapes (optional)


def _read_captions(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Captions file not found: {path}")
    captions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Allow tab-separated files; take first column as caption
            cap = line.split("\t")[0]
            captions.append(cap)
    if len(captions) == 0:
        raise ValueError(f"No captions loaded from {path}")
    return captions


class SyntheticJourneyDBDataset(Dataset):
    """Synthetic dataset using captions to drive one-step generation.

    Args:
        config: Dict-like config containing keys used here. Expected structure aligns with configs/defaults.yaml:
            - training.stage1.image_resolution
            - paths.datasets.journeydb_captions
            - models.vae.*, models.generator.*, models.clip.*, models.ip_adapter.projector.*
            - system.device, system.dtype
        captions_path: Optional override for the captions file path.
        num_samples: Optional limit on number of samples (for quick runs).
        device: Torch device string; defaults to config["system"]["device"].
        dtype: Compute dtype for generator/vae; defaults to config["system"]["dtype"].
        seed: Optional seed for reproducibility of epsilon sampling.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        captions_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.device = torch.device(device or self._cfg_val(["system", "device"], default="cuda"))
        self.autocast_enabled = bool(self._cfg_val(["system", "autocast"], default=True))
        self.dtype_str = dtype or self._cfg_val(["system", "dtype"], default="float16")
        self.compute_dtype = self._str_to_dtype(self.dtype_str)
        self.image_resolution = int(self._cfg_val(["training", "stage1", "image_resolution"], default=512))

        # latent spatial dims follow SDXL VAE: downsample by factor 8
        self.latent_channels = int(self._cfg_val(["models", "vae", "latent_channels"], default=4))
        ds_factor = 8
        self.latent_h = self.image_resolution // ds_factor
        self.latent_w = self.image_resolution // ds_factor

        # Seed for reproducibility
        self.rng = random.Random(seed if seed is not None else self._cfg_val(["system", "seed"], default=42))
        torch.manual_seed(self.rng.randint(0, 2**31 - 1))

        # Captions
        default_caps = self._cfg_val(["paths", "datasets", "journeydb_captions"], required=True)
        path_caps = captions_path or default_caps
        self.captions = _read_captions(path_caps)
        if num_samples is not None:
            self.captions = self.captions[: num_samples]

        # Lazy model handles (initialized on first use to avoid heavy startup in __init__)
        self._text_encoder = None
        self._image_encoder = None
        self._projector = None
        self._generator = None
        self._vae = None

    def __len__(self) -> int:
        return len(self.captions)

    def _init_models(self) -> None:
        if self._text_encoder is None:
            from swiftedit.models.clip.text_encoder import CLIPTextEncoder
            clip_cfg = self.cfg.get("models", {}).get("clip", {}).get("text", {})
            self._text_encoder = CLIPTextEncoder(
                model_name=clip_cfg.get("model_name", "ViT-L-14"),
                pretrained=clip_cfg.get("pretrained", "laion2b_s32b_b82k"),
                max_length=int(clip_cfg.get("max_length", 77)),
                device=str(self.device),
                dtype=torch.float32,  # keep text in fp32 for stability
                freeze=bool(clip_cfg.get("freeze", True)),
                repo_dir=self.cfg.get("paths", {}).get("openclip_dir", None),
            )
        if self._image_encoder is None:
            from swiftedit.models.clip.image_encoder import CLIPImageEncoder
            clip_img_cfg = self.cfg.get("models", {}).get("clip", {}).get("image", {})
            self._image_encoder = CLIPImageEncoder(
                model_name=clip_img_cfg.get("model_name", "ViT-L-14"),
                pretrained=clip_img_cfg.get("pretrained", "laion2b_s32b_b82k"),
                device=str(self.device),
                dtype=torch.float32,
                freeze=bool(clip_img_cfg.get("freeze", True)),
                repo_dir=self.cfg.get("paths", {}).get("openclip_dir", None),
            )
        if self._projector is None:
            from swiftedit.models.ip_adapter.projector import Projector
            proj_cfg = self.cfg.get("models", {}).get("ip_adapter", {}).get("projector", {})
            self._projector = Projector(
                in_dim=int(proj_cfg.get("in_dim", 768)),
                out_dim=int(proj_cfg.get("out_dim", 768)),
                num_tokens=int(proj_cfg.get("num_tokens", 4)),
                device=str(self.device),
                dtype=torch.float32,
            )
        if self._generator is None:
            from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
            gen_cfg = self.cfg.get("models", {}).get("generator", {})
            self._generator = SwiftBrushV2(
                repo_dir=self.cfg.get("paths", {}).get("one_step_gen_dir", None),
                latent_channels=int(gen_cfg.get("latent_channels", 4)),
                text_embed_dim=int(gen_cfg.get("text_embed_dim", 768)),
                device=str(self.device),
                dtype=self.compute_dtype,
            )
            self._generator.eval()  # frozen in Stage 1
        if self._vae is None:
            from swiftedit.models.vae.vae_sdxl import VAESDXL
            vae_cfg = self.cfg.get("models", {}).get("vae", {})
            self._vae = VAESDXL(
                repo_dir=self.cfg.get("paths", {}).get("sdxl_vae_dir", None),
                scaling_factor=float(vae_cfg.get("scaling_factor", 0.18215)),
                image_norm=str(vae_cfg.get("image_norm", "[-1,1]")),
                device=str(self.device),
                dtype=self.compute_dtype,
            )
            self._vae.eval()

    def _sample_epsilon(self, batch_size: int = 1) -> torch.Tensor:
        eps = torch.randn(
            (batch_size, self.latent_channels, self.latent_h, self.latent_w),
            device=self.device,
            dtype=self.compute_dtype,
        )
        return eps

    def _to_01(self, x: torch.Tensor, image_norm: str) -> torch.Tensor:
        if image_norm == "[-1,1]":
            return (x.clamp(-1, 1) + 1.0) / 2.0
        elif image_norm == "[0,1]":
            return x.clamp(0, 1)
        else:
            # default assume [-1,1]
            return (x.clamp(-1, 1) + 1.0) / 2.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        caption = self.captions[idx]
        self._init_models()

        # Text conditioning tokens
        c_y = self._text_encoder.encode([caption])  # (B=1, seq, dim) or (B=1, dim)

        # Sample epsilon
        eps = self._sample_epsilon(batch_size=1)

        # One-step generation to latent z
        with torch.no_grad():
            if self.autocast_enabled and self.compute_dtype in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16 if self.compute_dtype == torch.bfloat16 else torch.float16
                with torch.autocast(device_type=str(self.device), dtype=amp_dtype):
                    z = self._generator.forward(eps, c_y)
            else:
                z = self._generator.forward(eps, c_y)

        # Decode to image for image encoder
        vae_norm = str(self.cfg.get("models", {}).get("vae", {}).get("image_norm", "[-1,1]"))
        with torch.no_grad():
            if self.autocast_enabled and self.compute_dtype in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16 if self.compute_dtype == torch.bfloat16 else torch.float16
                with torch.autocast(device_type=str(self.device), dtype=amp_dtype):
                    x_hat = self._vae.decode(z)
            else:
                x_hat = self._vae.decode(z)
        x_hat_01 = self._to_01(x_hat, image_norm=vae_norm)

        # Image encoder and projector to tokens
        c_x_global = self._image_encoder.encode_images(x_hat_01)  # (B=1, D)
        c_x_tokens = self._projector.project(c_x_global)  # (B=1, num_tokens, D)

        sample: Dict[str, Any] = {
            "epsilon": eps.squeeze(0),  # (C,H,W)
            "z": z.detach().squeeze(0),
            "c_y": c_y.detach().squeeze(0),
            "c_x_tokens": c_x_tokens.detach().squeeze(0),
            "caption": caption,
            "x_hat": x_hat_01.detach().squeeze(0),
        }
        return sample

    # ---------------------- Utility helpers ----------------------
    def _cfg_val(self, keys: List[str], default: Any = None, required: bool = False) -> Any:
        cur = self.cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                if required:
                    raise KeyError(f"Missing required config key: {'.'.join(keys)}")
                return default
            cur = cur[k]
        return cur

    @staticmethod
    def _str_to_dtype(dtype_str: str) -> torch.dtype:
        map_ = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return map_.get(str(dtype_str).lower(), torch.float16)


# Convenience factory
def build_stage1_dataset(config: Dict[str, Any], **kwargs) -> SyntheticJourneyDBDataset:
    return SyntheticJourneyDBDataset(config=config, **kwargs)

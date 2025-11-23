from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except Exception as e:  # pragma: no cover - handled at runtime
    open_clip = None


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _discover_local_pretrained(openclip_dir: Optional[str]) -> Optional[str]:
    if not openclip_dir or not os.path.isdir(openclip_dir):
        return None
    # Heuristic: find first .pt or .bin checkpoint
    for root, _dirs, files in os.walk(openclip_dir):
        for f in files:
            if f.endswith('.pt') or f.endswith('.bin'):
                return os.path.join(root, f)
    return None


class CLIPImageEncoder(nn.Module):
    """
    Wrapper around OpenCLIP image encoder to produce pooled, normalized global image embeddings.

    Public methods:
    - encode(images, normalize=True): returns dict with key 'pooled' -> (B, D)
    - forward(images, normalize=True): returns (B, D)
    - from_config(cfg): factory using nested config dict
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[str] = "float32",
        freeze: bool = True,
        openclip_dir: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open-clip-torch is required. Please install open-clip-torch~=2.24.0"
            )
        self.device = torch.device(device)
        self.image_size = int(image_size)
        # For numerical stability, keep the model in float32
        self.compute_dtype = torch.float32 if dtype is None else getattr(torch, dtype)

        local_pretrained = _discover_local_pretrained(openclip_dir)
        pretrained_arg = local_pretrained if local_pretrained else pretrained

        # create_model_and_transforms returns (model, preprocess_train, preprocess_val)
        # Always create on CPU first to avoid CUDA initialization issues
        self.model, _pre_t, _pre_v = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_arg, device='cpu'
        )
        self.model = self.model.eval()
        # Move to target device and ensure float32 for stability
        self.model.to(self.device, dtype=torch.float32)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # Register mean/std buffers for normalization
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    @torch.no_grad()
    def _preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a batch tensor to CLIP input:
        - Accepts x shape (B, 3, H, W) with range [0,1] or [-1,1]
        - Resizes to (self.image_size, self.image_size) using bicubic
        - Normalizes by CLIP mean/std
        Returns float32 tensor on self.device.
        """
        assert x.ndim == 4 and x.shape[1] == 3, "Expected (B,3,H,W) tensor"
        x = x.to(self.device)
        x = x.to(torch.float32)
        # If appears to be in [-1,1], convert to [0,1]
        if x.min() < 0.0:
            x = (x + 1.0) / 2.0
        # Clamp to [0,1]
        x = x.clamp(0.0, 1.0)
        # Resize
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        # Normalize
        x = (x - self._mean) / self._std
        return x

    @torch.no_grad()
    def encode(
        self,
        images: Union[torch.Tensor, List["PIL.Image.Image"]],
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode images to global pooled embeddings.
        - images: either a (B,3,H,W) torch tensor in [0,1] or [-1,1],
                  or a list of PIL Images.
        - normalize: L2-normalize output features.
        Returns dict with key 'pooled' -> (B, D)
        """
        if isinstance(images, torch.Tensor):
            x = self._preprocess_tensor(images)
        else:
            # Convert PIL list using OpenCLIP's preprocess if available
            if len(images) == 0:
                raise ValueError("Empty image list provided to CLIPImageEncoder.encode")
            # Use open_clip's transforms for PIL Images
            # Recreate transforms for safety
            _, _pre_t, pre_v = open_clip.create_model_and_transforms(
                self.model.visual.__class__.__name__, pretrained=None
            )
            x_list = [pre_v(img).to(self.device) for img in images]
            x = torch.stack(x_list, dim=0)
        feats = self.model.encode_image(x)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return {"pooled": feats}

    @torch.no_grad()
    def forward(
        self,
        images: Union[torch.Tensor, List["PIL.Image.Image"]],
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.encode(images, normalize=normalize)["pooled"]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CLIPImageEncoder":
        provider = cfg.get("provider", "openclip").lower()
        if provider != "openclip":
            raise ValueError(f"Unsupported CLIP image provider: {provider}")
        model_name = cfg.get("model_name", "ViT-L-14")
        pretrained = cfg.get("pretrained", "laion2b_s32b_b82k")
        dtype = cfg.get("dtype", "float32")
        freeze = bool(cfg.get("freeze", True))
        openclip_dir = cfg.get("openclip_dir", None)
        image_size = cfg.get("image_size", 224)
        device = cfg.get("device", "cuda")
        return cls(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            freeze=freeze,
            openclip_dir=openclip_dir,
            image_size=image_size,
        )


def build_image_encoder(config: Dict[str, Any], device: Optional[str] = None) -> CLIPImageEncoder:
    """Convenience builder. Expects dict with keys:
    - models.clip.image: {provider, model_name, pretrained, dtype, freeze}
    - paths.openclip_dir (optional)
    """
    clip_cfg = (
        config.get("models", {})
        .get("clip", {})
        .get("image", {})
        .copy()
    )
    if device is not None:
        clip_cfg["device"] = device
    if "openclip_dir" not in clip_cfg:
        openclip_dir = config.get("paths", {}).get("openclip_dir", None)
        if openclip_dir:
            clip_cfg["openclip_dir"] = openclip_dir
    return CLIPImageEncoder.from_config(clip_cfg)

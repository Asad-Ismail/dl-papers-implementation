import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in ["float16", "fp16", "half"]:
        return torch.float16
    if s in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if s in ["float32", "fp32", "single", "float"]:
        return torch.float32
    return None


def _load_prompts(root: str, prompts_file: Optional[str]) -> List[Dict[str, Any]]:
    """Load prompts metadata.

    Expects a JSON file with a list of objects containing at least:
      {"image_path": "relative/or/absolute/path.jpg", "prompt": "text"}

    If prompts_file is None or missing, falls back to scanning common image extensions
    under root and assigns a dummy prompt (file stem).
    """
    items: List[Dict[str, Any]] = []
    if prompts_file is None:
        # Fallback: scan for images
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    stem = os.path.splitext(os.path.basename(fn))[0]
                    items.append({"image_path": rel, "prompt": stem})
        return items

    pf = prompts_file
    if not os.path.isabs(pf):
        pf = os.path.join(root, pf)
    if not os.path.isfile(pf):
        # Fallback to scan
        return _load_prompts(root, None)

    with open(pf, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"Prompts file at {pf} must contain a list of items or an object with 'items'.")
    for it in data:
        if "image_path" not in it or "prompt" not in it:
            # Attempt to adapt common schemas
            # e.g., {"image": "...", "caption": "..."}
            img_key = None
            for k in ["image", "image_file", "path", "filepath", "file"]:
                if k in it:
                    img_key = k
                    break
            prompt_key = None
            for k in ["caption", "text", "prompt_src", "prompt"]:
                if k in it:
                    prompt_key = k
                    break
            if img_key is None or prompt_key is None:
                continue
            items.append({"image_path": it[img_key], "prompt": it[prompt_key]})
        else:
            items.append({"image_path": it["image_path"], "prompt": it["prompt"]})
    return items


def _load_image(path: str, image_resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_resolution is not None and image_resolution > 0:
        img = img.resize((image_resolution, image_resolution), resample=Image.Resampling.BICUBIC)
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # Returns float32 tensor in [0,1], shape (3,H,W)
    x = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(img.size[1], img.size[0], 3)
         .numpy().astype("float32") / 255.0)
    )
    x = x.permute(2, 0, 1).contiguous()
    return x


class RealCommonCanvasDataset(Dataset):
    """Stage-2 real dataset that loads real images and their prompts and produces
    latents and conditioning tokens.

    Yields dict entries:
      - x: (3,H,W) float32 in [0,1]
      - z: (C=4,H/8,W/8) latent tensor (dtype per VAE)
      - c_y: (D=768) pooled text embedding (float32, L2-normalized)
      - c_x_tokens: (N_tokens, D=768) image conditioning tokens
      - prompt: str
      - path: str absolute path to image
    """

    def __init__(
        self,
        config: Dict[str, Any],
        root: Optional[str] = None,
        prompts_file: Optional[str] = None,
        image_resolution: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        sample_posterior: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.root = root or config.get("paths", {}).get("datasets", {}).get("commoncanvas_root", "")
        ds_cfg = config.get("training", {}).get("stage2", {}).get("dataset", {})
        self.prompts_file = prompts_file or ds_cfg.get("prompts_file", None)
        self.image_resolution = image_resolution or config.get("training", {}).get("stage2", {}).get("image_resolution", 512)
        dev = device or config.get("system", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.dtype = _map_dtype_str(dtype or config.get("system", {}).get("dtype", None)) or torch.float16
        self.sample_posterior = sample_posterior

        self.items = _load_prompts(self.root, self.prompts_file)
        if len(self.items) == 0:
            raise RuntimeError(f"No data items found under {self.root} with prompts_file={self.prompts_file}.")

        # Lazy models
        self._models_initialized = False
        self.text_encoder = None
        self.image_encoder = None
        self.projector = None
        self.vae = None

    def __len__(self) -> int:
        return len(self.items)

    def _init_models(self) -> None:
        if self._models_initialized:
            return
        from swiftedit.models.clip.text_encoder import CLIPTextEncoder
        from swiftedit.models.clip.image_encoder import CLIPImageEncoder
        from swiftedit.models.ip_adapter.projector import Projector
        from swiftedit.models.vae.vae_sdxl import VAESDXL

        # Build models from config
        self.text_encoder = CLIPTextEncoder.from_config(self.cfg.get("models", {}).get("clip", {}).get("text", {}))
        self.image_encoder = CLIPImageEncoder.from_config(self.cfg.get("models", {}).get("clip", {}).get("image", {}))
        self.projector = Projector.from_config(self.cfg.get("models", {}).get("ip_adapter", {}).get("projector", {}))
        self.vae = VAESDXL.from_config(self.cfg.get("models", {}).get("vae", {}))

        # Move to device and set modes
        for m in [self.text_encoder, self.image_encoder, self.projector, self.vae]:
            if m is None:
                continue
            m.to(self.device)
        # Freeze feature extractors and VAE; projector may be trainable in Stage 2 per config
        self.text_encoder.eval()
        self.image_encoder.eval()
        self.vae.eval()

        # Optionally freeze projector in Stage 2 per defaults
        freeze_proj = self.cfg.get("training", {}).get("stage2", {}).get("freeze", {}).get("projector", True)
        if freeze_proj:
            for p in self.projector.parameters():
                p.requires_grad = False
            self.projector.eval()
        else:
            self.projector.train()

        self._models_initialized = True

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._init_models()
        item = self.items[idx]
        img_path = item.get("image_path")
        if not os.path.isabs(img_path):
            img_path_full = os.path.join(self.root, img_path)
        else:
            img_path_full = img_path
        if not os.path.isfile(img_path_full):
            raise FileNotFoundError(f"Image file not found: {img_path_full}")

        prompt = item.get("prompt", "")

        # Load and preprocess image
        pil_img = _load_image(img_path_full, self.image_resolution)
        x = pil_to_tensor(pil_img)  # (3,H,W) in [0,1], float32

        # Prepare tensors on device
        x_device = x.to(self.device)

        # Encode with VAE (expects [-1,1] by default); convert if needed
        vae_norm = str(self.cfg.get("models", {}).get("vae", {}).get("image_norm", "[-1,1]")).lower()
        if vae_norm == "[-1,1]":
            x_vae = x_device * 2.0 - 1.0
        else:
            x_vae = x_device

        with torch.no_grad():
            z = self.vae.encode(x_vae, sample_posterior=self.sample_posterior)
            if isinstance(z, tuple):
                # diffusers AutoencoderKL returns a DiagonalGaussianDistribution; wrapper may return latents
                # Ensure z is a tensor (latent)
                z = z[0]

        # CLIP text embedding
        with torch.no_grad():
            c_y = self.text_encoder.forward([prompt], normalize=True)
            if isinstance(c_y, dict):
                c_y = c_y.get("pooled", None)
            if isinstance(c_y, (list, tuple)):
                c_y = c_y[0]
            if c_y.dim() == 2 and c_y.size(0) == 1:
                c_y = c_y.squeeze(0)
            c_y = c_y.to(self.device)

        # CLIP image embedding -> Projector tokens
        with torch.no_grad():
            # image encoder expects [0,1]; it will internally remap if provided [-1,1]
            img_emb = self.image_encoder.forward(x_device.unsqueeze(0), normalize=True)  # (1,D)
            if isinstance(img_emb, dict):
                img_emb = img_emb.get("pooled", None)
            if img_emb.dim() == 2 and img_emb.size(0) == 1:
                img_emb = img_emb.squeeze(0)
            c_x_tokens = self.projector.forward(img_emb.unsqueeze(0))  # (1,N,D)
            c_x_tokens = c_x_tokens.squeeze(0)

        sample = {
            "x": x,  # keep CPU float32 for potential CPU-side ops; move in trainer as needed
            "z": z.detach(),
            "c_y": c_y.detach(),
            "c_x_tokens": c_x_tokens.detach(),
            "prompt": prompt,
            "path": img_path_full,
        }
        return sample


def build_stage2_dataset(config: Dict[str, Any], **kwargs) -> RealCommonCanvasDataset:
    return RealCommonCanvasDataset(config, **kwargs)

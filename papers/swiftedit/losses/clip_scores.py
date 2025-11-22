from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None  # type: ignore

Tensor = torch.Tensor
ImageLike = Union[Tensor, "PIL.Image.Image"]
Texts = Union[str, Sequence[str]]

__all__ = [
    "CLIPScorer",
    "build_scorer_from_config",
    "clip_score_whole",
    "clip_score_edited",
]


def _map_dtype_str(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    s = str(dtype_str).lower()
    if s in {"float16", "fp16"}:
        return torch.float16
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if s in {"float32", "fp32"}:
        return torch.float32
    return None


def _to_bchw(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


def _ensure_01(x: Tensor) -> Tensor:
    x = x.to(torch.float32)
    if torch.any(x < 0):
        x = (x + 1.0) * 0.5
    return x.clamp(0, 1)


def _resize_mask(mask: Tensor, hw: Tuple[int, int]) -> Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        if mask.size(0) == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
    elif mask.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    mask = mask.to(torch.float32)
    mask = F.interpolate(mask, size=hw, mode="bilinear", align_corners=False)
    return mask.clamp(0, 1)


def _composite_with_mask(img: Tensor, mask: Tensor, fill: float = 0.5) -> Tensor:
    # img: (B,3,H,W) in [0,1]; mask: (B,1,H,W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1:
        mask3 = mask.repeat(1, 3, 1, 1)
    else:
        mask3 = mask
    bg = torch.full_like(img, fill)
    return img * mask3 + bg * (1.0 - mask3)


def _bbox_from_mask(mask: Tensor, thr: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
    # mask: (H,W) in [0,1]
    m = (mask >= thr).to(torch.uint8)
    if m.sum() == 0:
        return None
    ys = torch.where(m.sum(dim=1) > 0)[0]
    xs = torch.where(m.sum(dim=0) > 0)[0]
    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    return x0, y0, x1 + 1, y1 + 1


class CLIPScorer(nn.Module):
    """
    OpenCLIP-based scorer to compute CLIP-Whole and CLIP-Edited similarity scores.

    Scores are cosine similarities between normalized image and text embeddings, scaled by 100.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        openclip_dir: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if not _HAS_OPENCLIP:
            raise ImportError("open-clip-torch is required for CLIP scoring. Please install open-clip-torch~=2.24.0.")
        self.device = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.compute_dtype = _map_dtype_str(dtype) or torch.float32
        # Create model and preprocess
        if openclip_dir is not None and os.path.isdir(openclip_dir):
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None, device=self.device)
            # attempt to load weights if a checkpoint exists
            ckpt_path = None
            for root, _, files in os.walk(openclip_dir):
                for f in files:
                    if f.endswith(".pt") or f.endswith(".bin"):
                        ckpt_path = os.path.join(root, f)
                        break
                if ckpt_path:
                    break
            if ckpt_path is not None:
                sd = torch.load(ckpt_path, map_location=self.device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=False)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.preprocess = preprocess  # Callable for PIL; we also implement tensor path
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.image_size = image_size

    @torch.no_grad()
    def encode_text(self, texts: Texts) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(list(texts)).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat.float()
        feat = F.normalize(feat, dim=-1)
        return feat

    def _preprocess_tensor_image(self, img: Tensor) -> Tensor:
        # img: (B,3,H,W) in [0,1]
        img = _to_bchw(img)
        img = _ensure_01(img)
        img = F.interpolate(img, size=self.image_size, mode="bilinear", align_corners=False)
        # Normalize with CLIP mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        img = (img - mean) / std
        return img

    @torch.no_grad()
    def encode_image(self, images: Union[ImageLike, Sequence[ImageLike]]) -> Tensor:
        # Accept tensor or PILs; return normalized features (B, D)
        if isinstance(images, (list, tuple)):
            if len(images) == 0:
                return torch.empty(0, self.model.text_projection.shape[1], device=self.device)
            if PILImage is not None and isinstance(images[0], PILImage):
                # Use preprocess pipeline for PIL
                proc = torch.stack([self.preprocess(im).to(self.device) for im in images], dim=0)
            else:
                # Assume tensor batch
                proc = self._preprocess_tensor_image(torch.stack([img if isinstance(img, Tensor) else self._pil_to_tensor(img) for img in images], dim=0).to(self.device))
        else:
            if PILImage is not None and isinstance(images, PILImage):
                proc = self.preprocess(images).unsqueeze(0).to(self.device)
            else:
                proc = self._preprocess_tensor_image(images.to(self.device))
        feat = self.model.encode_image(proc)
        feat = feat.float()
        feat = F.normalize(feat, dim=-1)
        return feat

    def _pil_to_tensor(self, img: "PIL.Image.Image") -> Tensor:
        # Convert PIL to [0,1] tensor (3,H,W)
        if PILImage is None:
            raise RuntimeError("PIL is not available to process images.")
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # type: ignore
        if arr.size(0) == 4:
            arr = arr[:3]
        return arr

    @torch.no_grad()
    def score_image_text(self, images: ImageLike, texts: Texts) -> Tensor:
        """Return CLIP-Whole score(s) scaled by 100. images can be tensor (B,3,H,W) or PIL image; texts can be str or list of str.
        If texts is a single string and images is a batch, it is broadcast to batch.
        """
        img_feat = self.encode_image(images)
        txt_feat = self.encode_text(texts)
        if txt_feat.shape[0] == 1 and img_feat.shape[0] > 1:
            txt_feat = txt_feat.expand(img_feat.shape[0], -1)
        sim = (img_feat * txt_feat).sum(dim=-1)
        return sim * 100.0

    @torch.no_grad()
    def score_edited(
        self,
        images: Tensor,
        texts: Texts,
        mask: Tensor,
        mode: str = "composite",
        crop_threshold: float = 0.3,
        background_fill: float = 0.5,
    ) -> Tensor:
        """
        Compute CLIP-Edited score(s).
        - mode="composite": apply soft mask to image and fill background; encode masked composite.
        - mode="crop": crop the bounding box of mask>threshold; if empty, fall back to composite.
        Returns scores scaled by 100.
        """
        if isinstance(images, Tensor):
            img = _to_bchw(images).to(self.device)
        else:
            # single PIL image
            img = self._pil_to_tensor(images).unsqueeze(0).to(self.device)
        img = _ensure_01(img)
        B, _, H, W = img.shape
        mask = _resize_mask(mask.to(self.device), (H, W))
        if mode == "crop":
            scores: List[Tensor] = []
            for b in range(B):
                mb = mask[b, 0]
                bbox = _bbox_from_mask(mb, thr=crop_threshold)
                if bbox is None:
                    # fallback to composite for this sample
                    comp = _composite_with_mask(img[b:b+1], mask[b:b+1], fill=background_fill)
                    s = self.score_image_text(comp, texts if isinstance(texts, str) else [texts[b]])
                    scores.append(s.squeeze(0))
                else:
                    x0, y0, x1, y1 = bbox
                    crop = img[b:b+1, :, y0:y1, x0:x1]
                    s = self.score_image_text(crop, texts if isinstance(texts, str) else [texts[b]])
                    scores.append(s.squeeze(0))
            return torch.stack(scores, dim=0)
        else:
            comp = _composite_with_mask(img, mask, fill=background_fill)
            return self.score_image_text(comp, texts)


def build_scorer_from_config(cfg: Dict[str, Any], device: Optional[Union[str, torch.device]] = None) -> CLIPScorer:
    clip_cfg = (
        (((cfg.get("eval") or {}).get("piebench") or {}).get("clip_model"))
        or (((cfg.get("models") or {}).get("clip") or {}).get("image"))
        or {}
    )
    model_name = clip_cfg.get("model_name", "ViT-L-14")
    pretrained = clip_cfg.get("pretrained", "laion2b_s32b_b82k")
    openclip_dir = ((cfg.get("paths") or {}).get("openclip_dir"))
    dtype = ((cfg.get("models") or {}).get("clip") or {}).get("dtype", "float32")
    return CLIPScorer(
        model_name=model_name,
        pretrained=pretrained,
        device=device or (cfg.get("system", {}).get("device", None)),
        dtype=dtype,
        openclip_dir=openclip_dir,
        image_size=224,
    )


@torch.no_grad()
def clip_score_whole(scorer: CLIPScorer, images: ImageLike, texts: Texts) -> Tensor:
    return scorer.score_image_text(images, texts)


@torch.no_grad()
def clip_score_edited(
    scorer: CLIPScorer,
    images: ImageLike,
    texts: Texts,
    mask: Tensor,
    mode: str = "composite",
    crop_threshold: float = 0.3,
    background_fill: float = 0.5,
) -> Tensor:
    # Ensure tensor image for edited scoring
    if not isinstance(images, Tensor):
        if PILImage is None:
            raise RuntimeError("PIL not available to process images")
        # Convert PIL to tensor [0,1]
        import numpy as np  # local import to avoid hard dependency when unused
        arr = torch.from_numpy(np.array(images)).permute(2, 0, 1).float() / 255.0
        if arr.size(0) == 4:
            arr = arr[:3]
        images = arr
    return scorer.score_edited(images, texts, mask, mode=mode, crop_threshold=crop_threshold, background_fill=background_fill)

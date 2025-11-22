"""
Visualization utilities for SwiftEdit.

Provides helpers to:
- Convert between torch tensors and PIL images
- Save grids of images for qualitative comparison
- Overlay soft masks on images for ARaM/mask quality visualization

All functions are designed to be robust to inputs in either [0,1] or [-1,1], and
work with both CHW (channels-first) tensors and PIL images. Shapes are
normalized to 3-channel RGB images when saving/overlaying.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import os

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    F = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# Optional torchvision for grid utility
_HAS_TORCHVISION = False
try:
    import torchvision
    from torchvision.utils import make_grid
    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover
    make_grid = None  # type: ignore


def _ensure_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for visualization utilities.")


def _to_01(t: Tensor) -> Tensor:
    """Map tensor to [0,1] range if necessary.

    If any value < 0, assume input in [-1,1] and remap: x01 = (x+1)/2.
    Otherwise assume already in [0,1].
    """
    _ensure_torch()
    t = t.detach().float()
    if (t.min() < 0).item():
        t = (t + 1.0) * 0.5
    return t.clamp(0.0, 1.0)


def _ensure_chw(t: Tensor) -> Tensor:
    """Ensure tensor is CHW (or BCHW) channels-first.
    Accepts HWC/BHWC and converts to CHW/BCHW.
    """
    _ensure_torch()
    if t.ndim == 2:
        # grayscale HxW -> 1xHxW
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        # could be CHW or HWC
        c, h, w = t.shape
        if c in (1, 3):
            return t
        else:
            # assume HWC -> CHW
            t = t.permute(2, 0, 1)
    elif t.ndim == 4:
        b, c, h, w = t.shape
        if c in (1, 3):
            return t
        else:
            # assume BHWC -> BCHW
            t = t.permute(0, 3, 1, 2)
    return t


def _ensure_rgb(t: Tensor) -> Tensor:
    """Ensure tensor has 3 channels; if 1, replicate. If >3, take first 3."""
    _ensure_torch()
    t = _ensure_chw(t)
    if t.ndim == 3:
        c = t.shape[0]
        if c == 1:
            t = t.repeat(3, 1, 1)
        elif c > 3:
            t = t[:3]
    elif t.ndim == 4:
        c = t.shape[1]
        if c == 1:
            t = t.repeat(1, 3, 1, 1)
        elif c > 3:
            t = t[:, :3]
    return t


def pil_to_tensor(img: Image.Image) -> Tensor:
    """Convert a PIL RGB image to a float32 torch tensor in [0,1], CHW.

    If the image is not RGB, it is converted to RGB.
    """
    _ensure_torch()
    if Image is None:
        raise ImportError("PIL (Pillow) is required for pil_to_tensor.")
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0 if np is not None else None
    if arr is None:
        raise ImportError("numpy is required for PIL->Tensor conversion.")
    t = torch.from_numpy(arr)
    # HWC -> CHW
    t = t.permute(2, 0, 1).contiguous()
    return t


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    """Convert a CHW/BCHW tensor (in [0,1] or [-1,1]) to a PIL RGB image.

    If BCHW, only the first element of the batch is used.
    """
    _ensure_torch()
    if Image is None:
        raise ImportError("PIL (Pillow) is required for tensor_to_pil.")
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    t = _ensure_rgb(t)
    t = _to_01(t)
    t = t.clamp(0.0, 1.0)
    # CHW -> HWC
    t = t.permute(1, 2, 0).contiguous()
    arr = (t.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def save_image_grid(
    images: Union[Sequence[Tensor], Sequence[Image.Image]],
    path: str,
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    pil_resize: Optional[int] = None,
) -> None:
    """Save a grid of images to path.

    - images: list/sequence of tensors (CHW/BCHW/HWC) or PIL Images
    - nrow: number of images per row
    - padding: pixel padding between images
    - normalize: if True, map tensors from [-1,1] to [0,1]
    - value_range: optional (min, max) for normalization; if None, auto
    - pil_resize: optional final resize of the grid to a square size (e.g., 1024)
    """
    _ensure_torch()
    if len(images) == 0:
        raise ValueError("save_image_grid: 'images' is empty.")

    tensors: List[Tensor] = []
    for img in images:
        if isinstance(img, Image.Image):
            t = pil_to_tensor(img)
        else:
            t = img.detach().cpu().float()
        t = _ensure_rgb(t)
        if normalize:
            if value_range is not None:
                lo, hi = value_range
                t = t.clamp(lo, hi)
                t = (t - lo) / max(hi - lo, 1e-8)
            else:
                t = _to_01(t)
        t = t.clamp(0.0, 1.0)
        tensors.append(t)

    if _HAS_TORCHVISION and make_grid is not None:
        grid = make_grid(torch.stack(tensors, dim=0), nrow=nrow, padding=padding)
        grid_img = tensor_to_pil(grid)
    else:
        # Manual grid using PIL
        if Image is None:
            raise ImportError("PIL is required to save image grid.")
        # Assume all tensors same size
        c, h, w = tensors[0].shape
        cols = nrow
        rows = (len(tensors) + cols - 1) // cols
        grid_w = cols * w + (cols - 1) * padding
        grid_h = rows * h + (rows - 1) * padding
        grid_img = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
        for i, t in enumerate(tensors):
            img_i = tensor_to_pil(t)
            r = i // cols
            cidx = i % cols
            x = cidx * (w + padding)
            y = r * (h + padding)
            grid_img.paste(img_i, (x, y))

    if pil_resize is not None:
        grid_img = grid_img.resize((pil_resize, pil_resize), resample=Image.BICUBIC)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grid_img.save(path)


def _ensure_mask_tensor(mask: Union[Tensor, np.ndarray, Image.Image]) -> Tensor:
    _ensure_torch()
    if isinstance(mask, Tensor):
        m = mask.detach().cpu().float()
    elif isinstance(mask, Image.Image):
        m = pil_to_tensor(mask)
        # Convert to single channel by averaging if RGB
        if m.shape[0] == 3:
            m = m.mean(dim=0, keepdim=True)
    elif np is not None and isinstance(mask, np.ndarray):
        m = torch.from_numpy(mask).float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 3 and m.shape[2] in (1, 3):
            m = torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            # fallback to single channel by averaging last dim
            m = torch.from_numpy(mask).mean(axis=-1, keepdims=True)
            m = torch.from_numpy(m).permute(2, 0, 1).float()
    else:
        raise TypeError("Unsupported mask type. Expected Tensor, PIL.Image, or numpy array.")

    m = _ensure_chw(m)
    if m.shape[0] == 3:
        m = m.mean(dim=0, keepdim=True)
    m = m.clamp(0.0, 1.0)
    return m


def draw_mask_overlay(
    image: Union[Tensor, Image.Image],
    mask: Union[Tensor, np.ndarray, Image.Image],
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
    resize_mask_to_image: bool = True,
) -> Image.Image:
    """Overlay a (soft) mask on an image with the given color and alpha.

    - image: torch Tensor (CHW/BCHW/HWC; [0,1] or [-1,1]) or PIL Image
    - mask: torch Tensor / numpy array / PIL Image; values in [0,1] preferred
    - color: RGB tuple for overlay color
    - alpha: transparency factor in [0,1]
    - resize_mask_to_image: if True, bilinear-resize mask to image spatial size

    Returns a PIL Image with overlay.
    """
    _ensure_torch()
    if Image is None:
        raise ImportError("PIL (Pillow) is required for draw_mask_overlay.")

    # Convert image to tensor [0,1], CHW
    if isinstance(image, Image.Image):
        img_t = pil_to_tensor(image)
    else:
        img_t = image.detach().cpu().float()
        img_t = _ensure_rgb(img_t)
        img_t = _to_01(img_t).clamp(0.0, 1.0)
        if img_t.ndim == 4:
            img_t = img_t[0]

    # Convert mask to tensor [0,1], 1xHxW
    mask_t = _ensure_mask_tensor(mask)

    # Resize mask to image spatial size
    if resize_mask_to_image and F is not None:
        mh, mw = mask_t.shape[-2:]
        ih, iw = img_t.shape[-2:]
        if (mh != ih) or (mw != iw):
            mask_t = F.interpolate(mask_t.unsqueeze(0), size=(ih, iw), mode="bilinear", align_corners=False)[0]
    mask_t = mask_t.clamp(0.0, 1.0)

    # Create overlay color tensor
    color_t = torch.tensor(color, dtype=torch.float32).view(3, 1, 1) / 255.0
    overlay = color_t.expand_as(img_t)

    # Composite: img * (1 - alpha * mask) + overlay * (alpha * mask)
    comp = img_t * (1.0 - alpha * mask_t) + overlay * (alpha * mask_t)
    comp = comp.clamp(0.0, 1.0)

    return tensor_to_pil(comp)


def save_tensor_image(tensor: Tensor, path: str) -> None:
    """Save a single image tensor to disk as PNG/JPEG.

    Accepts CHW/BCHW/HWC tensors in [0,1] or [-1,1]; converts to PIL and saves.
    """
    img = tensor_to_pil(tensor)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


__all__ = [
    "pil_to_tensor",
    "tensor_to_pil",
    "save_image_grid",
    "draw_mask_overlay",
    "save_tensor_image",
]

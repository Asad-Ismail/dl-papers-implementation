"""
PieBench dataset loader

This module provides a flexible PyTorch Dataset for the PieBench evaluation suite.
It attempts to load 700 samples (or however many available under the root) with
source images, edit prompts, source prompts, and ground-truth masks when present.

Supported layouts:
1) Manifest JSON under root (preferred). File names probed in order:
   - piebench_manifest.json
   - piebench.json
   - manifest.json
   The manifest should be a list of items or an object with key "items" listing dicts:
   {
       "image": "path/to/image.png",
       "mask": "path/to/mask.png",  # optional
       "prompt_src": "...",          # source prompt
       "prompt_edit": "...",         # edit prompt
       "edit_type": "color",         # optional
       "id": "sample_0001"           # optional
   }
   Paths can be relative to root or absolute.

2) Directory layout with prompts JSON:
   - images/ (or imgs/) containing source images
   - masks/ containing matching masks named by stem (e.g., image.png -> image_mask.png or image.png)
   - prompts.json mapping image stems/paths to prompts (keys probed: prompt_src, source_prompt, prompt_edit, edit_prompt)

3) Fallback scanning:
   - Finds all image files under root (recursively); attempts to find a mask file with the same stem
     under a sibling directory named masks/ or in the same directory; prompts are blank unless a
     prompts.json exists.

Returns items with tensors:
- x_src: torch.Tensor (3, H, W), float32 in [0,1]
- gt_mask: torch.Tensor (1, H, W) in [0,1] if available, else None
- prompt_src: str
- prompt_edit: str
- path: str (absolute image path)
- meta: dict (id/edit_type/mask_path/etc.)

This dataset does NOT perform any model encoding; it prepares raw inputs for evaluation.
Use swiftedit/edit/inference.py to run editing and swiftedit/losses/* to compute metrics.
"""
from __future__ import annotations

import os
import json
import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

# Allowed image extensions
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def _find_manifest(root: str) -> Optional[str]:
    candidates = [
        os.path.join(root, "piebench_manifest.json"),
        os.path.join(root, "piebench.json"),
        os.path.join(root, "manifest.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _load_manifest(root: str, manifest_path: str) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Manifest at {manifest_path} has unexpected format: {type(items)}")
    normalized: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        image_path = it.get("image") or it.get("image_path") or it.get("src") or it.get("file")
        if not image_path:
            continue
        mask_path = it.get("mask") or it.get("mask_path") or it.get("gt_mask")
        prompt_src = (
            it.get("prompt_src")
            or it.get("source_prompt")
            or it.get("origin_prompt")
            or it.get("src_prompt")
            or ""
        )
        prompt_edit = (
            it.get("prompt_edit")
            or it.get("edit_prompt")
            or it.get("target_prompt")
            or it.get("tgt_prompt")
            or ""
        )
        normalized.append(
            {
                "image_path": _resolve_path(root, image_path),
                "mask_path": _resolve_path(root, mask_path) if mask_path else None,
                "prompt_src": str(prompt_src),
                "prompt_edit": str(prompt_edit),
                "edit_type": it.get("edit_type"),
                "id": it.get("id") or os.path.splitext(os.path.basename(image_path))[0],
            }
        )
    return normalized


def _find_prompts_json(root: str) -> Optional[str]:
    candidates = [
        os.path.join(root, "prompts.json"),
        os.path.join(root, "piebench_prompts.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _load_prompts_map(prompts_path: str) -> Dict[str, Dict[str, str]]:
    """Return a map: stem -> {prompt_src, prompt_edit}.
    Accepts formats:
    - list of items with keys image/path/file and prompt_src/source_prompt and prompt_edit/edit_prompt
    - dict of stem -> {prompt_src, prompt_edit}
    """
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Assume stem -> dict
        out: Dict[str, Dict[str, str]] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                ps = v.get("prompt_src") or v.get("source_prompt") or v.get("origin_prompt") or ""
                pe = v.get("prompt_edit") or v.get("edit_prompt") or v.get("target_prompt") or ""
                out[str(k)] = {"prompt_src": str(ps), "prompt_edit": str(pe)}
        return out
    elif isinstance(data, list):
        out: Dict[str, Dict[str, str]] = {}
        for it in data:
            if not isinstance(it, dict):
                continue
            image_path = it.get("image") or it.get("path") or it.get("file")
            if not image_path:
                continue
            stem = os.path.splitext(os.path.basename(image_path))[0]
            ps = it.get("prompt_src") or it.get("source_prompt") or it.get("origin_prompt") or ""
            pe = it.get("prompt_edit") or it.get("edit_prompt") or it.get("target_prompt") or ""
            out[stem] = {"prompt_src": str(ps), "prompt_edit": str(pe)}
        return out
    else:
        raise ValueError(f"Unsupported prompts format: {type(data)}")


def _scan_images(root: str) -> List[str]:
    files: List[str] = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(set(files))


def _candidate_mask_paths(img_path: str, masks_dir: Optional[str] = None) -> List[str]:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    candidates = []
    dirs_to_check = [os.path.dirname(img_path)]
    if masks_dir and os.path.isdir(masks_dir):
        dirs_to_check.insert(0, masks_dir)
    for d in dirs_to_check:
        for ext in _MASK_EXTS:
            candidates.append(os.path.join(d, f"{stem}{ext}"))
            candidates.append(os.path.join(d, f"{stem}_mask{ext}"))
            candidates.append(os.path.join(d, f"{stem}-mask{ext}"))
    return candidates


def _load_image(path: str, image_resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if image_resolution is not None and image_resolution > 0:
        img = img.resize((image_resolution, image_resolution), resample=Image.BICUBIC)
    return img


def _load_mask(path: str, image_resolution: int) -> Optional[Image.Image]:
    if path is None:
        return None
    if not os.path.isfile(path):
        return None
    try:
        m = Image.open(path).convert("L")
    except Exception:
        return None
    if image_resolution is not None and image_resolution > 0:
        m = m.resize((image_resolution, image_resolution), resample=Image.BICUBIC)
    return m


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(torch.ByteTensor(bytearray(img.tobytes())).numpy())
    # Fallback robust conversion using PIL -> numpy -> torch
    import numpy as _np
    arr = torch.from_numpy(_np.array(img)).float() / 255.0
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    if arr.shape[-1] == 3:
        arr = arr.permute(2, 0, 1).contiguous()
    elif arr.shape[-1] == 1:
        arr = arr.permute(2, 0, 1).contiguous()
    else:
        # Unexpected channels; try to reduce to 3
        if arr.shape[-1] > 3:
            arr = arr[..., :3].permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    return arr.to(torch.float32)


def mask_to_tensor(mask_img: Optional[Image.Image]) -> Optional[torch.Tensor]:
    if mask_img is None:
        return None
    import numpy as _np
    arr = torch.from_numpy(_np.array(mask_img)).float() / 255.0
    if arr.ndim == 2:
        arr = arr.unsqueeze(0)  # (1, H, W)
    elif arr.ndim == 3:
        arr = arr.permute(2, 0, 1)
        if arr.shape[0] != 1:
            arr = arr[:1, ...]
    arr = arr.clamp(0.0, 1.0)
    return arr.to(torch.float32)


class PieBenchDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_resolution: int = 512,
        use_gt_masks: bool = False,
        prompts_path: Optional[str] = None,
        return_paths: bool = True,
    ) -> None:
        super().__init__()
        self.root = os.path.abspath(root)
        self.image_resolution = int(image_resolution)
        self.use_gt_masks = bool(use_gt_masks)
        self.return_paths = bool(return_paths)

        manifest = _find_manifest(self.root)
        items: List[Dict[str, Any]] = []
        prompts_map: Dict[str, Dict[str, str]] = {}

        if manifest:
            items = _load_manifest(self.root, manifest)
        else:
            # Try directory-based assembly
            images_dir = None
            masks_dir = None
            for dn in ["images", "imgs", "src", "source"]:
                candidate = os.path.join(self.root, dn)
                if os.path.isdir(candidate):
                    images_dir = candidate
                    break
            for dn in ["masks", "gt", "gt_masks"]:
                candidate = os.path.join(self.root, dn)
                if os.path.isdir(candidate):
                    masks_dir = candidate
                    break
            img_files = _scan_images(images_dir or self.root)
            # Load prompts
            if prompts_path and os.path.isfile(prompts_path):
                prompts_map = _load_prompts_map(prompts_path)
            else:
                pjson = _find_prompts_json(self.root)
                if pjson:
                    prompts_map = _load_prompts_map(pjson)
            for img_path in img_files:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = None
                for cand in _candidate_mask_paths(img_path, masks_dir):
                    if os.path.isfile(cand):
                        mask_path = cand
                        break
                pm = prompts_map.get(stem, {"prompt_src": "", "prompt_edit": ""})
                items.append(
                    {
                        "image_path": os.path.abspath(img_path),
                        "mask_path": os.path.abspath(mask_path) if mask_path else None,
                        "prompt_src": pm.get("prompt_src", ""),
                        "prompt_edit": pm.get("prompt_edit", ""),
                        "edit_type": None,
                        "id": stem,
                    }
                )
        # Filter items that actually exist
        filtered: List[Dict[str, Any]] = []
        for it in items:
            ip = it.get("image_path")
            if not ip or not os.path.isfile(ip):
                continue
            mp = it.get("mask_path")
            if mp and not os.path.isfile(mp):
                it["mask_path"] = None
            filtered.append(it)
        self.items = filtered

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = _load_image(it["image_path"], self.image_resolution)
        x = pil_to_tensor(img)  # (3, H, W) [0,1]
        mask_img = _load_mask(it.get("mask_path"), self.image_resolution) if self.use_gt_masks else None
        m = mask_to_tensor(mask_img) if mask_img is not None else None
        sample: Dict[str, Any] = {
            "x_src": x,
            "gt_mask": m,
            "prompt_src": it.get("prompt_src", ""),
            "prompt_edit": it.get("prompt_edit", ""),
            "meta": {
                "id": it.get("id"),
                "edit_type": it.get("edit_type"),
                "mask_path": it.get("mask_path"),
            },
        }
        if self.return_paths:
            sample["path"] = it.get("image_path")
        return sample


def build_piebench_dataset(
    config: Dict[str, Any],
    root: Optional[str] = None,
    use_gt_masks: Optional[bool] = None,
    image_resolution: Optional[int] = None,
    prompts_path: Optional[str] = None,
) -> PieBenchDataset:
    """Factory to construct PieBenchDataset from config.

    Config keys consulted:
    - config["eval"]["piebench"]["root"]
    - config["eval"]["piebench"]["use_gt_masks"]
    - config["inference"]["image_resolution"] or config["eval"]["piebench"]["image_resolution"]
    """
    root = root or (
        (config.get("eval", {}).get("piebench", {}).get("root"))
        or os.path.join("data", "piebench")
    )
    use_gt_masks = (
        use_gt_masks
        if use_gt_masks is not None
        else config.get("eval", {}).get("piebench", {}).get("use_gt_masks", False)
    )
    image_resolution = (
        image_resolution
        if image_resolution is not None
        else config.get("eval", {}).get("piebench", {}).get("image_resolution")
        or config.get("inference", {}).get("image_resolution", 512)
    )
    return PieBenchDataset(
        root=root,
        image_resolution=int(image_resolution),
        use_gt_masks=bool(use_gt_masks),
        prompts_path=prompts_path,
    )


__all__ = ["PieBenchDataset", "build_piebench_dataset"]

import os
import sys
import time
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# Dataset and editing/inference
from swiftedit.eval.piebench_loader import build_piebench_dataset, PieBenchDataset
from swiftedit.edit.inference import edit_image

# Metrics
from swiftedit.losses.psnr_mse import masked_psnr_mse
from swiftedit.losses.clip_scores import (
    CLIPScorer,
    build_scorer_from_config,
    clip_score_whole,
    clip_score_edited,
)

# Logging and reproducibility
from swiftedit.utils.logger import ExperimentLogger, build_logger_from_config
from swiftedit.utils.seed import set_seed


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(defaults_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.isfile(defaults_path):
        raise FileNotFoundError(f"Defaults config not found: {defaults_path}")
    with open(defaults_path, "r") as f:
        base = yaml.safe_load(f) if _HAS_YAML else json.load(f)
    if override_path and os.path.isfile(override_path):
        with open(override_path, "r") as f:
            override = yaml.safe_load(f) if _HAS_YAML else json.load(f)
        _deep_update(base, override)
    return base


def _tensor_to_cpu_numpy(x: torch.Tensor) -> Any:
    return x.detach().cpu().numpy()


def _ensure_mask_tensor(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("Mask must be a torch.Tensor")
    # Expect (1, H, W); if (H, W) add channel
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    # If batch present, drop it for metrics later (we will re-batch)
    if mask.ndim == 4:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            # keep as is (B,1,H,W)
            pass
    return mask.float().clamp(0.0, 1.0)


def evaluate_sample(
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    scorer: Optional[CLIPScorer] = None,
    use_gt_mask_for_edit: bool = False,
    inversion_ckpt: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one edit and compute metrics.
    Returns a dict with metrics and metadata.
    """
    x_src: torch.Tensor = sample.get("x_src")
    prompt_src: str = sample.get("prompt_src", "")
    prompt_edit: str = sample.get("prompt_edit", "")
    gt_mask: Optional[torch.Tensor] = sample.get("gt_mask")
    img_path: Optional[str] = sample.get("path")
    meta: Dict[str, Any] = sample.get("meta", {})

    # For editing, optionally pass GT mask; otherwise rely on self-guided mask
    user_mask = gt_mask if use_gt_mask_for_edit else None

    start = time.time()
    try:
        edit_out = edit_image(
            image=x_src,  # tensor in [0,1]
            prompt_src=prompt_src,
            prompt_edit=prompt_edit,
            cfg=cfg,
            inversion_ckpt=inversion_ckpt,
            user_mask=user_mask,
            scales=None,
            device=device,
        )
    except Exception as e:
        # Fallback: return NaNs but capture error
        return {
            "error": str(e),
            "id": meta.get("id"),
            "edit_type": meta.get("edit_type"),
            "path": img_path,
            "runtime": float("nan"),
            "psnr_bg": float("nan"),
            "mse_bg": float("nan"),
            "clip_whole": float("nan"),
            "clip_edited": float("nan"),
        }

    end = time.time()
    x_edit: torch.Tensor = edit_out.get("x_edit")
    x_src_out: torch.Tensor = edit_out.get("x_src", x_src)
    timings: Dict[str, float] = edit_out.get("timings", {})
    runtime = float(timings.get("total", end - start))

    # Ensure GT mask for metrics
    m = _ensure_mask_tensor(gt_mask)
    if m is None:
        # Without mask, background metrics not defined; return NaN
        psnr_bg = torch.tensor(float("nan"))
        mse_bg = torch.tensor(float("nan"))
    else:
        # Add batch dimension for masked metrics
        x_e = x_edit.unsqueeze(0)
        x_s = x_src_out.unsqueeze(0)
        m_b = m.unsqueeze(0) if m.ndim == 3 else m
        metrics_bg = masked_psnr_mse(x_e, x_s, mask=m_b, use_background=True, reduction="none")
        psnr_bg = metrics_bg["psnr"][0]
        mse_bg = metrics_bg["mse"][0]

    # CLIP scores
    clip_whole_val = float("nan")
    clip_edited_val = float("nan")
    if scorer is not None:
        try:
            clip_whole_t = clip_score_whole(scorer, x_edit, prompt_edit)
            clip_whole_val = float(clip_whole_t.detach().cpu().mean().item())
        except Exception:
            pass
        try:
            if m is not None:
                clip_edited_t = clip_score_edited(scorer, x_edit, prompt_edit, mask=m, mode="composite")
                clip_edited_val = float(clip_edited_t.detach().cpu().mean().item())
        except Exception:
            pass

    result = {
        "id": meta.get("id"),
        "edit_type": meta.get("edit_type"),
        "path": img_path,
        "runtime": float(runtime),
        "psnr_bg": float(psnr_bg.detach().cpu().item()) if isinstance(psnr_bg, torch.Tensor) else float(psnr_bg),
        "mse_bg": float(mse_bg.detach().cpu().item()) if isinstance(mse_bg, torch.Tensor) else float(mse_bg),
        "clip_whole": clip_whole_val,
        "clip_edited": clip_edited_val,
    }
    return result


def evaluate_piebench(cfg: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run PieBench evaluation; returns summary metrics and writes CSV.
    """
    if override:
        _deep_update(cfg, override)

    # Seed and device
    seed = int(cfg.get("system", {}).get("seed", 42))
    deterministic = bool(cfg.get("system", {}).get("deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = cfg.get("system", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    pie_cfg = cfg.get("eval", {}).get("piebench", {})
    root = pie_cfg.get("root", cfg.get("paths", {}).get("datasets", {}).get("piebench_root", "data/piebench"))
    image_resolution = pie_cfg.get("image_resolution", cfg.get("inference", {}).get("image_resolution", 512))
    # Always load GT masks for metrics computation
    ds = build_piebench_dataset(cfg, root=root, use_gt_masks=True, image_resolution=image_resolution)

    # CLIP Scorer
    try:
        scorer = build_scorer_from_config(cfg, device=device)
    except Exception:
        scorer = None

    # Flags
    use_gt_masks_for_edit = bool(pie_cfg.get("use_gt_masks", False))
    inversion_ckpt = cfg.get("inference", {}).get("inversion_checkpoint", None) or cfg.get("models", {}).get("inversion_net", {}).get("checkpoint", None)

    # Logger
    logger = build_logger_from_config(cfg, base_log_dir=cfg.get("paths", {}).get("logs_dir", "logs"), results_csv_dir=cfg.get("logging", {}).get("results_csv_dir", "results"))
    logger.log(f"Loaded PieBench dataset from {root} with {len(ds)} samples; use_gt_masks_for_edit={use_gt_masks_for_edit}")

    results: List[Dict[str, Any]] = []
    runtimes: List[float] = []
    psnrs: List[float] = []
    mses: List[float] = []
    clips_whole: List[float] = []
    clips_edited: List[float] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        res = evaluate_sample(
            sample=sample,
            cfg=cfg,
            scorer=scorer,
            use_gt_mask_for_edit=use_gt_masks_for_edit,
            inversion_ckpt=inversion_ckpt,
            device=device,
        )
        results.append(res)

        # Aggregate if valid
        if not (isinstance(res.get("runtime"), float) and (res.get("runtime") != float("nan"))):
            pass
        else:
            runtimes.append(float(res.get("runtime", 0.0)))
        for k, lst in [("psnr_bg", psnrs), ("mse_bg", mses), ("clip_whole", clips_whole), ("clip_edited", clips_edited)]:
            v = res.get(k)
            if isinstance(v, float) and not (v != v):  # NaN check
                lst.append(v)

        if (idx + 1) % max(1, int(cfg.get("logging", {}).get("visualize_every", 1000))) == 0:
            logger.log(f"Processed {idx + 1}/{len(ds)}")

    # Summary
    def _mean(lst: List[float]) -> float:
        return float(sum(lst) / max(1, len(lst))) if len(lst) > 0 else float("nan")

    summary = {
        "num_samples": len(ds),
        "runtime_avg": _mean(runtimes),
        "psnr_bg_avg": _mean(psnrs),
        "mse_bg_avg": _mean(mses),
        "clip_whole_avg": _mean(clips_whole),
        "clip_edited_avg": _mean(clips_edited),
    }

    logger.log(f"Summary: {summary}")

    # Save per-sample CSV
    csv_dir = cfg.get("logging", {}).get("results_csv_dir", "results")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "piebench_results.csv")
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
    except Exception:
        # Minimal CSV writer
        import csv
        keys = list(results[0].keys()) if results else ["id", "edit_type", "path", "runtime", "psnr_bg", "mse_bg", "clip_whole", "clip_edited"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    logger.log(f"Saved results CSV to {csv_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SwiftEdit on PieBench")
    parser.add_argument("--defaults", type=str, default="swiftedit/configs/defaults.yaml", help="Path to defaults.yaml")
    parser.add_argument("--config", type=str, default=None, help="Optional override config YAML")
    parser.add_argument("--piebench_root", type=str, default=None, help="Override PieBench root directory")
    parser.add_argument("--use_gt_masks", action="store_true", help="Use GT masks during editing (ARaM) rather than self-guided")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda|cpu)")
    parser.add_argument("--resolution", type=int, default=None, help="Image resolution override (square)")
    parser.add_argument("--inversion_ckpt", type=str, default=None, help="Path to inversion-net checkpoint (Stage 2 EMA recommended)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_config(args.defaults, args.config)

    # Overrides
    override: Dict[str, Any] = {}
    if args.piebench_root:
        override.setdefault("eval", {}).setdefault("piebench", {})["root"] = args.piebench_root
    if args.use_gt_masks:
        override.setdefault("eval", {}).setdefault("piebench", {})["use_gt_masks"] = True
    if args.device:
        override.setdefault("system", {})["device"] = args.device
    if args.resolution:
        override.setdefault("inference", {})["image_resolution"] = int(args.resolution)
    if args.inversion_ckpt:
        override.setdefault("inference", {})["inversion_checkpoint"] = args.inversion_ckpt

    summary = evaluate_piebench(cfg, override=override)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

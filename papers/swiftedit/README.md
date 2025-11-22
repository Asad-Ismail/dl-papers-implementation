# SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion (Unofficial Reproduction)

This repository implements an end-to-end, one-step text-guided image editing system that couples a fast one-step generator with an inversion network and mask-aware Attention Rescaling (ARaM). It supports self-guided mask extraction (no user mask required), localized edits with strong background preservation, and a streamlined training and evaluation pipeline.

Core features implemented:
- One-step latent generator G(ε, c_y) -> z (placeholder SwiftBrushV2-compatible module)
- SDXL-compatible VAE encoder/decoder (diffusers AutoencoderKL wrapper)
- Frozen CLIP text/image encoders (OpenCLIP ViT-L/14)
- IP-Adapter image-conditioning (projector + learnable image KV branch)
- Decoupled cross-attention and ARaM integration for region-aware editing
- Inversion network Fθ(z, c_y) -> ε̂ with EMA tracking
- Stage 1 synthetic training (L_rec + L_regr)
- Stage 2 real training (DISTS perceptual + SDS-inspired regularization)
- Self-guided mask extractor from ε̂ differences
- Inference pipeline (CLI + Python API) with optional user mask
- PieBench evaluation (PSNR/MSE on background, CLIP-Whole/Edited, runtime)

Note: If SwiftBrushv2 pretrained weights are not available, the repo defaults to a strong one-step substitute (SDXL-Turbo-like latent generator). Teacher diffusion is stubbed by default but can be wired to SDXL base UNet if assets are provided.


## Table of Contents
- Quickstart
- Installation and Environment
- Downloading Model Assets
- Repo Structure
- Smoke Tests
- Training
  - Stage 1 (Synthetic)
  - Stage 2 (Real)
- Inference (Editing)
- Evaluation (PieBench)
- Configuration Overview
- ARaM and Self-Guided Masks
- Datasets
- Checkpoints and Logs
- Troubleshooting
- Acknowledgments and License


## Quickstart
```bash
# 1) Create environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 2) Download all required model assets (SDXL base/vae, OpenCLIP, one-step generator)
bash scripts/download_models.sh --all -o assets

# 3) (Optional) Source the paths manifest to set env variables for scripts
source assets/model_paths.env

# 4) Run a simple smoke test for inference
bash scripts/run_edit.sh \
  --image path/to/image.jpg \
  --src-prompt "a cute corgi in a park" \
  --edit-prompt "a cute corgi wearing sunglasses" \
  -o corgi_edit.png

# 5) Evaluate on a small PieBench subset (after placing the dataset)
bash scripts/eval_piebench.sh --piebench-root data/piebench
```


## Installation and Environment
- Python >= 3.10
- PyTorch >= 2.1 with CUDA 12.x (recommended: install official wheels for your CUDA)
- torchvision >= 0.16
- diffusers ~= 0.27
- transformers ~= 4.43
- open-clip-torch ~= 2.24.0
- timm >= 0.9.0,<0.10.0
- DISTS-pytorch (optional but recommended for Stage 2 perceptual loss)
- numpy, pillow, opencv-python, scikit-image, scipy, pyyaml, tqdm, einops, pandas (optional for CSV)

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip (Alternative)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you need CUDA-matched PyTorch wheels, consult https://pytorch.org/get-started/locally/ for +cu121 builds.

Hardware:
- A100 40GB was used in the paper for ~0.23 s per image. 24GB+ GPUs work with smaller batches and mixed precision.
- Inference < 8–12GB VRAM with autocast fp16.


## Downloading Model Assets
Use the provided script to fetch assets from Hugging Face and stage them under `assets/`:
```bash
bash scripts/download_models.sh --all -o assets
```
This downloads:
- SDXL base UNet (teacher) – optional for SDS regularization
- SDXL VAE – required
- OpenCLIP ViT-L/14 – required
- One-step generator – defaults to `stabilityai/sdxl-turbo` substitute

A manifest `assets/model_paths.env` is created. You can `source` it or set paths in `configs/defaults.yaml`.


## Repo Structure
```
.
├── configs/           # YAMLs: defaults, stage1, stage2, inference
├── scripts/           # Shell scripts for training, inference, evaluation, downloads
├── swiftedit/         # Main package
│   ├── models/        # VAE, CLIP, generator, IP-Adapter, inversion, attention
│   ├── losses/        # DISTS, PSNR/MSE, CLIP scores
│   ├── schedulers/    # Teacher diffusion + noise scheduler wrapper
│   ├── train/         # Stage 1/2 datasets and trainers
│   ├── edit/          # Mask extractor, ARaM utils, inference pipeline
│   ├── eval/          # PieBench dataset and evaluator
│   └── utils/         # Logging, checkpoints, timers, seeding, visualization
└── requirements.txt
```


## Smoke Tests
Simple import checks:
```bash
python - <<'PY'
import torch
from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
from swiftedit.models.vae.vaesdxl import VAESDXL as _  # noqa: just import
print("Imports OK; CUDA:", torch.cuda.is_available())
PY
```
One-step generation + decode (requires assets configured):
```python
import torch, yaml
from swiftedit.models.generator.swiftbrushv2 import SwiftBrushV2
from swiftedit.models.vae.vae_sdxl import VAESDXL
from swiftedit.models.clip.text_encoder import CLIPTextEncoder
cfg = yaml.safe_load(open('swiftedit/configs/defaults.yaml'))
G = SwiftBrushV2.from_config(cfg)
VAE = VAESDXL.from_config(cfg)
TE = CLIPTextEncoder.from_config(cfg)

B,H,W = 1, 64, 64  # latent H,W (images 512x512 => latents 64x64)
eps = torch.randn(B, 4, H, W).to(G.conv_in.weight.device)
text = TE(["a white cat"])
z_hat = G(eps, text)
x_hat = VAE.decode(z_hat)
print(x_hat.shape, x_hat.min().item(), x_hat.max().item())
```


## Training
Training uses YAML configs under `swiftedit/configs/`. You can run via Python modules or provided shell scripts.

### Stage 1 (Synthetic)
Objective: Train inversion network Fθ and IP-Adapter image branch on synthetic latent pairs.
- Losses: L_rec = ||z − ẑ||², L_regr = ||ε − ε̂||²; total L = L_rec + λ L_regr (λ=1)
- Trainable: Fθ, IP-Adapter projector, IP-Adapter branch; others frozen
- Batch size: 4 by default; iterations: 100k; EMA decay ≈ 0.999

Run:
```bash
# Script wrapper
bash scripts/train_stage1.sh \
  -d configs/defaults.yaml \
  -c configs/stage1.yaml

# Or Python module
python -m swiftedit.train.stage1.trainer_stage1 \
  --defaults configs/defaults.yaml \
  --config configs/stage1.yaml
```
Outputs:
- Checkpoints under `checkpoints/` (configurable)
- Logs under `logs/` with loss curves and optional recon samples

### Stage 2 (Real)
Objective: Train inversion network Fθ on real images using DISTS perceptual loss and SDS-style regularization.
- Losses: L_perc = DISTS(x, x̂) + 0.5 w(t) ||ε̂ − ε_ϕ(z_t, t, c_y)||²
- Trainable: Fθ only; encoders and IP-Adapter branch frozen
- Batch size: 1; iterations: 180k; EMA decay ≈ 0.999
- Teacher diffusion is optional; the provided stub returns zeros (acts as mild L2 on ε̂)

Run:
```bash
bash scripts/train_stage2.sh \
  -d configs/defaults.yaml \
  -c configs/stage2.yaml
# Or
python -m swiftedit.train.stage2.trainer_stage2 \
  --defaults configs/defaults.yaml \
  --config configs/stage2.yaml
```
Outputs:
- Fθ checkpoints (with EMA) under `checkpoints/`
- Logs under `logs/` with DISTS and regularization curves


## Inference (Editing)
Edit an image with source and target prompts, using self-guided masks (or a provided mask) and ARaM scaling.

Script:
```bash
bash scripts/run_edit.sh \
  --image examples/corgi.jpg \
  --src-prompt "a cute corgi in a park" \
  --edit-prompt "a cute corgi wearing sunglasses" \
  -o outputs/corgi_edit.png \
  --inversion-ckpt checkpoints/stage2_ema.pt \
  --s-y 1.0 --s-edit 0.3 --s-non-edit 1.5
```
Python API:
```python
import yaml, torch
from swiftedit.edit.inference import edit_image
cfg = yaml.safe_load(open('configs/inference.yaml'))
res = edit_image(
    image='examples/corgi.jpg',
    prompt_src='a cute corgi in a park',
    prompt_edit='a cute corgi wearing sunglasses',
    cfg=cfg,
)
# Save
from swiftedit.edit.inference import save_image
save_image(res['x_edit'], 'outputs/corgi_edit.png')
```


## Evaluation (PieBench)
Load the PieBench dataset and compute metrics: PSNR/MSE on unedited background, CLIP-Whole, CLIP-Edited, and runtime.
```bash
bash scripts/eval_piebench.sh \
  --piebench-root data/piebench \
  --inversion-ckpt checkpoints/stage2_ema.pt
```
Outputs: CSV under `results/` with per-sample metrics and summary.

Self-guided masks are used by default; you can enable GT masks during editing via `--use-gt-masks` for small improvements.


## Configuration Overview
Primary configs:
- `configs/defaults.yaml` – global defaults for paths, models, training, ARaM, etc.
- `configs/stage1.yaml` – overrides for Stage 1
- `configs/stage2.yaml` – overrides for Stage 2
- `configs/inference.yaml` – overrides for inference/evaluation

Notes:
- Some YAML values reference other keys using `${...}` placeholders; the current loader reads literal values. Scripts merge YAML files; if you need variable interpolation, resolve values in your driver code or provide absolute paths.
- Key fields:
  - paths.* – asset and dataset locations
  - models.vae/generator/clip – model settings
  - ip_adapter.projector/branch – image-conditioning tokens and learnable KV projections
  - inversion_net – Fθ with EMA
  - aram – ARaM scales and mask post-processing
  - training.stage1/2 – hyperparameters and freeze settings
  - schedulers.teacher – teacher/scheduler settings for SDS regularization
  - inference – resolution, mask options, precision, checkpoint path


## ARaM and Self-Guided Masks
- Self-guided mask M: computed from differences of inversion noise predictions under source vs edit prompts: m = ||Fθ(z, c_src) − Fθ(z, c_edit)||₂ over channels, min-max normalized.
- Optional Gaussian blur and soft thresholding keep the mask continuous; values clamped to [0,1].
- ARaM scaling combines decoupled attention:
  - Without mask: h = Attn_text + s_x Attn_image
  - With mask: h = s_y M Attn_text + s_edit M Attn_image + s_non-edit (1−M) Attn_image
- Default scales favor background preservation: s_y=1.0, s_edit=0.3, s_non-edit=1.5.
- You can supply a user mask (e.g., a binary PNG) to guide edits more precisely. The pipeline resizes/broadcasts masks to latent/attention shapes.

Implementation note: This reproduction applies ARaM at a fused conditioning stage (single step). Per-layer mask injection can be added by extending `GeneratorIP` with per-layer attention hooks.


## Datasets
- JourneyDB captions (Stage 1 synthetic): Provide ~40k captions, one per line (tab-separated first column also supported). Update `paths.datasets.journeydb_captions` in defaults.yaml as needed.
- CommonCanvas (Stage 2 real): place images and a prompts JSON (default `prompts.json`) under `data/commoncanvas`. The dataset loader supports various manifest schemas (list of items or dict with "items").
- PieBench (Evaluation): place the benchmark under `data/piebench`. Masks are optional for editing; evaluation uses GT masks to compute background PSNR/MSE.


## Checkpoints and Logs
- Checkpoints saved to `checkpoints/` by default with EMA shadows included when enabled.
- Logs saved to `logs/` (loss curves, optional visualizations).
- Results CSVs saved to `results/` during evaluation.


## Troubleshooting
- "ModuleNotFoundError: open_clip": Install `open-clip-torch~=2.24.0` (already in requirements.txt). Ensure `timm<0.10`.
- CUDA/runtime mismatch: Verify PyTorch build matches your CUDA (e.g., +cu121). Check `torch.version.cuda` and `nvidia-smi` outputs.
- OOM during training: Reduce batch size (Stage 1 from 4 to 2/1), enable autocast (fp16), or resize images. Stage 2 typically fits in <16GB.
- diffusers import/weights: Ensure `diffusers~=0.27` and that VAE assets exist under the path configured.
- Performance: Enable autocast (cfg.system.autocast: true) and consider `torch.compile` for `GeneratorIP` and `InversionNet` if your PyTorch version supports it.
- Teacher diffusion: The provided `TeacherDiffusion` returns zeros unless configured with SDXL base UNet. If you need stronger SDS regularization, integrate your UNet and scheduler via `schedulers/noise_scheduler.py`.


## Acknowledgments and License
This is an unofficial reproduction of "SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion". It relies on:
- Stability AI SDXL assets (VAE, base UNet)
- OpenCLIP (LAION) for text/image encoders
- diffusers (Hugging Face) for model wrappers and schedulers
- DISTS-pytorch for perceptual loss

Please refer to the licenses of these dependencies. This repository is provided for research/educational purposes.

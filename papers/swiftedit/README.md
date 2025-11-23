# SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion (Unofficial Reproduction)

⚠️ **Implementation Status: Active Development**

This repository contains an implementation of SwiftEdit, bootstrapped with DeepCode and refined through human review.

**✅ Implemented:**
- Model architecture definitions (VAE, CLIP, Generator, IP-Adapter, Inversion Network)
- Training implementations (Stage 1: Synthetic, Stage 2: Real)
- Dataset loaders (synthetic and real data pipelines)
- Loss function modules (DISTS, CLIP scores, PSNR/MSE)
- Configuration system (YAML configs for training/inference)
- Training scripts and utilities
- Checkpoint management and logging

**⚠️ In Progress:**
- Inference pipeline (editing implementation)
- Evaluation scripts (PieBench metrics)
- Model weights and pre-trained checkpoints

**Note:** Training scripts are functional. Inference and evaluation are under active development.


## Table of Contents
- Installation
- Available Components
- Repository Structure
- Development Status
- Configuration Overview
- Acknowledgments and License


## Installation

```bash
# Navigate to swiftedit directory
cd papers/swiftedit

# Create environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Verify installation
python -c "import swiftedit; print('SwiftEdit package installed')"
```

## Quick Start - Training

**First time setup:** Pre-download model weights to avoid long waits during training:

```bash
# Option 1: Use the download script (recommended)
python scripts/download_models.py

# Option 2: Manual download with wget (faster)
mkdir -p ~/.cache/huggingface/hub
cd ~/.cache/huggingface/hub

# Download OpenCLIP ViT-L-14 (1.71GB)
wget https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin

# Download SDXL VAE
huggingface-cli download stabilityai/sdxl-vae --local-dir models--stabilityai--sdxl-vae

# Download SDXL base model
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir models--stabilityai--stable-diffusion-xl-base-1.0
```

**Run training:**

```bash
# Stage 1 Training (Synthetic)
bash scripts/train_stage1.sh -d configs/defaults.yaml -c configs/stage1.yaml

# Stage 2 Training (Real)
bash scripts/train_stage2.sh -d configs/defaults.yaml -c configs/stage2.yaml

# Or run directly with Python
python -m swiftedit.train.stage1.trainer_stage1 --defaults configs/defaults.yaml --config configs/stage1.yaml
```

## Requirements
- Python >= 3.10
- PyTorch >= 2.1
- See `requirements.txt` for full dependencies

## Repository Structure
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


## Available Components

The following modules are implemented and ready for integration:

**Models:**
- `swiftedit.models.vae.vae_sdxl` - SDXL VAE encoder/decoder
- `swiftedit.models.clip.text_encoder` - CLIP text encoder
- `swiftedit.models.clip.image_encoder` - CLIP image encoder
- `swiftedit.models.generator.swiftbrushv2` - One-step generator
- `swiftedit.models.generator.generator_ip` - Generator with IP-Adapter
- `swiftedit.models.inversion.inversion_net` - Inversion network
- `swiftedit.models.ip_adapter` - IP-Adapter components

**Losses:**
- `swiftedit.losses.dists_loss` - Perceptual DISTS loss
- `swiftedit.losses.clip_scores` - CLIP similarity metrics
- `swiftedit.losses.psnr_mse` - PSNR and MSE metrics

**Utilities:**
- `swiftedit.utils.checkpoint` - Model checkpoint handling
- `swiftedit.utils.logger` - Training logger
- `swiftedit.schedulers.noise_scheduler` - Diffusion noise scheduling


## Training

Training is implemented for both stages. See configuration files in `configs/` for hyperparameters.

### Stage 1: Synthetic Training
Trains the inversion network and IP-Adapter on synthetic latent pairs.

```bash
bash scripts/train_stage1.sh -d configs/defaults.yaml -c configs/stage1.yaml
```

### Stage 2: Real Data Training
Fine-tunes on real images with perceptual losses.

```bash
bash scripts/train_stage2.sh -d configs/defaults.yaml -c configs/stage2.yaml
```

## Next Steps

**To complete this implementation:**
1. ✅ Training scripts - Implemented
2. ⚠️ Inference pipeline - Implement `swiftedit/edit/inference.py`
3. ⚠️ Evaluation scripts - Complete PieBench evaluation
4. 📦 Pre-trained weights - Train and release checkpoints


## Acknowledgments and License
This is an unofficial reproduction of "SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion". It relies on:
- Stability AI SDXL assets (VAE, base UNet)
- OpenCLIP (LAION) for text/image encoders
- diffusers (Hugging Face) for model wrappers and schedulers
- DISTS-pytorch for perceptual loss

Please refer to the licenses of these dependencies. This repository is provided for research/educational purposes.

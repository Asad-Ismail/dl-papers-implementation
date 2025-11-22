# SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion (Unofficial Reproduction)

⚠️ **Implementation Status: Partial/In Development**

This repository contains a **partial implementation** of SwiftEdit, bootstrapped with DeepCode. Currently available:

**✅ Implemented:**
- Model architecture definitions (VAE, CLIP, Generator, IP-Adapter, Inversion Network)
- Loss function modules (DISTS, CLIP scores, PSNR/MSE)
- Configuration templates (YAML configs for training/inference)
- Shell script templates for training and evaluation
- Utility modules (checkpoints, logging, schedulers)

**❌ Not Yet Implemented:**
- Training loop implementations (trainer_stage1.py, trainer_stage2.py)
- Dataset loaders (dataset_synthetic.py, dataset_real.py)
- Inference pipeline implementation
- Evaluation scripts
- Model weights and pre-trained checkpoints

This is a work-in-progress reproduction. The architecture and supporting code are in place, but the training and inference implementations need to be completed.


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


## Development Status

**What's Missing:**
- Training loop implementations (need to create `trainer_stage1.py`, `trainer_stage2.py`, dataset loaders)
- Inference pipeline implementation
- Evaluation script implementations
- Model weights/checkpoints

**How to Contribute:**
If you want to complete this implementation:
1. Implement training scripts in `swiftedit/train/stage1/` and `swiftedit/train/stage2/`
2. Implement dataset loaders for synthetic and real data
3. Implement the inference pipeline in `swiftedit/edit/inference.py`
4. Test with the provided configuration files

The architecture is defined and the infrastructure (configs, scripts, utilities) is in place - it needs the core training and inference logic.


## Acknowledgments and License
This is an unofficial reproduction of "SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion". It relies on:
- Stability AI SDXL assets (VAE, base UNet)
- OpenCLIP (LAION) for text/image encoders
- diffusers (Hugging Face) for model wrappers and schedulers
- DISTS-pytorch for perceptual loss

Please refer to the licenses of these dependencies. This repository is provided for research/educational purposes.

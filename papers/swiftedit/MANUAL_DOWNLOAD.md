# Manual Download Instructions for SwiftEdit Models

If automated downloads fail due to SSL/certificate issues, follow these manual steps:

## Option 1: Use Web Browser (Most Reliable)

1. **SDXL VAE** (most essential, ~335 MB)
   - Visit: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main
   - Download to `assets/sdxl_vae/`:
     - `config.json`
     - `diffusion_pytorch_model.safetensors`

2. **SDXL Base** (for scheduler, ~3-7 GB)
   - Visit: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main
   - Download to `assets/sdxl_base/`:
     - `model_index.json`
     - `scheduler/scheduler_config.json`
     - `unet/config.json`
     - `unet/diffusion_pytorch_model.safetensors` (large!)

3. **SDXL-Turbo** (one-step generator, ~7 GB)
   - Visit: https://huggingface.co/stabilityai/sdxl-turbo/tree/main
   - Download to `assets/generator_one_step/`:
     - `model_index.json`
     - `unet/config.json`
     - `unet/diffusion_pytorch_model.safetensors`
     - `vae/config.json`
     - `vae/diffusion_pytorch_model.safetensors`

4. **OpenCLIP** (~1.7 GB)
   - Visit: https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/tree/main
   - Download to `assets/openclip_vitl14/`:
     - `open_clip_pytorch_model.bin`
     - `config.json`

## Option 2: Use wget with SSL disabled

```bash
# Create directories
mkdir -p assets/sdxl_vae assets/sdxl_base/unet assets/sdxl_base/scheduler
mkdir -p assets/generator_one_step/unet assets/generator_one_step/vae
mkdir -p assets/openclip_vitl14

# SDXL VAE
wget --no-check-certificate -P assets/sdxl_vae \
  https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json \
  https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors

# SDXL Base (minimal files)
wget --no-check-certificate -P assets/sdxl_base \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json

wget --no-check-certificate -P assets/sdxl_base/scheduler \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/scheduler/scheduler_config.json

wget --no-check-certificate -P assets/sdxl_base/unet \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/config.json \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors

# Continue for other models...
```

## Option 3: Use curl with SSL disabled

```bash
# SDXL VAE
curl -k -L -o assets/sdxl_vae/config.json \
  https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json

curl -k -L -o assets/sdxl_vae/diffusion_pytorch_model.safetensors \
  https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors
```

## Minimum Required Files

To get started quickly, you need **at minimum**:

1. **SDXL VAE** (essential for training):
   - `assets/sdxl_vae/config.json`
   - `assets/sdxl_vae/diffusion_pytorch_model.safetensors`

2. **SDXL Base Scheduler**:
   - `assets/sdxl_base/scheduler/scheduler_config.json`

3. **Generator (SDXL-Turbo)**:
   - `assets/generator_one_step/unet/config.json`
   - `assets/generator_one_step/unet/diffusion_pytorch_model.safetensors`

After downloading, verify with:
```bash
ls -lh assets/*/
```

Then run training:
```bash
bash scripts/train_stage1.sh -d configs/defaults.yaml -c configs/stage1.yaml
```

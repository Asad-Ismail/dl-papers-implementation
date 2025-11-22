#!/usr/bin/env bash
set -euo pipefail

# SwiftEdit model asset downloader
#
# This script downloads all pretrained assets required by the SwiftEdit pipeline:
# - SDXL base (teacher UNet and config)
# - SDXL VAE (decoder/encoder for latent space)
# - OpenCLIP ViT-L/14 (text+image encoders)
# - One-step generator (defaults to SDXL-Turbo as a substitute for SwiftBrushv2)
#
# Requirements:
# - Python with huggingface_hub installed (pip install huggingface_hub safetensors)
# - Sufficient disk space (10–30 GB)
#
# Usage examples:
#   bash swiftedit/scripts/download_models.sh --all -o assets
#   bash swiftedit/scripts/download_models.sh --sdxl-base --sdxl-vae --openclip -o ~/.cache/swiftedit
#   GENERATOR_REPO="stabilityai/sd-turbo" bash swiftedit/scripts/download_models.sh --generator -o assets
#
# After download, set the paths in configs/defaults.yaml accordingly.

# Defaults (can be overridden by env)
OUT_DIR="assets"
SDXL_BASE_REPO="stabilityai/stable-diffusion-xl-base-1.0"
# Popular community VAE with fp16 fix for SDXL
SDXL_VAE_REPO="madebyollin/sdxl-vae-fp16-fix"
# OpenCLIP ViT-L/14 trained on LAION-2B
OPENCLIP_REPO="laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
# Fallback one-step generator if SwiftBrushv2 is unavailable
GENERATOR_REPO=${GENERATOR_REPO:-"stabilityai/sdxl-turbo"}

# Parse args
DO_BASE=0
DO_VAE=0
DO_OPENCLIP=0
DO_GEN=0
DO_ALL=0
REVISION="main"

print_help() {
  cat <<EOF
SwiftEdit Model Downloader

Options:
  --all                 Download all assets (SDXL base, SDXL VAE, OpenCLIP, generator)
  --sdxl-base           Download SDXL base (teacher UNet)
  --sdxl-vae            Download SDXL VAE
  --openclip            Download OpenCLIP ViT-L/14 weights
  --generator           Download one-step generator (default: \"$GENERATOR_REPO\")
  -o, --out DIR         Output base directory (default: $OUT_DIR)
  -r, --rev REV         Revision/branch to pin (default: $REVISION)
  -h, --help            Show this help

Environment overrides:
  GENERATOR_REPO        Repo ID for the one-step generator (default: $GENERATOR_REPO)
  SDXL_BASE_REPO        Repo ID for SDXL base (default: $SDXL_BASE_REPO)
  SDXL_VAE_REPO         Repo ID for SDXL VAE (default: $SDXL_VAE_REPO)
  OPENCLIP_REPO         Repo ID for OpenCLIP (default: $OPENCLIP_REPO)

Examples:
  bash swiftedit/scripts/download_models.sh --all -o assets
  GENERATOR_REPO=stabilityai/sd-turbo bash swiftedit/scripts/download_models.sh --generator -o assets
EOF
}

# Simple arg parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) DO_ALL=1; shift ;;
    --sdxl-base) DO_BASE=1; shift ;;
    --sdxl-vae) DO_VAE=1; shift ;;
    --openclip) DO_OPENCLIP=1; shift ;;
    --generator) DO_GEN=1; shift ;;
    -o|--out) OUT_DIR="$2"; shift 2 ;;
    -r|--rev) REVISION="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 1 ;;
  esac
done

if [[ $DO_ALL -eq 1 ]]; then
  DO_BASE=1; DO_VAE=1; DO_OPENCLIP=1; DO_GEN=1
fi

mkdir -p "$OUT_DIR"

# Python helper for snapshot_download with optional allow_patterns
py_download() {
  local repo_id="$1"; shift
  local dest="$1"; shift
  local revision="$1"; shift
  local patterns_json="$1"; shift
  python - "$repo_id" "$dest" "$revision" "$patterns_json" <<'PY'
import json, os, sys
from huggingface_hub import snapshot_download
repo_id, dest, revision, patterns_json = sys.argv[1:5]
allow_patterns = None
if patterns_json and patterns_json != "null":
    allow_patterns = json.loads(patterns_json)
os.makedirs(dest, exist_ok=True)
print(f"[HF] Downloading {repo_id} -> {dest} (rev={revision})")
snapshot_download(repo_id=repo_id, local_dir=dest, revision=revision,
                  local_dir_use_symlinks=False,
                  allow_patterns=allow_patterns)
print("[HF] Done.")
PY
}

# SDXL base (teacher)
if [[ $DO_BASE -eq 1 ]]; then
  BASE_DIR="$OUT_DIR/sdxl_base"
  export SDXL_BASE_REPO=${SDXL_BASE_REPO:-$SDXL_BASE_REPO}
  echo "Downloading SDXL base from ${SDXL_BASE_REPO} to ${BASE_DIR}"
  # Keep core components to reduce size
  py_download "${SDXL_BASE_REPO}" "$BASE_DIR" "$REVISION" '["*.json","*.yaml","*.txt","scheduler/*","unet/*","tokenizer*/*","text_encoder*/*","vae/*","*.safetensors"]'
fi

# SDXL VAE
if [[ $DO_VAE -eq 1 ]]; then
  VAE_DIR="$OUT_DIR/sdxl_vae"
  export SDXL_VAE_REPO=${SDXL_VAE_REPO:-$SDXL_VAE_REPO}
  echo "Downloading SDXL VAE from ${SDXL_VAE_REPO} to ${VAE_DIR}"
  py_download "${SDXL_VAE_REPO}" "$VAE_DIR" "$REVISION" 'null'
fi

# OpenCLIP ViT-L/14 (LAION2B)
if [[ $DO_OPENCLIP -eq 1 ]]; then
  CLIP_DIR="$OUT_DIR/openclip_vitl14"
  export OPENCLIP_REPO=${OPENCLIP_REPO:-$OPENCLIP_REPO}
  echo "Downloading OpenCLIP ViT-L/14 from ${OPENCLIP_REPO} to ${CLIP_DIR}"
  py_download "${OPENCLIP_REPO}" "$CLIP_DIR" "$REVISION" 'null'
fi

# One-step generator (SwiftBrushv2 substitute)
if [[ $DO_GEN -eq 1 ]]; then
  GEN_DIR="$OUT_DIR/generator_one_step"
  echo "Downloading one-step generator from ${GENERATOR_REPO} to ${GEN_DIR}"
  py_download "${GENERATOR_REPO}" "$GEN_DIR" "$REVISION" 'null'
fi

# Write a small paths manifest to help configure configs/defaults.yaml
MANIFEST="$OUT_DIR/model_paths.env"
echo "# Auto-generated by download_models.sh" > "$MANIFEST"
[[ $DO_BASE -eq 1 || $DO_ALL -eq 1 ]] && echo "SDXL_BASE_DIR=\"$OUT_DIR/sdxl_base\"" >> "$MANIFEST"
[[ $DO_VAE -eq 1 || $DO_ALL -eq 1 ]] && echo "SDXL_VAE_DIR=\"$OUT_DIR/sdxl_vae\"" >> "$MANIFEST"
[[ $DO_OPENCLIP -eq 1 || $DO_ALL -eq 1 ]] && echo "OPENCLIP_DIR=\"$OUT_DIR/openclip_vitl14\"" >> "$MANIFEST"
[[ $DO_GEN -eq 1 || $DO_ALL -eq 1 ]] && echo "ONE_STEP_GEN_DIR=\"$OUT_DIR/generator_one_step\"" >> "$MANIFEST"

echo "\nDownload complete. Paths manifest written to: $MANIFEST"
echo "Next: set these paths in configs/defaults.yaml (will be created) under:"
echo "  paths:"
echo "    sdxl_base: \"$OUT_DIR/sdxl_base\""
echo "    sdxl_vae: \"$OUT_DIR/sdxl_vae\""
echo "    openclip: \"$OUT_DIR/openclip_vitl14\""
echo "    generator: \"$OUT_DIR/generator_one_step\""

#!/usr/bin/env bash
# SwiftEdit — Stage 1 training launcher
# Runs the synthetic training stage (IP-Adapter branch + inversion network)
#
# Usage:
#   bash swiftedit/scripts/train_stage1.sh \
#       --defaults swiftedit/configs/defaults.yaml \
#       --config swiftedit/configs/stage1.yaml \
#       [--assets-env assets/model_paths.env] \
#       [--python python]
#
# Tips:
#  - Run scripts/download_models.sh --all to fetch model assets, then pass --assets-env
#    pointing to the generated manifest to export helper environment variables.
#  - You can override paths and hyperparameters in the YAML config files.
#  - This script simply forwards the config paths to the Python trainer.

set -euo pipefail

print_help() {
  cat <<EOF
SwiftEdit Stage 1 Training

Options:
  -d, --defaults FILE       Path to defaults YAML (default: swiftedit/configs/defaults.yaml)
  -c, --config FILE         Path to Stage 1 YAML (default: swiftedit/configs/stage1.yaml)
      --assets-env FILE     Optional env manifest to source (from scripts/download_models.sh)
      --python CMD          Python executable/command (default: python)
  -h, --help                Show this help message

Examples:
  bash swiftedit/scripts/train_stage1.sh \
    --defaults swiftedit/configs/defaults.yaml \
    --config swiftedit/configs/stage1.yaml \
    --assets-env assets/model_paths.env
EOF
}

DEFAULTS="swiftedit/configs/defaults.yaml"
CONFIG="swiftedit/configs/stage1.yaml"
ASSETS_ENV=""
PYTHON_CMD="python"
EXTRA_ARGS=()

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--defaults)
      DEFAULTS="$2"; shift 2 ;;
    -c|--config)
      CONFIG="$2"; shift 2 ;;
    --assets-env)
      ASSETS_ENV="$2"; shift 2 ;;
    --python)
      PYTHON_CMD="$2"; shift 2 ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      # Forward any extra args to the Python trainer (e.g., logging overrides)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Source assets manifest if provided
if [[ -n "$ASSETS_ENV" ]]; then
  if [[ -f "$ASSETS_ENV" ]]; then
    echo "Sourcing assets manifest: $ASSETS_ENV"
    # shellcheck disable=SC1090
    source "$ASSETS_ENV"
    echo "Asset paths:"
    [[ -n "${SDXL_BASE_DIR:-}" ]] && echo "  SDXL_BASE_DIR=$SDXL_BASE_DIR"
    [[ -n "${SDXL_VAE_DIR:-}" ]] && echo "  SDXL_VAE_DIR=$SDXL_VAE_DIR"
    [[ -n "${OPENCLIP_DIR:-}" ]] && echo "  OPENCLIP_DIR=$OPENCLIP_DIR"
    [[ -n "${ONE_STEP_GEN_DIR:-}" ]] && echo "  ONE_STEP_GEN_DIR=$ONE_STEP_GEN_DIR"
  else
    echo "[WARN] assets manifest not found: $ASSETS_ENV"
  fi
fi

# Basic checks
if [[ ! -f "$DEFAULTS" ]]; then
  echo "[ERROR] Defaults config not found: $DEFAULTS" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Stage 1 config not found: $CONFIG" >&2
  exit 1
fi

# Print environment summary
echo "Running Stage 1 training with:"
echo "  Defaults: $DEFAULTS"
echo "  Config:   $CONFIG"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA devices:"
  nvidia-smi || true
else
  echo "nvidia-smi not found; proceeding without GPU summary."
fi

# Launch Python trainer
set -x
"$PYTHON_CMD" -m swiftedit.train.stage1.trainer_stage1 \
  --defaults "$DEFAULTS" \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}"
set +x

#!/usr/bin/env bash
# SwiftEdit Stage 2 Training Launcher
# Runs real-image training with DISTS + SDS-inspired regularization.

set -euo pipefail

print_help() {
  cat << 'EOF'
SwiftEdit Stage 2 Training

Usage:
  swiftedit/scripts/train_stage2.sh [options] [-- extra args forwarded]

Options:
  -d, --defaults FILE     Path to defaults YAML (default: swiftedit/configs/defaults.yaml)
  -c, --config FILE       Path to Stage 2 YAML (default: swiftedit/configs/stage2.yaml)
      --assets-env FILE   Optional environment manifest from scripts/download_models.sh
      --python CMD        Python executable/command (default: python)
  -h, --help              Show this help and exit

Notes:
- Extra args after options are forwarded to the Python trainer (e.g., --resume CKPT, etc.).
- If --assets-env is provided, the file is sourced to populate model asset paths.
EOF
}

DEFAULTS="swiftedit/configs/defaults.yaml"
CONFIG="swiftedit/configs/stage2.yaml"
ASSETS_ENV=""
PYTHON_CMD="python"
EXTRA_ARGS=()

# Parse CLI
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--defaults)
      DEFAULTS="$2"; shift 2;;
    -c|--config)
      CONFIG="$2"; shift 2;;
    --assets-env)
      ASSETS_ENV="$2"; shift 2;;
    --python)
      PYTHON_CMD="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    --)
      shift; while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done; break;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

# Validate config paths
if [[ ! -f "$DEFAULTS" ]]; then
  echo "[ERROR] Defaults YAML not found: $DEFAULTS" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Stage 2 YAML not found: $CONFIG" >&2
  exit 1
fi

# Source assets manifest if provided
if [[ -n "$ASSETS_ENV" ]]; then
  if [[ -f "$ASSETS_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$ASSETS_ENV"
    echo "[INFO] Sourced assets manifest: $ASSETS_ENV"
    [[ -n "${SDXL_BASE_DIR:-}" ]] && echo "  SDXL_BASE_DIR=$SDXL_BASE_DIR"
    [[ -n "${SDXL_VAE_DIR:-}" ]] && echo "  SDXL_VAE_DIR=$SDXL_VAE_DIR"
    [[ -n "${OPENCLIP_DIR:-}" ]] && echo "  OPENCLIP_DIR=$OPENCLIP_DIR"
    [[ -n "${ONE_STEP_GEN_DIR:-}" ]] && echo "  ONE_STEP_GEN_DIR=$ONE_STEP_GEN_DIR"
  else
    echo "[WARN] Assets env file not found: $ASSETS_ENV" >&2
  fi
fi

# GPU summary (optional)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU Summary (nvidia-smi):"
  nvidia-smi || true
else
  echo "[INFO] nvidia-smi not found; proceeding without GPU summary."
fi

# Final command
CMD=("$PYTHON_CMD" -m swiftedit.train.stage2.trainer_stage2 --defaults "$DEFAULTS" --config "$CONFIG")
# Append extra args
for arg in "${EXTRA_ARGS[@]}"; do CMD+=("$arg"); done

# Execute
set -x
"${CMD[@]}"
set +x

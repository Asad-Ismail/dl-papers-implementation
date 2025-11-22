#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat << 'EOF'
SwiftEdit — PieBench Evaluation Runner

Usage:
  eval_piebench.sh [options] [-- EXTRA_ARGS]

Required/Primary options:
  -d, --defaults FILE         Path to defaults YAML (default: swiftedit/configs/defaults.yaml)
  -c, --config FILE           Path to eval/inference YAML override (default: swiftedit/configs/inference.yaml)
      --piebench-root DIR     Root directory of PieBench dataset (overrides config)
      --use-gt-masks          Use ground-truth masks for ARaM during editing in addition to metrics
      --device DEV            Device to run on (e.g., cuda, cuda:0, cpu)
      --resolution INT        Target image resolution override (e.g., 512)
      --inversion-ckpt FILE   Path to inversion-net checkpoint (overrides config)

Utility options:
      --assets-env FILE       Source environment manifest produced by scripts/download_models.sh
      --python CMD            Python executable/command (default: python)
  -h, --help                  Show this help and exit

Notes:
- Any arguments after "--" are forwarded verbatim to the Python evaluator module.
- The evaluator writes a CSV summary into cfg.logging.results_csv_dir by default.

Examples:
  bash swiftedit/scripts/eval_piebench.sh \
    --piebench-root data/piebench --device cuda --use-gt-masks

EOF
}

# Defaults
DEFAULTS="swiftedit/configs/defaults.yaml"
CONFIG="swiftedit/configs/inference.yaml"
PIEBENCH_ROOT=""
USE_GT_MASKS=false
DEVICE=""
RESOLUTION=""
INVERSION_CKPT=""
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
    --piebench-root)
      PIEBENCH_ROOT="$2"; shift 2;;
    --use-gt-masks)
      USE_GT_MASKS=true; shift 1;;
    --device)
      DEVICE="$2"; shift 2;;
    --resolution)
      RESOLUTION="$2"; shift 2;;
    --inversion-ckpt)
      INVERSION_CKPT="$2"; shift 2;;
    --assets-env)
      ASSETS_ENV="$2"; shift 2;;
    --python)
      PYTHON_CMD="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    --)
      shift; # consume "--"
      while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done;;
    *)
      # Forward unknown args
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

# Validate config files
if [[ ! -f "$DEFAULTS" ]]; then
  echo "[ERROR] Defaults YAML not found: $DEFAULTS" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config YAML not found: $CONFIG" >&2
  exit 1
fi

# Optionally source assets env manifest
if [[ -n "$ASSETS_ENV" ]]; then
  if [[ -f "$ASSETS_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$ASSETS_ENV"
    echo "[INFO] Sourced assets env: $ASSETS_ENV"
    [[ -n "${SDXL_BASE_DIR:-}" ]] && echo "  SDXL_BASE_DIR=$SDXL_BASE_DIR"
    [[ -n "${SDXL_VAE_DIR:-}" ]] && echo "  SDXL_VAE_DIR=$SDXL_VAE_DIR"
    [[ -n "${OPENCLIP_DIR:-}" ]] && echo "  OPENCLIP_DIR=$OPENCLIP_DIR"
    [[ -n "${ONE_STEP_GEN_DIR:-}" ]] && echo "  ONE_STEP_GEN_DIR=$ONE_STEP_GEN_DIR"
  else
    echo "[WARN] --assets-env file not found: $ASSETS_ENV" >&2
  fi
fi

# GPU summary (optional)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU status:"; nvidia-smi || true
fi

# Build Python command
CMD=("$PYTHON_CMD" -m swiftedit.eval.evaluate_piebench --defaults "$DEFAULTS" --config "$CONFIG")

# Append optional overrides
if [[ -n "$PIEBENCH_ROOT" ]]; then
  CMD+=(--piebench_root "$PIEBENCH_ROOT")
fi
if [[ "$USE_GT_MASKS" == "true" ]]; then
  CMD+=(--use_gt_masks)
fi
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ -n "$RESOLUTION" ]]; then
  CMD+=(--resolution "$RESOLUTION")
fi
if [[ -n "$INVERSION_CKPT" ]]; then
  CMD+=(--inversion_ckpt "$INVERSION_CKPT")
fi

# Append extra args
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("--")
  for arg in "${EXTRA_ARGS[@]}"; do CMD+=("$arg"); done
fi

# Execute
set -x
"${CMD[@]}"
set +x

echo "[INFO] PieBench evaluation completed. See results CSV in configured directory (logging.results_csv_dir)."

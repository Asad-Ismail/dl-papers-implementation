#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
SwiftEdit - One-step Text-guided Image Editing (Inference Runner)
Usage:
  run_edit.sh [options]

Required arguments:
  -i, --image PATH              Path to the source image to edit
  --src-prompt TEXT             Source (original) prompt describing the image
  --edit-prompt TEXT            Edit prompt describing the desired change

Common options:
  -o, --out PATH                Output path for the edited image (default: alongside input, suffix _edit.png)
  -d, --defaults FILE           Defaults YAML config (default: swiftedit/configs/defaults.yaml)
  -c, --config FILE             Inference YAML config (default: swiftedit/configs/inference.yaml)
  -m, --mask PATH               Optional user-supplied mask image; if omitted, self-guided mask is used
  --inversion-ckpt PATH         Optional inversion-net checkpoint to load (overrides config)
  --device DEV                  Device to use (e.g., cuda, cuda:0, or cpu)
  --resolution INT              Image resolution for inference (default from config)
  --s-y FLOAT                   ARaM scale for text attention (default from config)
  --s-edit FLOAT                ARaM scale for image attention on edit regions (default from config)
  --s-non-edit FLOAT            ARaM scale for image attention on non-edit regions (default from config)
  --assets-env FILE             Optional env manifest from scripts/download_models.sh to source paths
  --python CMD                  Python executable/command (default: python)
  -h, --help                    Show this help message and exit

Any extra arguments after -- will be forwarded to the Python module.
EOF
}

# Defaults
DEFAULTS="swiftedit/configs/defaults.yaml"
CONFIG="swiftedit/configs/inference.yaml"
IMAGE=""
SRC_PROMPT=""
EDIT_PROMPT=""
OUT=""
MASK=""
INVERSION_CKPT=""
DEVICE=""
RESOLUTION=""
S_Y=""
S_EDIT=""
S_NON_EDIT=""
ASSETS_ENV=""
PYTHON_CMD="python"
EXTRA_ARGS=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--defaults)
      DEFAULTS="$2"; shift 2;;
    -c|--config)
      CONFIG="$2"; shift 2;;
    -i|--image)
      IMAGE="$2"; shift 2;;
    --src-prompt)
      SRC_PROMPT="$2"; shift 2;;
    --edit-prompt)
      EDIT_PROMPT="$2"; shift 2;;
    -o|--out)
      OUT="$2"; shift 2;;
    -m|--mask)
      MASK="$2"; shift 2;;
    --inversion-ckpt)
      INVERSION_CKPT="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --resolution)
      RESOLUTION="$2"; shift 2;;
    --s-y)
      S_Y="$2"; shift 2;;
    --s-edit)
      S_EDIT="$2"; shift 2;;
    --s-non-edit)
      S_NON_EDIT="$2"; shift 2;;
    --assets-env)
      ASSETS_ENV="$2"; shift 2;;
    --python)
      PYTHON_CMD="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    --)
      shift; while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done; break;;
    *)
      # Forward unknown args
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

# Validate required inputs
if [[ -z "$IMAGE" ]] || [[ -z "$SRC_PROMPT" ]] || [[ -z "$EDIT_PROMPT" ]]; then
  echo "[ERROR] Missing required arguments.\n" >&2
  print_help
  exit 1
fi

# Verify config files exist
if [[ ! -f "$DEFAULTS" ]]; then
  echo "[ERROR] Defaults YAML not found: $DEFAULTS" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Inference YAML not found: $CONFIG" >&2
  exit 1
fi

# Source assets env if provided
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
    echo "[WARN] --assets-env provided but file not found: $ASSETS_ENV" >&2
  fi
fi

# Derive default output path if not provided
if [[ -z "$OUT" ]]; then
  stem="${IMAGE%.*}"
  OUT="${stem}_edit.png"
fi

# GPU summary
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU summary:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

# Build Python command
CMD=("$PYTHON_CMD" -m swiftedit.edit.inference \
  --defaults "$DEFAULTS" \
  --config "$CONFIG" \
  --image "$IMAGE" \
  --prompt-src "$SRC_PROMPT" \
  --prompt-edit "$EDIT_PROMPT" \
  --out "$OUT"
)

# Optional flags
if [[ -n "$MASK" ]]; then CMD+=(--mask "$MASK"); fi
if [[ -n "$INVERSION_CKPT" ]]; then CMD+=(--inversion-ckpt "$INVERSION_CKPT"); fi
if [[ -n "$DEVICE" ]]; then CMD+=(--device "$DEVICE"); fi
if [[ -n "$RESOLUTION" ]]; then CMD+=(--resolution "$RESOLUTION"); fi
if [[ -n "$S_Y" ]]; then CMD+=(--s-y "$S_Y"); fi
if [[ -n "$S_EDIT" ]]; then CMD+=(--s-edit "$S_EDIT"); fi
if [[ -n "$S_NON_EDIT" ]]; then CMD+=(--s-non-edit "$S_NON_EDIT"); fi

# Forward any extra args
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("--" "${EXTRA_ARGS[@]}")
fi

echo "[INFO] Running inference..."
set -x
"${CMD[@]}"
set +x

echo "[OK] Edited image saved to: $OUT"

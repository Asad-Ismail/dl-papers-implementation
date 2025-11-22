#!/usr/bin/env bash
set -euo pipefail

# REGLA: From-scratch training on WikiText-103
# Usage:
#   bash regla/scripts/run_train_wt.sh [CONFIG_YAML]
# If CONFIG_YAML is not provided, defaults to regla/configs/regla_wt103.yaml

# Recommended environment variables for determinism and memory behavior
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
# Uncomment to pin a specific GPU
# export CUDA_VISIBLE_DEVICES=0

# Default config
DEFAULT_CONF="configs/regla_wt103.yaml"
CONF_PATH="${1:-$DEFAULT_CONF}"

if [ ! -f "$CONF_PATH" ]; then
  echo "Config file not found: $CONF_PATH" >&2
  exit 1
fi

# Create output directories if they can be inferred from the config defaults
mkdir -p checkpoints || true
mkdir -p logs || true

# Run training via the package entrypoint to avoid relying on CLI parser specifics
python -c "
import sys
import traceback

try:
    import regla
    print('[run_train_wt] Using config:', '$CONF_PATH')
    regla.train_lm('$CONF_PATH')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"

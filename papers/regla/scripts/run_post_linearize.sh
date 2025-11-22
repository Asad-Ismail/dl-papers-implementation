#!/usr/bin/env bash
set -euo pipefail

# Post-linearization + continual pretraining runner
# Usage:
#   ./regla/scripts/run_post_linearize.sh [CONFIG_YAML]
#
# Defaults to regla/configs/post_linearize_sp.yaml

export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM="false"

CONF_PATH=${1:-"regla/configs/post_linearize_sp.yaml"}

if [[ ! -f "$CONF_PATH" ]]; then
  echo "Config not found: $CONF_PATH" >&2
  exit 1
fi

mkdir -p checkpoints logs

python - <<'PY'
import sys, traceback
from regla.train.post_linearize_and_continual import main as run
try:
    # Forward the config path as argv if provided
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "${CONF_PATH}"])
    run()
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
PY

#!/usr/bin/env bash
set -euo pipefail

# Speed/Memory benchmarking runner
# Usage:
#   ./regla/scripts/run_bench.sh [CONFIG_YAML]
#
# Defaults to regla/configs/regla_wt103.yaml

export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM="false"

CONF_PATH=${1:-"regla/configs/regla_wt103.yaml"}

if [[ ! -f "$CONF_PATH" ]]; then
  echo "Config not found: $CONF_PATH" >&2
  exit 1
fi

python - <<'PY'
import sys, json, traceback
import regla
try:
    regla.run_bench("${CONF_PATH}")
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY

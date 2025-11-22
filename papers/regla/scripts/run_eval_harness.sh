#!/usr/bin/env bash
set -euo pipefail

# lm-evaluation-harness runner via REGLA wrapper
# Usage:
#   ./regla/scripts/run_eval_harness.sh [CONFIG_YAML] [TASKS_CSV] [SHOTS]
#
# Defaults:
#   CONFIG_YAML = regla/configs/post_linearize_sp.yaml
#   TASKS_CSV   = boolq,piqa,hellaswag,winogrande,truthfulqa_mc1,truthfulqa_mc2
#   SHOTS       = 0

export TOKENIZERS_PARALLELISM="false"

CONF_PATH=${1:-"regla/configs/post_linearize_sp.yaml"}
TASKS=${2:-"boolq,piqa,hellaswag,winogrande,truthfulqa_mc1,truthfulqa_mc2"}
SHOTS=${3:-"0"}

if [[ ! -f "$CONF_PATH" ]]; then
  echo "Config not found: $CONF_PATH" >&2
  exit 1
fi

python - <<'PY'
import sys, traceback
try:
    from regla.eval.harness_wrapper import run_harness_cli
    # Synthesize CLI args for the wrapper
    sys.argv = [
        "regla-harness",
        "--config", "${CONF_PATH}",
        "--tasks", "${TASKS}",
        "--shots", "${SHOTS}"
    ]
    run_harness_cli()
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY

#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-6h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"

CONFIGS=(
  "conf/paper_icdm/cn69_pilot_20260526/pilot_lago_acm_local_9_10.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log="$LOG_DIR/pilot_gpu7_lago_real-data_bridge_${name}_${STAMP}.log"
  echo "[LAGO real-data bridge] $(date -Is) running $config with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[LAGO real-data bridge] $config exited with status $status; see $log"
    FAILED=1
  fi
done


cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-6h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

if [[ ! -d "$DSBM_ROOT" ]]; then
  echo "DSBM root not found: $DSBM_ROOT"
  echo "Set DSBM_ROOT=/path/to/datasets-sbm or pass a valid datasets-sbm directory."
  exit 2
fi

log="$LOG_DIR/pilot_gpu7_pilot_dsbm_lago_random_mc290_5b_gpu7_${STAMP}.log"
echo "[LAGO DSBM feasibility] $(date -Is) running DSBM pilot with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes random \
  --max-changes 290 \
  --methods lago \
  --modes dynamic smart \
  --smart-depth 3 \
  --smart-radius 1 \
  --catch-errors \
  --output-dir results/paper_icdm \
  --name "pilot_dsbm_lago_random_mc290_5b_gpu7_${STAMP}" \
  --limit 1 \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[LAGO DSBM feasibility] DSBM pilot exited with status $status; checkpointed files, if any, are in results/paper_icdm"
fi
if [[ "$status" -ne 0 ]]; then
  FAILED=1
fi
exit "$FAILED"

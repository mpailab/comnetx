#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-48h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

if [[ ! -d "$DSBM_ROOT" ]]; then
  echo "DSBM root not found: $DSBM_ROOT"
  exit 2
fi

log="$LOG_DIR/gpu7_final_dsbm_update_size_gpu7_${STAMP}.log"
echo "[DSBM update-size final] $(date -Is) running DSBM job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes random hubs community \
  --max-changes 290 1450 2900 \
  --methods leidenalg dfleiden \
  --modes naive smart dynamic \
  --smart-depth 3 \
  --smart-radius 1 \
  --use-gpu \
  --catch-errors \
  --output-dir results/paper_icdm \
  --name "final_dsbm_update_size_gpu7_${STAMP}" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[DSBM update-size final] exited with status $status; checkpointed files, if any, are in results/paper_icdm"
fi
exit "$status"

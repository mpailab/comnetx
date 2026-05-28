#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
SKIP_REGISTRY="${SKIP_REGISTRY:-results/registry/all_results.json}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-6h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm


log="$LOG_DIR/retry04_dsbm_random_5_batches_mc29000_${STAMP}.log"
echo "[retry04_dsbm_random_5_batches_mc29000] $(date -Is) running focused five-batch DSBM recovery with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes random \
  --max-changes 29000 \
  --methods leidenalg \
  --modes smart naive \
  --use-gpu \
  --catch-errors \
  --skip-registry "$SKIP_REGISTRY" \
  --output-dir results/paper_icdm \
  --name "retry04_dsbm_random_5_batches_mc29000_${STAMP}" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[retry04_dsbm_random_5_batches_mc29000] exited with status $status; see $log"
fi
exit "$status"

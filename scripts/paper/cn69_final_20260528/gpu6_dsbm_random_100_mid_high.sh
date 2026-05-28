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
TIMEOUT="${TIMEOUT:-12h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_dsbm() {
  local title="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/${name}_${STAMP}.log"
  echo "[${title}] $(date -Is) running DSBM stress with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \
    "$@" \
    --methods leidenalg dfleiden \
    --modes naive smart dynamic \
    --use-gpu \
    --catch-errors \
    --skip-registry "$SKIP_REGISTRY" \
    --output-dir results/paper_icdm \
    --name "${name}_${STAMP}" \
    2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[${title}] exited with status $status; see $log"
    FAILED=1
  fi
}

run_dsbm "DSBM random 100 batches mid/high" "gpu6_dsbm_random_100_batches_mid_high" \
  --root "$DSBM_ROOT" \
  --batch-suffix 100_batches \
  --regimes random \
  --max-changes 2900 14500 29000

exit "$FAILED"

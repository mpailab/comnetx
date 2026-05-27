#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-9h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

log="$LOG_DIR/extra24_after3_dfleiden_closure_variants_arxivmath_9500_long_extra24_${STAMP}.log"
echo "[DF-Leiden long closure variants arxivmath 9:500] $(date -Is) running workload/profile job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \
  --datasets arxivmath \
  --batches 9:500 \
  --methods dfleiden \
  --variants no_closure no_contraction \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 120 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "after3_dfleiden_closure_variants_arxivmath_9500_long_extra24_${STAMP}" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[DF-Leiden long closure variants arxivmath 9:500] exited with status $status; see $log"
fi
exit "$status"

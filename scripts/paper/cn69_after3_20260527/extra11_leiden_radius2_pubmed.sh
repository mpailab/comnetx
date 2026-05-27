#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-12h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

log="$LOG_DIR/extra11_after3_leiden_radius2_pubmed_extra11_${STAMP}.log"
echo "[Leiden radius-2 dyn_pubmed workload] $(date -Is) running workload/profile job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \
  --datasets dyn_pubmed \
  --batches 999:50 \
  --methods leidenalg \
  --variants full \
  --smart-depth 3 \
  --smart-radius 2 \
  --aggregation-mode sum \
  --max-updates 20 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "after3_leiden_radius2_pubmed_extra11_${STAMP}" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[Leiden radius-2 dyn_pubmed workload] exited with status $status; see $log"
fi
exit "$status"

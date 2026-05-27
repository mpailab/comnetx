#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-24h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

log="$LOG_DIR/gpu1_after3_workload_memory_topology_gpu1_${STAMP}.log"
echo "[workload/memory topology] $(date -Is) running workload/profile job with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \
  --datasets dyn_pubmed arxivmath \
  --batches 999:10 999:50 9:500 \
  --methods leidenalg dfleiden \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors \
  --paths-config "$PATHS_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --output-dir results/paper_icdm \
  --name "after3_workload_memory_topology_gpu1_${STAMP}" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[workload/memory topology] exited with status $status; see $log"
fi
exit "$status"

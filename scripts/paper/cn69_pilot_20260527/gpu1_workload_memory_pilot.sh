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
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_profile() {
  local name="$1"
  shift
  local log="$LOG_DIR/gpu1_${name}_${STAMP}.log"
  echo "[workload/memory pilot] $(date -Is) running ${name} with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py "$@" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[workload/memory pilot] ${name} exited with status $status; see $log"
    FAILED=1
  fi
}

run_profile "topology_workload_memory"   --datasets dyn_pubmed arxivmath   --batches 999:10 999:50   --methods leidenalg dfleiden   --variants full   --smart-depth 3   --smart-radius 1   --aggregation-mode sum   --max-updates 50   --use-gpu   --force-undirected   --ground-truth-metrics   --catch-errors   --paths-config "$PATHS_CONFIG"   --cache-dir "$CACHE_DIR"   --output-dir results/paper_icdm   --name "profile_topology_workload_memory_gpu1_${STAMP}"

run_profile "s2cag_workload_memory"   --datasets dyn_pubmed arxivmath   --batches 999:10 999:50   --methods s2cag   --feature-modes random   --baseline-iter 10   --variants full   --smart-depth 3   --smart-radius 1   --aggregation-mode norm   --max-updates 10   --use-gpu   --force-undirected   --ground-truth-metrics   --catch-errors   --paths-config "$PATHS_CONFIG"   --cache-dir "$CACHE_DIR"   --output-dir results/paper_icdm   --name "profile_s2cag_workload_memory_gpu1_${STAMP}"

exit "$FAILED"

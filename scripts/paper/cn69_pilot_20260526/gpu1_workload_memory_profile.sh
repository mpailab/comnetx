#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-6h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_profile() {
  local name="$1"
  shift
  local log="$LOG_DIR/pilot_gpu1_${name}_${STAMP}.log"
  echo "[GPU workload profile] $(date -Is) running ${name} with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py "$@" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[GPU workload profile] ${name} exited with status $status; see $log"
    FAILED=1
  fi
}

run_profile "topology_workload"   --datasets dyn_pubmed arxivmath   --batches 999:100 99:200   --methods leidenalg dfleiden   --smart-depth 3   --smart-radius 1   --aggregation-mode sum   --max-updates 20   --use-gpu   --force-undirected   --ground-truth-metrics   --catch-errors   --paths-config "$PATHS_CONFIG"   --cache-dir "$CACHE_DIR"   --output-dir results/paper_icdm   --name "profile_topology_workload_gpu1_${STAMP}"

run_profile "s2cag_workload"   --datasets dyn_cora dyn_pubmed   --batches 999:100   --methods s2cag   --feature-modes dataset   --baseline-iter 10   --smart-depth 3   --smart-radius 1   --aggregation-mode norm   --max-updates 10   --use-gpu   --force-undirected   --ground-truth-metrics   --catch-errors   --paths-config "$PATHS_CONFIG"   --cache-dir "$CACHE_DIR"   --output-dir results/paper_icdm   --name "profile_s2cag_workload_gpu1_${STAMP}"

exit "$FAILED"

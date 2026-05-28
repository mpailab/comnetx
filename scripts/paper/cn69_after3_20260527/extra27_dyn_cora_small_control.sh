#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-2h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

run_profile() {
  local title="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/extra27_${name}_${STAMP}.log"
  echo "[dyn_cora small control] $(date -Is) running ${title} with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \
    "$@" \
    --paths-config "$PATHS_CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --output-dir results/paper_icdm \
    --name "${name}_${STAMP}" \
    2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[dyn_cora small control] ${title} exited with status $status; see $log"
    FAILED=1
  fi
}

run_profile "topology full profiles" "dyn_cora_workload_topology_extra27" \
  --datasets dyn_cora \
  --batches 999:50 \
  --methods leidenalg dfleiden \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile "S2CAG full profile" "dyn_cora_workload_s2cag_extra27" \
  --datasets dyn_cora \
  --batches 999:50 \
  --methods s2cag \
  --feature-modes random \
  --baseline-iter 10 \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode norm \
  --max-updates 10 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile "Leiden closure/contraction ablation" "dyn_cora_closure_contraction_extra27" \
  --datasets dyn_cora \
  --batches 999:50 \
  --methods leidenalg \
  --variants full no_closure no_contraction \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

exit "$FAILED"

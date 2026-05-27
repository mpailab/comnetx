#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
LOG_DIR="${LOG_DIR:-output}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "conf/paper_icdm/cn69/lago_temporal_batch_sweep.json"
)

COMPLETED_CONFIGS=(
  "conf/paper_icdm/cn69/lago_temporal_batch_sweep.json"
)

for config in "${CONFIGS[@]}"; do
  if [[ "${RERUN_COMPLETED_CN69:-0}" != "1" ]]; then
    for completed in "${COMPLETED_CONFIGS[@]}"; do
      if [[ "$config" == "$completed" ]]; then
        echo "[LAGO temporal batch sweep] skipping already ingested $config from results/paper_icdm/3; set RERUN_COMPLETED_CN69=1 to rerun"
        continue 2
      fi
    done
  fi
  name="$(basename "$config" .json)"
  echo "[LAGO temporal batch sweep] $(date -Is) running $config on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$LOG_DIR/gpu5_lago_temporal_batch_sweep_${name}_${STAMP}.log"
done

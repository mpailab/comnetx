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
  "conf/paper_icdm/cn69/real_topology_long_horizon.json"
)

COMPLETED_CONFIGS=(
  "conf/paper_icdm/cn69/real_topology_long_horizon.json"
)

for config in "${CONFIGS[@]}"; do
  if [[ "${RERUN_COMPLETED_CN69:-0}" != "1" ]]; then
    for completed in "${COMPLETED_CONFIGS[@]}"; do
      if [[ "$config" == "$completed" ]]; then
        echo "[real topology long horizon] skipping already ingested $config from results/paper_icdm/3; set RERUN_COMPLETED_CN69=1 to rerun"
        continue 2
      fi
    done
  fi
  name="$(basename "$config" .json)"
  echo "[real topology long horizon] $(date -Is) running $config on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$LOG_DIR/gpu1_real_topology_long_horizon_${name}_${STAMP}.log"
done

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
LOG_DIR="${LOG_DIR:-logs/paper_icdm/cn69}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "conf/paper_icdm/cn69/real_topology_batch_sweep.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  echo "[real topology batch sweep] $(date -Is) running $config on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$LOG_DIR/gpu0_${name}_${STAMP}.log"
done

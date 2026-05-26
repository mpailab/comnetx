#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
LOG_DIR="${LOG_DIR:-logs/paper_icdm/cn69}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "conf/paper_icdm/cn69/dmon_dataset_batch_sweep.json"
  "conf/paper_icdm/cn69/dmon_random_batch_sweep_seed_1.json"
  "conf/paper_icdm/cn69/dmon_random_batch_sweep_seed_2.json"
  "conf/paper_icdm/cn69/dmon_random_batch_sweep_seed_3.json"
  "conf/paper_icdm/cn69/dmon_random_batch_sweep_seed_4.json"
  "conf/paper_icdm/cn69/dmon_random_batch_sweep_seed_5.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  echo "[DMoN batch and seed sweep] $(date -Is) running $config on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$LOG_DIR/gpu3_${name}_${STAMP}.log"
done

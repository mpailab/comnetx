#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1

LOG_DIR="${LOG_DIR:-logs/paper_icdm/cn69}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

echo "[DSBM topology stress] $(date -Is) running DSBM stress on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python scripts/paper/run_dsbm_stress.py \
  --all-batches \
  --methods leidenalg dfleiden \
  --modes naive smart dynamic \
  --smart-depth 3 \
  --smart-radius 1 --use-gpu \
  --catch-errors \
  --output-dir results/paper_icdm \
  --name "dsbm_topology_stress_gpu6_${STAMP}" \
  2>&1 | tee "$LOG_DIR/gpu6_dsbm_topology_stress_gpu6_${STAMP}.log"

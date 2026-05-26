#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
LOG_DIR="${LOG_DIR:-output}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

if [[ ! -d "$DSBM_ROOT" ]]; then
  echo "DSBM root not found: $DSBM_ROOT"
  echo "Set DSBM_ROOT=/path/to/datasets-sbm or pass a valid datasets-sbm directory."
  exit 2
fi

echo "[DSBM topology stress] $(date -Is) running DSBM stress on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
python scripts/paper/run_dsbm_stress.py \
  --root "$DSBM_ROOT" \
  --all-batches \
  --methods leidenalg dfleiden \
  --modes naive smart dynamic \
  --smart-depth 3 \
  --smart-radius 1 --use-gpu \
  --catch-errors \
  --output-dir results/paper_icdm \
  --name "dsbm_topology_stress_gpu6_${STAMP}" \
  2>&1 | tee "$LOG_DIR/gpu6_dsbm_topology_stress_gpu6_${STAMP}.log"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <max-changes> <seed>" >&2
  exit 2
fi

MAX_CHANGES="$1"
SEED="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-10}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_dsbm_seed_robustness/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_dsbm_seed_robustness/series_${SERIES}}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PYTHONUNBUFFERED=1

for REGIME in hubs community; do
  DATASET="dsbm-${REGIME}-1024-58-mc${MAX_CHANGES}-${SEED}"
  DATASET_ROOT="datasets-sbm/dsbm-${REGIME}-1024-58/${DATASET}"
  RUN_ID="${REGIME}_mc${MAX_CHANGES}_seed${SEED}_100b"

  mkdir -p "${RESULTS_ROOT}/${RUN_ID}" "${LOG_ROOT}"

  export PARENT_HOSTNAME="cn69_dsbm_${REGIME}_mc${MAX_CHANGES}_seed${SEED}"

  echo "Run: ${RUN_ID}"
  echo "DATASET_ROOT=${DATASET_ROOT}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
  echo "OUTPUT_DIR=${RESULTS_ROOT}/${RUN_ID}"

  python scripts/paper/run_dsbm_stress.py \
    --root "${DATASET_ROOT}" \
    --batch-suffix 100_batches \
    --regimes "${REGIME}" \
    --max-changes "${MAX_CHANGES}" \
    --methods leidenalg \
    --modes naive smart \
    --smart-depth 3 \
    --smart-radius 1 \
    --use-gpu \
    --output-dir "${RESULTS_ROOT}/${RUN_ID}" \
    --name "${RUN_ID}" \
    --catch-errors \
    2>&1 | tee "${LOG_ROOT}/${RUN_ID}.log"
done

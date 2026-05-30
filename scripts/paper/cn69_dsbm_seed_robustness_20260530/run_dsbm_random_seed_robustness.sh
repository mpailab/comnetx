#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <max-changes> <seed>" >&2
  exit 2
fi

MAX_CHANGES="$1"
SEED="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-9}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
DATASET="dsbm-random-1024-58-mc${MAX_CHANGES}-${SEED}"
DATASET_ROOT="datasets-sbm/dsbm-random-1024-58/${DATASET}"
RUN_ID="random_mc${MAX_CHANGES}_seed${SEED}_100b"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_dsbm_seed_robustness/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_dsbm_seed_robustness/series_${SERIES}}"

mkdir -p "${RESULTS_ROOT}/${RUN_ID}" "${LOG_ROOT}"

if [[ ! -f "${DATASET_ROOT}/out.${DATASET}.100_batches" ]]; then
  echo "Missing stream: ${DATASET_ROOT}/out.${DATASET}.100_batches" >&2
  exit 1
fi
if [[ ! -f "${DATASET_ROOT}/coms.${DATASET}.100_batches.npz" ]]; then
  echo "Missing communities: ${DATASET_ROOT}/coms.${DATASET}.100_batches.npz" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69_dsbm_random_mc${MAX_CHANGES}_seed${SEED}}"
export PYTHONUNBUFFERED=1

echo "Run: ${RUN_ID}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
echo "OUTPUT_DIR=${RESULTS_ROOT}/${RUN_ID}"

python scripts/paper/run_dsbm_stress.py \
  --root "${DATASET_ROOT}" \
  --batch-suffix 100_batches \
  --regimes random \
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

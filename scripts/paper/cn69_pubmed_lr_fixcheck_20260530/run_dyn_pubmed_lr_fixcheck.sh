#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu-id> <repeat-id>" >&2
  exit 2
fi

GPU_SLOT="$1"
REPEAT_ID="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-8}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
RUN_ID="r${REPEAT_ID}_gpu${GPU_SLOT}_dyn_pubmed_lr_fixcheck"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_fixcheck/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_fixcheck/series_${SERIES}}"

mkdir -p "${RESULTS_ROOT}/${RUN_ID}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69_lr_fixcheck_r${REPEAT_ID}}"
export PYTHONUNBUFFERED=1
export RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"

echo "Run: ${RUN_ID}"
echo "GPU_SLOT=${GPU_SLOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
echo "RESULTS_DIR=${RESULTS_DIR}"

python scripts/launch.py \
  "${SCRIPT_DIR}/dyn_pubmed_lr_fixcheck.json" \
  --paths-config datasets-info/paths/cn69.json \
  2>&1 | tee "${LOG_ROOT}/${RUN_ID}.log"

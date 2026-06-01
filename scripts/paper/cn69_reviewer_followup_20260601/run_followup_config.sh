#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpu-slot> <run-id> <config> [<config> ...]" >&2
  exit 2
fi

GPU_SLOT="$1"
RUN_ID="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-12}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_reviewer_followup/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_reviewer_followup/series_${SERIES}}"

mkdir -p "${RESULTS_ROOT}/${RUN_ID}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69_reviewer_${RUN_ID}}"
export PYTHONUNBUFFERED=1
export RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"

echo "Run: ${RUN_ID}"
echo "GPU_SLOT=${GPU_SLOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
echo "RESULTS_DIR=${RESULTS_DIR}"

for CONFIG in "$@"; do
  echo "Config: ${CONFIG}"
  python scripts/launch.py \
    "${SCRIPT_DIR}/${CONFIG}" \
    --paths-config datasets-info/paths/cn69.json \
    2>&1 | tee -a "${LOG_ROOT}/${RUN_ID}.log"
done

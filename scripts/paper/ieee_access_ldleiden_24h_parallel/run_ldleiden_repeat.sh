#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <short|long> <repeat-number>" >&2
  exit 2
fi

PHASE="$1"
REPEAT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

case "${PHASE}" in
  short)
    CONFIG="${SCRIPT_DIR}/ldleiden_short_999_10.json"
    TIMEOUT_VALUE="${LDLEIDEN_SHORT_TIMEOUT:-12h}"
    ;;
  long)
    CONFIG="${SCRIPT_DIR}/ldleiden_long_9_500.json"
    TIMEOUT_VALUE="${LDLEIDEN_LONG_TIMEOUT:-23h}"
    ;;
  *)
    echo "Unknown phase: ${PHASE}; expected short or long" >&2
    exit 2
    ;;
esac

if ! [[ "${REPEAT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Repeat number must be a positive integer: ${REPEAT}" >&2
  exit 2
fi

printf -v REPEAT_PADDED '%02d' "${REPEAT}"
CAMPAIGN_ID="${LDLEIDEN_CAMPAIGN_ID:-ldleiden-cn69-20260828-v3}"
RUN_ID="${PHASE}-r${REPEAT_PADDED}"
RESULTS_ROOT="${LDLEIDEN_RESULTS_ROOT:-results/ieee-access-2026-1/raw/ldleiden/${CAMPAIGN_ID}}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"
LOG_FILE="${RESULTS_DIR}/run.log"
RESULT_FILE="${RESULTS_DIR}/$(basename "${CONFIG}" .json)_now.json"
DONE_MARKER="${RESULTS_DIR}/.done"
FAILED_MARKER="${RESULTS_DIR}/.failed"
PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"

mkdir -p "${RESULTS_DIR}"

if [[ -f "${DONE_MARKER}" ]]; then
  echo "Already completed: ${RESULTS_DIR}"
  exit 0
fi

rm -f "${FAILED_MARKER}"

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1
export RESULTS_DIR
# A private cache avoids races between containers and preserves the exact
# bootstrap used by this repetition alongside its measurement JSON.
export COMNETX_CACHE_DIR="${LDLEIDEN_CACHE_DIR:-${RESULTS_DIR}/bootstrap-cache}"
THREAD_COUNT="${LDLEIDEN_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export OPENBLAS_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

{
  echo
  echo "[$(date -Is)] Starting ${RUN_ID}"
  echo "Config: ${CONFIG}"
  echo "Results: ${RESULTS_DIR}"
  echo "Bootstrap cache: ${COMNETX_CACHE_DIR}"
  echo "Timeout: ${TIMEOUT_VALUE}"
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Python: $(python --version 2>&1)"
} | tee -a "${LOG_FILE}"

set +e
timeout --kill-after=2m "${TIMEOUT_VALUE}" \
  python scripts/launch.py "${CONFIG}" --paths-config "${PATHS_CONFIG}" \
  2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${STATUS}" -eq 0 && ! -s "${RESULT_FILE}" ]]; then
  echo "Expected result JSON was not created: ${RESULT_FILE}" \
    | tee -a "${LOG_FILE}" >&2
  STATUS=1
fi

if [[ "${STATUS}" -eq 0 ]]; then
  printf 'completed_at=%s\n' "$(date -Is)" >"${DONE_MARKER}"
  echo "[$(date -Is)] Completed ${RUN_ID}" | tee -a "${LOG_FILE}"
else
  printf 'exit_status=%s\nfailed_at=%s\n' \
    "${STATUS}" "$(date -Is)" >"${FAILED_MARKER}"
  echo "[$(date -Is)] Failed ${RUN_ID} with status ${STATUS}" \
    | tee -a "${LOG_FILE}" >&2
fi

exit "${STATUS}"

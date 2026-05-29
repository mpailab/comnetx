#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
LOG_DIR="${LOG_DIR:-output}"
PAPER_ICDM_SERIES="${PAPER_ICDM_SERIES:-9}"
RESULTS_DIR="${RESULTS_DIR:-results/paper_icdm/$PAPER_ICDM_SERIES}"
ROBUSTNESS_TIMEOUT="${ROBUSTNESS_TIMEOUT:-2h}"
export RESULTS_DIR

STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

run_config() {
  local label="$1"
  local config="$2"
  local timeout_value="${3:-$ROBUSTNESS_TIMEOUT}"
  local name log status
  name="$(basename "$config" .json)"
  log="$LOG_DIR/${label}_${name}_${STAMP}.log"
  echo "[${label}] $(date -Is) running ${config} timeout=${timeout_value} output=${RESULTS_DIR}"
  timeout --kill-after=2m "$timeout_value" python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[${label}] ${config} exited with status ${status}; see ${log}"
    FAILED=1
  fi
}

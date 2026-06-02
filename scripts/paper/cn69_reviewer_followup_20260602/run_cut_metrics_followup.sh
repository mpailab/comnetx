#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-15}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_reviewer_followup/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_reviewer_followup/series_${SERIES}}"
PATHS_CONFIG="${PAPER_ICDM_PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CONTINUE_ON_ERROR="${PAPER_ICDM_CONTINUE_ON_ERROR:-1}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

LOCK_DIR="${RESULTS_ROOT}/.reviewer_followup.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another reviewer follow-up queue appears to be active: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PARENT_HOSTNAME="${PAPER_ICDM_PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

MASTER_LOG="${LOG_ROOT}/reviewer_followup.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"
}

format_command() {
  local arg
  local quoted
  local out=""
  for arg in "$@"; do
    printf -v quoted '%q' "${arg}"
    out+="${quoted} "
  done
  printf '%s' "${out% }"
}

format_duration() {
  local seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((seconds / 3600))" \
    "$(((seconds % 3600) / 60))" \
    "$((seconds % 60))"
}

run_or_continue() {
  local task_id="$1"
  local label="$2"
  shift 2
  local task_dir="${RESULTS_ROOT}/${task_id}"
  local task_log="${LOG_ROOT}/${task_id}.log"
  local done_marker="${task_dir}/.done"
  local failed_marker="${task_dir}/.failed"

  mkdir -p "${task_dir}" "${LOG_ROOT}"
  if [[ -f "${done_marker}" ]]; then
    log "Skip completed ${task_id}: ${label}"
    return 0
  fi

  rm -f "${failed_marker}"
  local started_at
  local start_epoch
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  start_epoch="$(date '+%s')"
  log "Start ${task_id}: ${label}"
  log "Started at: ${started_at}"
  log "Output: ${task_dir}"
  log "Task log: ${task_log}"
  log "Command: $(format_command "$@")"

  if "$@" 2>&1 | tee -a "${task_log}"; then
    local finished_at
    local finish_epoch
    local elapsed
    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
    finish_epoch="$(date '+%s')"
    elapsed="$((finish_epoch - start_epoch))"
    touch "${done_marker}"
    log "Done ${task_id}: ${label}"
    log "Finished at: ${finished_at}; duration: $(format_duration "${elapsed}")"
  else
    local failed_at
    local fail_epoch
    local elapsed
    failed_at="$(date '+%Y-%m-%d %H:%M:%S')"
    fail_epoch="$(date '+%s')"
    elapsed="$((fail_epoch - start_epoch))"
    touch "${failed_marker}"
    log "Failed ${task_id}: ${label}"
    log "Failed at: ${failed_at}; duration: $(format_duration "${elapsed}")"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
}

run_cut_metrics() {
  local task_id="$1"
  local label="$2"
  shift 2
  local task_dir="${RESULTS_ROOT}/${task_id}"

  run_or_continue \
    "${task_id}" \
    "${label}" \
    python scripts/paper/compute_leiden_cut_metrics.py \
      --out "${task_dir}/${task_id}.json" \
      --cache-dir "${task_dir}/communities_cache" \
      --paths-config "${PATHS_CONFIG}" \
      "$@"
}

log "Reviewer follow-up queue started"
log "SERIES=${SERIES}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
log "RESULTS_ROOT=${RESULTS_ROOT}"

# Targeted structural-quality check for the reviewer comment about modularity.
# This repeats only the key Leiden final partitions used in the main
# large-graph 999:10 comparison: full recomputation and Local L=3,r=1 on
# dyn_pubmed and arxivmath. The script writes JSON after every completed
# dataset/mode pair, so partial results survive interruption.
run_cut_metrics \
  00_leiden_cut_metrics_99910 \
  "Leiden final cut metrics on key 999:10 graphs" \
  --datasets dyn_pubmed arxivmath \
  --modes naive smart \
  --batch 999:10 \
  --smart-depth 3 \
  --smart-radius 1 \
  --resolution 1.0 \
  --use-gpu \
  --force-undirected \
  --verbose 1

log "Reviewer follow-up queue completed"

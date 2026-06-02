#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-16}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_article_completion/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_article_completion/series_${SERIES}}"
PATHS_CONFIG="${PAPER_ICDM_PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CONTINUE_ON_ERROR="${PAPER_ICDM_CONTINUE_ON_ERROR:-1}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

LOCK_DIR="${RESULTS_ROOT}/.article_completion.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another article-completion follow-up appears to be active: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PARENT_HOSTNAME="${PAPER_ICDM_PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

MASTER_LOG="${LOG_ROOT}/article_completion.log"

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

run_launch_repeat() {
  local task_prefix="$1"
  local label="$2"
  local repeats="$3"
  local config="$4"

  for rep in $(seq 1 "${repeats}"); do
    local rep_id
    rep_id="$(printf '%s_r%02d' "${task_prefix}" "${rep}")"
    local rep_dir="${RESULTS_ROOT}/${rep_id}"
    run_or_continue \
      "${rep_id}" \
      "${label} (repeat ${rep}/${repeats})" \
      env RESULTS_DIR="${rep_dir}" \
      python scripts/launch.py "${SCRIPT_DIR}/${config}" --paths-config "${PATHS_CONFIG}"
  done
}

run_profile() {
  local task_id="$1"
  local label="$2"
  shift 2
  local task_dir="${RESULTS_ROOT}/${task_id}"

  run_or_continue \
    "${task_id}" \
    "${label}" \
    python scripts/paper/profile_smart_workload.py \
      --output-dir "${task_dir}" \
      --name "${task_id}" \
      --paths-config "${PATHS_CONFIG}" \
      "$@"
}

log "Article-completion follow-up queue started"
log "SERIES=${SERIES}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
log "RESULTS_ROOT=${RESULTS_ROOT}"

# Fill the only missing topology-ablation graph in icdm-2026-1. Full Leiden and
# the default Local L=3,r=1 row for dyn_cora already exist in the
# single-container Table II small-graph block, so this queue measures only the
# remaining grid points.
run_launch_repeat \
  00_leiden_dyn_cora_topology_main \
  "Leiden dyn_cora topology L x r main grid 999:10" \
  1 \
  00_leiden_dyn_cora_topology_main_99910.json

run_launch_repeat \
  01_leiden_dyn_cora_topology_l3 \
  "Leiden dyn_cora topology L=3 side points 999:10" \
  1 \
  01_leiden_dyn_cora_topology_l3_99910.json

run_profile \
  02_profile_leiden_dfleiden_workload_99910 \
  "Leiden and DF-Leiden workload profiles 999:10" \
  --datasets dyn_cora dyn_pubmed arxivmath \
  --batches 999:10 \
  --methods leidenalg dfleiden \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile \
  03_profile_s2cag_dataset_workload_99910 \
  "S2CAG dataset-feature workload profiles 999:10" \
  --datasets dyn_cora dyn_pubmed arxivmath \
  --batches 999:10 \
  --methods s2cag \
  --feature-modes dataset \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --baseline-iter 10 \
  --random-feature-dim 64 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

log "Article-completion follow-up queue completed"

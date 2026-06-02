#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SERIES="${PAPER_ICDM_SERIES:-14}"
VISIBLE_GPU="${PAPER_ICDM_VISIBLE_GPU:-0}"
RESULTS_ROOT="${PAPER_ICDM_RESULTS_ROOT:-results/paper_icdm_single_container_rerun/series_${SERIES}}"
LOG_ROOT="${PAPER_ICDM_LOG_ROOT:-logs/paper_icdm_single_container_rerun/series_${SERIES}}"
PATHS_CONFIG="${PAPER_ICDM_PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CONTINUE_ON_ERROR="${PAPER_ICDM_CONTINUE_ON_ERROR:-1}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

LOCK_DIR="${RESULTS_ROOT}/.single_container_rerun.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another single-container rerun appears to be active: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PARENT_HOSTNAME="${PAPER_ICDM_PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

MASTER_LOG="${LOG_ROOT}/single_container_rerun.log"

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

run_dsbm() {
  local task_id="$1"
  local label="$2"
  shift 2
  local task_dir="${RESULTS_ROOT}/${task_id}"

  run_or_continue \
    "${task_id}" \
    "${label}" \
    python scripts/paper/run_dsbm_stress.py \
      --output-dir "${task_dir}" \
      --name "${task_id}" \
      "$@"
}

log "Sequential rerun started"
log "SERIES=${SERIES}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "PARENT_HOSTNAME=${PARENT_HOSTNAME}"
log "RESULTS_ROOT=${RESULTS_ROOT}"

# Main paper baselines, ordered from shorter to longer. The four smaller
# Table II graphs are measured once; dyn_pubmed and arxivmath keep repeated
# runs because they drive the runtime-stability claims.
run_launch_repeat 00_leiden_table2_small "Leiden 999:10 Table II small/medium graphs" 1 00_leiden_table2_small_99910.json
run_launch_repeat 00_dfleiden_table2_small "DF-Leiden 999:10 Table II small/medium graphs" 1 00_dfleiden_table2_small_99910.json
run_launch_repeat 00_s2cag_dataset_table2_small "S2CAG dataset-feature 999:10 Table II small/medium graphs" 1 00_s2cag_dataset_table2_small_99910.json

run_launch_repeat 01_leiden_core "Leiden 999:10 core stability" 5 01_leiden_core_99910.json
run_launch_repeat 02_dfleiden_core "DF-Leiden 999:10 core stability" 5 02_dfleiden_core_99910.json
run_launch_repeat 03_s2cag_dataset_core "S2CAG dataset-feature 999:10 core stability" 5 03_s2cag_dataset_core_99910.json

# Long-horizon real-data topology endpoints from the article. They are expensive
# and mostly deterministic in quality, so one clean sequential pass is enough.
run_launch_repeat 04_leiden_long "Leiden 9:500 long horizon" 1 04_leiden_long_9_500.json
run_launch_repeat 05_dfleiden_long "DF-Leiden 9:500 long horizon" 1 05_dfleiden_long_9_500.json

# Auxiliary article evidence. These are single-pass because they are used for
# ablations/frontiers rather than repeated-run statistics.
run_launch_repeat 06_s2cag_feature_ablation "S2CAG feature-mode ablation 999:100" 1 06_s2cag_feature_ablation_999100.json
run_launch_repeat 07_leiden_topology_ablation_main "Leiden L x r topology ablation 999:10, excluding core L=3:r=1" 1 07_leiden_topology_ablation_main_99910.json
run_launch_repeat 07_leiden_topology_ablation_l3 "Leiden L=3 topology ablation 999:10, excluding core r=1" 1 07_leiden_topology_ablation_l3_99910.json

run_profile \
  08_profile_leiden_closure \
  "Leiden workload and closure/contraction profile" \
  --datasets dyn_cora dyn_pubmed arxivmath \
  --batches 999:50 \
  --methods leidenalg \
  --variants full no_closure no_contraction \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 50 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile \
  09_profile_dfleiden_workload \
  "DF-Leiden workload profile" \
  --datasets dyn_cora dyn_pubmed arxivmath \
  --batches 999:50 \
  --methods dfleiden \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode sum \
  --max-updates 10 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

run_profile \
  10_profile_s2cag_workload \
  "S2CAG workload profile" \
  --datasets dyn_cora dyn_pubmed arxivmath \
  --batches 999:50 \
  --methods s2cag \
  --feature-modes random \
  --variants full \
  --smart-depth 3 \
  --smart-radius 1 \
  --aggregation-mode norm \
  --baseline-iter 10 \
  --random-feature-dim 64 \
  --max-updates 10 \
  --use-gpu \
  --force-undirected \
  --ground-truth-metrics \
  --catch-errors

# DSBM seed robustness was originally distributed across eight containers. It is
# placed last because it is the longest block and checkpoints after every
# dataset/method/mode result.
run_dsbm \
  11_dsbm_seed_core_100b \
  "DSBM five-seed 100-batch core regimes" \
  --root datasets-sbm \
  --batch-suffix 100_batches \
  --regimes random hubs community \
  --max-changes 290 1450 \
  --methods leidenalg \
  --modes naive smart \
  --smart-depth 3 \
  --smart-radius 1 \
  --use-gpu \
  --catch-errors

log "Sequential rerun completed"

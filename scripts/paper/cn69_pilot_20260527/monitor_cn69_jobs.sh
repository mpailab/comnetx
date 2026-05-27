#!/usr/bin/env bash
set -uo pipefail

REMOTE_WORKDIR="${REMOTE_WORKDIR:-/home/dev/users/bokov/comnetx}"
WATCH_INTERVAL=0
TAIL_LINES=0
PROCESS_PATTERN='[s]cripts/launch.py|[p]rofile_smart_workload.py|[r]un_dsbm_stress.py'

CONTAINERS=(
  dev_bokov
  dev_uporova
  dev_konovalov
  dev_egorov
  dev_egorov2
  dev_drobyshev
  dev_drobyshev2
  dev_drobyshev3
)

GPUS=(0 1 2 3 4 5 6 7)

SCRIPTS=(
  gpu0_closure_contraction_pilot.sh
  gpu1_workload_memory_pilot.sh
  gpu2_dsbm_update_size_pilot.sh
  gpu3_topology_variance_final.sh
  gpu4_topology_long_horizon_final.sh
  gpu5_s2cag_focused_final.sh
  gpu6_closure_contraction_final.sh
  gpu7_dsbm_update_size_final.sh
)

usage() {
  cat <<'EOF'
Usage: scripts/paper/cn69_pilot_20260527/monitor_cn69_jobs.sh [options]

Options:
  --watch SECONDS   Repeat the status check every SECONDS.
  --tail LINES      Show the last LINES lines of each latest GPU log.
  -h, --help        Show this help.

Environment:
  REMOTE_WORKDIR    Project directory inside each container.
                   Default: /home/dev/users/bokov/comnetx
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH_INTERVAL="${2:-}"
      shift 2
      ;;
    --tail)
      TAIL_LINES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$WATCH_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "--watch must be a non-negative integer" >&2
  exit 2
fi

if ! [[ "$TAIL_LINES" =~ ^[0-9]+$ ]]; then
  echo "--tail must be a non-negative integer" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command is not available" >&2
  exit 127
fi

container_state() {
  local container="$1"
  docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true
}

running_processes() {
  local container="$1"
  docker exec "$container" bash -lc "pgrep -af '$PROCESS_PATTERN' || true" 2>/dev/null || true
}

remote_log_info() {
  local container="$1"
  local gpu="$2"
  docker exec "$container" bash -lc "
    cd '$REMOTE_WORKDIR' 2>/dev/null || { echo 'NO_WORKDIR'; exit 0; }
    latest_log=\$(ls -t output/gpu${gpu}_*.log 2>/dev/null | head -1 || true)
    if [[ -z \"\$latest_log\" ]]; then
      echo 'LOG: none'
      echo 'LOG_MTIME: n/a'
      echo 'LAST: n/a'
      echo 'ERROR_HINT: no'
    else
      echo \"LOG: \$latest_log\"
      echo \"LOG_MTIME: \$(stat -c '%y' \"\$latest_log\" 2>/dev/null | cut -d'.' -f1)\"
      echo \"LAST: \$(tail -n 1 \"\$latest_log\" 2>/dev/null | tr '\t' ' ')\"
      if tail -n 100 \"\$latest_log\" 2>/dev/null | grep -Eiq 'exited with status|Traceback|ModuleNotFoundError|RuntimeError|CUDA.*error|Killed|timed out|No such file|not found|Error on '; then
        echo 'ERROR_HINT: yes'
      else
        echo 'ERROR_HINT: no'
      fi
      if [[ '$TAIL_LINES' -gt 0 ]]; then
        echo 'TAIL_BEGIN'
        tail -n '$TAIL_LINES' \"\$latest_log\" 2>/dev/null
        echo 'TAIL_END'
      fi
    fi
    latest_json=\$(find results results/paper_icdm -maxdepth 2 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -3 | cut -d' ' -f2- || true)
    if [[ -n \"\$latest_json\" ]]; then
      echo 'RECENT_JSON:'
      echo \"\$latest_json\" | sed 's/^/  /'
    else
      echo 'RECENT_JSON: none'
    fi
  " 2>&1 || true
}

field_value() {
  local name="$1"
  awk -F': ' -v key="$name" '$1 == key {print substr($0, length(key) + 3); exit}'
}

run_once() {
  echo "cn69 paper job status at $(date -Is)"
  printf '%-16s %-5s %-36s %-14s %s\n' "container" "gpu" "script" "status" "latest_log"
  printf '%-16s %-5s %-36s %-14s %s\n' "---------" "---" "------" "------" "----------"

  local idx
  for idx in "${!CONTAINERS[@]}"; do
    local container="${CONTAINERS[$idx]}"
    local gpu="${GPUS[$idx]}"
    local script_name="${SCRIPTS[$idx]}"
    local state
    state="$(container_state "$container")"

    if [[ "$state" != "true" ]]; then
      printf '%-16s %-5s %-36s %-14s %s\n' "$container" "gpu$gpu" "$script_name" "CONTAINER_DOWN" "n/a"
      continue
    fi

    local processes
    processes="$(running_processes "$container" "$script_name")"

    local info
    info="$(remote_log_info "$container" "$gpu")"
    local latest_log
    latest_log="$(printf '%s\n' "$info" | field_value LOG)"
    local log_mtime
    log_mtime="$(printf '%s\n' "$info" | field_value LOG_MTIME)"
    local last_line
    last_line="$(printf '%s\n' "$info" | field_value LAST)"
    local error_hint
    error_hint="$(printf '%s\n' "$info" | field_value ERROR_HINT)"

    local status
    if [[ -n "$processes" ]]; then
      status="RUNNING"
    elif [[ "$latest_log" == "none" || -z "$latest_log" ]]; then
      status="NOT_STARTED"
    elif [[ "$error_hint" == "yes" ]]; then
      status="CHECK_LOG"
    else
      status="DONE"
    fi

    printf '%-16s %-5s %-36s %-14s %s\n' "$container" "gpu$gpu" "$script_name" "$status" "$latest_log"
    echo "  log_mtime: ${log_mtime:-n/a}"
    echo "  last_line: ${last_line:-n/a}"
    if [[ "$status" == "RUNNING" ]]; then
      echo "$processes" | sed 's/^/  process: /'
    fi
    printf '%s\n' "$info" | awk '
      /^RECENT_JSON:/ {show=1; print "  recent_json:"; next}
      /^TAIL_BEGIN$/ {tail=1; print "  tail:"; next}
      /^TAIL_END$/ {tail=0; next}
      show && /^  / {print "  " $0; next}
      show && !/^  / {show=0}
      tail {print "    " $0}
    '
  done
}

while true; do
  run_once
  if [[ "$WATCH_INTERVAL" -eq 0 ]]; then
    break
  fi
  echo
  echo "Sleeping ${WATCH_INTERVAL}s; press Ctrl-C to stop."
  sleep "$WATCH_INTERVAL"
  echo
done

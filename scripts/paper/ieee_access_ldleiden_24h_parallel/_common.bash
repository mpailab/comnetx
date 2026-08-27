set -Eeuo pipefail

readonly LAUNCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$LAUNCH_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
REPAIRED_CAMPAIGN_ID="${REPAIRED_CAMPAIGN_ID:-}"
LD_PARALLEL_RUN_ID="${LD_PARALLEL_RUN_ID:-ldleiden-cn69-20260827}"
LD_PARALLEL_BUDGET_HOURS="${LD_PARALLEL_BUDGET_HOURS:-24}"
EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA:-}"
PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"

readonly PYTHON_BIN PATHS_CONFIG REPAIRED_CAMPAIGN_ID LD_PARALLEL_RUN_ID
readonly LD_PARALLEL_BUDGET_HOURS EXPECTED_GIT_SHA PARENT_HOSTNAME
readonly RUNNER="scripts/paper/ieee_access_ldleiden_24h_parallel/run_shard.py"
readonly VALIDATOR="scripts/paper/ieee_access_ldleiden_24h_parallel/validate_parallel.py"
readonly RESULTS_ROOT="results/ieee-access-2026-1/raw/ldleiden/$LD_PARALLEL_RUN_ID"
readonly REPAIRED_ROOT="results/ieee-access-2026-1/raw/repaired-comnetx"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_environment() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "Python executable not found: $PYTHON_BIN"
  command -v taskset >/dev/null 2>&1 \
    || die "taskset is required for isolated timing runs"
  command -v flock >/dev/null 2>&1 \
    || die "flock is required for duplicate-launch protection"
  command -v git >/dev/null 2>&1 || die "git is required"
  [[ -f "$PATHS_CONFIG" ]] || die "Dataset path map not found: $PATHS_CONFIG"
  [[ -n "$REPAIRED_CAMPAIGN_ID" ]] \
    || die "Set REPAIRED_CAMPAIGN_ID to the exact paired ComNetX campaign"
  [[ "$REPAIRED_CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "Invalid repaired-ComNetX campaign id: $REPAIRED_CAMPAIGN_ID"
  [[ -f "$REPAIRED_ROOT/$REPAIRED_CAMPAIGN_ID/manifest.json" ]] \
    || die "Paired repaired-ComNetX campaign not found: $REPAIRED_CAMPAIGN_ID"
  [[ -f "$RUNNER" ]] || die "Shard runner not found: $RUNNER"
  [[ -f "$VALIDATOR" ]] || die "Parallel validator not found: $VALIDATOR"
  [[ -n "$EXPECTED_GIT_SHA" ]] \
    || die "Set EXPECTED_GIT_SHA to the reviewed full commit SHA"
  "$PYTHON_BIN" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' \
    || die "Python 3.10 is required"
  local actual_sha status
  actual_sha="$(git rev-parse HEAD)"
  [[ "$actual_sha" == "$EXPECTED_GIT_SHA" ]] \
    || die "Expected git SHA $EXPECTED_GIT_SHA, found $actual_sha"
  status="$(git status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$status" ]]; then
    printf '%s\n' "$status" >&2
    die "Measurement launch requires a clean checkout"
  fi
}

assignment() {
  local phase="$1" repeat="$2"
  case "$phase:$repeat" in
    short:1) printf '01 0\n' ;;
    short:2) printf '02 1\n' ;;
    short:3) printf '03 2\n' ;;
    short:4) printf '04 3\n' ;;
    short:5) printf '05 4\n' ;;
    long:1) printf '06 5\n' ;;
    long:2) printf '07 6\n' ;;
    long:3) printf '08 7\n' ;;
    *) die "Unknown LD-Leiden shard assignment: $phase repeat $repeat" ;;
  esac
}

cpu_for_shard_slot() {
  local shard_slot="$1"
  "$PYTHON_BIN" - "$shard_slot" <<'PY'
import os
from pathlib import Path
import sys

shard_slot = int(sys.argv[1])
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((root / "physical_package_id").read_text().strip())
        core = int((root / "core_id").read_text().strip())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot resolve physical topology for CPU {cpu}: {exc}")
    cores.setdefault((package, core), cpu)
choices = sorted(cores.values())
if not 0 <= shard_slot < 8:
    raise SystemExit(f"invalid shard CPU slot: {shard_slot}")
if len(choices) < 8:
    raise SystemExit(
        f"eight physical cores are required; only {len(choices)} cores allowed"
    )
position = shard_slot * len(choices) // 8
print(choices[position])
PY
}

run_python_on_shard_cpu() {
  run_supervised taskset -c "$LD_CPU_SET" "$PYTHON_BIN" "$RUNNER" "$@"
}

run_python() {
  run_supervised "$PYTHON_BIN" "$@"
}

ACTIVE_CHILD_PID=""

run_supervised() {
  local return_code
  "$@" &
  ACTIVE_CHILD_PID=$!
  if wait "$ACTIVE_CHILD_PID"; then
    return_code=0
  else
    return_code=$?
  fi
  ACTIVE_CHILD_PID=""
  return "$return_code"
}

forward_signal() {
  local signal_name="$1" exit_code="$2"
  trap - TERM HUP
  if [[ -n "${ACTIVE_CHILD_PID:-}" ]]; then
    kill -s "$signal_name" "$ACTIVE_CHILD_PID" 2>/dev/null || true
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    ACTIVE_CHILD_PID=""
  fi
  exit "$exit_code"
}

run_shard() {
  local phase="$1" repeat="$2" shard_id shard_slot
  read -r shard_id shard_slot < <(assignment "$phase" "$repeat")

  [[ "$LD_PARALLEL_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "Invalid LD_PARALLEL_RUN_ID: $LD_PARALLEL_RUN_ID"
  mkdir -p "$RESULTS_ROOT/driver-logs" "$RESULTS_ROOT/markers" \
    "$RESULTS_ROOT/.locks"
  exec > >(tee -a "$RESULTS_ROOT/driver-logs/$shard_id.log") 2>&1

  record_shell_failure() {
    local return_code=$?
    if (( return_code != 0 )); then
      local temporary="$RESULTS_ROOT/markers/failure-$shard_id.shell.tmp.$$"
      printf 'run_id=%s\nshard=%s\nexit_code=%d\nrecorded_at_utc=%s\n' \
        "$LD_PARALLEL_RUN_ID" "$shard_id" "$return_code" \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >"$temporary" || true
      mv -f "$temporary" \
        "$RESULTS_ROOT/markers/failure-$shard_id.shell" 2>/dev/null || true
    fi
  }
  trap record_shell_failure EXIT
  trap 'forward_signal TERM 143' TERM
  trap 'forward_signal HUP 129' HUP

  command -v flock >/dev/null 2>&1 \
    || die "flock is required for duplicate-launch protection"
  exec {SHARD_LOCK_FD}>"$RESULTS_ROOT/.locks/shard-$shard_id.lock"
  if ! flock -n "$SHARD_LOCK_FD"; then
    trap - EXIT
    die "Shard $shard_id is already running for $LD_PARALLEL_RUN_ID"
  fi
  rm -f "$RESULTS_ROOT/markers/failure-$shard_id.json" \
    "$RESULTS_ROOT/markers/failure-$shard_id.json.tmp" \
    "$RESULTS_ROOT/markers/failure-$shard_id.shell" \
    "$RESULTS_ROOT/markers/failure-$shard_id.shell.tmp."*

  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "Python executable not found: $PYTHON_BIN"
  LD_CPU_SET="${LD_CPU_SET:-$(cpu_for_shard_slot "$shard_slot")}"
  readonly LD_CPU_SET
  export PARENT_HOSTNAME

  require_environment

  note "Preparing shard $shard_id: $phase repeat $repeat on CPU $LD_CPU_SET"
  run_python_on_shard_cpu prepare \
    --run-id "$LD_PARALLEL_RUN_ID" \
    --shard "$shard_id" \
    --paths-config "$PATHS_CONFIG" \
    --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
    --expected-git-sha "$EXPECTED_GIT_SHA" \
    --budget-hours "$LD_PARALLEL_BUDGET_HOURS" \
    --cpu-set "$LD_CPU_SET"

  if [[ "$shard_id" == "01" ]]; then
    note "Waiting for all eight sealed preflights"
    run_python "$RUNNER" wait \
      --run-id "$LD_PARALLEL_RUN_ID" --stage preflight
    run_python "$VALIDATOR" "$RESULTS_ROOT" --stage preflight
    note "Running the single all-six diagnostic smoke gate"
    run_python_on_shard_cpu smoke \
      --run-id "$LD_PARALLEL_RUN_ID" \
      --shard "$shard_id" \
      --paths-config "$PATHS_CONFIG" \
      --cpu-set "$LD_CPU_SET"
    run_python "$VALIDATOR" "$RESULTS_ROOT" --stage smoke
  else
    note "Waiting for the validated all-six smoke gate"
    run_python "$RUNNER" wait \
      --run-id "$LD_PARALLEL_RUN_ID" --stage smoke
  fi

  note "Running measured $phase repeat $repeat"
  run_python_on_shard_cpu measure \
    --run-id "$LD_PARALLEL_RUN_ID" \
    --shard "$shard_id" \
    --paths-config "$PATHS_CONFIG" \
    --cpu-set "$LD_CPU_SET"

  if [[ "$shard_id" == "01" ]]; then
    note "Waiting for the complete five-short/three-long result set"
    run_python "$RUNNER" wait \
      --run-id "$LD_PARALLEL_RUN_ID" --stage done
    run_python "$VALIDATOR" "$RESULTS_ROOT" --stage final
    note "All eight LD-Leiden shards passed final validation"
  else
    note "Shard $shard_id is complete"
  fi
  trap - EXIT TERM HUP
}

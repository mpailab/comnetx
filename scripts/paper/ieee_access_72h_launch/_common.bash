set -Eeuo pipefail

readonly LAUNCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$LAUNCH_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
REPAIRED_CAMPAIGN_ID="${REPAIRED_CAMPAIGN_ID:-repaired-comnetx-cn69-20260825}"
LD_CAMPAIGN_ID="${LD_CAMPAIGN_ID:-ldleiden-cn69-20260825}"
DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
CAMPAIGN_BUDGET_HOURS="${CAMPAIGN_BUDGET_HOURS:-72}"
EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA:-}"

readonly PYTHON_BIN PATHS_CONFIG REPAIRED_CAMPAIGN_ID LD_CAMPAIGN_ID
readonly DSBM_ROOT CAMPAIGN_BUDGET_HOURS EXPECTED_GIT_SHA

readonly REPAIRED_RUNNER="scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py"
readonly REPAIRED_VALIDATOR="scripts/paper/ieee_access_repaired_comnetx_72h/validate_campaign.py"
readonly LD_RUNNER="scripts/paper/ieee_access_ldleiden_72h/run_protocol.py"
readonly LD_VALIDATOR="scripts/paper/ieee_access_ldleiden_72h/validate_results.py"
readonly BOOTSTRAP_SYNC="scripts/paper/ieee_access_72h_launch/sync_bootstrap.py"
readonly PAIR_VALIDATOR="scripts/paper/ieee_access_72h_launch/validate_campaign_pair.py"
readonly REPAIRED_RESULTS_ROOT="results/ieee-access-2026-1/raw/repaired-comnetx"
readonly LD_RESULTS_ROOT="results/ieee-access-2026-1/raw/ldleiden"
readonly REPAIRED_CAMPAIGN_DIR="$REPAIRED_RESULTS_ROOT/$REPAIRED_CAMPAIGN_ID"
readonly LD_CAMPAIGN_DIR="$LD_RESULTS_ROOT/$LD_CAMPAIGN_ID"
readonly BUDGET_STATE_FILE="$REPAIRED_CAMPAIGN_DIR/launch_budget.json"
readonly LOCK_FILE="results/ieee-access-2026-1/raw/.ieee-access-launch.lock"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_common_environment() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "Python executable not found: $PYTHON_BIN"
  command -v flock >/dev/null 2>&1 \
    || die "flock is required; run this launcher inside the project container"
  command -v timeout >/dev/null 2>&1 \
    || die "GNU timeout is required; run this launcher inside the project container"
  command -v git >/dev/null 2>&1 || die "git is required"

  "$PYTHON_BIN" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)' \
    || die "Python 3.10 is required"

  [[ -f "$PATHS_CONFIG" ]] || die "Dataset path map not found: $PATHS_CONFIG"
  [[ -f "$REPAIRED_RUNNER" ]] || die "Runner not found: $REPAIRED_RUNNER"
  [[ -f "$LD_RUNNER" ]] || die "Runner not found: $LD_RUNNER"
  [[ -f "$REPAIRED_VALIDATOR" ]] || die "Validator not found: $REPAIRED_VALIDATOR"
  [[ -f "$LD_VALIDATOR" ]] || die "Validator not found: $LD_VALIDATOR"
  [[ -f "$BOOTSTRAP_SYNC" ]] || die "Bootstrap helper not found: $BOOTSTRAP_SYNC"
  [[ -f "$PAIR_VALIDATOR" ]] || die "Pair validator not found: $PAIR_VALIDATOR"
  [[ "$REPAIRED_CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "Invalid repaired campaign id: $REPAIRED_CAMPAIGN_ID"
  [[ "$LD_CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "Invalid LD-Leiden campaign id: $LD_CAMPAIGN_ID"

  local actual_sha status
  actual_sha="$(git rev-parse HEAD)"
  if [[ -n "$EXPECTED_GIT_SHA" && "$actual_sha" != "$EXPECTED_GIT_SHA" ]]; then
    die "Expected git SHA $EXPECTED_GIT_SHA, found $actual_sha"
  fi
  status="$(git status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$status" ]]; then
    printf '%s\n' "$status" >&2
    die "Measurement launch requires a clean checkout"
  fi
  note "Source SHA: $actual_sha"
}

acquire_campaign_lock() {
  mkdir -p "$(dirname -- "$LOCK_FILE")"
  exec {CAMPAIGN_LOCK_FD}>"$LOCK_FILE"
  flock -n "$CAMPAIGN_LOCK_FD" \
    || die "Another IEEE Access launcher is already running in this checkout"
}

require_preflight_campaigns() {
  [[ -f "$REPAIRED_CAMPAIGN_DIR/preflight/validation.json" ]] \
    || die "Run 01_preflight.sh first (repaired preflight is missing)"
  [[ -f "$LD_CAMPAIGN_DIR/manifest.json" ]] \
    || die "Run 01_preflight.sh first (LD-Leiden preflight is missing)"
  [[ -f "$BUDGET_STATE_FILE" ]] \
    || die "Run 01_preflight.sh first (the shared 72-hour clock is missing)"
}

initialize_budget_clock() {
  local actual_sha
  actual_sha="$(git rev-parse HEAD)"
  "$PYTHON_BIN" - "$BUDGET_STATE_FILE" "$CAMPAIGN_BUDGET_HOURS" "$actual_sha" \
    "$REPAIRED_CAMPAIGN_ID" "$LD_CAMPAIGN_ID" "$PATHS_CONFIG" <<'PY'
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
hours = float(sys.argv[2])
git_sha = sys.argv[3]
repaired_campaign_id = sys.argv[4]
ld_campaign_id = sys.argv[5]
paths_config = Path(sys.argv[6]).resolve()
if not math.isfinite(hours) or hours <= 0:
    raise SystemExit("CAMPAIGN_BUDGET_HOURS must be a positive finite number")
if hours > 72:
    raise SystemExit("CAMPAIGN_BUDGET_HOURS cannot exceed the registered 72 hours")
paths_sha256 = hashlib.sha256(paths_config.read_bytes()).hexdigest()
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "comnetx-ieee-access-launch-budget-v1",
        "git_sha": git_sha,
        "budget_hours": hours,
        "repaired_campaign_id": repaired_campaign_id,
        "ld_campaign_id": ld_campaign_id,
        "paths_config_sha256": paths_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(f"saved campaign clock mismatch for {key}")
    started = payload.get("started_at_epoch")
    deadline = payload.get("deadline_epoch")
    if not isinstance(started, int) or not isinstance(deadline, int):
        raise SystemExit("saved campaign clock has malformed timestamps")
    if deadline != started + int(round(hours * 3600)):
        raise SystemExit("saved campaign deadline is inconsistent with its start")
    print(f"Reusing deadline epoch {payload['deadline_epoch']}")
    raise SystemExit(0)

started = int(time.time())
payload = {
    "schema": "comnetx-ieee-access-launch-budget-v1",
    "git_sha": git_sha,
    "budget_hours": hours,
    "repaired_campaign_id": repaired_campaign_id,
    "ld_campaign_id": ld_campaign_id,
    "paths_config_sha256": paths_sha256,
    "started_at_epoch": started,
    "deadline_epoch": started + int(round(hours * 3600)),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
print(f"Started {hours:g}-hour clock; deadline epoch {payload['deadline_epoch']}")
PY
}

hours_left() {
  "$PYTHON_BIN" - "$BUDGET_STATE_FILE" "$(git rev-parse HEAD)" \
    "$REPAIRED_CAMPAIGN_ID" "$LD_CAMPAIGN_ID" "$CAMPAIGN_BUDGET_HOURS" \
    "$PATHS_CONFIG" <<'PY'
import hashlib
import json
import math
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "schema": "comnetx-ieee-access-launch-budget-v1",
    "git_sha": sys.argv[2],
    "repaired_campaign_id": sys.argv[3],
    "ld_campaign_id": sys.argv[4],
    "budget_hours": float(sys.argv[5]),
    "paths_config_sha256": hashlib.sha256(Path(sys.argv[6]).read_bytes()).hexdigest(),
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"saved campaign clock mismatch for {key}")
started = payload.get("started_at_epoch")
deadline = payload.get("deadline_epoch")
if not isinstance(started, int) or not isinstance(deadline, int):
    raise SystemExit("saved campaign clock has malformed timestamps")
if deadline != started + int(round(float(sys.argv[5]) * 3600)):
    raise SystemExit("saved campaign deadline is inconsistent with its start")
remaining = max(0.0, (float(deadline) - time.time()) / 3600.0)
if not math.isfinite(remaining):
    raise SystemExit("invalid remaining-time calculation")
print(f"{remaining:.6f}")
PY
}

budget_deadline_epoch() {
  "$PYTHON_BIN" - "$BUDGET_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema") != "comnetx-ieee-access-launch-budget-v1":
    raise SystemExit("saved campaign clock has an unexpected schema")
deadline = payload.get("deadline_epoch")
if not isinstance(deadline, int):
    raise SystemExit("saved campaign clock has a malformed deadline")
print(deadline)
PY
}

seconds_left() {
  local remaining
  remaining="$(hours_left)"
  "$PYTHON_BIN" - "$remaining" <<'PY'
import math
import sys

seconds = int(math.floor(float(sys.argv[1]) * 3600.0))
print(max(0, seconds))
PY
}

run_ld_phase() {
  local phase="$1" seconds deadline_epoch watchdog_seconds return_code
  shift
  seconds="$(seconds_left)"
  deadline_epoch="$(budget_deadline_epoch)"
  watchdog_seconds=$((seconds + 65))
  (( seconds > 0 )) || die "The registered 72-hour measurement budget is exhausted"
  note "Starting LD-Leiden $phase with $(hours_left) hours left"
  if timeout --signal=TERM --kill-after=10s "${watchdog_seconds}s" \
    "$PYTHON_BIN" "$LD_RUNNER" \
      --paths-config "$PATHS_CONFIG" \
      --campaign-id "$LD_CAMPAIGN_ID" \
      --phase "$phase" --deadline-epoch "$deadline_epoch" --resume "$@"; then
    return 0
  else
    return_code=$?
  fi
  if [[ "$return_code" -eq 124 ]]; then
    die "LD-Leiden $phase reached the registered 72-hour deadline"
  fi
  return "$return_code"
}

hours_at_least() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import math
import sys

value = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if math.isfinite(value) and value >= threshold else 1)
PY
}

stage_status() {
  "$PYTHON_BIN" - "$REPAIRED_CAMPAIGN_DIR/manifest.json" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("stage_status", {}).get(sys.argv[2])
if not isinstance(status, str):
    raise SystemExit(f"missing stage status: {sys.argv[2]}")
print(status)
PY
}

stage_is_resolved() {
  local status
  status="$(stage_status "$1")"
  [[ "$status" == "validated" || "$status" == "skipped_for_budget" \
    || "$status" == "skipped_by_stage2_no_go" ]]
}

require_repaired_stage_validated() {
  local status
  status="$(stage_status "$1")"
  [[ "$status" == "validated" ]] \
    || die "Required predecessor $1 is $status, not validated"
}

ld_phase_status() {
  "$PYTHON_BIN" - "$LD_CAMPAIGN_DIR/manifest.json" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("phase_status", {}).get(sys.argv[2])
if not isinstance(status, str):
    raise SystemExit(f"missing LD-Leiden phase status: {sys.argv[2]}")
print(status)
PY
}

require_ld_phase_completed() {
  local status
  status="$(ld_phase_status "$1")"
  [[ "$status" == "completed" ]] \
    || die "Required LD-Leiden predecessor $1 is $status, not completed"
}

stage_failed_at_budget_boundary() {
  "$PYTHON_BIN" - "$REPAIRED_CAMPAIGN_DIR/manifest.json" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
registration = payload.get("time_boundary_stops", {}).get(sys.argv[2])
raise SystemExit(0 if isinstance(registration, dict) else 1)
PY
}

stage_failed_at_interruption() {
  "$PYTHON_BIN" - "$REPAIRED_CAMPAIGN_DIR/manifest.json" "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
registration = payload.get("interrupted_stages", {}).get(sys.argv[2])
raise SystemExit(0 if isinstance(registration, dict) else 1)
PY
}

require_optional_stage_can_be_deferred() {
  local stage="$1" status
  status="$(stage_status "$stage")"
  if [[ "$status" == "running" ]]; then
    note "$stage has a stale running marker; auditing process-group absence"
    "$PYTHON_BIN" "$REPAIRED_RUNNER" \
      --paths-config "$PATHS_CONFIG" \
      --campaign-id "$REPAIRED_CAMPAIGN_ID" \
      --recover-interrupted-stage "$stage" --resume
    status="$(stage_status "$stage")"
  fi
  if [[ "$status" == "failed" ]] \
    && ! stage_failed_at_budget_boundary "$stage" \
    && ! stage_failed_at_interruption "$stage"; then
    die "$stage failed for a non-budget reason; inspect it instead of recording a skip"
  fi
}

stage2_breadth_decision() {
  "$PYTHON_BIN" - \
    "$REPAIRED_CAMPAIGN_DIR/stages/stage2_core_short/validation.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("optional_breadth_go")
if value is True:
    print("go")
elif value is False:
    print("no-go")
else:
    raise SystemExit("Stage 2 lacks a Boolean optional_breadth_go decision")
PY
}

run_repaired_stage() {
  local stage="$1"
  shift
  local remaining deadline_epoch
  remaining="$(hours_left)"
  deadline_epoch="$(budget_deadline_epoch)"
  note "Starting $stage with $remaining hours left"
  "$PYTHON_BIN" "$REPAIRED_RUNNER" \
    --paths-config "$PATHS_CONFIG" \
    --campaign-id "$REPAIRED_CAMPAIGN_ID" \
    --stage "$stage" --hours-left "$remaining" \
    --deadline-epoch "$deadline_epoch" \
    --stop-with-hours-left 0 --resume "$@"
  "$PYTHON_BIN" "$REPAIRED_VALIDATOR" "$REPAIRED_CAMPAIGN_ID" --stage "$stage"
}

repaired_stage_can_finalize() {
  local stage="$1" remaining deadline_epoch
  shift
  remaining="$(hours_left)"
  deadline_epoch="$(budget_deadline_epoch)"
  "$PYTHON_BIN" "$REPAIRED_RUNNER" \
    --paths-config "$PATHS_CONFIG" \
    --campaign-id "$REPAIRED_CAMPAIGN_ID" \
    --stage "$stage" --hours-left "$remaining" \
    --deadline-epoch "$deadline_epoch" --resume --dry-run "$@" \
    >/dev/null 2>&1
}

record_budget_skip() {
  local stage="$1" remaining deadline_epoch
  require_optional_stage_can_be_deferred "$stage"
  remaining="$(hours_left)"
  deadline_epoch="$(budget_deadline_epoch)"
  note "Recording budget skip for $stage with $remaining hours left"
  "$PYTHON_BIN" "$REPAIRED_RUNNER" \
    --paths-config "$PATHS_CONFIG" \
    --campaign-id "$REPAIRED_CAMPAIGN_ID" \
    --skip-for-budget "$stage" --hours-left "$remaining" \
    --deadline-epoch "$deadline_epoch" --resume
}

record_stage2_no_go() {
  local stage="$1" status
  status="$(stage_status "$stage")"
  [[ "$status" == "pending" ]] \
    || die "Stage-2 no-go can resolve only a pending stage, not $stage=$status"
  note "Recording the pre-registered Stage-2 no-go for $stage"
  "$PYTHON_BIN" "$REPAIRED_RUNNER" \
    --paths-config "$PATHS_CONFIG" \
    --campaign-id "$REPAIRED_CAMPAIGN_ID" \
    --skip-for-stage2-no-go "$stage" --resume
}

require_all_optional_stages_resolved() {
  local stage
  for stage in \
    stage4_long_repeatability \
    stage5_topology_controls \
    stage6_dsbm \
    stage7_dfleiden_interface \
    stage7_s2cag_interface; do
    stage_is_resolved "$stage" \
      || die "Optional stage remains unresolved: $stage=$(stage_status "$stage")"
  done
}

repeatability_completed_count() {
  "$PYTHON_BIN" - \
    "$REPAIRED_CAMPAIGN_DIR/stages/stage4_long_repeatability" <<'PY'
import json
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
count = 0
for command_dir in sorted(stage_dir.glob("smart_long_repeat_*")):
    if not command_dir.is_dir():
        continue
    completed = 0
    for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        completed += payload.get("status") == "completed"
    if completed > 1:
        raise SystemExit(f"multiple completed attempts in {command_dir}")
    count += completed
print(count)
PY
}

stage_completed_command_count() {
  "$PYTHON_BIN" - "$REPAIRED_CAMPAIGN_DIR/stages/$1" <<'PY'
import json
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
count = 0
for command_dir in sorted(path for path in stage_dir.glob("*") if path.is_dir()):
    completed = 0
    for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        completed += payload.get("status") == "completed"
    if completed > 1:
        raise SystemExit(f"multiple completed attempts in {command_dir}")
    count += completed
print(count)
PY
}

prepare_launcher() {
  require_common_environment
  acquire_campaign_lock
}

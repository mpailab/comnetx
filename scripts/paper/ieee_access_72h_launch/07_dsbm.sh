#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage4_long_core
require_ld_phase_completed measured_9_500

next_dsbm_seed() {
  "$PYTHON_BIN" - \
    "$REPAIRED_CAMPAIGN_DIR/stages/stage6_dsbm" <<'PY'
import json
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
seed_order = (42, 43, 44, 45, 46)
conditions = tuple(
    (regime, max_changes)
    for max_changes in (290, 1450)
    for regime in ("random", "hubs", "community")
)
completed_seeds = []
saw_incomplete = False
for seed in seed_order:
    completed_conditions = 0
    for regime, max_changes in conditions:
        command_dir = (
            stage_dir / f"dsbm_seed_{seed}_{regime}_mc{max_changes}_paired"
        )
        completed = 0
        for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            completed += payload.get("status") == "completed"
        if completed > 1:
            raise SystemExit(f"multiple completed attempts in {command_dir}")
        completed_conditions += completed
    if completed_conditions == len(conditions):
        if saw_incomplete:
            raise SystemExit("completed DSBM seed appears after an incomplete seed")
        completed_seeds.append(seed)
        continue
    saw_incomplete = True
    print(seed)
    break
PY
}

unfinalized_dsbm_seed() {
  "$PYTHON_BIN" - \
    "$REPAIRED_CAMPAIGN_DIR/stages/stage6_dsbm" <<'PY'
import json
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
conditions = tuple(
    (regime, max_changes)
    for max_changes in (290, 1450)
    for regime in ("random", "hubs", "community")
)
completed = []
for seed in (42, 43, 44, 45, 46):
    count = 0
    for regime, max_changes in conditions:
        command_dir = stage_dir / f"dsbm_seed_{seed}_{regime}_mc{max_changes}_paired"
        for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            count += payload.get("status") == "completed"
    if count == len(conditions):
        completed.append(seed)
    else:
        break
validation_path = stage_dir / "validation.json"
validated = (
    json.loads(validation_path.read_text(encoding="utf-8")).get("completed_seeds", [])
    if validation_path.is_file()
    else []
)
if validated != completed[: len(validated)]:
    raise SystemExit("saved DSBM validation is not the completed fixed seed prefix")
if len(validated) < len(completed):
    print(completed[-1])
PY
}

stage="stage6_dsbm"
if stage_is_resolved "$stage"; then
  note "$stage is already resolved: $(stage_status "$stage")"
else
  require_optional_stage_can_be_deferred "$stage"
  seed="$(unfinalized_dsbm_seed)"
  if [[ -n "$seed" ]]; then
    [[ -d "$DSBM_ROOT" ]] || die "DSBM dataset root not found: $DSBM_ROOT"
    note "Finalizing already completed paired DSBM seed $seed before new work"
    run_repaired_stage "$stage" --dsbm-root "$DSBM_ROOT" --dsbm-seed "$seed"
  fi
  while (( $(dsbm_completed_seed_count) < 3 )); do
    seed="$(next_dsbm_seed)"
    completed_conditions="$(dsbm_current_seed_completed_condition_count)"
    seed_gate=17
    if (( completed_conditions > 0 )); then
      seed_gate=4
    fi
    remaining="$(hours_left)"
    if [[ -z "$seed" ]] || ! hours_at_least "$remaining" "$seed_gate"; then
      note "$stage has not reached its three-seed minimum: seed $seed needs the registered ${seed_gate}-hour start margin"
      break
    fi
    [[ -d "$DSBM_ROOT" ]] || die "DSBM dataset root not found: $DSBM_ROOT"
    note "Prioritizing fresh paired DSBM seed $seed before breadth controls"
    run_repaired_stage "$stage" --dsbm-root "$DSBM_ROOT" --dsbm-seed "$seed"
  done
fi

control_stage="stage5_topology_controls"
if (( $(dsbm_completed_seed_count) < 3 )); then
  note "$control_stage remains pending until three complete paired DSBM seeds validate"
elif stage_is_resolved "$control_stage"; then
  note "$control_stage is already resolved: $(stage_status "$control_stage")"
else
  require_optional_stage_can_be_deferred "$control_stage"
  if [[ "$(stage2_breadth_decision)" == "no-go" ]]; then
    record_stage2_no_go "$control_stage"
  else
    remaining="$(hours_left)"
    if hours_at_least "$remaining" 4; then
      run_repaired_stage "$control_stage"
    elif repaired_stage_can_finalize "$control_stage"; then
      note "Finalizing completed $control_stage commands below its start gate"
      run_repaired_stage "$control_stage"
    else
      note "$control_stage remains pending for a later registered window"
    fi
  fi
fi

precision_ready=true
for prerequisite in \
  stage7_dfleiden_interface \
  stage7_s2cag_interface \
  stage4_long_repeatability \
  stage5_topology_controls; do
  if ! stage_is_resolved "$prerequisite"; then
    precision_ready=false
  fi
done

if ! stage_is_resolved "$stage" \
  && (( $(dsbm_completed_seed_count) >= 3 )) \
  && [[ "$precision_ready" == true ]]; then
  require_optional_stage_can_be_deferred "$stage"
  while true; do
    if stage_is_resolved "$stage"; then
      break
    fi
    seed="$(next_dsbm_seed)"
    if [[ -z "$seed" ]]; then
      break
    fi
    completed_conditions="$(dsbm_current_seed_completed_condition_count)"
    seed_gate=17
    if (( completed_conditions > 0 )); then
      seed_gate=4
    fi
    remaining="$(hours_left)"
    if ! hours_at_least "$remaining" "$seed_gate"; then
      note "$stage remains pending: precision seed $seed needs the registered ${seed_gate}-hour start margin"
      break
    fi
    [[ -d "$DSBM_ROOT" ]] || die "DSBM dataset root not found: $DSBM_ROOT"
    note "Advancing the fixed paired DSBM order with precision seed $seed"
    run_repaired_stage "$stage" --dsbm-root "$DSBM_ROOT" --dsbm-seed "$seed"
  done
elif ! stage_is_resolved "$stage" \
  && (( $(dsbm_completed_seed_count) >= 3 )); then
  note "The publishable DSBM minimum is complete; precision seeds wait until interface, repeatability, and Stage 5 evidence are resolved"
fi

if stage_is_resolved "$stage"; then
  note "DSBM target resolved: all five fixed seeds are validated"
else
  note "DSBM remains $(stage_status "$stage"); completed seed blocks are preserved for the next registered window"
fi

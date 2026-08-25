#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage4_long_core
require_ld_phase_completed measured_9_500

repeat_stage="stage4_long_repeatability"
if stage_is_resolved "$repeat_stage"; then
  note "$repeat_stage is already resolved: $(stage_status "$repeat_stage")"
else
  require_optional_stage_can_be_deferred "$repeat_stage"
  completed_repeats="$(repeatability_completed_count)"
  if (( completed_repeats > 0 )); then
    note "Finalizing $completed_repeats completed repeatability run(s)"
    run_repaired_stage "$repeat_stage" --repetitions 1
  else
    remaining="$(hours_left)"
    if hours_at_least "$remaining" 40; then
      run_repaired_stage "$repeat_stage" \
        --repetitions 2 --stop-with-hours-left 32
    elif hours_at_least "$remaining" 33; then
      run_repaired_stage "$repeat_stage" \
        --repetitions 1 --stop-with-hours-left 32
    elif hours_at_least "$remaining" 12 && ! hours_at_least "$remaining" 30; then
      run_repaired_stage "$repeat_stage" --repetitions 1
    elif ! hours_at_least "$remaining" 12; then
      record_budget_skip "$repeat_stage"
    else
      note "Deferring $repeat_stage inside the 30-33h DSBM handoff window"
    fi
  fi
fi

control_stage="stage5_topology_controls"
if stage_is_resolved "$control_stage"; then
  note "$control_stage is already resolved: $(stage_status "$control_stage")"
else
  require_optional_stage_can_be_deferred "$control_stage"
  if [[ "$(stage2_breadth_decision)" == "no-go" ]]; then
    record_stage2_no_go "$control_stage"
  else
    remaining="$(hours_left)"
    if hours_at_least "$remaining" 38; then
      run_repaired_stage "$control_stage"
    elif repaired_stage_can_finalize "$control_stage"; then
      note "Finalizing completed $control_stage commands below its start gate"
      run_repaired_stage "$control_stage"
    else
      record_budget_skip "$control_stage"
    fi
  fi
fi

note "Repeatability/control phase is resolved as far as the registered budget allows"

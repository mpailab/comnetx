#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage4_long_core
require_ld_phase_completed measured_9_500

decision="$(stage2_breadth_decision)"
raw_dsbm_seeds="$(dsbm_completed_seed_count)"
validated_dsbm_seeds="$(dsbm_validated_seed_count)"
if (( raw_dsbm_seeds < validated_dsbm_seeds )); then
  die "Saved DSBM validation refers to missing completed seed commands"
fi
if (( raw_dsbm_seeds > validated_dsbm_seeds )); then
  note "A complete DSBM seed still needs validation; File 06 will not start optional work before File 07 finalizes it"
  exit 0
fi
dsbm_reserve=0
if ! stage_is_resolved stage6_dsbm \
  && (( validated_dsbm_seeds < 3 )); then
  dsbm_reserve="$(dsbm_protected_reserve_hours "$(hours_left)")"
fi

df_stage="stage7_dfleiden_interface"
if stage_is_resolved "$df_stage"; then
  note "$df_stage is already resolved: $(stage_status "$df_stage")"
else
  require_optional_stage_can_be_deferred "$df_stage"
  if [[ "$decision" == "no-go" ]]; then
    record_stage2_no_go "$df_stage"
  else
    remaining="$(hours_left)"
    if hours_at_least "$remaining" "$((dsbm_reserve + 1))"; then
      run_repaired_stage "$df_stage" --stop-with-hours-left "$dsbm_reserve"
    elif repaired_stage_can_finalize "$df_stage"; then
      note "Finalizing completed $df_stage commands below its start gate"
      run_repaired_stage "$df_stage"
    else
      note "$df_stage remains pending for a later registered window"
    fi
  fi
fi

s2cag_stage="stage7_s2cag_interface"
if stage_is_resolved "$s2cag_stage"; then
  note "$s2cag_stage is already resolved: $(stage_status "$s2cag_stage")"
elif [[ "$decision" == "no-go" ]]; then
  require_optional_stage_can_be_deferred "$s2cag_stage"
  record_stage2_no_go "$s2cag_stage"
elif [[ "$(stage_status "$df_stage")" != "validated" ]]; then
  note "$s2cag_stage remains pending because $df_stage is not yet validated"
else
  require_optional_stage_can_be_deferred "$s2cag_stage"
  remaining="$(hours_left)"
  if hours_at_least "$remaining" "$((dsbm_reserve + 2))"; then
    run_repaired_stage "$s2cag_stage" \
      --repetitions 1 --stop-with-hours-left "$dsbm_reserve"
  elif repaired_stage_can_finalize "$s2cag_stage" --repetitions 1; then
    note "Finalizing completed $s2cag_stage commands below its start gate"
    run_repaired_stage "$s2cag_stage" --repetitions 1
  else
    note "$s2cag_stage remains pending for a later registered window"
  fi
fi

repeat_stage="stage4_long_repeatability"
if stage_is_resolved "$repeat_stage"; then
  note "$repeat_stage is already resolved: $(stage_status "$repeat_stage")"
else
  require_optional_stage_can_be_deferred "$repeat_stage"
  completed_repeats="$(repeatability_completed_count)"
  remaining="$(hours_left)"
  if (( completed_repeats > 0 )); then
    note "Finalizing $completed_repeats completed smart-only long repeat(s)"
    run_repaired_stage "$repeat_stage" --repetitions "$completed_repeats"
  elif hours_at_least "$remaining" "$((dsbm_reserve + 2))"; then
    note "Launching two smart-only long repetitions while preserving the registered DSBM reserve"
    run_repaired_stage "$repeat_stage" \
      --repetitions 2 --stop-with-hours-left "$dsbm_reserve"
  else
    note "$repeat_stage remains pending: two repeats require 2 hours beyond the protected DSBM reserve"
  fi
fi

note "First-day interface and repeatability pass complete; unfinished optional evidence remains pending"

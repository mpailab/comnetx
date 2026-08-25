#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage4_long_core
require_ld_phase_completed measured_9_500
stage_is_resolved stage5_topology_controls \
  || die "Stage 5 is unresolved; re-run file 06"
stage_is_resolved stage6_dsbm || die "Stage 6 is unresolved; re-run file 07"

decision="$(stage2_breadth_decision)"

repeat_stage="stage4_long_repeatability"
if stage_is_resolved "$repeat_stage"; then
  note "$repeat_stage is already resolved: $(stage_status "$repeat_stage")"
else
  require_optional_stage_can_be_deferred "$repeat_stage"
  completed_repeats="$(repeatability_completed_count)"
  remaining="$(hours_left)"
  if (( completed_repeats > 0 )); then
    run_repaired_stage "$repeat_stage" --repetitions 1
  elif hours_at_least "$remaining" 12; then
    run_repaired_stage "$repeat_stage" --repetitions 1
  else
    record_budget_skip "$repeat_stage"
  fi
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
    if hours_at_least "$remaining" 4; then
      run_repaired_stage "$df_stage"
    elif repaired_stage_can_finalize "$df_stage"; then
      note "Finalizing completed $df_stage commands below its start gate"
      run_repaired_stage "$df_stage"
    else
      record_budget_skip "$df_stage"
    fi
  fi
fi

s2cag_stage="stage7_s2cag_interface"
if stage_is_resolved "$s2cag_stage"; then
  note "$s2cag_stage is already resolved: $(stage_status "$s2cag_stage")"
else
  require_optional_stage_can_be_deferred "$s2cag_stage"
  if [[ "$decision" == "no-go" ]]; then
    record_stage2_no_go "$s2cag_stage"
  else
    remaining="$(hours_left)"
    if hours_at_least "$remaining" 12; then
      run_repaired_stage "$s2cag_stage" --repetitions 1
    elif repaired_stage_can_finalize "$s2cag_stage" --repetitions 1; then
      note "Finalizing completed $s2cag_stage commands below its start gate"
      run_repaired_stage "$s2cag_stage" --repetitions 1
    else
      record_budget_skip "$s2cag_stage"
    fi
  fi
fi

note "Running complete campaign validation"
require_all_optional_stages_resolved
"$PYTHON_BIN" "$REPAIRED_VALIDATOR" "$REPAIRED_CAMPAIGN_ID"
"$PYTHON_BIN" "$LD_VALIDATOR" "$LD_CAMPAIGN_ID"
"$PYTHON_BIN" "$PAIR_VALIDATOR" \
  --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
  --ld-campaign "$LD_CAMPAIGN_ID"

note "All required measurements and every affordable optional phase are validated"

#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage4_long_core
require_ld_phase_completed measured_9_500
stage_is_resolved stage5_topology_controls \
  || die "Resolve Stage 5 by re-running 06_repeatability_and_controls.sh first"
if ! stage_is_resolved stage4_long_repeatability; then
  require_optional_stage_can_be_deferred stage4_long_repeatability
fi
completed_repeats="$(repeatability_completed_count)"
if ! stage_is_resolved stage4_long_repeatability \
  && (( completed_repeats > 0 )); then
  die "Completed repeatability evidence is not validated; re-run file 06 before DSBM"
fi

stage="stage6_dsbm"
if stage_is_resolved "$stage"; then
  note "$stage is already resolved: $(stage_status "$stage")"
else
  require_optional_stage_can_be_deferred "$stage"
  remaining="$(hours_left)"
  if hours_at_least "$remaining" 30; then
    [[ -d "$DSBM_ROOT" ]] || die "DSBM dataset root not found: $DSBM_ROOT"
    run_repaired_stage "$stage" --dsbm-root "$DSBM_ROOT"
  else
    completed_commands="$(stage_completed_command_count "$stage")"
    if (( completed_commands > 0 )); then
      [[ -d "$DSBM_ROOT" ]] \
        || die "Completed DSBM output cannot be validated without $DSBM_ROOT"
      repaired_stage_can_finalize "$stage" --dsbm-root "$DSBM_ROOT" \
        || die "Completed DSBM output failed its finalize-only audit"
      note "Finalizing completed DSBM output below its start gate"
      run_repaired_stage "$stage" --dsbm-root "$DSBM_ROOT"
    else
      record_budget_skip "$stage"
    fi
  fi
fi

note "DSBM phase resolved: $(stage_status "$stage")"

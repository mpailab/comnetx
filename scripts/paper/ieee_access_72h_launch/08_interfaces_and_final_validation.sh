#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage4_long_core
require_ld_phase_completed measured_9_500

note "Running the end-of-window reproducibility checkpoint"
"$PYTHON_BIN" "$REPAIRED_VALIDATOR" "$REPAIRED_CAMPAIGN_ID"
"$PYTHON_BIN" "$LD_VALIDATOR" "$LD_CAMPAIGN_ID"
"$PYTHON_BIN" "$PAIR_VALIDATOR" \
  --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
  --ld-campaign "$LD_CAMPAIGN_ID"

for stage in \
  stage7_dfleiden_interface \
  stage7_s2cag_interface \
  stage4_long_repeatability \
  stage5_topology_controls; do
  note "Checkpoint status: $stage=$(stage_status "$stage")"
done

dsbm_status="$(stage_status stage6_dsbm)"
dsbm_seed_count="$(dsbm_validated_seed_count)"
if [[ "$dsbm_status" == "validated" ]]; then
  note "Checkpoint status: stage6_dsbm=validated (all five fixed seeds)"
elif (( dsbm_seed_count >= 3 )); then
  note "Checkpoint status: stage6_dsbm=$dsbm_status; the paired three-seed publishable minimum is met ($dsbm_seed_count/5), while precision seeds remain pending"
else
  note "Checkpoint status: stage6_dsbm=$dsbm_status; only $dsbm_seed_count/3 publishable-minimum seeds are complete"
  note "Do not present this DSBM pilot as the final robustness result; at least three complete seeds are required"
fi

note "Core, all three LD-Leiden phases, and all currently completed optional evidence pass the checkpoint"

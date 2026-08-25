#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage1_correctness_smoke
require_ld_phase_completed smoke_999_10

run_repaired_stage stage2_core_short

run_ld_phase short
"$PYTHON_BIN" "$LD_VALIDATOR" "$LD_CAMPAIGN_ID" --allow-partial

decision="$(stage2_breadth_decision)"
note "Short core validated; optional-breadth decision: $decision"

#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage3_mechanism
require_ld_phase_completed measured_999_10

run_repaired_stage stage4_long_core

"$PYTHON_BIN" "$BOOTSTRAP_SYNC" \
  --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
  --ld-campaign "$LD_CAMPAIGN_ID" \
  --initial-batch 9 --datasets dyn_pubmed arxivmath
run_ld_phase long
"$PYTHON_BIN" "$LD_VALIDATOR" "$LD_CAMPAIGN_ID"

note "The required repaired-ComNetX core and principal LD comparator are complete"

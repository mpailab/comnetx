#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns

run_repaired_stage stage1_correctness_smoke

"$PYTHON_BIN" "$BOOTSTRAP_SYNC" \
  --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
  --ld-campaign "$LD_CAMPAIGN_ID" \
  --initial-batch 999 \
  --datasets dyn_cora dyn_acm dyn_citeseer patent dyn_pubmed arxivmath
run_ld_phase smoke
"$PYTHON_BIN" "$LD_VALIDATOR" "$LD_CAMPAIGN_ID" --allow-partial

note "Both correctness smoke gates passed"

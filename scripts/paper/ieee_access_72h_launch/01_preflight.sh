#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
[[ -n "$EXPECTED_GIT_SHA" ]] \
  || die "Set EXPECTED_GIT_SHA to the reviewed full commit SHA before preflight"

if [[ -f "$BUDGET_STATE_FILE" ]]; then
  require_preflight_campaigns
  "$PYTHON_BIN" "$PAIR_VALIDATOR" \
    --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
    --ld-campaign "$LD_CAMPAIGN_ID" --preflight-only
  initialize_budget_clock
  note "Preflights and the shared clock already exist; nothing was reset"
  exit 0
fi

"$PYTHON_BIN" "$PAIR_VALIDATOR" \
  --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
  --ld-campaign "$LD_CAMPAIGN_ID" --assert-clock-creation-safe

repaired_args=(
  "$PYTHON_BIN" "$REPAIRED_RUNNER"
  --paths-config "$PATHS_CONFIG"
  --campaign-id "$REPAIRED_CAMPAIGN_ID"
  --preflight
)
if [[ -f "$REPAIRED_CAMPAIGN_DIR/manifest.json" ]]; then
  repaired_args+=(--resume)
else
  repaired_args+=(--ack-production-api-settled)
fi
note "Preflighting repaired ComNetX"
"${repaired_args[@]}"

ld_args=(
  "$PYTHON_BIN" "$LD_RUNNER"
  --paths-config "$PATHS_CONFIG"
  --campaign-id "$LD_CAMPAIGN_ID"
  --preflight-only
)
if [[ -f "$LD_CAMPAIGN_DIR/manifest.json" ]]; then
  ld_args+=(--resume)
fi
note "Preflighting LD-Leiden"
"${ld_args[@]}"

"$PYTHON_BIN" "$PAIR_VALIDATOR" \
  --repaired-campaign "$REPAIRED_CAMPAIGN_ID" \
  --ld-campaign "$LD_CAMPAIGN_ID" --preflight-only

initialize_budget_clock
note "Both campaigns are sealed; no graph measurement was run"

#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_HEAVY_TIMEOUT:-4h}"
run_leiden_pair "heavy02_dsbm_hubs_100_mc290" hubs 100_batches 290 "$JOB_TIMEOUT"
run_leiden_pair "heavy02_dsbm_hubs_100_mc1450" hubs 100_batches 1450 "$JOB_TIMEOUT"
exit "$FAILED"

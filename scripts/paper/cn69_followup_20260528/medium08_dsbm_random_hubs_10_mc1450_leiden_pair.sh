#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_MEDIUM_TIMEOUT:-10h}"
run_leiden_pair "medium08_dsbm_random_10_mc1450" random 10_batches 1450 "$JOB_TIMEOUT"
run_leiden_pair "medium08_dsbm_hubs_10_mc1450" hubs 10_batches 1450 "$JOB_TIMEOUT"
exit "$FAILED"

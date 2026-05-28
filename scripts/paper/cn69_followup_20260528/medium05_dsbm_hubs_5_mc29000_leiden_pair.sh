#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_MEDIUM_TIMEOUT:-10h}"
run_leiden_pair "medium05_dsbm_hubs_5_mc29000" hubs 5_batches 29000 "$JOB_TIMEOUT"
exit "$FAILED"

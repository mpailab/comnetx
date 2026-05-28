#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_MEDIUM_TIMEOUT:-10h}"
run_leiden_pair "medium03_dsbm_random_5_mc14500" random 5_batches 14500 "$JOB_TIMEOUT"
exit "$FAILED"

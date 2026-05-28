#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_MEDIUM_TIMEOUT:-8h}"
run_leiden_pair "medium02_dsbm_random_5_mc2900" random 5_batches 2900 "$JOB_TIMEOUT"
exit "$FAILED"

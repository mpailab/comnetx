#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_MEDIUM_TIMEOUT:-8h}"
run_leiden_pair "medium06_dsbm_random_10_mc290" random 10_batches 290 "$JOB_TIMEOUT"
exit "$FAILED"

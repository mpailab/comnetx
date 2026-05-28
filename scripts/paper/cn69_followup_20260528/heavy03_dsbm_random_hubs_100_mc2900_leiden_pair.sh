#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_HEAVY_TIMEOUT:-4h}"
run_leiden_pair "heavy03_dsbm_random_100_mc2900" random 100_batches 2900 "$JOB_TIMEOUT"
exit "$FAILED"

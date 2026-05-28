#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_HEAVY_TIMEOUT:-4h}"
run_leiden_pair "heavy01_dsbm_random_100_mc290" random 100_batches 290 "$JOB_TIMEOUT"
run_leiden_pair "heavy01_dsbm_random_100_mc1450" random 100_batches 1450 "$JOB_TIMEOUT"
exit "$FAILED"

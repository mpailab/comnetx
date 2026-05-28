#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

JOB_TIMEOUT="${CN69_MEDIUM_TIMEOUT:-8h}"
run_leiden_pair "medium01_dsbm_random_5_mc1450" random 5_batches 1450 "$JOB_TIMEOUT"
exit "$FAILED"

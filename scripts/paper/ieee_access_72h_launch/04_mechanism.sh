#!/usr/bin/env bash

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

prepare_launcher
require_preflight_campaigns
require_repaired_stage_validated stage2_core_short
require_ld_phase_completed measured_999_10

run_repaired_stage stage3_mechanism

note "Production parity, mechanism profiles, ablations, and certificates passed"

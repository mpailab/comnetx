#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run06 arxivmath DF-Leiden robustness" \
  "conf/paper_icdm/cn69_robustness_20260529/robust06_arxivmath_dfleiden.json" \
  "${DFLEIDEN_ROBUST_TIMEOUT:-30m}"

exit "$FAILED"

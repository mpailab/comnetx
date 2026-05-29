#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run01 arxivmath S2CAG robustness A" \
  "conf/paper_icdm/cn69_robustness_20260529/robust01_arxivmath_s2cag_a.json" \
  "${S2CAG_ARXIV_TIMEOUT:-3h}"

exit "$FAILED"

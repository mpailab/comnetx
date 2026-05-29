#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run02 arxivmath S2CAG robustness B" \
  "conf/paper_icdm/cn69_robustness_20260529/robust02_arxivmath_s2cag_b.json" \
  "${S2CAG_ARXIV_TIMEOUT:-3h}"

exit "$FAILED"

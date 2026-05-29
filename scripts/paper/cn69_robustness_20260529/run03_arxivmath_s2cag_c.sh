#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run03 arxivmath S2CAG robustness C" \
  "conf/paper_icdm/cn69_robustness_20260529/robust03_arxivmath_s2cag_c.json" \
  "${S2CAG_ARXIV_TIMEOUT:-3h}"

exit "$FAILED"

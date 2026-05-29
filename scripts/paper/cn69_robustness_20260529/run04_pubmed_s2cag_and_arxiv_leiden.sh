#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run04 dyn_pubmed S2CAG robustness A" \
  "conf/paper_icdm/cn69_robustness_20260529/robust04_dyn_pubmed_s2cag_a.json" \
  "${S2CAG_PUBMED_TIMEOUT:-1h}"
run_config "run04 arxivmath Leiden robustness" \
  "conf/paper_icdm/cn69_robustness_20260529/robust04_arxivmath_leiden.json" \
  "${TOPOLOGY_ROBUST_TIMEOUT:-1h}"

exit "$FAILED"

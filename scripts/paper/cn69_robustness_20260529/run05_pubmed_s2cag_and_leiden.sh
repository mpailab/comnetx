#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run05 dyn_pubmed S2CAG robustness B" \
  "conf/paper_icdm/cn69_robustness_20260529/robust05_dyn_pubmed_s2cag_b.json" \
  "${S2CAG_PUBMED_TIMEOUT:-1h}"
run_config "run05 dyn_pubmed Leiden robustness" \
  "conf/paper_icdm/cn69_robustness_20260529/robust05_dyn_pubmed_leiden.json" \
  "${TOPOLOGY_ROBUST_TIMEOUT:-1h}"

exit "$FAILED"

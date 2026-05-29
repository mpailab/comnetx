#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run07 dyn_pubmed DF-Leiden robustness A" \
  "conf/paper_icdm/cn69_robustness_20260529/robust07_dyn_pubmed_dfleiden_a.json" \
  "${DFLEIDEN_ROBUST_TIMEOUT:-30m}"

exit "$FAILED"

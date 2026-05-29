#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "$0")/common.sh"

run_config "run08 dyn_pubmed DF-Leiden robustness B" \
  "conf/paper_icdm/cn69_robustness_20260529/robust08_dyn_pubmed_dfleiden_b.json" \
  "${DFLEIDEN_ROBUST_TIMEOUT:-30m}"

exit "$FAILED"

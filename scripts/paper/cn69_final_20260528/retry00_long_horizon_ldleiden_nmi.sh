#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
CACHE_DIR="${CACHE_DIR:-/home/dev/communities}"
DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
SKIP_REGISTRY="${SKIP_REGISTRY:-results/registry/all_results.json}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-2h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm


log="$LOG_DIR/retry00_long_horizon_ldleiden_nmi_${STAMP}.log"
echo "[long-horizon LD-Leiden NMI retry] $(date -Is) running two dyn_pubmed 999:100 cells after LD-Leiden apply() compatibility fix"
timeout --kill-after=2m "$TIMEOUT" python scripts/launch.py \
  conf/paper_icdm/cn69_final_20260528/long_horizon_missing_nmi_dyn_pubmed_999100.json \
  --paths-config "$PATHS_CONFIG" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[long-horizon LD-Leiden NMI retry] exited with status $status; see $log"
fi
exit "$status"

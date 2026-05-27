#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${PATHS_CONFIG:-datasets-info/paths/cn69.json}"
LOG_DIR="${LOG_DIR:-output}"
TIMEOUT="${TIMEOUT:-24h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"

CONFIGS=(
  "conf/paper_icdm/cn69_pilot_20260527/topology_variance_final.json"
)

for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .json)"
  log="$LOG_DIR/gpu3_topology_variance_final_${name}_${STAMP}.log"
  echo "[topology variance final] $(date -Is) running $config with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/launch.py "$config" --paths-config "$PATHS_CONFIG" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[topology variance final] $config exited with status $status; see $log"
    FAILED=1
  fi
done

exit "$FAILED"

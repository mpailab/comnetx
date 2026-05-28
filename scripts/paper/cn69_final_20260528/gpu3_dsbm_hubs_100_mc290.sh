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
DEFAULT_JOB_TIMEOUT="10h"
JOB_TIMEOUT="${CN69_JOB_TIMEOUT:-$DEFAULT_JOB_TIMEOUT}"
PAPER_ICDM_SERIES="${PAPER_ICDM_SERIES:-7}"
RESULTS_DIR="${RESULTS_DIR:-results/paper_icdm/$PAPER_ICDM_SERIES}"
export RESULTS_DIR
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

mark_manifest_failed() {
  local manifest_path="$1"
  local status_code="$2"
  local context="$3"
  if [[ ! -f "$manifest_path" ]]; then
    return 0
  fi
  python - "$manifest_path" "$status_code" "$context" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
status_code = sys.argv[2]
context = sys.argv[3]
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
data["wrapper_exit_status"] = status_code
data["wrapper_context"] = context
data["wrapper_updated_at"] = datetime.now().isoformat()
if data.get("status") == "running":
    data["status"] = "wrapper_failed"
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_dsbm() {
  local title="$1"
  local name="$2"
  shift 2
  local method mode run_name log status
  for method in leidenalg dfleiden; do
    for mode in naive smart dynamic; do
      if [[ "$method" == "leidenalg" && "$mode" == "dynamic" ]]; then
        continue
      fi
      run_name="${name}_${method}_${mode}_${STAMP}"
      log="$LOG_DIR/${run_name}.log"
      echo "[${title} / ${method}-${mode}] $(date -Is) running DSBM stress with timeout=$JOB_TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
      timeout --kill-after=2m "$JOB_TIMEOUT" python scripts/paper/run_dsbm_stress.py \
        "$@" \
        --methods "$method" \
        --modes "$mode" \
        --use-gpu \
        --catch-errors \
        --skip-registry "$SKIP_REGISTRY" \
        --output-dir "$RESULTS_DIR" \
        --name "$run_name" \
        2>&1 | tee "$log"
      status="${PIPESTATUS[0]}"
      if [[ "$status" -ne 0 ]]; then
        mark_manifest_failed "$RESULTS_DIR/manifest_${run_name}.json" "$status" "${title} / ${method}-${mode}"
        echo "[${title} / ${method}-${mode}] exited with status $status; see $log"
        FAILED=1
      fi
    done
  done
}

run_dsbm "DSBM hubs 100 batches mc290" "gpu3_dsbm_hubs_100_batches_mc290" \
  --root "$DSBM_ROOT" \
  --batch-suffix 100_batches \
  --regimes hubs \
  --max-changes 290

exit "$FAILED"

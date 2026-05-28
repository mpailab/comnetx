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
DEFAULT_JOB_TIMEOUT="2h"
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


log="$LOG_DIR/retry00_long_horizon_ldleiden_nmi_${STAMP}.log"
echo "[long-horizon LD-Leiden NMI retry] $(date -Is) running two dyn_pubmed 999:100 cells after LD-Leiden apply() compatibility fix with timeout=$JOB_TIMEOUT"
timeout --kill-after=2m "$JOB_TIMEOUT" python scripts/launch.py \
  conf/paper_icdm/cn69_final_20260528/long_horizon_missing_nmi_dyn_pubmed_999100.json \
  --paths-config "$PATHS_CONFIG" \
  2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[long-horizon LD-Leiden NMI retry] exited with status $status; see $log"
fi
exit "$status"

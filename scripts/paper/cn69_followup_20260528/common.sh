#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${PARENT_HOSTNAME:-cn69}"
export PYTHONUNBUFFERED=1

DSBM_ROOT="${DSBM_ROOT:-datasets-sbm}"
SKIP_REGISTRY="${SKIP_REGISTRY:-results/registry/all_results.json}"
LOG_DIR="${LOG_DIR:-output}"
PAPER_ICDM_SERIES="${PAPER_ICDM_SERIES:-8}"
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

run_leiden_cell() {
  local title="$1"
  local regime="$2"
  local batch_suffix="$3"
  local max_changes="$4"
  local mode="$5"
  local timeout_value="$6"
  local safe_title run_name log status
  safe_title="$(echo "$title" | tr ' /' '__')"
  run_name="${safe_title}_leidenalg_${mode}_${STAMP}"
  log="$LOG_DIR/${run_name}.log"
  echo "[${title} / leidenalg-${mode}] $(date -Is) timeout=${timeout_value} output=${RESULTS_DIR}"
  timeout --kill-after=2m "$timeout_value" python scripts/paper/run_dsbm_stress.py \
    --root "$DSBM_ROOT" \
    --batch-suffix "$batch_suffix" \
    --regimes "$regime" \
    --max-changes "$max_changes" \
    --methods leidenalg \
    --modes "$mode" \
    --use-gpu \
    --catch-errors \
    --skip-registry "$SKIP_REGISTRY" \
    --output-dir "$RESULTS_DIR" \
    --name "$run_name" \
    2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    mark_manifest_failed "$RESULTS_DIR/manifest_${run_name}.json" "$status" "${title} / leidenalg-${mode}"
    echo "[${title} / leidenalg-${mode}] exited with status $status; see $log"
    FAILED=1
  fi
}

run_leiden_pair() {
  local title="$1"
  local regime="$2"
  local batch_suffix="$3"
  local max_changes="$4"
  local timeout_value="$5"
  run_leiden_cell "$title" "$regime" "$batch_suffix" "$max_changes" smart "$timeout_value"
  run_leiden_cell "$title" "$regime" "$batch_suffix" "$max_changes" naive "$timeout_value"
}

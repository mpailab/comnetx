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
DEFAULT_JOB_TIMEOUT="8h"
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

need_dyn_cora_profile() {
  python - <<'PY'
import json
from pathlib import Path

registry = Path("results/registry/all_results.json")
targets = {
    ("dyn_cora", "999:50", "leidenalg", None, "full"),
    ("dyn_cora", "999:50", "dfleiden", None, "full"),
    ("dyn_cora", "999:50", "s2cag", "random", "full"),
    ("dyn_cora", "999:50", "leidenalg", None, "no_closure"),
    ("dyn_cora", "999:50", "leidenalg", None, "no_contraction"),
}
if not registry.exists():
    print("registry missing; running dyn_cora small-control block")
    raise SystemExit(0)
rows = json.loads(registry.read_text(encoding="utf-8"))
seen = set()
for row in rows:
    if row.get("measurement_type") != "workload_profile":
        continue
    key = (
        row.get("base_dataset"),
        str(row.get("batch_strategy")),
        row.get("method"),
        row.get("feature_mode"),
        row.get("variant", "full"),
    )
    if key in targets:
        seen.add(key)
missing = sorted(targets - seen, key=str)
if missing:
    print("missing dyn_cora workload-profile cells:")
    for item in missing:
        print("  ", item)
    raise SystemExit(0)
print("all target dyn_cora workload-profile cells already exist; skipping")
raise SystemExit(1)
PY
}

if need_dyn_cora_profile; then
  echo "[dyn_cora small control] $(date -Is) using the already prepared extra27 script"
  RESULTS_DIR="$RESULTS_DIR" CN69_JOB_TIMEOUT="${DYN_CORA_TIMEOUT:-2h}" \
    scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh
  status="$?"
  if [[ "$status" -ne 0 ]]; then
    echo "[dyn_cora small control] exited with status $status"
    FAILED=1
  fi
fi

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

run_dsbm "DSBM hubs 5 batches mc290" "gpu1_dsbm_hubs_5_batches_mc290" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes hubs \
  --max-changes 290

run_dsbm "DSBM hubs 5 batches mc1450" "gpu1_dsbm_hubs_5_batches_mc1450" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes hubs \
  --max-changes 1450

run_dsbm "DSBM hubs 5 batches mc2900" "gpu1_dsbm_hubs_5_batches_mc2900" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes hubs \
  --max-changes 2900

run_dsbm "DSBM hubs 5 batches mc14500" "gpu1_dsbm_hubs_5_batches_mc14500" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes hubs \
  --max-changes 14500

run_dsbm "DSBM hubs 5 batches mc29000" "gpu1_dsbm_hubs_5_batches_mc29000" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes hubs \
  --max-changes 29000

run_dsbm "DSBM hubs 10 batches mc290" "gpu1_dsbm_hubs_10_batches_mc290" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes hubs \
  --max-changes 290

run_dsbm "DSBM hubs 10 batches mc1450" "gpu1_dsbm_hubs_10_batches_mc1450" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes hubs \
  --max-changes 1450

run_dsbm "DSBM hubs 10 batches mc2900" "gpu1_dsbm_hubs_10_batches_mc2900" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes hubs \
  --max-changes 2900

run_dsbm "DSBM hubs 10 batches mc14500" "gpu1_dsbm_hubs_10_batches_mc14500" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes hubs \
  --max-changes 14500

run_dsbm "DSBM hubs 10 batches mc29000" "gpu1_dsbm_hubs_10_batches_mc29000" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes hubs \
  --max-changes 29000

exit "$FAILED"

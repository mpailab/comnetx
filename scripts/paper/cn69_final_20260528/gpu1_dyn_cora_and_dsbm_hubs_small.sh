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
TIMEOUT="${TIMEOUT:-8h}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm

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
  TIMEOUT="${DYN_CORA_TIMEOUT:-2h}" scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh
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
  local log="$LOG_DIR/${name}_${STAMP}.log"
  echo "[${title}] $(date -Is) running DSBM stress with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \
    "$@" \
    --methods leidenalg dfleiden \
    --modes naive smart dynamic \
    --use-gpu \
    --catch-errors \
    --skip-registry "$SKIP_REGISTRY" \
    --output-dir results/paper_icdm \
    --name "${name}_${STAMP}" \
    2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[${title}] exited with status $status; see $log"
    FAILED=1
  fi
}

run_dsbm "DSBM hubs 5 batches" "gpu1_dsbm_hubs_5_batches" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes hubs \
  --max-changes 290 1450 2900 14500 29000

run_dsbm "DSBM hubs 10 batches" "gpu1_dsbm_hubs_10_batches" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes hubs \
  --max-changes 290 1450 2900 14500 29000

exit "$FAILED"

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

need_long_horizon_nmi() {
  python - <<'PY'
import json
from pathlib import Path

registry = Path("results/registry/all_results.json")
targets = {
    ("dyn_pubmed", "999:100", "ldleiden", "naive"),
    ("dyn_pubmed", "999:100", "ldleiden", "smart"),
}
if not registry.exists():
    print("registry missing; running the NMI completion block")
    raise SystemExit(0)
rows = json.loads(registry.read_text(encoding="utf-8"))
seen = set()
for row in rows:
    if row.get("measurement_type") != "experiment":
        continue
    key = (
        row.get("base_dataset"),
        str(row.get("batch_strategy")),
        row.get("method"),
        row.get("mode"),
    )
    if key in targets and row.get("final_nmi") is not None:
        seen.add(key)
missing = sorted(targets - seen)
if missing:
    print("missing long-horizon NMI cells:")
    for item in missing:
        print("  ", item)
    raise SystemExit(0)
print("all target long-horizon NMI cells already exist; skipping")
raise SystemExit(1)
PY
}

if need_long_horizon_nmi; then
  log="$LOG_DIR/gpu0_long_horizon_missing_nmi_${STAMP}.log"
  echo "[long-horizon NMI] $(date -Is) rerunning only the two dyn_pubmed 999:100 LD-Leiden cells that lack final_nmi"
  timeout --kill-after=2m "${NMI_TIMEOUT:-2h}" python scripts/launch.py \
    conf/paper_icdm/cn69_final_20260528/long_horizon_missing_nmi_dyn_pubmed_999100.json \
    --paths-config "$PATHS_CONFIG" \
    2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[long-horizon NMI] exited with status $status; see $log"
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

run_dsbm "DSBM random 5 batches" "gpu0_dsbm_random_5_batches" \
  --root "$DSBM_ROOT" \
  --batch-suffix 5_batches \
  --regimes random \
  --max-changes 290 1450 2900 14500 29000

run_dsbm "DSBM random 10 batches" "gpu0_dsbm_random_10_batches" \
  --root "$DSBM_ROOT" \
  --batch-suffix 10_batches \
  --regimes random \
  --max-changes 290 1450 2900 14500 29000

exit "$FAILED"

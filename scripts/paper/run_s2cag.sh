#!/usr/bin/env bash
# One-command server launch: 20 paired runs, no diagnostic campaign.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PATHS_CONFIG="datasets-info/paths/cn69.json"
DEVICE="cuda"
OUTPUT_DIR=""
PYTHON_BIN=""

usage() {
  cat <<'EOF'
Usage: bash scripts/paper/run_s2cag.sh [options]

Run full S2CAG and ComNetX+S2CAG on PubMed and arxivmath:
999:5 and 999:10, random features, five paired repeats (40 processes).
A dedicated environment is prepared automatically on the first launch.

  --paths-config FILE  Dataset paths JSON (default: datasets-info/paths/cn69.json)
  --output-dir DIR     New result directory (default: timestamped under results/)
  --device cuda|cpu   Tensor device (default: cuda; no silent CPU fallback)
  --python PROGRAM    Python 3.10 or 3.11 for creating the dedicated environment
  -h, --help          Show this help without installing or running anything
EOF
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --paths-config|--output-dir|--device|--python)
      (($# >= 2)) || fail "Missing value for $1"
      case "$1" in
        --paths-config) PATHS_CONFIG="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --device) DEVICE="$2" ;;
        --python) PYTHON_BIN="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1 (use --help)" ;;
  esac
done
[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || fail "--device must be cuda or cpu"
cd -- "$ROOT"
[[ -f "$PATHS_CONFIG" ]] || fail "Dataset paths file does not exist: $PATHS_CONFIG"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="results/s2cag-paired/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ ! -e "$OUTPUT_DIR" ]] || fail "Output directory already exists: $OUTPUT_DIR"

VENV="$ROOT/.venv-s2cag"
REQUIREMENTS="$ROOT/requirements/ieee-access-s2cag.txt"
CONFIG="$ROOT/conf/ieee_access/s2cag_pairs/confirmatory.json"
compatible_python() {
  "$1" -c 'import sys; sys.exit(sys.version_info[:2] not in ((3, 10), (3, 11)))' 2>/dev/null
}
if [[ ! -e "$VENV" ]]; then
  if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.11 python3.10 python3; do
      if command -v "$candidate" >/dev/null 2>&1 && compatible_python "$candidate"; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
  fi
  [[ -n "$PYTHON_BIN" ]] || fail "Install Python 3.10/3.11 or pass --python /path/to/python3.11"
  compatible_python "$PYTHON_BIN" || fail "S2CAG requires Python 3.10/3.11 with the pinned SciPy"
  printf 'Creating dedicated environment: %s\n' "$VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PYTHON_RUN="$VENV/bin/python"
[[ -x "$PYTHON_RUN" ]] || fail "Incomplete environment: $VENV (check or remove it before retrying)"
compatible_python "$PYTHON_RUN" || fail "Existing $VENV must use Python 3.10 or 3.11"
REQUIREMENTS_SHA="$("$PYTHON_RUN" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$REQUIREMENTS")"
STAMP="$VENV/.s2cag-requirements.sha256"
if [[ ! -f "$STAMP" ]] || [[ "$(cat -- "$STAMP")" != "$REQUIREMENTS_SHA" ]]; then
  printf 'Installing pinned S2CAG dependencies into %s\n' "$VENV"
  "$PYTHON_RUN" -m pip install --disable-pip-version-check -r "$REQUIREMENTS"
  printf '%s\n' "$REQUIREMENTS_SHA" > "$STAMP"
fi

# Validate selection, without any extra dataset run or diagnostic campaign.
"$PYTHON_RUN" - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    config = json.load(f)
if config['measurement']['comnetx_profile']:
    raise SystemExit('The server campaign must have ComNetX profiling disabled')
PY
printf 'Starting 20 pairs (40 processes). Results: %s\n' "$OUTPUT_DIR"
"$PYTHON_RUN" -u "$ROOT/scripts/paper/run_s2cag_pairs.py" run "$CONFIG" \
  --paths-config "$PATHS_CONFIG" --output-dir "$OUTPUT_DIR" --device "$DEVICE"
printf '\nFinished. Validated comparisons: %s/paired-summary.json\n' "$OUTPUT_DIR"

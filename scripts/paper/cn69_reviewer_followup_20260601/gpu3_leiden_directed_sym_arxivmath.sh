#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_followup_config.sh" 3 leiden_directed_sym_arxivmath leiden_directed_arxivmath.json leiden_sym_arxivmath.json

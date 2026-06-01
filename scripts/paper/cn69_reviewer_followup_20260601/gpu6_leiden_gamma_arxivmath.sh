#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_followup_config.sh" 6 leiden_gamma_arxivmath leiden_gamma_arxivmath.json

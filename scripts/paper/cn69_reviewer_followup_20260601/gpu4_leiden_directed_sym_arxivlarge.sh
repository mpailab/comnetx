#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_followup_config.sh" 4 leiden_directed_sym_arxivlarge leiden_directed_arxivlarge.json leiden_sym_arxivlarge.json

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_followup_config.sh" 0 s2cag_arxivlarge s2cag_large_arxivlarge.json

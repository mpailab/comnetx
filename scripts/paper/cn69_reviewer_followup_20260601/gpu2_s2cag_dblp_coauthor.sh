#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_followup_config.sh" 2 s2cag_dblp_coauthor s2cag_large_dblp_coauthor.json

#!/usr/bin/env bash
# Open the interactive analysis app on a results directory (default: results/).
set -euo pipefail
cd "$(dirname "$0")/.."
exec streamlit run src/analysis/app.py -- --results "${1:-results}"

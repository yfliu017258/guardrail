#!/usr/bin/env bash
# Run every evaluation of the suite that has no completed run yet, then say how to view the results.
#   scripts/run_suite.sh                 # uses ./suite.yaml
#   scripts/run_suite.sh suite.yaml --limit 50
#   scripts/run_suite.sh suite.yaml --only my-new-guardrail
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE="${1:-suite.yaml}"
[ -f "$SUITE" ] || { echo "no $SUITE — start from: cp src/config/evaluation/suite.example.yaml suite.yaml" >&2; exit 2; }
exec python -m src.evaluation.suite "$SUITE" "${@:2}"

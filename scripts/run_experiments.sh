#!/usr/bin/env bash
# Leave a set of experiments running unattended: in the background, with a log, resuming and retrying on failure.
#
#   scripts/run_experiments.sh start                         # everything in ./suite.yaml that is not done yet
#   scripts/run_experiments.sh start my-guardrail qwen3.6-27b@core_v1     # only these experiments
#   scripts/run_experiments.sh status                        # running? + the latest progress and summary
#   scripts/run_experiments.sh logs                          # follow the log
#   scripts/run_experiments.sh stop                          # stop it; `start` later continues where it stopped
#
# An experiment is a guardrail / model name, a dataset name, or <subject>@<dataset> (names from the suite file).
# Environment:  SUITE=suite.yaml   MAX_SAMPLES=<n per dataset>   RETRIES=5   RETRY_WAIT=300 (seconds)
#               RESULTS=<results dir>   EXTRA="--retry-errors"  (any other flags of `python -m src.evaluation.suite`)
set -euo pipefail
cd "$(dirname "$0")/.."

SUITE="${SUITE:-suite.yaml}"
RUN_DIR="${RUN_DIR:-logs}"
PID_FILE="$RUN_DIR/experiments.pid"
LATEST="$RUN_DIR/experiments.latest.log"

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-}" in
  start)
    shift
    [ -f "$SUITE" ] || { echo "no $SUITE — start from: cp src/config/evaluation/suite.example.yaml suite.yaml" >&2; exit 2; }
    if running; then echo "already running (pid $(cat "$PID_FILE")); see: $0 status" >&2; exit 1; fi
    mkdir -p "$RUN_DIR"
    LOG="$RUN_DIR/experiments_$(date +%Y%m%d-%H%M%S).log"
    ARGS=("$SUITE" --retries "${RETRIES:-5}" --retry-wait "${RETRY_WAIT:-300}")
    [ -n "${MAX_SAMPLES:-}" ] && ARGS+=(--max-samples "$MAX_SAMPLES")
    [ -n "${RESULTS:-}" ] && ARGS+=(--results "$RESULTS")
    [ "$#" -gt 0 ] && ARGS+=(--only "$@")
    # shellcheck disable=SC2086
    nohup python -m src.evaluation.suite "${ARGS[@]}" ${EXTRA:-} > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    ln -sf "$(basename "$LOG")" "$LATEST"
    echo "started (pid $!)   log: $LOG"
    echo "progress: $0 status      stop: $0 stop"
    ;;
  status)
    if running; then echo "RUNNING (pid $(cat "$PID_FILE"))"; else echo "not running"; fi
    [ -f "$LATEST" ] || { echo "no log yet"; exit 0; }
    echo "log: $RUN_DIR/$(readlink "$LATEST")"
    grep -E '^=== |^### ' "$LATEST" | tail -n 3 || true                   # which run it is on
    tr '\r' '\n' < "$LATEST" | grep -E '^eval\[' | tail -n 1 || true      # its progress bar
    if grep -q '^view:' "$LATEST"; then                                   # finished: the summary table
      awk '/^-{20,}/{n++} n>=1' "$LATEST" | tail -n 40
    fi
    ;;
  logs)
    [ -f "$LATEST" ] || { echo "no log yet" >&2; exit 1; }
    exec tail -n 50 -f "$LATEST"
    ;;
  stop)
    if running; then
      kill "$(cat "$PID_FILE")" && echo "stopped; finished rows are kept — \`$0 start\` continues from there"
    else
      echo "not running"
    fi
    rm -f "$PID_FILE"
    ;;
  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac

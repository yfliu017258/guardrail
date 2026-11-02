#!/usr/bin/env bash
# Start the guardrail server.
#
#   CONFIG=src/config/server/pipelines/two_stage.yaml PORT=8080 deploy/server/run_server.sh
#
# Needs: GUARDRAIL_API_KEY (clients send it as x-api-key; the server refuses to start without it unless the
# config sets server.api_key_env: null), GUARD_MODEL_BASE_URL (the vLLM serving the guard model, see
# deploy/vllm/serve_guard.sh). Both can live in .env at the repo root.
#
# One process per replica (uvicorn workers=1): stages hold model clients / weights in-process and the
# pipeline is async, so scale with replicas behind a load balancer, not with workers.
# Probes: liveness GET /health, readiness GET /ready (503 while the guard model is not serving).
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-src/config/server/default.yaml}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
LOG_LEVEL="${LOG_LEVEL:-info}"

exec python -m src.server --config ${CONFIG} --host "${HOST}" --port "${PORT}" --log-level "${LOG_LEVEL}" "$@"

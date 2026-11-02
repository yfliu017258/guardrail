#!/usr/bin/env bash
# Start the guardrail server.
#
#   CONFIG=src/config/server/pipelines/two_stage.yaml PORT=8080 deploy/server/run_server.sh [--set key=value ...]
#
# Needs: GUARDRAIL_API_KEY (clients send it as x-api-key; the server refuses to start without it unless the
# config sets server.api_key_env: null), GUARD_MODEL_BASE_URL (the vLLM serving the guard model, see
# deploy/vllm/serve_guard.sh). Both can live in .env at the repo root.
# HOST / PORT / LOG_LEVEL override the YAML's server.* only when set. SNAPSHOT=0 skips writing the resolved
# config to results/server/ (read-only filesystems).
#
# One process per replica (uvicorn workers=1): stages hold model clients / weights in-process and the
# pipeline is async, so scale with replicas behind a load balancer, not with workers.
# Probes: liveness GET /health, readiness GET /ready (503 while the guard model is not serving).
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-src/config/server/default.yaml}"
args=()
[[ -n "${HOST:-}" ]] && args+=(--host "$HOST")
[[ -n "${PORT:-}" ]] && args+=(--port "$PORT")
[[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "$LOG_LEVEL")
[[ "${SNAPSHOT:-1}" == "0" ]] && args+=(--no-snapshot)

# shellcheck disable=SC2086  # CONFIG may list several YAML files
exec python -m src.server --config ${CONFIG} ${args[@]+"${args[@]}"} "$@"

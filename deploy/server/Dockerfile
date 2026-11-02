# Guardrail server image (CPU; the LLM guard runs in a separate vLLM container on the MIG slice).
#   docker build -f deploy/server/Dockerfile -t ojt-guardrail .
#   docker run -p 8080:8080 -e GUARDRAIL_API_KEY=... -e GUARD_MODEL_BASE_URL=http://vllm-guard:8000/v1 \
#              -e CONFIG=src/config/server/models/qwen3guard_gen_4b.yaml ojt-guardrail
# For hf_classifier stages on GPU, build FROM a CUDA/PyTorch base and install ".[hf]" instead.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy ./deploy
RUN pip install --no-cache-dir . && useradd -m guard && mkdir -p results && chown guard results
USER guard

ENV CONFIG=src/config/server/default.yaml PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')" || exit 1
CMD ["deploy/server/run_server.sh"]

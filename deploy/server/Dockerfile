# Guardrail server image. The LLM guard runs in a separate vLLM container on the MIG slice.
#   docker build -f deploy/server/Dockerfile -t ojt-guardrail .                       # CPU: rules + openai_guard stages
#   docker build -f deploy/server/Dockerfile --build-arg EXTRAS=hf -t ojt-guardrail-hf .   # + hf_classifier stages
#   docker run -p 8080:8080 -e GUARDRAIL_API_KEY=... -e GUARD_MODEL_BASE_URL=http://vllm-guard:8000/v1 \
#              -e CONFIG=src/config/server/models/qwen3guard_gen_4b.yaml ojt-guardrail
# hf_classifier stages (e.g. pipelines/two_stage.yaml) need EXTRAS=hf; for GPU inference use a CUDA/PyTorch
# base image instead of python:slim.
FROM python:3.12-slim

ARG EXTRAS=""
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy ./deploy
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}" \
 && useradd -u 10001 -g 0 -M guard \
 && mkdir -p results && chgrp -R 0 /app/results && chmod -R g+rwX /app/results

# numeric UID (runAsNonRoot) in group 0, so arbitrary-UID platforms (OpenShift) can write results/
USER 10001
ENV CONFIG=src/config/server/default.yaml PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')" || exit 1
CMD ["deploy/server/run_server.sh"]

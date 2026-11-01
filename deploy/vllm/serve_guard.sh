#!/usr/bin/env bash
# Serve a guard model on one H200 MIG slice (1g.35gb) with vLLM's OpenAI-compatible server.
#
#   nvidia-smi -L                              # find the MIG UUID, e.g. MIG-xxxxxxxx-xxxx-...
#   MIG_UUID=MIG-... MODEL=Qwen/Qwen3Guard-Gen-4B ./deploy/vllm/serve_guard.sh
#
# Sizing notes for 35 GB: weights (bf16 ~2 bytes/param) + KV cache + CUDA graphs/activations.
#   4B  bf16 ~8 GB   -> plenty of KV cache
#   8B  bf16 ~16 GB  -> fine with max-model-len 8k
#   12B bf16 ~24 GB  -> tight; use fp8 or shorter context
#   gpt-oss-20b MXFP4 ~14 GB -> fits (MoE, check your vLLM build supports MXFP4 on Hopper)
# 1g = 1/7 of the SMs: throughput, not memory, is usually the limit -> keep max-num-seqs moderate,
# keep guard outputs short (max_tokens 16-64), and benchmark p95 latency under real concurrency.
set -euo pipefail

: "${MODEL:?set MODEL}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_SEQS="${MAX_SEQS:-32}"
DTYPE="${DTYPE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"          # e.g. "--quantization fp8" or "--reasoning-parser openai_gptoss"

if [[ -n "${MIG_UUID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$MIG_UUID"; fi

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs "$MAX_SEQS" \
  --max-logprobs 20 \
  --api-key "${VLLM_API_KEY:-EMPTY}" \
  ${EXTRA_ARGS}

#!/usr/bin/env bash
# Launch the repository's validated single-GPU Qwen3.6 vLLM profile.
#
# The defaults target an NVIDIA DGX Spark/GB10. Environment overrides make the
# launcher reusable on other Blackwell systems without duplicating the complete
# serving command in onboarding documentation.
set -euo pipefail

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed or is not on PATH" >&2
  exit 127
fi

model="${VLLM_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
revision="${VLLM_MODEL_REVISION:-491c2f1ea524c639598bf8fa787a93fed5a6fbce}"
host="${VLLM_HOST:-0.0.0.0}"
port="${VLLM_PORT:-8000}"
# Leave unified-memory headroom for FlakeGraph workers and Spark executors on
# systems such as GB10; standalone inference deployments can override this.
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
max_model_len="${VLLM_MAX_MODEL_LEN:-262144}"
max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
max_num_seqs="${VLLM_MAX_NUM_SEQS:-4}"

# Atomic accumulation is faster for Marlin MoE kernels on the GB10 profile.
export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"

exec vllm serve "$model" \
  --served-model-name "$model" \
  --revision "$revision" \
  --host "$host" \
  --port "$port" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice

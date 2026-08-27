#!/usr/bin/env bash
# Launch the repository's validated single-GPU vLLM profile.
#
# The defaults target an NVIDIA DGX Spark/GB10 and match the Helm chart's own
# serving profile, so a laptop-scale check and a fleet run exercise the same
# engine configuration. Environment overrides make the launcher reusable on
# other Blackwell systems without duplicating the command in onboarding docs.
set -euo pipefail

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed or is not on PATH" >&2
  exit 127
fi

model="${VLLM_MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
revision="${VLLM_MODEL_REVISION:-9e3d73c76eddb75f795cc24ccfbc5affe41c66bd}"
host="${VLLM_HOST:-0.0.0.0}"
port="${VLLM_PORT:-8000}"
# On unified-memory systems such as GB10 this competes with the OS and the
# kubelet rather than drawing from a separate pool; 0.70 has taken such a node
# off the network. A dedicated inference host can raise it.
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
max_model_len="${VLLM_MAX_MODEL_LEN:-262144}"
max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
# Must stay below the max_concurrent the KV budget allows. Check a change with:
#   flakegraph serving sizing --kv-heads 4 --head-dim 256 --attention-layers 16 \
#     --weights-gib 21.81 --device-memory-gib 119.2 --max-num-seqs <value>
max_num_seqs="${VLLM_MAX_NUM_SEQS:-24}"

# Atomic accumulation is faster for Marlin MoE kernels on the GB10 profile.
export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"

# Speculative decoding is deliberately absent. It conflicts with
# --async-scheduling and forfeits much of the reusable prefix on this model's
# hybrid cache, so it belongs behind a benchmark rather than in a default.
exec vllm serve "$model" \
  --served-model-name "$model" \
  --revision "$revision" \
  --host "$host" \
  --port "$port" \
  --scheduling-policy priority \
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
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice

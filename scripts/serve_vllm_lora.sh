#!/usr/bin/env bash
set -euo pipefail

# ---- User-editable parameters (can also be provided via env vars) ----
# Base model directory/path that vLLM should serve
MODEL_DIR=${MODEL_DIR:-"./"}

# The name that vLLM exposes to clients (EvalScope uses this in the 'model' field)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen3-8B"}

# GPU selection
GPU_ID=${GPU_ID:-"1"}

# vLLM server port
PORT=${PORT:-"8801"}

# How much GPU memory vLLM can use
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.9"}

# Enable ModelScope loader behavior if you rely on ModelScope
VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-"True"}

# LoRA settings
# vLLM expects: --lora-modules <name>=<path>  (you can pass multiple --lora-modules ...)
LORA_NAME=${LORA_NAME:-"qwen3-lora"}
LORA_PATH=${LORA_PATH:-""}  # REQUIRED: path to your LoRA checkpoint/adapters
MAX_LORA_RANK=${MAX_LORA_RANK:-"32"}

# Any extra args you want to pass to vLLM
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-""}

# ---- Safety checks ----
if [[ -z "${LORA_PATH}" ]]; then
  echo "[ERROR] LORA_PATH is empty. Example:" >&2
  echo "  LORA_PATH=/path/to/checkpoint-xxxx ./scripts/serve_vllm_lora.sh" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"${GPU_ID}"}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE}

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Serving model: ${MODEL_DIR} (served-model-name=${SERVED_MODEL_NAME})"
echo "[INFO] LoRA: ${LORA_NAME}=${LORA_PATH} (max-rank=${MAX_LORA_RANK})"
echo "[INFO] Port: ${PORT} | gpu-memory-utilization: ${GPU_MEMORY_UTILIZATION}"

exec vllm serve "${MODEL_DIR}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust_remote_code \
  --port "${PORT}" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${LORA_PATH}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  ${VLLM_EXTRA_ARGS}

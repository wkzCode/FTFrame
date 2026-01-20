#!/usr/bin/env bash
set -euo pipefail

# Example:
#   MMLU_PATH=/mnt/.../MMLU \
#   MODEL_NAME=DeepSeek-Qwen3-8b \
#   API_URL=http://127.0.0.1:8801/v1/chat/completions \
#   ./scripts/run_evalscope_mmlu.sh

MMLU_PATH=${MMLU_PATH:-""}
MODEL_NAME=${MODEL_NAME:-"DeepSeek-Qwen3-8b"}
API_URL=${API_URL:-"http://127.0.0.1:8801/v1/chat/completions"}
BATCH=${BATCH:-128}

if [[ -z "${MMLU_PATH}" ]]; then
  echo "[ERROR] MMLU_PATH is empty. Set it to your local MMLU directory." >&2
  exit 2
fi

python scripts/eval_evalscope.py \
  --model "${MODEL_NAME}" \
  --api-url "${API_URL}" \
  --mmlu-path "${MMLU_PATH}" \
  --batch "${BATCH}"

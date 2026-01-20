#!/usr/bin/env bash

VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=1 vllm serve ./ \
  --gpu-memory-utilization 0.9 \
  --served-model-name Qwen3-8B \
  --trust_remote_code \
  --port 8801 \
  --enable-lora \
  --lora-modules qwen3-lora=/outputs/checkpoint-6241 \
  --max-lora-rank 32
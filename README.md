# Qwen3-8B Finetune Framework (MMLU + LoRA/DoRA)

This repo is a **reproducible finetuning framework** for **Qwen3-8B** on **MMLU** using **PEFT adapters**.

It is structured so you can:
- run **baseline** MMLU evaluation
- run **LoRA finetuning** (SFT: answer is a single letter A/B/C/D)
- run **post-finetune evaluation** (fast logits-based or generation-based)
- optionally run **DoRA** (if your installed `peft` version supports it; auto-fallback to LoRA otherwise)
- optionally **merge adapter** back into base weights

## 0) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 1) Quickstart

### Baseline eval (no finetune)
```bash
python scripts/eval_mmlu.py --config configs/eval_mmlu.yaml \
  eval.model_name_or_path=/path/to/Qwen3-8B \
  eval.adapter_path=null
```

### LoRA finetune
```bash
python scripts/train.py --config configs/train_lora.yaml \
  model.model_name_or_path=/path/to/Qwen3-8B
```

### Evaluate finetuned adapter
```bash
python scripts/eval_mmlu.py --config configs/eval_mmlu.yaml \
  eval.model_name_or_path=/path/to/Qwen3-8B \
  eval.adapter_path=outputs/exp_lora/checkpoint-last
```

> Multi-GPU (recommended)
```bash
accelerate launch scripts/train.py --config configs/train_lora.yaml \
  model.model_name_or_path=/path/to/Qwen3-8B
```

## 2) What gets saved

Each run writes into `outputs/<run_name>/`:
- `run_config.yaml` (full resolved config snapshot)
- `trainer_state.json`, `train_results.json`, `eval_results.json`
- adapter weights/config (PEFT)
- optional `merged/` if enabled

## 3) Config overrides

Overrides are `key=value` pairs after the config. Examples:

```bash
python scripts/train.py --config configs/train_lora.yaml \
  training.learning_rate=1e-4 \
  training.num_train_epochs=1 \
  lora.r=16
```

To set `null` from CLI, use `null`:

```bash
python scripts/eval_mmlu.py --config configs/eval_mmlu.yaml \
  eval.adapter_path=null
```

## 4) Notes

- This framework uses **chat templates** via `tokenizer.apply_chat_template` and masks the prompt tokens so the loss is only on the assistant answer tokens.
- For MMLU, this repo provides two evaluation modes:
  - `first_token` (fast): compares the first unmasked label token vs argmax logits
  - `generate` (true MCQ): greedy-generate a few tokens and parse the first valid letter



## 5) MMLU evaluation via vLLM + EvalScope (your workflow)

If you prefer evaluating **through a vLLM OpenAI-compatible server** and **EvalScope** (instead of `scripts/eval_mmlu.py`), use the included helpers.

### Install eval stack

```bash
pip install -r requirements_evalscope.txt
```

### Step A: start vLLM server with LoRA enabled

In one terminal:

```bash
# REQUIRED: point to the LoRA checkpoint directory (e.g. outputs/.../checkpoint-xxxx)
LORA_PATH=/path/to/qwen3_8b_lora_mmlu/checkpoint-6241 MODEL_DIR=/path/to/base/Qwen3-8B SERVED_MODEL_NAME=Qwen3-8B GPU_ID=1 PORT=8801 ./scripts/serve_vllm_lora.sh
```

This is equivalent to your command:

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=1 vllm serve ./   --gpu-memory-utilization 0.9 --served-model-name Qwen3-8B --trust_remote_code   --port 8801 --enable-lora --lora-modules qwen3-lora=/path/to/checkpoint-6241 --max-lora-rank 32
```

### Step B: run EvalScope MMLU eval

In another terminal:

```bash
MMLU_PATH=/path/to/MMLU API_URL=http://127.0.0.1:8801/v1/chat/completions MODEL_NAME=DeepSeek-Qwen3-8b ./scripts/run_evalscope_mmlu.sh
```

Or call the python runner directly:

```bash
python scripts/eval_evalscope.py   --model DeepSeek-Qwen3-8b   --api-url http://127.0.0.1:8801/v1/chat/completions   --mmlu-path /path/to/MMLU   --batch 128
```

> `scripts/eval_evalscope.py` is a parameterized version of your `eval.py` (same TaskConfig fields, just made CLI-friendly).

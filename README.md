# LLM PEFT Framework

This repo is a **finetuning framework** for LLM using **PEFT adapters**.

It is structured so you can:
- Run baseline evaluation
- Run PEFT 
- Run Post-finetune evaluation 
- Merge adapter back into base weights

## 0) Install

```bash
conda create -n ftframe python=3.10
conda activate ftframe
pip install -r requirements.txt
```

## 1) Quickstart

### Baseline eval
```bash
bash scripts/serve_vllm.sh # Don't use enable-lora, lora-modules and max-lora-rank

# Another Terminal
python scripts/eval_evalscope.py
```

### LoRA finetune
```bash
# 单卡或常规Trainer分布式（torchrun/accelerate launch均可）
python scripts/train.py --config configs/train_lora.yaml

# 自定义Accelerate引擎（需要传 --engine accelerate）
accelerate launch scripts/train.py --config configs/train_lora.yaml --engine accelerate

# DeepSpeed Zero-2（示例）
accelerate launch scripts/train.py \
  --config configs/train_lora.yaml \
  --engine accelerate \
  distributed.deepspeed_config=configs/deepspeed_zero2.json
```

### Evaluate finetuned adapter
```bash
bash scripts/serve_vllm.sh

# Another Terminal
python scripts/eval_evalscope.py
```


## Distributed options
- 配置项：`run.engine`/`distributed.engine` 选择 `trainer` 或 `accelerate`；`distributed.deepspeed_config` 指向 DeepSpeed 配置（示例见 `configs/deepspeed_zero2.json`）；`distributed.mixed_precision` 控制混合精度（auto/bf16/fp16/no）。
- Trainer 路径使用 `TrainingArguments` 原生分布式/DeepSpeed 支持；Accelerate 路径使用自定义训练循环并内置断点保存、指标聚合。


## 2) What gets saved

Each run writes into `outputs/<run_name>/`:
- `run_config.yaml` (full resolved config snapshot)
- `trainer_state.json`, `train_results.json`, `eval_results.json`
- adapter weights/config (PEFT)
- optional `merged/` if enabled


## 3) Notes

- This framework uses **chat templates** via `tokenizer.apply_chat_template` and masks the prompt tokens so the loss is only on the assistant answer tokens.
- For MMLU, this repo provides two evaluation modes:
  - `first_token` (fast): compares the first unmasked label token vs argmax logits
  - `generate` (true MCQ): greedy-generate a few tokens and parse the first valid letter

## 4) TODO
- [ ] More dataset (only MMLU now)
- [ ] More PEFT method
- [x] DDP/DeepSpeed supported

> Thanks for Huggingface, vLLM, EvalScope, etc.

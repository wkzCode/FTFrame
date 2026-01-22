#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.data.collator import DataCollatorForCausalLMWithPadding
from src.data.mmlu import MMLUConfig, load_mmlu_splits, tokenize_mmlu_for_sft
from src.models.adapters import AdapterConfig, apply_lora_or_dora
from src.models.load_model import load_causal_lm, load_tokenizer
from src.utils.config import apply_overrides, load_yaml, save_yaml
from src.utils.io import save_json
from src.utils.logging import setup_logging
from src.utils.metrics import first_token_accuracy
from src.utils.seed import set_all_seeds


def _dict_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def _resolve_resume_checkpoint(out_dir: str, resume_arg):
    """
    resume_arg:
      - None: no resume
      - "auto": pick latest checkpoint under out_dir
      - path: resume from that checkpoint path
    """
    if resume_arg is None:
        return None

    # bool True -> auto
    if resume_arg is True:
        resume_arg = "auto"

    if isinstance(resume_arg, str):
        v = resume_arg.strip()
        if v.lower() in {"", "null", "none"}:
            return None
        if v.lower() == "auto":
            ckpt = get_last_checkpoint(out_dir)
            return ckpt
        # explicit path
        if os.path.isdir(v):
            return v
        # if user passes relative path, try under out_dir
        rel = os.path.join(out_dir, v)
        if os.path.isdir(rel):
            return rel
        raise ValueError(f"resume_from_checkpoint path not found: {v}")

    raise TypeError(f"Unsupported resume_from_checkpoint type: {type(resume_arg)}")

def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3-8B on MMLU with LoRA/DoRA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint. Use 'auto' to pick latest under outputs/<run_name>/, or pass a checkpoint dir path.",
    )
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides like training.learning_rate=1e-4")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    run_name = cfg["run"]["name"]
    output_root = cfg["run"]["output_root"]
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(out_dir, name=run_name)
    # If resuming and run_config.yaml already exists, don't overwrite it silently.
    run_cfg_path = os.path.join(out_dir, "run_config.yaml")
    if os.path.exists(run_cfg_path) and args.resume_from_checkpoint:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_yaml(cfg, os.path.join(out_dir, f"run_config_resume_{ts}.yaml"))
    else:
        save_yaml(cfg, run_cfg_path)

    seed = int(cfg["run"].get("seed", 42))
    set_all_seeds(seed)

    logger.info(f"Run: {run_name} | Output: {out_dir}")

    # tokenizer/model
    model_cfg = cfg.get("model", {})
    tokenizer = load_tokenizer(
        model_cfg["model_name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    model = load_causal_lm(
        model_cfg["model_name_or_path"],
        torch_dtype=str(model_cfg.get("torch_dtype", "auto")),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_flash_attn=bool(model_cfg.get("use_flash_attn", False)),
        attn_implementation=model_cfg.get("attn_implementation", None),
    )

    # dataset
    data_cfg = cfg.get("data", {})
    prompt_cfg = data_cfg.get("prompt", {})
    mmlu_cfg = MMLUConfig(
        dataset_name=str(data_cfg.get("dataset_name", "cais/mmlu")),
        dataset_config=str(data_cfg.get("dataset_config", "all")),
        train_split=str(data_cfg.get("train_split", "auxiliary_train")),
        eval_split=str(data_cfg.get("eval_split", "validation")),
        max_seq_len=int(data_cfg.get("max_seq_len", 1024)),
        num_proc=int(data_cfg.get("num_proc", 8)),
        prompt_template=str(prompt_cfg.get("template", "simple_exam")),
        enable_thinking=bool(prompt_cfg.get("enable_thinking", False)),
    )

    logger.info("Loading MMLU dataset...")
    train_raw, eval_raw = load_mmlu_splits(mmlu_cfg)
    logger.info(f"Train samples: {len(train_raw)} | Eval samples: {len(eval_raw)}")

    logger.info("Tokenizing train split...")
    train_ds = tokenize_mmlu_for_sft(train_raw, tokenizer, mmlu_cfg, split_name="train")
    logger.info("Tokenizing eval split...")
    eval_ds = tokenize_mmlu_for_sft(eval_raw, tokenizer, mmlu_cfg, split_name="eval")

    # adapters
    lora_cfg_raw = cfg.get("lora", {})
    adapter_cfg = AdapterConfig(
        enabled=bool(lora_cfg_raw.get("enabled", True)),
        r=int(lora_cfg_raw.get("r", 32)),
        alpha=int(lora_cfg_raw.get("alpha", 64)),
        dropout=float(lora_cfg_raw.get("dropout", 0.05)),
        bias=str(lora_cfg_raw.get("bias", "none")),
        target_modules=lora_cfg_raw.get("target_modules", "auto"),
    )

    enable_dora = bool(cfg.get("dora", {}).get("enabled", False))
    if enable_dora:
        logger.info("DoRA enabled (best-effort). If your peft lacks use_dora, it will fall back to LoRA.")

    model = apply_lora_or_dora(model, adapter_cfg, enable_dora=enable_dora)

    # Print trainable params if available
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # training args
    tcfg = cfg.get("training", {})

    # resume logic: CLI arg wins; else allow config key training.resume_from_checkpoint
    resume_value = args.resume_from_checkpoint
    if resume_value is None:
        resume_value = tcfg.get("resume_from_checkpoint", None)
    resume_ckpt = _resolve_resume_checkpoint(out_dir, resume_value)
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    use_bf16 = False
    use_fp16 = False
    # dtype auto: if cuda + bf16 supported, we loaded bf16. For Trainer flags, we follow hardware.
    try:
        import torch

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and (not use_bf16)
    except Exception:
        pass

    # If resuming, avoid overwrite_output_dir=True
    overwrite_output_dir = bool(tcfg.get("overwrite_output_dir", True))
    if resume_ckpt:
        overwrite_output_dir = False

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=overwrite_output_dir,

        num_train_epochs=float(tcfg.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),

        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),

        bf16=use_bf16,
        fp16=use_fp16,

        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_strategy=str(tcfg.get("eval_strategy", "steps")),
        eval_steps=int(tcfg.get("eval_steps", 10000)),
        save_strategy=str(tcfg.get("save_strategy", "steps")),
        save_steps=int(tcfg.get("save_steps", 1000)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),

        report_to=list(tcfg.get("report_to", ["tensorboard"])),
        remove_unused_columns=False,

        optim=str(tcfg.get("optim", "adamw_torch")),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        dataloader_pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=first_token_accuracy,
    )

    logger.info("Starting training...")
    # If resume_ckpt is None, this behaves like normal training.
    # If resume_ckpt is a path, Trainer will restore optimizer/scheduler/rng state, etc.
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    logger.info("Saving adapter + tokenizer...")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    save_json(train_metrics, os.path.join(out_dir, "train_results.json"))

    # logger.info("Running evaluation...")
    # eval_metrics = trainer.evaluate()
    # eval_metrics["eval_samples"] = len(eval_ds)
    # trainer.log_metrics("eval", eval_metrics)
    # trainer.save_metrics("eval", eval_metrics)
    # save_json(eval_metrics, os.path.join(out_dir, "eval_results.json"))

    # merge
    if bool(cfg.get("merge", {}).get("enabled", False)):
        logger.info("Merging adapter into base model...")
        merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        merged_dir = os.path.join(out_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to: {merged_dir}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

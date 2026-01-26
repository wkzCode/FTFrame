#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from datetime import datetime
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import Trainer, TrainingArguments, get_scheduler
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

    if resume_arg is True:
        resume_arg = "auto"

    if isinstance(resume_arg, str):
        v = resume_arg.strip()
        if v.lower() in {"", "null", "none"}:
            return None
        if v.lower() == "auto":
            ckpt = get_last_checkpoint(out_dir)
            return ckpt
        if os.path.isdir(v):
            return v
        rel = os.path.join(out_dir, v)
        if os.path.isdir(rel):
            return rel
        raise ValueError(f"resume_from_checkpoint path not found: {v}")

    raise TypeError(f"Unsupported resume_from_checkpoint type: {type(resume_arg)}")


def _build_mmlu(cfg: Dict[str, Any], tokenizer, logger):
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

    return train_ds, eval_ds, DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)


def _save_with_accelerate(accelerator: Accelerator, model, tokenizer, save_dir: str, logger, save_state: bool = True):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        unwrapped.save_pretrained(save_dir, save_function=accelerator.save, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Checkpoint saved to: {save_dir}")
    if save_state:
        accelerator.save_state(save_dir)


def _evaluate_first_token(accelerator: Accelerator, model, eval_dataloader: DataLoader, logger):
    model.eval()
    correct = torch.tensor(0, device=accelerator.device)
    total = torch.tensor(0, device=accelerator.device)

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

        labels = batch["labels"]
        valid_mask = (labels != -100).any(dim=1)
        if valid_mask.sum().item() == 0:
            continue

        first_idx = torch.argmax((labels != -100).int(), dim=1)
        batch_indices = torch.arange(labels.size(0), device=labels.device)
        preds = logits[batch_indices, first_idx, :].argmax(dim=-1)
        gold = labels[batch_indices, first_idx]

        preds = preds[valid_mask]
        gold = gold[valid_mask]
        correct += (preds == gold).sum()
        total += gold.numel()

    correct = accelerator.reduce(correct, reduction="sum")
    total = accelerator.reduce(total, reduction="sum")

    correct_val = correct.item()
    total_val = total.item()
    acc = correct_val / total_val if total_val > 0 else 0.0

    if accelerator.is_main_process:
        logger.info(f"Eval first_token_acc={acc:.4f} | samples={total_val}")

    model.train()
    return {"first_token_acc": acc, "eval_samples": total_val}


def train_with_accelerate(
    cfg: Dict[str, Any],
    model,
    tokenizer,
    train_ds,
    eval_ds,
    data_collator,
    out_dir: str,
    logger,
    resume_ckpt: str | None,
    use_bf16: bool,
    use_fp16: bool,
):
    tcfg = cfg.get("training", {})
    dist_cfg = cfg.get("distributed", {})

    grad_accum = int(tcfg.get("gradient_accumulation_steps", 1))
    per_device_train_bs = int(tcfg.get("per_device_train_batch_size", 1))
    per_device_eval_bs = int(tcfg.get("per_device_eval_batch_size", 1))
    dataloader_num_workers = int(tcfg.get("dataloader_num_workers", 0))

    num_epochs = float(tcfg.get("num_train_epochs", 1.0))
    learning_rate = float(tcfg.get("learning_rate", 2e-4))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.03))
    lr_scheduler_type = str(tcfg.get("lr_scheduler_type", "cosine"))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))

    eval_strategy = str(tcfg.get("eval_strategy", "none"))
    eval_steps = int(tcfg.get("eval_steps", 0))
    save_strategy = str(tcfg.get("save_strategy", "steps"))
    save_steps = int(tcfg.get("save_steps", 1000))
    report_to = list(tcfg.get("report_to", ["tensorboard"]))

    mixed_precision = dist_cfg.get("mixed_precision", "auto")
    if mixed_precision == "auto":
        if use_bf16:
            mixed_precision = "bf16"
        elif use_fp16:
            mixed_precision = "fp16"
        else:
            mixed_precision = "no"

    ds_config = dist_cfg.get("deepspeed_config", None) or tcfg.get("deepspeed_config", None)
    ds_plugin = None
    if ds_config:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
        logger.info(f"Using DeepSpeed config: {ds_config}")

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        log_with=report_to if report_to else None,
        project_dir=out_dir,
        mixed_precision=mixed_precision,
        deepspeed_plugin=ds_plugin,
    )

    if accelerator.is_main_process:
        logger.info(f"Accelerate initialized | devices={accelerator.num_processes} | mp={mixed_precision}")
        if report_to:
            accelerator.init_trackers(run_name=cfg["run"]["name"], config=cfg)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=per_device_train_bs,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=per_device_eval_bs,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
        pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
    )

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=learning_rate)

    # Prepare for distributed
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    max_train_steps = int(num_epochs * num_update_steps_per_epoch)
    num_warmup_steps = int(max_train_steps * warmup_ratio)

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if resume_ckpt:
        accelerator.load_state(resume_ckpt)
        if accelerator.is_main_process:
            logger.info(f"Loaded accelerator state from: {resume_ckpt}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    train_loss_sum = 0.0
    last_eval_metrics = {}

    logger.info("Starting training with Accelerate...")
    for epoch in range(int(math.ceil(num_epochs))):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            if completed_steps >= max_train_steps:
                break

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if max_grad_norm and max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                gathered_loss = accelerator.gather(loss.detach())
                train_loss_sum += gathered_loss.mean().item()
                completed_steps += 1
                progress_bar.update(1)

                if save_strategy == "steps" and save_steps > 0 and (completed_steps % save_steps == 0):
                    _save_with_accelerate(
                        accelerator,
                        model,
                        tokenizer,
                        os.path.join(out_dir, f"checkpoint-{completed_steps}"),
                        logger,
                        save_state=True,
                    )

                if eval_strategy == "steps" and eval_steps > 0 and (completed_steps % eval_steps == 0):
                    last_eval_metrics = _evaluate_first_token(accelerator, model, eval_dataloader, logger)
                    if report_to:
                        accelerator.log(last_eval_metrics, step=completed_steps)

        if eval_strategy == "epoch":
            last_eval_metrics = _evaluate_first_token(accelerator, model, eval_dataloader, logger)
            if report_to:
                accelerator.log(last_eval_metrics, step=completed_steps)

    accelerator.wait_for_everyone()
    _save_with_accelerate(accelerator, model, tokenizer, out_dir, logger, save_state=True)

    train_metrics = {
        "train_loss": train_loss_sum / max(1, completed_steps),
        "train_steps": completed_steps,
        "train_samples": len(train_ds),
    }
    if last_eval_metrics:
        train_metrics.update({f"last_{k}": v for k, v in last_eval_metrics.items()})

    if accelerator.is_main_process:
        save_json(train_metrics, os.path.join(out_dir, "train_results.json"))
        logger.info(f"Training done. Steps={completed_steps}")
        if report_to:
            accelerator.log(train_metrics, step=completed_steps)
            accelerator.end_training()

    # merge adapter if needed
    if bool(cfg.get("merge", {}).get("enabled", False)) and accelerator.is_main_process:
        logger.info("Merging adapter into base model...")
        unwrapped = accelerator.unwrap_model(model)
        merged = unwrapped.merge_and_unload() if hasattr(unwrapped, "merge_and_unload") else unwrapped
        merged_dir = os.path.join(out_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to: {merged_dir}")


def train_with_hf_trainer(
    cfg: Dict[str, Any],
    model,
    tokenizer,
    train_ds,
    eval_ds,
    data_collator,
    out_dir: str,
    logger,
    resume_ckpt: str | None,
    use_bf16: bool,
    use_fp16: bool,
):
    tcfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    dist_cfg = cfg.get("distributed", {})

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
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        deepspeed=dist_cfg.get("deepspeed_config", None) or tcfg.get("deepspeed_config", None),
        ddp_find_unused_parameters=bool(dist_cfg.get("ddp_find_unused_parameters", False)),
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

    logger.info("Starting training with HF Trainer...")
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

    if bool(cfg.get("merge", {}).get("enabled", False)):
        logger.info("Merging adapter into base model...")
        merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        merged_dir = os.path.join(out_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to: {merged_dir}")

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3-8B on MMLU with LoRA/DoRA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint. Use 'auto' to pick latest under outputs/<run_name>/, or pass a checkpoint dir path.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["trainer", "accelerate"],
        default=None,
        help="trainer: HuggingFace Trainer (default). accelerate: custom Accelerate loop supporting DeepSpeed.",
    )
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides like training.learning_rate=1e-4")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    run_cfg = cfg.get("run", {})
    run_name = run_cfg["name"]
    output_root = run_cfg["output_root"]
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)

    seed = int(run_cfg.get("seed", 42))
    set_all_seeds(seed)

    engine = args.engine or cfg.get("distributed", {}).get("engine") or run_cfg.get("engine") or "trainer"
    if engine not in {"trainer", "accelerate"}:
        raise ValueError(f"Unsupported engine: {engine}")

    # logging
    logger_name = run_name if is_main_process else f"{run_name}_rank{local_rank}"
    logger = setup_logging(out_dir, name=logger_name)

    logger.info(f"Run: {run_name} | Output: {out_dir} | Engine: {engine}")

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

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_ds, eval_ds, data_collator = _build_mmlu(cfg, tokenizer, logger)

    # resume logic: CLI arg wins; else allow config key training.resume_from_checkpoint
    resume_value = args.resume_from_checkpoint
    if resume_value is None:
        resume_value = _dict_get(cfg.get("training", {}), "resume_from_checkpoint", None)
    resume_ckpt = _resolve_resume_checkpoint(out_dir, resume_value)
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    # dtype choices for mixed precision
    use_bf16 = False
    use_fp16 = False
    try:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and (not use_bf16)
    except Exception:
        pass

    # Save run config (avoid overwrite when resuming)
    run_cfg_path = os.path.join(out_dir, "run_config.yaml")
    if is_main_process:
        if os.path.exists(run_cfg_path) and resume_ckpt:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_yaml(cfg, os.path.join(out_dir, f"run_config_resume_{ts}.yaml"))
        else:
            save_yaml(cfg, run_cfg_path)

    if engine == "accelerate":
        train_with_accelerate(cfg, model, tokenizer, train_ds, eval_ds, data_collator, out_dir, logger, resume_ckpt, use_bf16, use_fp16)
    else:
        train_with_hf_trainer(cfg, model, tokenizer, train_ds, eval_ds, data_collator, out_dir, logger, resume_ckpt, use_bf16, use_fp16)


if __name__ == "__main__":
    main()

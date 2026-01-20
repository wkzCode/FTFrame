#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Any, Dict, Optional

import torch

from src.data.collator import DataCollatorForCausalLMWithPadding
from src.data.mmlu import MMLUConfig, build_eval_prompts, load_mmlu_splits, make_texts_with_chat_template
from src.data.prompt_templates import build_prompt
from src.models.load_model import load_causal_lm, load_tokenizer
from src.models.peft_io import load_with_optional_adapter
from src.utils.config import apply_overrides, load_yaml, save_yaml
from src.utils.io import save_json
from src.utils.logging import setup_logging
from src.utils.seed import set_all_seeds


VALID_LETTERS = {"A", "B", "C", "D"}


def _parse_letter(text: str) -> Optional[str]:
    # find first occurrence of A/B/C/D as a standalone letter
    m = re.search(r"\b([ABCD])\b", text.strip())
    if m:
        return m.group(1)
    # fallback: look at first char
    if text and text[0].upper() in VALID_LETTERS:
        return text[0].upper()
    return None


@torch.inference_mode()
def eval_generate(
    model,
    tokenizer,
    ds,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    max_length: int,
) -> Dict[str, Any]:
    """Greedy generation + parse letter."""
    model.eval()

    correct = 0
    total = 0
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    for start in range(0, len(ds), batch_size):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = batch["prompt_text"]
        golds = batch["gold"]
        subjects = batch["subject"]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature and temperature > 0),
            temperature=float(temperature) if temperature else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_suffix = gen[:, enc["input_ids"].shape[1] :]
        outs = tokenizer.batch_decode(gen_suffix, skip_special_tokens=True)

        for out, gold, subj in zip(outs, golds, subjects):
            pred = _parse_letter(out)
            total += 1
            by_subject[subj]["total"] += 1
            if pred == gold:
                correct += 1
                by_subject[subj]["correct"] += 1

    overall = correct / total if total else 0.0
    per_subject = {
        k: (v["correct"] / v["total"] if v["total"] else 0.0)
        for k, v in by_subject.items()
    }

    return {
        "metric": "generate",
        "overall_acc": float(overall),
        "n": int(total),
        "per_subject": dict(sorted(per_subject.items(), key=lambda x: x[0])),
    }


def _build_sft_eval_dataset(eval_raw, tokenizer, cfg: MMLUConfig):
    """Build tokenized examples with labels for fast first-token accuracy."""

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["question"]
        choices = example["choices"]
        subject = example.get("subject", "") or ""
        # gold label in dataset is int or string
        gold = example["answer"]

        # resolve to A/B/C/D using same logic in src.data.mmlu
        from src.data.mmlu import resolve_answer_to_letter

        answer_letter = resolve_answer_to_letter(gold)

        user_prompt = build_prompt(cfg.prompt_template, subject, question, choices)
        texts = make_texts_with_chat_template(
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            assistant_answer=answer_letter,
            enable_thinking=cfg.enable_thinking,
        )
        prompt_text, full_text = texts["prompt_text"], texts["full_text"]

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=cfg.max_seq_len,
            add_special_tokens=False,
        )
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=cfg.max_seq_len,
            add_special_tokens=False,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = input_ids.copy()

        prompt_len = len(prompt["input_ids"])
        labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "subject": subject,
        }

    # Keep subject column for per-subject aggregation
        import os
    num_proc = max(1, int(cfg.num_proc or 1))
    num_proc = min(num_proc, os.cpu_count() or 1)

    return eval_raw.map(
        preprocess,
        remove_columns=eval_raw.column_names,
        desc="Tokenizing eval for first_token",
        num_proc=num_proc,
    )


@torch.inference_mode()
def eval_first_token(
    model,
    tokenizer,
    ds,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Compute first-token accuracy without generation."""
    model.eval()

    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    correct = 0
    total = 0
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    for start in range(0, len(ds), batch_size):
        batch_ds = ds.select(range(start, min(start + batch_size, len(ds))))
        subjects = batch_ds["subject"]
        collated = collator([{k: batch_ds[i][k] for k in ["input_ids", "attention_mask", "labels"]} for i in range(len(batch_ds))])

        collated = {k: v.to(device) for k, v in collated.items()}
        outputs = model(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
        )
        logits = outputs.logits  # [bs, seq, vocab]
        pred_ids = torch.argmax(logits, dim=-1)
        labels = collated["labels"]

        for i in range(labels.shape[0]):
            subj = subjects[i] or ""
            # first index where label != -100
            idxs = torch.nonzero(labels[i] != -100, as_tuple=False).view(-1)
            if idxs.numel() == 0:
                continue
            j = int(idxs[0].item())
            total += 1
            by_subject[subj]["total"] += 1
            if int(pred_ids[i, j].item()) == int(labels[i, j].item()):
                correct += 1
                by_subject[subj]["correct"] += 1

    overall = correct / total if total else 0.0
    per_subject = {
        k: (v["correct"] / v["total"] if v["total"] else 0.0)
        for k, v in by_subject.items()
    }

    return {
        "metric": "first_token",
        "overall_acc": float(overall),
        "n": int(total),
        "per_subject": dict(sorted(per_subject.items(), key=lambda x: x[0])),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMLU for baseline or adapter")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides like eval.adapter_path=... or eval.metric=first_token")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    run_name = cfg["run"]["name"]
    output_root = cfg["run"]["output_root"]
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(out_dir, name=run_name)
    save_yaml(cfg, os.path.join(out_dir, "run_config.yaml"))

    seed = int(cfg["run"].get("seed", 42))
    set_all_seeds(seed)

    ecfg = cfg["eval"]

    tokenizer = load_tokenizer(
        ecfg["model_name_or_path"],
        trust_remote_code=bool(ecfg.get("trust_remote_code", True)),
    )

    model = load_causal_lm(
        ecfg["model_name_or_path"],
        torch_dtype=str(ecfg.get("torch_dtype", "auto")),
        gradient_checkpointing=False,
        trust_remote_code=bool(ecfg.get("trust_remote_code", True)),
    )

    adapter_path = ecfg.get("adapter_path", None)
    model = load_with_optional_adapter(model, adapter_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_cfg = MMLUConfig(
        dataset_name=str(ecfg.get("dataset_name", "cais/mmlu")),
        dataset_config=str(ecfg.get("dataset_config", "all")),
        train_split="auxiliary_train",  # unused
        eval_split=str(ecfg.get("split", "validation")),
        max_seq_len=int(ecfg.get("max_seq_len", 1024)),
        num_proc=int(ecfg.get("num_proc", 8)),
        prompt_template=str(ecfg.get("prompt", {}).get("template", "simple_exam")),
        enable_thinking=bool(ecfg.get("prompt", {}).get("enable_thinking", False)),
    )

    logger.info("Loading eval split...")
    _train_unused, eval_raw = load_mmlu_splits(data_cfg)
    logger.info(f"Eval samples: {len(eval_raw)}")

    metric = str(ecfg.get("metric", "generate"))

    if metric == "generate":
        eval_ds = build_eval_prompts(eval_raw, tokenizer, data_cfg)
        results = eval_generate(
            model=model,
            tokenizer=tokenizer,
            ds=eval_ds,
            batch_size=int(ecfg.get("per_device_batch_size", 4)),
            max_new_tokens=int(ecfg.get("max_new_tokens", 5)),
            temperature=float(ecfg.get("temperature", 0.0)),
            device=device,
            max_length=int(ecfg.get("prompt_max_length", 2048)),
        )
    elif metric == "first_token":
        eval_ds = _build_sft_eval_dataset(eval_raw, tokenizer, data_cfg)
        results = eval_first_token(
            model=model,
            tokenizer=tokenizer,
            ds=eval_ds,
            batch_size=int(ecfg.get("per_device_batch_size", 8)),
            device=device,
        )
    else:
        raise ValueError("eval.metric must be one of: generate, first_token")

    logger.info(f"Overall accuracy: {results['overall_acc']:.4f} (n={results['n']})")

    save_json(results, os.path.join(out_dir, "mmlu_results.json"))
    logger.info(f"Saved: {os.path.join(out_dir, 'mmlu_results.json')}")


if __name__ == "__main__":
    main()

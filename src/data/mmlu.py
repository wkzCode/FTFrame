from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from .prompt_templates import build_prompt


def resolve_answer_to_letter(ans: Union[int, str]) -> str:
    """Resolve labels to A/B/C/D like your current script."""
    if isinstance(ans, int):
        mapping = ["A", "B", "C", "D"]
        if 0 <= ans < 4:
            return mapping[ans]
        raise ValueError(f"Unexpected int answer: {ans}")
    if isinstance(ans, str):
        ans = ans.strip()
        if ans in ["A", "B", "C", "D"]:
            return ans
        if ans.isdigit():
            return resolve_answer_to_letter(int(ans))
        raise ValueError(f"Unexpected str answer: {ans}")
    raise TypeError(f"Unexpected answer type: {type(ans)}")


def make_texts_with_chat_template(
    tokenizer: Any,
    user_prompt: str,
    assistant_answer: str,
    enable_thinking: bool = False,
) -> Dict[str, str]:
    """Build prompt_text / full_text using chat template."""
    prompt_messages = [{"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    full_messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_answer},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return {"prompt_text": prompt_text, "full_text": full_text}


@dataclass
class MMLUConfig:
    dataset_name: str = "cais/mmlu"
    dataset_config: str = "all"
    train_split: str = "auxiliary_train"
    eval_split: str = "validation"
    max_seq_len: int = 1024
    num_proc: int = 8

    prompt_template: str = "simple_exam"
    enable_thinking: bool = False


def load_mmlu_splits(cfg: MMLUConfig):
    train_ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.train_split)
    eval_ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.eval_split)
    return train_ds, eval_ds


def tokenize_mmlu_for_sft(ds, tokenizer: Any, cfg: MMLUConfig, split_name: str = "train"):
    """Tokenize MMLU into (input_ids, attention_mask, labels).

    The loss is masked on the prompt portion, so it's only on assistant answer tokens.
    This matches your current approach.
    """

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["question"]
        choices = example["choices"]
        subject = example.get("subject", None)

        answer_letter = resolve_answer_to_letter(example["answer"])

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

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    num_proc = max(1, int(cfg.num_proc or 1))
    num_proc = min(num_proc, os.cpu_count() or 1)

    return ds.map(
        preprocess,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {split_name}",
        num_proc=num_proc,
    )


def build_eval_prompts(ds, tokenizer: Any, cfg: MMLUConfig):
    """For generation-based eval, build prompt_text (user only, assistant generation prefix) and labels."""

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["question"]
        choices = example["choices"]
        subject = example.get("subject", None)
        gold = resolve_answer_to_letter(example["answer"])

        user_prompt = build_prompt(cfg.prompt_template, subject, question, choices)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        return {"prompt_text": prompt_text, "gold": gold, "subject": subject or ""}

    num_proc = max(1, int(cfg.num_proc or 1))
    num_proc = min(num_proc, os.cpu_count() or 1)

    return ds.map(
        preprocess,
        remove_columns=ds.column_names,
        desc="Building eval prompts",
        num_proc=num_proc,
    )

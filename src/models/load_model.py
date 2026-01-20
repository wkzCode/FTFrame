from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {dtype}")


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    torch_dtype: str = "auto",
    use_flash_attn: bool = False,
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = True,
):
    dtype = _resolve_dtype(torch_dtype)
    attn_impl = attn_implementation
    if attn_impl is None and use_flash_attn:
        # HF uses this name for FlashAttention 2
        attn_impl = "flash_attention_2"

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    # Some custom model classes (trust_remote_code) or older Transformers versions
    # might not accept attn_implementation as a kwarg. Fallback gracefully.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        # Best-effort: set the config field if present.
        if attn_impl and hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = attn_impl
        elif attn_impl and hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = attn_impl

    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union

import torch


def detect_lora_target_modules(model: torch.nn.Module) -> List[str]:
    """Auto-detect common projection module names.

    This matches your current helper.
    """
    common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    existing = set()
    for name, _module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in common:
            existing.add(leaf)
    targets = [x for x in common if x in existing]
    if not targets:
        raise RuntimeError(
            "Could not auto-detect LoRA target modules. "
            "Please inspect model.named_modules() and set target_modules manually."
        )
    return targets


@dataclass
class AdapterConfig:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Union[str, List[str]] = "auto"


def apply_lora_or_dora(
    model: torch.nn.Module,
    cfg: AdapterConfig,
    enable_dora: bool = False,
):
    """Apply LoRA (and optionally DoRA) via PEFT.

    - If `enable_dora=True` but your installed PEFT lacks DoRA support, we fall back to LoRA.
    """
    if not cfg.enabled:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "peft is required. Please `pip install peft>=0.12.0`."
        ) from e

    if cfg.target_modules == "auto":
        target_modules = detect_lora_target_modules(model)
    else:
        target_modules = list(cfg.target_modules)

    kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.r),
        lora_alpha=int(cfg.alpha),
        lora_dropout=float(cfg.dropout),
        bias=str(cfg.bias),
        target_modules=target_modules,
    )

    # DoRA support (best-effort).
    if enable_dora:
        sig = inspect.signature(LoraConfig.__init__)
        if "use_dora" in sig.parameters:
            kwargs["use_dora"] = True
        else:
            # fallback silently; caller should log
            pass

    lora_config = LoraConfig(**kwargs)
    model = get_peft_model(model, lora_config)
    return model

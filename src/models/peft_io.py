from __future__ import annotations

from typing import Optional

import torch


def load_with_optional_adapter(model, adapter_path: Optional[str]):
    if adapter_path in (None, "", "null"):
        return model

    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft is required to load adapters.") from e

    return PeftModel.from_pretrained(model, adapter_path)

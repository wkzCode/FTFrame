from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class DataCollatorForCausalLMWithPadding:
    """Pads input_ids/attention_mask/labels to batch max length.

    Mirrors the logic from your current script.
    """

    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lb in labels:
            if len(lb) < max_len:
                lb = lb + [self.label_pad_token_id] * (max_len - len(lb))
            else:
                lb = lb[:max_len]
            padded_labels.append(lb)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

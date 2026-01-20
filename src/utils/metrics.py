from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def first_token_accuracy(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Accuracy on the first unmasked label token.

    This mirrors the fast metric in your current script.
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_ids = np.argmax(predictions, axis=-1)
    correct = 0
    total = 0

    for i in range(labels.shape[0]):
        idxs = np.where(labels[i] != -100)[0]
        if len(idxs) == 0:
            continue
        j = int(idxs[0])
        total += 1
        if int(pred_ids[i, j]) == int(labels[i, j]):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return {"first_token_acc": float(acc), "eval_samples": float(total)}

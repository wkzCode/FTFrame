from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def _coerce_value(v: str) -> Any:
    # CLI override parsing
    low = v.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"

    # int/float
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dotlist overrides like: training.learning_rate=1e-4"""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item}")
        key, val = item.split("=", 1)
        keys = key.split(".")
        cur = out
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = _coerce_value(val)
    return out


def save_yaml(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

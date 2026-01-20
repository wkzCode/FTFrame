#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

from src.models.load_model import load_causal_lm, load_tokenizer
from src.models.peft_io import load_with_optional_adapter


def main():
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into base model and save")
    parser.add_argument("--base", type=str, required=True, help="Base model path or HF id")
    parser.add_argument("--adapter", type=str, required=True, help="Adapter path")
    parser.add_argument("--out", type=str, required=True, help="Output dir")
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tokenizer = load_tokenizer(args.base, trust_remote_code=args.trust_remote_code)
    model = load_causal_lm(
        args.base,
        torch_dtype=args.torch_dtype,
        gradient_checkpointing=False,
        trust_remote_code=args.trust_remote_code,
    )

    model = load_with_optional_adapter(model, args.adapter)

    if not hasattr(model, "merge_and_unload"):
        raise RuntimeError("Loaded model does not support merge_and_unload (is this a PEFT model?)")

    merged = model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    print(f"Merged model saved to: {args.out}")


if __name__ == "__main__":
    main()

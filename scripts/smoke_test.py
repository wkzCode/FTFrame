#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Quick smoke test: load tokenizer+model and run one forward pass."""

import argparse
import torch

from src.models.load_model import load_causal_lm, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    model = load_causal_lm(args.model, trust_remote_code=args.trust_remote_code, gradient_checkpointing=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text = tok.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = model(**enc)

    print("OK. logits shape:", tuple(out.logits.shape))


if __name__ == "__main__":
    main()

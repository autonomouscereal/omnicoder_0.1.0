from __future__ import annotations

import argparse
import json
from pathlib import Path


def exact_match(pred: str, ref: str) -> bool:
    return pred.strip() == ref.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal text exact-match eval from JSONL")
    ap.add_argument("--tasks", type=str, required=True, help="JSONL with {prompt, reference, candidate?}")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(args.tasks, "r", encoding="utf-8", errors="ignore") if l.strip()]
    matched = 0
    total = 0
    for ex in data:
        # Expect a structure like {prompt, reference, candidates:[...]?}
        ref = ex.get("reference") or ex.get("answer") or ""
        if not ref:
            continue
        total += 1
        # Use provided candidate if present; this tool doesn't run the model
        cand = ex.get("candidate") or ex.get("prediction") or ""
        matched += int(exact_match(cand, ref))
    print(f"exact_match: {matched}/{total} = {(matched/total if total else 0.0):.3f}")


if __name__ == "__main__":
    main()

import argparse, json
from pathlib import Path

import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.inference.generate import generate
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=str, required=True, help='JSONL with {input, target}')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--ckpt', type=str, default='')
    args = ap.parse_args()

    tokenizer = get_text_tokenizer(prefer_hf=True)
    model = OmniTransformer()
    if args.ckpt:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False)
    model.to(args.device)

    total, correct = 0, 0
    with open(args.tasks, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            prompt = ex['input']
            target = ex['target']
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            out_ids = generate(model, input_ids, max_new_tokens=args.max_new_tokens)
            pred = tokenizer.decode(out_ids[0].tolist())
            total += 1
            correct += int(target.strip() in pred)
    print(f"exact_match_contains: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()

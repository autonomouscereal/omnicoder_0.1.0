from __future__ import annotations

"""
Expert-branching decoding (prototype):

At each step, propose W candidate next tokens from the base model logits and
rank them using a verifier head when present (fallback to base probability).
Picks the best candidate, appends, and repeats. This approximates a
beam-in-one-pass style strategy without external draft models.
"""

import argparse
import math
from typing import List, Optional, Tuple

import torch


def _topk_candidates(logits: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    probs = torch.softmax(logits, dim=-1)
    k = max(1, min(int(k), int(probs.size(-1))))
    p, idx = torch.topk(probs, k=k, dim=-1)
    return idx.squeeze(0).tolist(), p.squeeze(0).tolist()


def expert_branch_decode(
    prompt: str,
    max_new_tokens: int = 64,
    width: int = 3,
    device: str = "cpu",
    mobile_preset: str = "mobile_4gb",
) -> str:
    from omnicoder.inference.generate import build_mobile_model_by_name  # type: ignore
    from omnicoder.training.simple_tokenizer import get_text_tokenizer  # type: ignore

    model = build_mobile_model_by_name(mobile_preset, mem_slots=4)
    model.eval().to(device)
    tok = get_text_tokenizer(prefer_hf=True)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out_tokens: List[int] = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(ids)
        # Unified outputs API (logits first)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            verifier = None
            for extra in outputs[1:]:
                if isinstance(extra, torch.Tensor) and extra.dim() == 3 and extra.size(-1) == logits.size(-1):
                    verifier = extra[:, -1:, :]  # last-step verifier logits
                    break
        else:
            logits = outputs
            verifier = None

        step_logits = logits[:, -1:, :]  # (1,1,V)
        cand_ids, cand_p = _topk_candidates(step_logits.squeeze(1), width)
        # Score with verifier when present; else base prob
        scores: List[float] = []
        if verifier is not None:
            v = torch.softmax(verifier, dim=-1).squeeze(1)  # (1,V)
            if cand_ids:
                idx = torch.tensor(cand_ids, dtype=torch.long, device=v.device)
                scores = v[0, idx].detach().tolist()
            else:
                scores = []
        else:
            scores = cand_p
        # pick best
        best_idx = max(enumerate(scores), key=lambda kv: kv[1])[0]
        next_id = cand_ids[best_idx]
        out_tokens.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return tok.decode(out_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expert-branching decoding (rank W candidates by verifier/base probs)")
    ap.add_argument("--prompt", type=str, default="Hello")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    args = ap.parse_args()
    text = expert_branch_decode(prompt=args.prompt, max_new_tokens=int(args.max_new_tokens), width=int(args.width), device=args.device, mobile_preset=args.mobile_preset)
    print(text)


if __name__ == "__main__":
    main()



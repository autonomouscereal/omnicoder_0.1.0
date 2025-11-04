from __future__ import annotations

"""
Reward model (RM) trainer for RLHF-style preference data.

Expects a JSONL file with pairwise preferences:
 {"prompt": str, "chosen": str, "rejected": str}

Trains a small scalar reward head on top of OmniTransformer embeddings via a
pairwise Bradley-Terry objective: r(chosen) > r(rejected).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.inference.generate import build_mobile_model_by_name


def _load_pairs(path: str) -> List[Tuple[str,str,str]]:
    items: List[Tuple[str,str,str]] = []
    p = Path(path)
    if not p.exists():
        return items
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            pr = str(obj.get("prompt", ""))
            ch = str(obj.get("chosen", ""))
            rj = str(obj.get("rejected", ""))
            if pr and ch and rj:
                items.append((pr, ch, rj))
        except Exception:
            continue
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Reward model trainer (pairwise)")
    ap.add_argument("--data", type=str, required=True, help="JSONL with {prompt, chosen, rejected}")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--preset", type=str, default="mobile_4gb")
    ap.add_argument("--out", type=str, default="weights/reward_model.pt")
    args = ap.parse_args()

    pairs = _load_pairs(args.data)
    if not pairs:
        print("[rm_train] no data; exiting")
        return

    tok = get_text_tokenizer(prefer_hf=True)
    model: OmniTransformer = build_mobile_model_by_name(args.preset)
    # Add a tiny reward head on top of final hidden mean
    d = int(model.d_model) if hasattr(model, 'd_model') else 512
    reward_head = nn.Linear(d, 1)
    model.to(args.device).eval()  # keep frozen
    reward_head.to(args.device).train()
    opt = torch.optim.AdamW(reward_head.parameters(), lr=args.lr)

    def _score(text: str) -> torch.Tensor:
        ids = torch.tensor([tok.encode(text)[:512]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(ids, use_cache=False)
            hid = out[-1] if isinstance(out, tuple) else model.embed(ids)  # best-effort hidden
        if isinstance(hid, torch.Tensor) and hid.dim() == 3:
            h = hid.mean(dim=1)  # (B, D)
        else:
            h = model.embed(ids).mean(dim=1)
        return reward_head(h).squeeze(-1)  # (B,)

    step = 0
    import random
    for step in range(1, int(args.steps) + 1):
        pr, ch, rj = random.choice(pairs)
        r_pos = _score(pr + "\n" + ch)
        r_neg = _score(pr + "\n" + rj)
        # Bradley-Terry pairwise loss: log(sigmoid(r_pos - r_neg))
        loss = -torch.log(torch.sigmoid(r_pos - r_neg) + 1e-8).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_head.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[rm] step {step}/{args.steps} loss={float(loss.item()):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _safe_save({"reward_head": reward_head.state_dict(), "d_model": d, "preset": args.preset}, args.out)
    print(f"[rm] saved reward model head to {args.out}")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Train a small reward model on pairs or triples with preferences.
Input JSONL: {"prompt": "...", "candidates": ["a","b",...], "preferred": 0}
The reward model scores text continuations; we optimize cross-entropy over pairwise comparisons.
"""

import argparse
import json
from typing import List

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _load(jsonl: str) -> List[dict]:
    return [json.loads(l) for l in open(jsonl, 'r', encoding='utf-8', errors='ignore') if l.strip()]


class RewardHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, C) -> score last token
        last = hidden[:, -1, :]
        return self.scorer(last).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a reward model for preferences')
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--out', type=str, default='weights/reward_model.pt')
    args = ap.parse_args()

    tok = get_text_tokenizer(prefer_hf=True)
    from omnicoder.inference.generate import build_mobile_model_by_name
    base = build_mobile_model_by_name(args.mobile_preset)
    base.to(args.device)
    head = RewardHead(d_model=base.ln_f.normalized_shape[0]).to(args.device)
    # Resume from best-known if present
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        _loaded = load_best_or_latest(base, args.out)
        if _loaded is not None:
            print(f"[resume] loaded {_loaded}")
    except Exception:
        pass
    opt = torch.optim.AdamW(list(base.parameters()) + list(head.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    data = _load(args.data)
    step = 0
    for epoch in range(10**9):
        for ex in data:
            prompt = ex.get('prompt', '')
            cands = ex.get('candidates', [])
            pref = int(ex.get('preferred', 0))
            if not cands:
                continue
            scores = []
            for c in cands:
                text = prompt + c
                ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=args.device)
                out = base(ids, return_hidden=True)
                if isinstance(out, tuple) and len(out) >= 2:
                    hidden = out[-1]
                else:
                    # derive hidden via last pre-head layer if needed
                    hidden = base.ln_f(base.embed(ids))  # type: ignore
                s = head(hidden)
                scores.append(s)
            scores_t = torch.stack(scores, dim=0).squeeze(-1)  # (K,)
            target = torch.tensor([pref], dtype=torch.long, device=args.device)
            loss = loss_fn(scores_t.unsqueeze(0), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(base.parameters()) + list(head.parameters()), max_norm=1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss={loss.item():.4f}")
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    state = {'base': base.state_dict(), 'head': head.state_dict()}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, state, meta={'train_args': {'steps': int(args.steps)}})
    else:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        import torch
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in state.items()}, args.out)
        final = args.out
    # Best by last loss (lower is better)
    try:
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        if 'loss' in locals():
            maybe_save_best(args.out, base, 'reward_loss', float(loss.item()), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved reward model to {final}")


if __name__ == '__main__':
    main()



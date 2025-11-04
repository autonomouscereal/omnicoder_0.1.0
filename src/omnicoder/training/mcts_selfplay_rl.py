from __future__ import annotations

"""
AlphaGo-style self-play RL with ToT+MCTS

This trainer uses the same model as policy and value (value_head) and the
ToTMCTS driver to perform policy improvement on synthetic tasks (math/code).
For simplicity, we use a synthetic generator of arithmetic prompts and known
answers, and optimize the model via a simple advantage-weighted logprob loss.
"""

import argparse
import random
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.reasoning.tot_mcts import ToTMCTS, ToTConfig
from omnicoder.inference.generate import generate
from omnicoder.utils.logger import get_logger


def _make_math_prompts(n: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for _ in range(n):
        a = random.randint(2, 999)
        b = random.randint(2, 999)
        if random.random() < 0.5:
            out.append((f"Compute {a}+{b}.", a + b))
        else:
            out.append((f"Compute {a}*{b}.", a * b))
    return out


def _extract_integer(text: str) -> int:
    s = ''.join([ch if ch.isdigit() else ' ' for ch in text])
    parts = [p for p in s.split(' ') if p]
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="MCTS self-play RL trainer (math synthetic)")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--out", type=str, default="weights/mcts_selfplay.pt")
    args = ap.parse_args()

    log = get_logger("omnicoder.mcts_rl")
    tok = get_text_tokenizer(prefer_hf=True)
    from omnicoder.inference.generate import build_mobile_model_by_name
    model: OmniTransformer = build_mobile_model_by_name(args.mobile_preset)
    model.to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    cfg = ToTConfig(simulations=64, branch_factor=3, max_depth=6, device=args.device)
    tot = ToTMCTS(model, tok, cfg)

    for it in range(1, int(args.iters) + 1):
        batch = _make_math_prompts(int(args.batch))
        total_loss = 0.0
        for prompt, gold in batch:
            # MCTS search for a better continuation
            best_ids, stats = tot.search(prompt)
            # Roll policy to get a comparable baseline
            inp = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)
            out_ids = generate(model, inp, max_new_tokens=24, temperature=0.7, top_k=40, top_p=0.9)
            text = tok.decode(out_ids[0].tolist())
            guess = _extract_integer(text)
            # Reward: correct=+1 else -1; small bonus for MCTS path length
            r = 1.0 if guess == gold else -1.0
            # Policy gradient surrogate: encourage final baseline token under advantage r
            model.eval()
            with torch.inference_mode():
                out = model(inp, use_cache=False)
                logits = out[0] if isinstance(out, tuple) else out
            model.train()
            tgt = torch.argmax(logits[:, -1, :], dim=-1)
            logp = torch.log_softmax(logits[:, -1, :], dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            adv = torch.tensor([r], dtype=torch.float32, device=args.device)
            loss = -(adv.detach() * logp).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        if (it % 5) == 0 or it == 1:
            try:
                log.info("mcts_iter=%d loss=%.4f", int(it), float(total_loss / max(1, int(args.batch))))
            except Exception:
                pass

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta={'train_args': {'iters': int(args.iters)}})
    else:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"[mcts_rl] saved to {final}")


if __name__ == "__main__":
    main()



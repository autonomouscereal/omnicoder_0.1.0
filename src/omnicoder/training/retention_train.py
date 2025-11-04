from __future__ import annotations

"""
Train a simple KV/memory retention head to learn compressive slots + window policy.

This trainer optimizes a token-level retention score produced by the model such that
older prefix tokens are down-weighted (compressed into a fixed number of slots), while
the most recent window is retained at full resolution.

Outputs:
- weights/kv_retention.json: {
    "compressive_slots": int,
    "window_size": int,
    "slots": int,           # backward-compat
    "window": int           # backward-compat
  } sidecar consumed by runners/exporters
- Optional: checkpoint with updated retention head (if applicable)
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from omnicoder.modeling.transformer_moe import OmniTransformer


def build_targets(seq_len: int, window: int, slots: int, device: torch.device) -> torch.Tensor:
    """Construct a target retention score per position in [0,1].

    - Most recent `window` tokens: 1.0 (keep)
    - Older prefix: a soft ramp toward 0 based on `slots`
    """
    t = torch.linspace(0, 1, steps=seq_len, device=device)
    # Newest near 1, oldest near 0
    ramp = t
    if window > 0:
        keep = torch.zeros(seq_len, device=device)
        keep[-window:] = 1.0
    else:
        keep = torch.zeros(seq_len, device=device)
    # Soften ramp by slot count (more slots → slower decay)
    decay = torch.pow(1.0 - ramp, 1.0 / max(1.0, float(slots)))
    target = torch.maximum(keep, decay)
    return target  # (T,)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train retention head for KV/memory compression policy")
    ap.add_argument("--seq_len", type=int, default=int(os.getenv("OMNICODER_RET_SEQ_LEN", "2048")))
    ap.add_argument("--window", type=int, default=int(os.getenv("OMNICODER_RET_WINDOW", "512")))
    ap.add_argument("--slots", type=int, default=int(os.getenv("OMNICODER_RET_SLOTS", "8")))
    ap.add_argument("--steps", type=int, default=int(os.getenv("OMNICODER_RET_STEPS", "200")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_RET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--lr", type=float, default=float(os.getenv("OMNICODER_RET_LR", "3e-4")))
    ap.add_argument("--mobile_preset", type=str, default=os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb"))
    ap.add_argument("--out", type=str, default=os.getenv("OMNICODER_RET_OUT", "weights/kv_retention.json"))
    ap.add_argument("--save_ckpt", type=str, default=os.getenv("OMNICODER_RET_CKPT", ""))
    args = ap.parse_args()

    device = torch.device(args.device)
    # Small model with retention head present
    model = OmniTransformer()
    # Resume from best-known
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        _loaded = load_best_or_latest(model, args.save_ckpt or 'weights/kv_retention.pt')
        if _loaded is not None:
            print(f"[resume] loaded {_loaded}")
    except Exception:
        pass
    model.to(device).train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    bce = nn.BCEWithLogitsLoss()

    # Synthetic tokens (no need for tokenizer)
    B = 1
    T = int(args.seq_len)
    vocab = getattr(model, 'vocab_size', 32000)

    for step in range(int(args.steps)):
        ids = torch.randint(0, vocab, (B, T), dtype=torch.long, device=device)
        out = model(ids, past_kv=None, use_cache=False, return_hidden=False)
        # Expect retention score among outputs; handle tuples robustly
        if isinstance(out, tuple) and len(out) >= 7:
            retention = out[6]
        else:
            # If retention head is unavailable, stop early
            print("[warn] model does not expose retention_score; exiting")
            break
        # retention shape: (B,T,1) or (B,T)
        if retention.dim() == 3 and retention.size(-1) == 1:
            retention = retention.squeeze(-1)
        target = build_targets(T, int(args.window), int(args.slots), device=device).unsqueeze(0)
        loss = bce(retention, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 10 == 0:
            print(f"[ret] step {step+1}/{args.steps} loss={loss.item():.4f}")

    # Write a retention sidecar for runners/exporters
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "compressive_slots": int(args.slots),
        "window_size": int(args.window),
        # Backward-compat keys
        "slots": int(args.slots),
        "window": int(args.window),
    }
    Path(args.out).write_text(json.dumps(sidecar, indent=2), encoding='utf-8')
    print(f"[ret] wrote retention sidecar to {args.out}")

    # Optional: save model checkpoint with updated retention head
    if args.save_ckpt:
        try:
            from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
        except Exception:
            save_with_sidecar = None  # type: ignore
            maybe_save_best = None  # type: ignore
        try:
            if callable(save_with_sidecar):
                final = save_with_sidecar(args.save_ckpt, model.state_dict(), meta={'train_args': {'steps': int(args.steps)}, 'retention': sidecar})
            else:
                from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
                _safe_save(model.state_dict(), args.save_ckpt)
                final = Path(args.save_ckpt)
            # Best by lower retention BCE on last batch (loss variable)
            try:
                if callable(maybe_save_best) and 'loss' in locals():
                    maybe_save_best(args.save_ckpt, model, 'retention_bce', float(loss.item()), higher_is_better=False)
            except Exception:
                pass
            print(f"[ret] saved checkpoint to {final}")
        except Exception as e:
            print(f"[warn] could not save checkpoint: {e}")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Train the retrieval write-policy head using teacher marks.

Input: JSONL with entries {"text": str, "write_marks": [indices]}
We tokenize text, run the model forward to get hidden states, then optimize
the write head (sigmoid) to match marks using BCE loss.

Logs acceptance ratio and loss periodically, and saves the tuned checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.data.datamodule import DataModule


def main() -> None:
    ap = argparse.ArgumentParser(description="Train write-policy head from teacher marks")
    ap.add_argument("--marks", type=str, required=True, help="Path to JSONL with {text, write_marks}")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="weights/omnicoder_write_head.pt")
    args = ap.parse_args()

    # Dataset loader
    dm = DataModule(train_folder=".", seq_len=int(args.seq_len), batch_size=int(args.batch_size))
    dl: DataLoader | None = dm.teacher_marks_loader(args.marks)
    if dl is None:
        raise RuntimeError(f"Could not build teacher marks loader from {args.marks}")

    # Model
    model = OmniTransformer()
    model.to(args.device).train()
    # Optimize only the write head
    for n, p in model.named_parameters():
        if "write_head" not in n:
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    bce = nn.BCEWithLogitsLoss()

    step = 0
    total_accept = 0.0
    total_marks = 0.0
    for batch in dl:
        step += 1
        ids: torch.Tensor = batch["input_ids"].to(args.device)  # (B,T)
        marks: torch.Tensor = batch["write_marks"].to(args.device)  # (B,T)

        # Run model to obtain hidden states aligned to tokens
        out = model(ids, use_cache=False, return_hidden=True)
        if isinstance(out, tuple):
            # (logits, mtp_logits?, hidden)
            hidden = out[-1]
            if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                # fallback: derive hidden from pre-projection
                hidden = model.ln_f(model.embed(ids))
        else:
            hidden = model.ln_f(model.embed(ids))

        logits_write = model.write_head(hidden).squeeze(-1)  # (B,T)
        loss = bce(logits_write, marks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            pred = (torch.sigmoid(logits_write) > 0.5).float()
            total_accept += float(pred.sum().item())
            total_marks += float(marks.sum().item())

        if step % 20 == 0:
            ratio = (total_accept / max(1.0, total_marks)) if total_marks > 0 else 0.0
            print(json.dumps({
                "step": int(step),
                "loss": float(loss.item()),
                "accept_to_mark_ratio": float(ratio),
            }))
        if step >= int(args.steps):
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    payload = {"write_head": model.write_head.state_dict()}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, payload, meta={'train_args': {'steps': int(args.steps)}})
    else:
        _safe_save(payload, args.out)
        final = args.out
    try:
        if callable(maybe_save_best) and 'loss' in locals():
            maybe_save_best(args.out, model.write_head, 'write_head_bce', float(loss.item()), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved write-head to {final}")


if __name__ == "__main__":
    main()



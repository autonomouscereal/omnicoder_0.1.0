from __future__ import annotations

"""
Distilled verifier training utility.

Trains a simple linear head on frozen hidden states to approximate an external
verifier signal (e.g., acceptance probability or token-level margin). Produces
a checkpoint consumable by AGoT via OMNICODER_VERIFIER_DISTILL_WEIGHTS.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple
import json
import os
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainConfig:
    dim: int
    vocab: int
    lr: float = 1e-2
    steps: int = 2000
    batch_size: int = 64
    out_path: str = 'weights/verifier_distill.pt'


class LinearVerifier(nn.Module):
    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.W = nn.Linear(dim, vocab, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B,C) -> (B,V)
        return self.W(h)


def iter_jsonl_hidden_stream(path: str, max_rows: int | None = None) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Yields (hidden, targets) from a JSONL with fields: hidden:[float], targets:[float]."""
    count = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if max_rows is not None and count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                h = torch.tensor(obj['hidden'], dtype=torch.float32)
                y = torch.tensor(obj['targets'], dtype=torch.float32)
                yield h, y
                count += 1
            except Exception:
                continue


def train_from_jsonl(cfg: TrainConfig, data_path: str) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinearVerifier(cfg.dim, cfg.vocab).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    batch_h: list[torch.Tensor] = []
    batch_y: list[torch.Tensor] = []
    step = 0
    for h, y in iter_jsonl_hidden_stream(data_path):
        batch_h.append(h)
        batch_y.append(y)
        if len(batch_h) >= cfg.batch_size:
            H = torch.stack(batch_h).to(device)
            Y = torch.stack(batch_y).to(device)
            logits = model(H)
            loss = loss_fn(logits, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch_h.clear()
            batch_y.clear()
            step += 1
            if step >= cfg.steps:
                break
    out = cfg.out_path
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    sd = {
        'weight': model.W.weight.detach().cpu(),
        'bias': model.W.bias.detach().cpu(),
        'dim': cfg.dim,
        'vocab': cfg.vocab,
    }
    _safe_save(sd, out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--vocab', type=int, required=True)
    ap.add_argument('--data', type=str, required=True, help='JSONL path with {hidden:[], targets:[]}')
    ap.add_argument('--out', type=str, default='weights/verifier_distill.pt')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-2)
    args = ap.parse_args()
    cfg = TrainConfig(dim=args.dim, vocab=args.vocab, lr=args.lr, steps=args.steps, batch_size=args.batch, out_path=args.out)
    path = train_from_jsonl(cfg, data_path=args.data)
    print(path)



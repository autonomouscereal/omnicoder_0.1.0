from __future__ import annotations

"""
Microbench for reasoning rollouts: compares default vs mixed-precision + KV hints.

Usage:
  python -m omnicoder.reasoning.microbench --tokens 64 --steps 64
"""

import os
import time
import torch
from typing import Optional


def _fake_model(d_model: int = 256, vocab: int = 32000):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d_model)
            self.proj = torch.nn.Linear(d_model, vocab)

        def forward(self, x, past_kv=None, use_cache=True, return_hidden=False, prefix_hidden=None):  # type: ignore[override]
            h = self.emb(x)
            logits = self.proj(h)
            kv = past_kv
            outs = (logits, kv, None, None, None, None, None, h)
            return outs if return_hidden else (logits, kv)

    return M()


def run(tokens: int = 64, steps: int = 64, mixed: bool = True) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _fake_model().to(device)
    x = torch.randint(0, 32000, (1, tokens), device=device)
    kv = None
    start = time.time()
    for _ in range(steps):
        if mixed and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outs = model(x[:, -1:], past_kv=kv, use_cache=True, return_hidden=True)
        else:
            outs = model(x[:, -1:], past_kv=kv, use_cache=True, return_hidden=True)
        if isinstance(outs, tuple):
            kv = outs[1]
        x = torch.cat([x, torch.argmax(outs[0][:, -1, :], dim=-1, keepdim=True)], dim=1)
    end = time.time()
    return float(steps / max(1e-6, (end - start)))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--steps', type=int, default=64)
    args = ap.parse_args()
    os.environ.setdefault('OMNICODER_REASONING_MIXED_PREC', 'bf16')
    tps_mixed = run(tokens=args.tokens, steps=args.steps, mixed=True)
    tps_fp32 = run(tokens=args.tokens, steps=args.steps, mixed=False)
    print({'tps_mixed': tps_mixed, 'tps_fp32': tps_fp32, 'speedup': (tps_mixed / max(1e-6, tps_fp32))})



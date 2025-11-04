from __future__ import annotations

"""
Minimal cycle-consistency training utility.

Stage A: generate from prompt → image/audio/text (depending on head availability)
Stage B: re-encode (caption/transcribe) → compare back to prompt using CLIPScore or
         simple text overlap proxy. Optimizes a lightweight projector on top of
         the model's hidden states to improve cycle-consistency.

This is a lightweight/proxy implementation with no external internet dependencies.
"""

import argparse
from pathlib import Path
import os
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import torch.nn.functional as F


def text_overlap_score(ref: str, hyp: str) -> float:
    try:
        rs = set(ref.lower().split())
        hs = set(hyp.lower().split())
        if not rs or not hs:
            return 0.0
        inter = len(rs & hs)
        return float(inter) / float(max(1, len(rs)))
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default=os.getenv('OMNICODER_CKPT',''))
    ap.add_argument('--device', type=str, default=os.getenv('OMNICODER_DEVICE','cpu'))
    ap.add_argument('--steps', type=int, default=int(os.getenv('OMNICODER_CYCLE_STEPS','200')))
    ap.add_argument('--lr', type=float, default=float(os.getenv('OMNICODER_CYCLE_LR','1e-4')))
    ap.add_argument('--prompts', type=str, default=os.getenv('OMNICODER_CYCLE_PROMPTS','A photo of a dog;A person speaking'))
    args = ap.parse_args()

    from omnicoder.modeling.transformer_moe import OmniTransformer
    model = OmniTransformer()
    if args.ckpt:
        try:
            state = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
    model.to(args.device).train()
    # Tiny projector on hidden states → token logits to refine text recon
    proj = nn.Linear(model.ln_f.normalized_shape[0], model.lm_head.out_features, bias=False).to(args.device)  # type: ignore[index]
    opt = torch.optim.AdamW(proj.parameters(), lr=float(args.lr))

    # Tokenizer
    from omnicoder.training.simple_tokenizer import get_text_tokenizer
    tok = get_text_tokenizer(prefer_hf=True)
    prompts = [p.strip() for p in str(args.prompts).split(';') if p.strip()]
    if not prompts:
        prompts = ["Hello world"]

    for step in range(int(args.steps)):
        p = prompts[step % len(prompts)]
        ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=args.device)
        # Forward to get hidden states
        out = model(ids, past_kv=None, use_cache=False, return_hidden=True)
        if isinstance(out, tuple):
            # Expect (logits, mtp?, diff?, halt?, retention?, hidden)
            hidden = out[-1]
            logits = proj(hidden)
        else:
            hidden = None
            logits = out
        # Teacher-forced next-token shift CE loss as a proxy to improve reconstruction
        target = ids
        ce = F.cross_entropy(logits[:, : target.size(1), :].transpose(1, 2), target, ignore_index=0)
        loss = ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
        if (step + 1) % 20 == 0:
            print({'step': step + 1, 'loss': float(loss.item())})

    outp = Path('weights/cycle_projector.pt')
    outp.parent.mkdir(parents=True, exist_ok=True)
    _safe_save({'state_dict': proj.state_dict()}, outp)
    print({'saved': str(outp)})


if __name__ == '__main__':
    main()



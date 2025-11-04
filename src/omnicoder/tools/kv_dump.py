from __future__ import annotations

"""
KV dump utility: runs the generator over a set of prompts and saves sampled
per-layer K/V tensors to disk for later analysis or autoencoder training.
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch


def _load_prompts(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return ["Hello OmniCoder"]
    if p.suffix.lower() == '.json' or p.suffix.lower() == '.jsonl':
        try:
            lines = p.read_text(encoding='utf-8').splitlines()
            out = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict) and 'prompt' in obj:
                        out.append(str(obj['prompt']))
                    else:
                        out.append(s)
                except Exception:
                    out.append(s)
            return out or ["Hello OmniCoder"]
        except Exception:
            return ["Hello OmniCoder"]
    return [p.read_text(encoding='utf-8')]


def main() -> None:
    ap = argparse.ArgumentParser(description='Dump K/V tensors from decode steps for prompts')
    ap.add_argument('--prompts', type=str, default='weights/kv_dump_prompts.jsonl')
    ap.add_argument('--out_dir', type=str, default='weights/kv_dump')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--max_new_tokens', type=int, default=16)
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    args = ap.parse_args()

    from omnicoder.modeling.transformer_moe import OmniTransformer  # lazy import
    from omnicoder.training.simple_tokenizer import get_text_tokenizer

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tok = get_text_tokenizer(prefer_hf=False)
    model = OmniTransformer(preset=args.mobile_preset).to(args.device)  # type: ignore[arg-type]
    model.eval()

    prompts = _load_prompts(args.prompts)
    with torch.no_grad():
        for pi, prompt in enumerate(prompts):
            ids = torch.tensor([[t for t in tok.encode(prompt)]], dtype=torch.long, device=args.device)
            out = ids.clone()
            past_kv = None
            for t in range(int(args.max_new_tokens)):
                step = out[:, -1:] if out.size(1) > 1 else out
                outs = model(step, past_kv=past_kv, use_cache=True)
                if isinstance(outs, tuple):
                    logits, past_kv = outs[0], outs[1]
                else:
                    logits, past_kv = outs, None
                # Save a snapshot every 4 steps
                if (t % 4 == 0) and past_kv is not None:
                    dump = []
                    for (k_t, v_t, meta) in past_kv:  # type: ignore[assignment]
                        dump.append({'k': k_t.detach().cpu(), 'v': v_t.detach().cpu(), 'meta': meta})
                    from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
                    _safe_save(dump, out_root / f"kv_pi{pi}_t{t}.pt")
                # Greedy next
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                out = torch.cat([out, next_id], dim=1)
    print({"out_dir": str(out_root.resolve())})


if __name__ == '__main__':
    main()



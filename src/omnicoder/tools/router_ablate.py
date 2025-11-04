from __future__ import annotations

import argparse
import json
import time

import torch

from omnicoder.inference.generate import build_mobile_model_by_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _bench(model, prompt: str, max_new: int = 64) -> dict:
    tok = get_text_tokenizer(prefer_hf=False)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    t0 = time.perf_counter()
    # Minimal decode loop to avoid requiring .generate
    device = next(model.parameters()).device
    ids = ids.to(device)
    out = ids
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(out[:, -1:], past_kv=None, use_cache=False)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
    dt = max(time.perf_counter() - t0, 1e-6)
    return {"tps": float(max_new / dt)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate LLMRouter vs baseline routing presets")
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--prompt', type=str, default='The quick brown fox jumps over the lazy dog.')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--out_json', type=str, default='weights/router_ablation.json')
    args = ap.parse_args()

    res = {}
    for mode in ('baseline', 'llmrouter'):
        model = build_mobile_model_by_name(args.mobile_preset, mem_slots=0)
        model.to(args.device)
        model.eval()
        if hasattr(model, 'blocks') and mode == 'llmrouter':
            for blk in model.blocks:
                try:
                    from omnicoder.modeling.routing import LLMRouter  # type: ignore
                    n_e = getattr(blk.moe, 'n_experts', 4)
                    k = getattr(blk.moe, 'top_k', 2)
                    d = getattr(blk.moe, 'd_model', getattr(model, 'd_model', 512))
                    blk.moe.router = LLMRouter(int(d), int(n_e), k=int(k))  # type: ignore[attr-defined]
                except Exception:
                    pass
        res[mode] = _bench(model, args.prompt, max_new=int(args.max_new_tokens))

    out = {
        'preset': args.mobile_preset,
        'max_new_tokens': int(args.max_new_tokens),
        'baseline_tps': res.get('baseline', {}).get('tps'),
        'llmrouter_tps': res.get('llmrouter', {}).get('tps'),
        'speedup_x': (res.get('llmrouter', {}).get('tps', 0.0) / max(1e-6, res.get('baseline', {}).get('tps', 1e-6))),
    }
    print(json.dumps(out))
    try:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print('[router_ablate] wrote', args.out_json)
    except Exception:
        pass


if __name__ == '__main__':
    main()



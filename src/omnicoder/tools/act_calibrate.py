import argparse
import json
import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, MobilePreset2GB
from omnicoder.training.simple_tokenizer import get_text_tokenizer


@torch.inference_mode()
def collect_activation_scales(model: OmniTransformer, prompts: list[str], layer_names: list[str] | None = None) -> dict:
    """Run a few prompts and collect per-channel activation scales for selected modules.

    Returns a dict mapping module paths to {"scale": [..]} lists (float).
    """
    stats: dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_mod, _inp, out):
            ten = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else None)
            if ten is None:
                return
            # Per-channel over last dim
            scale = ten.detach().abs().amax(dim=tuple(range(ten.dim()-1)))
            if name in stats:
                stats[name] = torch.maximum(stats[name], scale)
            else:
                stats[name] = scale
        return fn

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.LayerNorm):
            if layer_names is None or any(n in name for n in layer_names):
                handles.append(mod.register_forward_hook(_hook(name)))

    tok = get_text_tokenizer(prefer_hf=True)
    device = next(model.parameters()).device
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=device)
        # a few steps of decoding to exercise cache path
        past_kv = None
        for _ in range(8):
            out = model(ids[:, -1:], past_kv=past_kv, use_cache=True)
            if isinstance(out, tuple):
                logits, past_kv = out[0], out[1]
            else:
                logits = out
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)

    for h in handles:
        h.remove()

    out: dict[str, list[float]] = {k: v.detach().float().cpu().tolist() for k, v in stats.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--out', type=str, default='weights/act_scales.json')
    ap.add_argument('--prompts', type=str, nargs='*', default=[
        "Explain SIMD on ARM.",
        "What is a transformer?",
        "Write a Python function to compute Fibonacci numbers.",
    ])
    args = ap.parse_args()

    if args.mobile_preset in ('mobile_4gb','mobile_2gb'):
        preset = MobilePreset() if args.mobile_preset == 'mobile_4gb' else MobilePreset2GB()
        model = OmniTransformer(
            vocab_size=preset.vocab_size,
            n_layers=preset.n_layers,
            d_model=preset.d_model,
            n_heads=preset.n_heads,
            mlp_dim=preset.mlp_dim,
            n_experts=preset.moe_experts,
            top_k=preset.moe_top_k,
            max_seq_len=preset.max_seq_len,
            use_rope=True,
            kv_latent_dim=preset.kv_latent_dim,
            multi_query=preset.multi_query,
        )
    else:
        model = OmniTransformer()
    model.eval()
    scales = collect_activation_scales(model, args.prompts)
    with open(args.out, 'w') as f:
        json.dump(scales, f, indent=2)
    print(f"Wrote activation scales to {args.out}")


if __name__ == '__main__':
    main()



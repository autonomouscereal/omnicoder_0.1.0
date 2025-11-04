from __future__ import annotations

"""
KV-cache quantization calibration utility.

Collects per-layer, per-head, per-group statistics over the latent-K/V streams
to derive static quantization parameters for u8/NF4 (optional).

Outputs a JSON with summary and optional .pt tensors of stats for advanced use.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from omnicoder.inference.generate import build_mobile_model_by_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _update_u8(running: Dict[str, torch.Tensor], x: torch.Tensor, group: int) -> Dict[str, torch.Tensor]:
    B, H, T, D = x.shape
    G = (D + group - 1) // group
    pad = G * group - D
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad), value=0.0)
    xv = torch.ops.aten.reshape.default(x, (B, H, T, G, group))
    cur_max = xv.amax(dim=(-1, -2), keepdim=False)  # (B,H,G)
    cur_min = xv.amin(dim=(-1, -2), keepdim=False)
    if not running:
        return {"max": cur_max.detach().cpu(), "min": cur_min.detach().cpu()}
    running["max"] = torch.maximum(running["max"], cur_max.detach().cpu())
    running["min"] = torch.minimum(running["min"], cur_min.detach().cpu())
    return running


def _update_nf4(running: Dict[str, torch.Tensor], x: torch.Tensor, group: int) -> Dict[str, torch.Tensor]:
    B, H, T, D = x.shape
    G = (D + group - 1) // group
    pad = G * group - D
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad), value=0.0)
    xv = torch.ops.aten.reshape.default(x, (B, H, T, G, group))
    # Reduce over time, batch for stability; track mean and std via sums
    cur_mean = xv.mean(dim=(-1, -2), keepdim=False)  # (B,H,G)
    cur_var = ((xv - cur_mean.unsqueeze(-1)) ** 2).mean(dim=(-1, -2), keepdim=False)
    if not running:
        return {"mean": cur_mean.detach().cpu(), "var": cur_var.detach().cpu(), "count": torch.ones_like(cur_mean, dtype=torch.float32)}
    # Simple running max over variance and running avg over mean
    running["mean"] = 0.9 * running["mean"] + 0.1 * cur_mean.detach().cpu()
    running["var"] = torch.maximum(running["var"], cur_var.detach().cpu())
    running["count"] = running["count"] + 1.0
    return running


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate KV-cache quantization params over sample prompts")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb", choices=["mobile_4gb","mobile_2gb"]) 
    ap.add_argument("--prompts", type=str, nargs='*', default=["Hello OmniCoder!"], help="List of prompts or a path to a text file with one prompt per line")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--kvq", type=str, default="u8", choices=["u8","nf4"])
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=str, default="weights/kvq_calibration.json")
    args = ap.parse_args()

    # Load prompts
    prompts = args.prompts
    if len(prompts) == 1 and Path(prompts[0]).exists():
        prompts = [l.strip() for l in Path(prompts[0]).read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
        prompts = prompts[:16]

    tok = get_text_tokenizer(prefer_hf=True)
    model = build_mobile_model_by_name(args.mobile_preset)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    stats_k: Dict[int, Dict[str, torch.Tensor]] = {}
    stats_v: Dict[int, Dict[str, torch.Tensor]] = {}

    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=device)
            past = None
            # warm-up over prompt
            for t in range(ids.size(1)):
                step = ids[:, t:t+1]
                out = model(step, past_kv=past, use_cache=True)
                if isinstance(out, tuple):
                    logits, past = out[0], out[1]
                else:
                    logits, past = out, past
            # generate N tokens, collecting KV stats per step
            cur_id = ids[:, -1:]
            for _ in range(args.max_new_tokens):
                out = model(cur_id, past_kv=past, use_cache=True)
                if not isinstance(out, tuple):
                    break
                logits, new_kv = out[0], out[1]
                # Last appended slice per layer is at time index -1
                for li, (k, v) in enumerate(new_kv):
                    # Shape (B,H,T,DL). Take only the newest time dim to approximate streaming distribution
                    k_new = k[:, :, -1:, :].to(torch.float32).detach()
                    v_new = v[:, :, -1:, :].to(torch.float32).detach()
                    if args.kvq == 'u8':
                        stats_k[li] = _update_u8(stats_k.get(li, {}), k_new, args.group)
                        stats_v[li] = _update_u8(stats_v.get(li, {}), v_new, args.group)
                    else:
                        stats_k[li] = _update_nf4(stats_k.get(li, {}), k_new, args.group)
                        stats_v[li] = _update_nf4(stats_v.get(li, {}), v_new, args.group)
                # greedy next id
                cur_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                past = new_kv

    # Serialize summary (schema includes group size and optional per-layer npz paths with stats)
    out_path = Path(args.out)
    out_dir = out_path.parent
    summary: Dict[str, dict] = {"scheme": args.kvq, "group": int(args.group), "layers": {}}
    for li in sorted(stats_k.keys()):
        sk = stats_k[li]
        sv = stats_v[li]
        entry: Dict[str, dict] = {}
        if args.kvq == 'u8':
            # Persist per-layer stats to .npz to avoid bloating JSON
            npz_path = out_dir / f"kvq_u8_layer{li}.npz"
            try:
                import numpy as _np
                _np.savez_compressed(
                    npz_path,
                    k_max=sk["max"].numpy(),
                    k_min=sk["min"].numpy(),
                    v_max=sv["max"].numpy(),
                    v_min=sv["min"].numpy(),
                )
                entry["u8_npz"] = str(npz_path)
            except Exception:
                entry["k_max_shape"] = list(sk["max"].shape)
                entry["k_min_shape"] = list(sk["min"].shape)
                entry["v_max_shape"] = list(sv["max"].shape)
                entry["v_min_shape"] = list(sv["min"].shape)
        else:
            npz_path = out_dir / f"kvq_nf4_layer{li}.npz"
            try:
                import numpy as _np
                _np.savez_compressed(
                    npz_path,
                    k_mean=sk["mean"].numpy(),
                    k_var=sk["var"].numpy(),
                    v_mean=sv["mean"].numpy(),
                    v_var=sv["var"].numpy(),
                )
                entry["nf4_npz"] = str(npz_path)
            except Exception:
                entry["k_mean_shape"] = list(sk["mean"].shape)
                entry["k_var_shape"] = list(sk["var"].shape)
                entry["v_mean_shape"] = list(sv["mean"].shape)
                entry["v_var_shape"] = list(sv["var"].shape)
        summary["layers"][str(li)] = entry

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



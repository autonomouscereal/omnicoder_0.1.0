from __future__ import annotations

"""
MLA micro-benchmark: measure decode throughput for dense vs block-sparse attention
and recommend OMNICODER_ATT_BLOCK_SPARSE / OMNICODER_BS_STRIDE.

Usage:
  python -m omnicoder.tools.mla_microbench --preset mobile_4gb --device cpu --steps 256 --out weights/mla_microbench.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from omnicoder.inference.generate import build_mobile_model_by_name


@torch.inference_mode()
def _bench(model: torch.nn.Module, device: str, steps: int, bs: int | None) -> float:
    model.eval().to(device)
    # Configure block-sparse flag on all blocks
    try:
        if hasattr(model, "blocks"):
            for blk in model.blocks:
                if hasattr(blk, "attn"):
                    setattr(blk.attn, "block_sparse", bool(bs is not None))
                    if bs is not None and hasattr(blk.attn, "bs_stride"):
                        setattr(blk.attn, "bs_stride", int(bs))
    except Exception:
        pass

    # Warm
    tok = torch.randint(0, getattr(model, "vocab_size", 32000), (1, 1), dtype=torch.long, device=device)
    kv = None
    for _ in range(8):
        out = model(tok, past_kv=kv, use_cache=True)
        if isinstance(out, tuple):
            kv = out[1]
        tok = torch.randint_like(tok, low=0, high=getattr(model, "vocab_size", 32000))

    # Time decode
    start = time.time()
    tok = torch.randint(0, getattr(model, "vocab_size", 32000), (1, 1), dtype=torch.long, device=device)
    kv = None
    for _ in range(max(1, int(steps))):
        out = model(tok, past_kv=kv, use_cache=True)
        if isinstance(out, tuple):
            kv = out[1]
            logits = out[0]
        else:
            logits = out
        tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    dt = max(1e-6, time.time() - start)
    tps = float(steps) / dt
    return tps


def main() -> None:
    ap = argparse.ArgumentParser(description="Micro-benchmark MLA dense vs block-sparse")
    ap.add_argument("--preset", type=str, default="mobile_4gb")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", type=str, default="weights/mla_microbench.json")
    ap.add_argument("--assert_dml_speedup", action="store_true", help="If DirectML is available, assert DML speedup vs CPU meets threshold")
    ap.add_argument("--dml_speedup_min", type=float, default=float(__import__('os').environ.get('OMNICODER_BENCH_DML_SPEEDUP_MIN', '1.50')))
    args = ap.parse_args()

    model = build_mobile_model_by_name(args.preset)
    # Bench configurations
    configs: list[tuple[str, int | None]] = [("dense", None), ("bs_32", 32), ("bs_64", 64), ("bs_128", 128)]
    results: dict[str, float] = {}
    for name, stride in configs:
        try:
            tps = _bench(model, args.device, steps=int(args.steps), bs=stride)
        except Exception:
            tps = 0.0
        results[name] = round(tps, 3)

    # Pick best
    best_name = max(results, key=results.get) if results else "dense"
    rec_sparse = best_name.startswith("bs_")
    rec_stride = int(best_name.split("_")[1]) if rec_sparse else 0
    out = {
        "results": results,
        "recommend": {
            "OMNICODER_ATT_BLOCK_SPARSE": 1 if rec_sparse else 0,
            "OMNICODER_BS_STRIDE": rec_stride if rec_sparse else 0,
        },
    }
    # Optional DirectML vs CPU speedup microbench
    dml_speedup_ok = None
    try:
        import torch_directml  # type: ignore
        dml_dev = torch_directml.device()
        t_cpu = _bench(build_mobile_model_by_name(args.preset), 'cpu', steps=int(args.steps), bs=None)
        t_dml = _bench(build_mobile_model_by_name(args.preset), dml_dev, steps=int(args.steps), bs=None)
        speedup = (t_dml / max(t_cpu, 1e-9)) if t_cpu > 0 else 0.0
        out['dml'] = {
            'cpu_tps': round(float(t_cpu), 3),
            'dml_tps': round(float(t_dml), 3),
            'speedup': round(float(speedup), 3),
            'threshold': float(args.dml_speedup_min),
        }
        dml_speedup_ok = bool(speedup >= float(args.dml_speedup_min))
        out['dml']['ok'] = dml_speedup_ok
    except Exception:
        pass
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))

    if bool(args.assert_dml_speedup) and dml_speedup_ok is not None and (not dml_speedup_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()



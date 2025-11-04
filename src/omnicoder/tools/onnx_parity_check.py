from __future__ import annotations

"""
ONNX parity checker for decode-step graphs.

Runs a few random decode steps through the PyTorch wrapper and ONNX Runtime and
reports per-component absolute/relative error statistics with configurable tolerances.

Usage:
  python -m omnicoder.tools.onnx_parity_check \
    --onnx weights/release/text/omnicoder_decode_step.onnx \
    --preset mobile_4gb --seq_len 1 --steps 4 --abs_tol 3e-3 --rel_tol 3e-3

Outputs a JSON report next to the ONNX file: <onnx_basename>_parity.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from omnicoder.export.onnx_export import DecodeStepWrapper  # type: ignore
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset, MobilePreset2GB


def _build_model(preset: str, seq_len: int) -> OmniTransformer:
    if preset == "mobile_4gb":
        p = MobilePreset()
    elif preset == "mobile_2gb":
        p = MobilePreset2GB()
    else:
        p = MobilePreset()
    m = OmniTransformer(
        vocab_size=p.vocab_size,
        n_layers=p.n_layers,
        d_model=p.d_model,
        n_heads=p.n_heads,
        mlp_dim=p.mlp_dim,
        n_experts=p.moe_experts,
        top_k=p.moe_top_k,
        max_seq_len=max(64, seq_len),
        use_rope=True,
        kv_latent_dim=p.kv_latent_dim,
        multi_query=p.multi_query,
        multi_token=1,
    )
    m.eval()
    return m


def _run_torch(wrapper: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
    with torch.no_grad():
        out = wrapper(*inputs)
    if isinstance(out, tuple):
        outs = list(out)
    else:
        outs = [out]
    return [o.detach().cpu() for o in outs]


def _run_onnx(onnx_path: str, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise SystemExit(f"onnxruntime required: {e}")
    so = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])  # type: ignore
    feed = {k: v for k, v in inputs.items()}
    out_names = [o.name for o in sess.get_outputs()]
    out = sess.run(out_names, feed)
    return out


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _abs_rel_err(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float]:
    diff = np.abs(a - b)
    abs_max = float(np.max(diff)) if diff.size else 0.0
    abs_mean = float(np.mean(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(a), 1e-6)
    rel = diff / denom
    rel_max = float(np.max(rel)) if rel.size else 0.0
    rel_mean = float(np.mean(rel)) if rel.size else 0.0
    return abs_max, abs_mean, rel_max, rel_mean


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode-step ONNX parity checker")
    ap.add_argument("--onnx", required=True, type=str)
    ap.add_argument("--preset", type=str, default="mobile_4gb")
    ap.add_argument("--seq_len", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--abs_tol", type=float, default=3e-3)
    ap.add_argument("--rel_tol", type=float, default=3e-3)
    args = ap.parse_args()

    model = _build_model(args.preset, args.seq_len)
    wrapper = DecodeStepWrapper(model)

    # Build inputs
    B = 1
    T_past = 0
    H = model.blocks[0].attn.n_heads
    DL = model.blocks[0].attn.kv_latent_dim
    input_ids = torch.randint(0, model.vocab_size, (B, 1), dtype=torch.long)
    past: List[torch.Tensor] = []
    for _ in model.blocks:
        past.append(torch.zeros(B, H, T_past, DL))
    for _ in model.blocks:
        past.append(torch.zeros(B, H, T_past, DL))
    torch_inputs = (input_ids, *past)

    # Torch forward
    torch_outs = _run_torch(wrapper, torch_inputs)

    # ONNX feed dict
    feed: Dict[str, np.ndarray] = {"input_ids": _to_numpy(input_ids)}
    nb = len(model.blocks)
    for i in range(nb):
        feed[f"k_lat_{i}"] = _to_numpy(past[i])
    for i in range(nb):
        feed[f"v_lat_{i}"] = _to_numpy(past[i + nb])
    onnx_outs = _run_onnx(args.onnx, feed)

    # Compare outputs component-wise: logits + per-layer caches (k/v stacked at end)
    report: Dict[str, Any] = {"ok": True, "components": []}
    names: List[str] = ["logits"] + [f"nk_lat_{i}" for i in range(nb)] + [f"nv_lat_{i}" for i in range(nb)]
    for name, t_out, o_out in zip(names, torch_outs, onnx_outs):
        t_np = _to_numpy(t_out)
        o_np = o_out.astype(np.float32)
        abs_max, abs_mean, rel_max, rel_mean = _abs_rel_err(t_np, o_np)
        comp = {
            "name": name,
            "shape_torch": list(t_np.shape),
            "shape_onnx": list(o_np.shape),
            "abs_max": abs_max,
            "abs_mean": abs_mean,
            "rel_max": rel_max,
            "rel_mean": rel_mean,
            "ok": (abs_max <= args.abs_tol) or (rel_max <= args.rel_tol),
        }
        report["components"].append(comp)
        if not comp["ok"]:
            report["ok"] = False

    out_path = Path(args.onnx).with_suffix(".parity.json")
    out_path.write_text(json.dumps(report, indent=2))
    status = "OK" if report["ok"] else "FAIL"
    print(f"[parity] {status} -> {out_path}")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Export lightweight open-vocab grounding heads to ONNX (and optionally Core ML/ExecuTorch).

Heads: SimpleGroundingHead, RepRTAHead

These operate on precomputed vision tokens and a text embedding. For export, we
emit a tiny wrapper that accepts (vision_tokens, text_embed) with dynamic batch
and sequence dims. By default, we export both heads into the specified directory.

Examples:
  python -m omnicoder.export.onnx_export_grounding --out weights/vision --d_model 384 --tokens 196
  python -m omnicoder.export.onnx_export_grounding --out weights/vision --head rep_rta --d_model 384 --tokens 196 --coreml --executorch
"""

import argparse
from pathlib import Path
from typing import Optional

import torch


def _export_head(kind: str, out_dir: Path, d_model: int, num_props: int, tokens: int, opset: int,
                 do_coreml: bool = False, do_executorch: bool = False) -> dict:
    from omnicoder.modeling.multimodal.vision_grounding import SimpleGroundingHead, RepRTAHead  # type: ignore

    class Wrap(torch.nn.Module):
        def __init__(self, head: torch.nn.Module):
            super().__init__()
            self.h = head
        def forward(self, vt: torch.Tensor, txt: torch.Tensor):  # type: ignore
            boxes, conf = self.h(vt, txt)
            return boxes, conf

    if kind == "simple":
        head = SimpleGroundingHead(d_model=int(d_model), num_props=int(num_props)).eval()
        name = "simple_grounding"
    elif kind == "rep_rta":
        head = RepRTAHead(d_model=int(d_model), num_props=int(num_props)).eval()
        name = "reprta_grounding"
    else:
        raise ValueError(f"Unknown head kind: {kind}")

    wrap = Wrap(head).eval()
    # Dummy shapes: (B,T,C) and (B,C); keep B,T dynamic
    B = 1
    T = int(tokens)
    C = int(d_model)
    vt = torch.randn(B, T, C)
    txt = torch.randn(B, C)
    onnx_path = out_dir / f"{name}.onnx"
    out: dict = {"onnx": None, "coreml": None, "executorch": None}
    try:
        torch.onnx.export(
            wrap, (vt, txt), str(onnx_path),
            input_names=["vision_tokens", "text_embed"],
            output_names=["boxes", "conf"],
            dynamic_axes={
                "vision_tokens": {0: "B", 1: "T"},
                "text_embed": {0: "B"},
                "boxes": {0: "B", 1: "N"},
                "conf": {0: "B", 1: "N"},
            },
            opset_version=int(opset),
        )
        out["onnx"] = str(onnx_path)
    except Exception as e:
        print(f"[onnx] {name} export failed: {e}")

    if do_coreml and out.get("onnx") is not None:
        try:
            import coremltools as ct  # type: ignore
            traced = torch.jit.trace(wrap, (vt, txt))
            mlp = out_dir / f"{name}.mlmodel"
            model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="vision_tokens", shape=vt.shape),
                    ct.TensorType(name="text_embed", shape=txt.shape),
                ],
                convert_to="mlprogram",
            )
            model.save(str(mlp))
            out["coreml"] = str(mlp)
        except Exception as e:
            print(f"[coreml] {name} export failed: {e}")

    if do_executorch:
        try:
            from torch.export import export as torch_export  # type: ignore
            pte = out_dir / f"{name}.pte"
            prog = torch_export(wrap, (vt, txt))
            with open(pte, "wb") as f:
                f.write(prog.to_pte())
            out["executorch"] = str(pte)
        except Exception:
            try:
                ts = torch.jit.trace(wrap, (vt, txt), check_trace=False)
                ts_path = out_dir / f"{name}.pt"
                ts.save(str(ts_path))
                out["executorch"] = str(ts_path)
            except Exception as e:
                print(f"[execu] {name} export failed: {e}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export grounding heads to ONNX (and optionally Core ML/ExecuTorch)")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--head", type=str, default="both", choices=["both", "simple", "rep_rta"])
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--num_props", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=196, help="Number of vision tokens (T) used for dummy export")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--coreml", action="store_true")
    ap.add_argument("--executorch", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["simple", "rep_rta"] if args.head == "both" else [args.head]
    results = {}
    for k in kinds:
        results[k] = _export_head(
            kind=k,
            out_dir=out_dir,
            d_model=int(args.d_model),
            num_props=int(args.num_props),
            tokens=int(args.tokens),
            opset=int(args.opset),
            do_coreml=bool(args.coreml),
            do_executorch=bool(args.executorch),
        )
    # Write a small manifest
    try:
        import json
        (out_dir / "grounding_export.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("[manifest] grounding_export.json written")
    except Exception:
        pass


if __name__ == "__main__":
    main()



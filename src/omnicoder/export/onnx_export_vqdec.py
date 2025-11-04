from __future__ import annotations

"""
Export a standalone Image VQ decoder (indices -> image) to ONNX/Core ML/ExecuTorch.

Example:
  python -m omnicoder.export.onnx_export_vqdec \
    --codebook weights/image_vq_codebook.pt \
    --onnx weights/image_vq_decoder.onnx --hq 14 --wq 14
"""

import argparse
from pathlib import Path

import torch

from omnicoder.modeling.multimodal.image_vq_decoder import ImageVQDecoder


def export_onnx(decoder: ImageVQDecoder, out_path: Path, hq: int, wq: int, opset: int = 17) -> bool:
    decoder.eval()
    dummy = torch.zeros(1, int(hq), int(wq), dtype=torch.long)
    try:
        torch.onnx.export(
            decoder,
            dummy,
            str(out_path),
            input_names=["indices"],
            output_names=["image"],
            dynamic_axes={"indices": {1: "hq", 2: "wq"}, "image": {2: "H", 3: "W"}},
            opset_version=int(opset),
        )
        print(f"[ONNX] Exported image VQ decoder to {out_path}")
        return True
    except Exception as e:
        print(f"[ONNX] export failed: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Image VQ decoder")
    ap.add_argument("--codebook", type=str, required=True, help="Path to ImageVQVAE.export_codebook blob")
    ap.add_argument("--onnx", type=str, default="", help="Output ONNX path")
    ap.add_argument("--hq", type=int, default=14)
    ap.add_argument("--wq", type=int, default=14)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    dec = ImageVQDecoder.from_codebook_file(args.codebook)
    ok_any = False
    if args.onnx:
        ok_any = export_onnx(dec, Path(args.onnx), args.hq, args.wq, args.opset) or ok_any
    if not ok_any:
        print("No exports completed. Provide --onnx.")


if __name__ == "__main__":
    main()



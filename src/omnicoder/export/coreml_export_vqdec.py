from __future__ import annotations

"""
Export Image VQ decoder to Core ML (MLProgram).

Example:
  python -m omnicoder.export.coreml_export_vqdec \
    --codebook weights/image_vq_codebook.pt \
    --out weights/image_vq_decoder.mlmodel --hq 14 --wq 14
"""

import argparse
from pathlib import Path

import torch

from omnicoder.modeling.multimodal.image_vq_decoder import ImageVQDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Image VQ decoder to Core ML")
    ap.add_argument("--codebook", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--hq", type=int, default=14)
    ap.add_argument("--wq", type=int, default=14)
    args = ap.parse_args()

    try:
        import coremltools as ct  # type: ignore
    except Exception as e:
        raise SystemExit("coremltools>=7 required for MLProgram export") from e

    dec = ImageVQDecoder.from_codebook_file(args.codebook).eval()
    # Trace with dummy long input; MLProgram supports shape flexibility, but keep it fixed here for simplicity
    dummy = torch.zeros(1, int(args.hq), int(args.wq), dtype=torch.long)
    traced = torch.jit.trace(dec, dummy)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="indices", shape=dummy.shape, dtype=ct.int64)],
    )
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(outp))
    print(f"[CoreML] Saved image VQ decoder to {outp}")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Export Video VQ decoder to Core ML (MLProgram) for a fixed number of frames T.

Example:
  python -m omnicoder.export.coreml_export_videovqdec \
    --out weights/vqdec/video_vq_decoder.mlmodel --t 8 --hq 16 --wq 16 --code_dim 192
  # optionally provide a codebook blob for better decoding initialization
  # --codebook weights/video_vq_codebook.pt
"""

import argparse
from pathlib import Path
from typing import List

import torch

from omnicoder.modeling.multimodal.video_vq_decoder import VideoVQDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Video VQ decoder to Core ML (fixed T)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--wq", type=int, default=16)
    ap.add_argument("--code_dim", type=int, default=192)
    ap.add_argument("--codebook", type=str, default="")
    args = ap.parse_args()

    try:
        import coremltools as ct  # type: ignore
    except Exception as e:
        raise SystemExit("coremltools>=7 required") from e

    dec = VideoVQDecoder.from_codebook_file(args.codebook or None, code_dim=int(args.code_dim)).eval()

    # Build a wrapper module with T inputs to satisfy Core ML's static interface
    class Wrapper(torch.nn.Module):
        def __init__(self, d: VideoVQDecoder, t: int) -> None:
            super().__init__()
            self.dec = d
            self.t = int(t)

        def forward(self, *frames: torch.Tensor) -> torch.Tensor:
            return self.dec(list(frames))

    wrap = Wrapper(dec, int(args.t)).eval()
    dummies = tuple(torch.zeros(1, int(args.hq), int(args.wq), dtype=torch.long) for _ in range(int(args.t)))
    traced = torch.jit.trace(wrap, dummies, check_trace=False)

    inputs = [ct.TensorType(name=f"indices_{i}", shape=dummies[i].shape, dtype=ct.int64) for i in range(int(args.t))]
    mlmodel = ct.convert(traced, convert_to="mlprogram", inputs=inputs)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(outp))
    print(f"[CoreML] Saved Video VQ decoder to {outp}")


if __name__ == "__main__":
    main()



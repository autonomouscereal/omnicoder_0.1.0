from __future__ import annotations

"""
Export Video VQ decoder to ExecuTorch .pte for a fixed number of frames T.

Example:
  python -m omnicoder.export.executorch_export_videovqdec \
    --out weights/vqdec/video_vq_decoder.pte --t 8 --hq 16 --wq 16 --code_dim 192
"""

import argparse
from pathlib import Path

import torch

from omnicoder.modeling.multimodal.video_vq_decoder import VideoVQDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Video VQ decoder to ExecuTorch (fixed T)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--wq", type=int, default=16)
    ap.add_argument("--code_dim", type=int, default=192)
    ap.add_argument("--codebook", type=str, default="")
    args = ap.parse_args()

    try:
        from torch.export import export as torch_export  # type: ignore
    except Exception as e:
        raise SystemExit("PyTorch >=2.3 required for ExecuTorch export") from e

    dec = VideoVQDecoder.from_codebook_file(args.codebook or None, code_dim=int(args.code_dim)).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, d: VideoVQDecoder, t: int) -> None:
            super().__init__()
            self.dec = d
            self.t = int(t)

        def forward(self, *frames: torch.Tensor) -> torch.Tensor:
            return self.dec(list(frames))

    wrap = Wrapper(dec, int(args.t)).eval()
    example = tuple(torch.zeros(1, int(args.hq), int(args.wq), dtype=torch.long) for _ in range(int(args.t)))
    prog = torch_export(wrap, example)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        f.write(prog.to_pte())
    print(f"[ExecuTorch] Saved Video VQ decoder to {outp}")


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from omnicoder.modeling.multimodal.video_vq_decoder import VideoVQDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Video VQ decoder to ONNX")
    ap.add_argument("--codebook", type=str, default="", help="Optional path to video VQ codebook blob")
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--wq", type=int, default=16)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--code_dim", type=int, default=192)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    dec = VideoVQDecoder.from_codebook_file(args.codebook or None, code_dim=int(args.code_dim)).eval()
    # Represent T frames as a pack of T inputs (simplest path)
    inputs = [torch.zeros(1, int(args.hq), int(args.wq), dtype=torch.long) for _ in range(int(args.t))]
    input_names = [f"indices_{i}" for i in range(int(args.t))]
    dynamic_axes = {name: {1: "hq", 2: "wq"} for name in input_names}

    def wrapper(*frames):
        return dec(list(frames))

    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,  # type: ignore[arg-type]
        tuple(inputs),
        str(out_path),
        input_names=input_names,
        output_names=["video"],
        dynamic_axes=dynamic_axes | {"video": {1: "t", 3: "H", 4: "W"}},
        opset_version=int(args.opset),
    )
    print(f"[ONNX] Exported Video VQ decoder to {out_path}")


if __name__ == "__main__":
    main()



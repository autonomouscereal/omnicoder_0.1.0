from __future__ import annotations

"""
Export Image VQ decoder to ExecuTorch .pte.

Example:
  python -m omnicoder.export.executorch_export_vqdec \
    --codebook weights/image_vq_codebook.pt \
    --out weights/image_vq_decoder.pte --hq 14 --wq 14
"""

import argparse
from pathlib import Path

import torch

from omnicoder.modeling.multimodal.image_vq_decoder import ImageVQDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Image VQ decoder to ExecuTorch")
    ap.add_argument("--codebook", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--hq", type=int, default=14)
    ap.add_argument("--wq", type=int, default=14)
    args = ap.parse_args()

    try:
        from torch.export import export as torch_export  # type: ignore
    except Exception as e:
        raise SystemExit("PyTorch >=2.3 with torch.export required for ExecuTorch export") from e

    dec = ImageVQDecoder.from_codebook_file(args.codebook).eval()
    dummy = (torch.zeros(1, int(args.hq), int(args.wq), dtype=torch.long),)
    prog = torch_export(dec, dummy)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        f.write(prog.to_pte())
    print(f"[ExecuTorch] Saved image VQ decoder to {outp}")


if __name__ == "__main__":
    main()



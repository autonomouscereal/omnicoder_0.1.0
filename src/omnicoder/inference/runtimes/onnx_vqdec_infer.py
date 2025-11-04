from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX VQ decoder demo (indices->image)")
    ap.add_argument("--model", type=str, required=True, help="Path to image_vq_decoder.onnx")
    ap.add_argument("--hq", type=int, default=14)
    ap.add_argument("--wq", type=int, default=14)
    ap.add_argument("--out", type=str, default="weights/vqdec_out.png")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("onnxruntime and Pillow required") from e

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])  # type: ignore
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Build a simple checkerboard indices pattern
    hq, wq = int(args.hq), int(args.wq)
    idx = np.arange(hq * wq, dtype=np.int64).reshape(hq, wq) % 16
    indices = idx[None, :, :]
    img = sess.run([out_name], {in_name: indices})[0][0]  # (3,H,W)
    # Convert to HWC and save
    hwc = np.transpose((img * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hwc).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()



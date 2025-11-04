from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Core ML VQ decoder demo (indices->image)")
    ap.add_argument("--model", type=str, required=True, help="Path to image_vq_decoder.mlmodel")
    ap.add_argument("--hq", type=int, default=14)
    ap.add_argument("--wq", type=int, default=14)
    ap.add_argument("--out", type=str, default="weights/vqdec_out_coreml.png")
    args = ap.parse_args()

    try:
        import coremltools as ct  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        raise SystemExit("coremltools and Pillow required") from e

    mlmodel = ct.models.MLModel(args.model)  # type: ignore
    hq, wq = int(args.hq), int(args.wq)
    idx = np.arange(hq * wq, dtype=np.int64).reshape(1, hq, wq)
    out = mlmodel.predict({"indices": idx})  # type: ignore
    img = out.get("image")
    if img is None:
        raise SystemExit("Model output 'image' not found")
    # Expect (1,3,H,W)
    arr = np.array(img) if hasattr(img, "__array__") else img
    if arr.ndim == 4:
        arr = arr[0]
    hwc = np.transpose((arr * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hwc).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()



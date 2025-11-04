from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX Video VQ decoder demo (indices list -> video)")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--wq", type=int, default=16)
    ap.add_argument("--out_dir", type=str, default="weights/vqvideo_out")
    args = ap.parse_args()

    import onnxruntime as ort  # type: ignore
    from PIL import Image  # type: ignore

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])  # type: ignore
    inputs = sess.get_inputs()
    assert len(inputs) == int(args.t)
    feeds = {}
    for i in range(int(args.t)):
        feeds[inputs[i].name] = np.zeros((1, int(args.hq), int(args.wq)), dtype=np.int64)
    out = sess.run([sess.get_outputs()[0].name], feeds)[0]  # (B,T,3,H,W)
    vid = out[0]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(vid.shape[0]):
        frame = np.transpose((vid[i] * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))
        Image.fromarray(frame).save(str(Path(args.out_dir) / f"frame_{i:03d}.png"))
    print(f"Saved {vid.shape[0]} frames to {args.out_dir}")


if __name__ == "__main__":
    main()



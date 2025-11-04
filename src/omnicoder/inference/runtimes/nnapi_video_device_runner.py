from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore

from .onnx_video_decode import ORTI2VCallable


def _save_mp4(frames_thwc: np.ndarray, out_path: str, fps: int = 24) -> Optional[str]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    if frames_thwc.ndim != 4:
        return None
    h, w = int(frames_thwc.shape[1]), int(frames_thwc.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fr in frames_thwc:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Device-side NNAPI i2v ORT runner (measures FPS; optional mp4 output)")
    ap.add_argument("--onnx_dir", type=str, required=True, help="Path to ONNX i2v export folder (contains generator.onnx)")
    ap.add_argument("--seed_image", type=str, default="", help="Path to seed image (RGB) on device; if empty, uses gray seed")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--provider", type=str, default="NNAPIExecutionProvider")
    ap.add_argument("--out", type=str, default="/data/local/tmp/oc_video_run.json")
    ap.add_argument("--mp4_out", type=str, default="/data/local/tmp/oc_video_out.mp4")
    args = ap.parse_args()

    if ort is None:
        result = {
            "error": "onnxruntime not available on device",
        }
        Path(args.out).write_text(json.dumps(result, indent=2))
        return

    # Build seed (1,3,H,W) float32 in [0,1]
    if args.seed_image:
        try:
            from PIL import Image  # type: ignore
            img = Image.open(args.seed_image).convert("RGB").resize((int(args.width), int(args.height)))
            seed = np.array(img).astype(np.float32) / 255.0
            seed = seed.transpose(2, 0, 1)[None, ...]
        except Exception:
            h, w = int(args.height), int(args.width)
            seed = np.full((1, 3, h, w), 0.5, dtype=np.float32)
    else:
        h, w = int(args.height), int(args.width)
        seed = np.full((1, 3, h, w), 0.5, dtype=np.float32)

    used_provider = args.provider
    provider_options = None
    # Try to instantiate with requested provider; if missing, fallback to CPU
    try:
        callable_i2v = ORTI2VCallable(args.onnx_dir, provider=used_provider, provider_options=provider_options)
        provider_ok = True
    except Exception:
        provider_ok = False
        used_provider = "CPUExecutionProvider"
        callable_i2v = ORTI2VCallable(args.onnx_dir, provider=used_provider, provider_options=None)

    t0 = time.perf_counter()
    frames = callable_i2v.generate(seed, int(args.frames))  # (T,H,W,3) uint8
    t1 = time.perf_counter()
    elapsed = float(t1 - t0)
    fps = (float(frames.shape[0]) / elapsed) if elapsed > 0 else 0.0
    mp4_path = _save_mp4(frames, args.mp4_out, fps=int(args.fps))

    result = {
        "frames": int(frames.shape[0]),
        "width": int(frames.shape[2]) if frames.ndim == 4 else int(args.width),
        "height": int(frames.shape[1]) if frames.ndim == 4 else int(args.height),
        "elapsed_s": elapsed,
        "fps": fps,
        "provider": used_provider,
        "requested_provider": args.provider,
        "provider_ok": bool(provider_ok),
        "mp4_out": (mp4_path or ""),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()



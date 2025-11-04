from __future__ import annotations

"""
Mixed discrete+continuous generation — latent refiner runner (image/audio).

Runs a tiny ONNX refiner on latent tensors produced by a blueprint generator
(e.g., VQ/image latent or audio mel latent) and writes refined latents.

Usage:
  python -m omnicoder.inference.mixed_latent_refine \
    --onnx weights/release/image/latent_refiner.onnx \
    --latents in_latents.npy \
    --out_latents out_latents.npy

For audio:
  python -m omnicoder.inference.mixed_latent_refine \
    --onnx weights/release/audio/audio_latent_refiner.onnx \
    --latents in_mel.npy --out_latents out_mel.npy
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _run_onnx(onnx_path: str, latents: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise RuntimeError(f"onnxruntime not available: {e}")

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inputs = {sess.get_inputs()[0].name: latents.astype(np.float32)}
    outs = sess.run(None, inputs)
    if not outs:
        raise RuntimeError("Refiner produced no outputs")
    return outs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a continuous latent refiner ONNX on image/audio latents")
    ap.add_argument('--onnx', type=str, required=True, help='Path to refiner ONNX model')
    ap.add_argument('--latents', type=str, required=True, help='Path to input latents .npy')
    ap.add_argument('--out_latents', type=str, required=True, help='Path to output refined latents .npy')
    args = ap.parse_args()

    lat = np.load(args.latents)
    ref = _run_onnx(args.onnx, lat)
    Path(os.path.dirname(args.out_latents) or '.').mkdir(parents=True, exist_ok=True)
    np.save(args.out_latents, ref)
    print({'in': args.latents, 'out': args.out_latents, 'shape_in': tuple(lat.shape), 'shape_out': tuple(ref.shape)})


if __name__ == '__main__':
    main()



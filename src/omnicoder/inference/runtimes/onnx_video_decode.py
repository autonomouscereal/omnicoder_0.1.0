from __future__ import annotations

"""
ORT image-to-video callable wrapper.

This expects an ONNX-exported directory with a single generator model:

  - generator.onnx: inputs { "image": (1,3,H,W) float32 in [0,1] }
                    outputs { "frames": (1,T,3,H,W) float32 in [0,1] }

If your export uses different tensor names, pass explicit names when
instantiating ORTI2VCallable.

Provider options can be supplied to steer NNAPI/CoreML/DML backends.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore


class ORTI2VCallable:
    def __init__(
        self,
        onnx_dir: str,
        provider: str = "CPUExecutionProvider",
        provider_options: Optional[dict] = None,
        input_name: str = "image",
        output_name: str = "frames",
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        self.onnx_dir = str(onnx_dir)
        self.input_name = input_name
        self.output_name = output_name
        gen = Path(onnx_dir) / "generator.onnx"
        if not gen.exists():
            raise FileNotFoundError(f"Missing generator.onnx in {onnx_dir}")
        providers = [provider]
        if provider_options is not None:
            sess_opts = ort.SessionOptions()
            self.sess = ort.InferenceSession(str(gen), sess_options=sess_opts, providers=providers, provider_options=[provider_options])
        else:
            self.sess = ort.InferenceSession(str(gen), providers=providers)

    def generate(self, image_bchw: np.ndarray, num_frames: int) -> np.ndarray:
        if image_bchw.dtype != np.float32:
            image_bchw = image_bchw.astype(np.float32)
        # Ensure [0,1]
        if image_bchw.max() > 1.0:
            image_bchw = np.clip(image_bchw / 255.0, 0.0, 1.0)
        out = self.sess.run([self.output_name], {self.input_name: image_bchw})[0]  # (1,T,3,H,W)
        if out.shape[0] == 1:
            out = out[0]
        if num_frames and out.shape[0] != int(num_frames):
            # Simple uniform resample to requested T
            t = out.shape[0]
            idx = np.linspace(0, t - 1, int(num_frames)).round().astype(int)
            out = out[idx]
        # Return (T,H,W,3) uint8
        out = np.clip(out.transpose(0, 2, 3, 1) * 255.0, 0.0, 255.0).astype(np.uint8)
        return out



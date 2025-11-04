from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class AudioConvFrontEnd(nn.Module):
    """
    Lightweight, export-friendly audio front-end that maps raw PCM (mono) to
    a log-energy feature grid analogous to mel spectrograms, using only Conv1d
    and pointwise ops (ONNX/mobile friendly; no complex STFT ops).

    Input:  (B, 1, T)  float32 PCM normalized to [-1, 1]
    Output: (B, T_out, M) where M ~= n_mels (default 80)
    """

    def __init__(self, n_mels: int = 80) -> None:
        super().__init__()
        c1 = max(32, n_mels // 2)
        c2 = n_mels
        # Two-stage temporal stride stack to downsample time and expand channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size=10, stride=5, padding=4, bias=False)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=8, stride=4, padding=3, bias=False)
        self.act2 = nn.GELU()
        self.ln = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # x: (B,1,T)
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        # (B, C, T') -> (B, T', C)
        h = h.transpose(1, 2)
        h = self.ln(h)
        return h


def main() -> None:
    ap = argparse.ArgumentParser(description="Export lightweight audio Conv1d front-end as ONNX")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n_mels", type=int, default=80)
    args = ap.parse_args()

    model = AudioConvFrontEnd(n_mels=int(args.n_mels))
    model.eval()

    outp = Path(args.out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input with dynamic time axis
    dummy = torch.randn(1, 1, 16000)
    torch.onnx.export(
        model,
        (dummy,),
        outp.as_posix(),
        input_names=["pcm"],
        output_names=["features"],
        opset_version=int(args.opset),
        dynamic_axes={"pcm": {2: "samples"}, "features": {1: "frames"}},
    )
    print(f"Exported audio front-end to {outp}")


if __name__ == "__main__":
    main()



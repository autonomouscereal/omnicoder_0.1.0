from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn


class TemporalSSM(nn.Module):
    """
    Minimal temporal state-space-like block realized as depthwise temporal conv
    with residual and feed-forward. Export-friendly for ONNX.

    Input/Output shape: (B, T, C)
    """

    def __init__(self, d_model: int = 384, kernel_size: int = 5, expansion: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(d_model * expansion, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        h = x.transpose(1, 2)  # (B,C,T)
        y = self.dw(h)
        y = self.pw2(self.act(self.pw1(y)))
        y = y.transpose(1, 2)
        return x + y


def main() -> None:
    ap = argparse.ArgumentParser(description="Export TemporalSSM to ONNX")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--expansion", type=int, default=2)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    mod = TemporalSSM(d_model=int(args.d_model), kernel_size=int(args.kernel_size), expansion=int(args.expansion)).eval()
    dummy = torch.randn(1, 8, int(args.d_model))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        mod,
        (dummy,),
        out_path.as_posix(),
        input_names=["x"],
        output_names=["y"],
        opset_version=int(args.opset),
        dynamic_axes={"x": {0: "batch", 1: "time"}, "y": {0: "batch", 1: "time"}},
    )
    print(f"Exported TemporalSSM to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class TemporalSSM(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        hidden = int(d_model * expansion)
        self.proj_in = nn.Linear(d_model, hidden * 2, bias=False)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, groups=hidden, padding=kernel_size // 2)
        self.proj_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        u, v = self.proj_in(x).chunk(2, dim=-1)
        v = torch.nn.functional.gelu(v)
        y = u * v
        y = y.transpose(1, 2)
        y = self.dw(y)
        y = y.transpose(1, 2)
        return self.proj_out(y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ONNX for temporal video SSM block")
    ap.add_argument("--out", type=str, default="weights/video/temporal_ssm.onnx")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--expansion", type=int, default=2)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    model = TemporalSSM(d_model=int(args.d_model), kernel_size=int(args.kernel), expansion=int(args.expansion))
    model.eval()

    # Dummy input: (B,T,C)
    dummy = torch.randn(1, 8, int(args.d_model), dtype=torch.float32)
    input_names = ["x"]
    output_names = ["y"]
    dynamic_axes = {"x": {0: "B", 1: "T"}, "y": {0: "B", 1: "T"}}

    # Prefer dynamo exporter for opset>=18
    exported = False
    try:
        dyn_export = getattr(torch.onnx, "dynamo_export", None)
        if callable(dyn_export) and int(args.opset) >= 18:
            onnx_model = dyn_export(model, (dummy,), input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=int(args.opset), dynamic_shapes=True)
            try:
                onnx_model.save(args.out)  # type: ignore[attr-defined]
            except Exception:
                Path(args.out).write_bytes(onnx_model)  # type: ignore[arg-type]
            exported = True
    except Exception:
        exported = False

    if not exported:
        torch.onnx.export(
            model,
            dummy,
            args.out,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
        )
    print(f"Exported temporal SSM ONNX to {args.out}")


if __name__ == "__main__":
    main()



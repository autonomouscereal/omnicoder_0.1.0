from __future__ import annotations

"""Int4 matmul kernel registry per provider.

Initial minimal backends default to CPU dequant + fp32 matmul, but the registry and
function signatures are stable so provider-specific kernels (NNAPI/CoreML/DML) can
be swapped in via environment variable or constructor flags.
"""

import os
from typing import Callable, Dict

import torch


KernelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _nibble_order() -> str:
    return os.environ.get("OMNICODER_INT4_NIBBLE_ORDER", "low_first").strip().lower()


def _aligned_in_features(n: int) -> int:
    try:
        align = int(os.environ.get("OMNICODER_INT4_ALIGN", "64"))
        if align > 0:
            return ((n + align - 1) // align) * align
    except Exception:
        pass
    return n


def _cpu_dequant_matmul(x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Reference path: unpack + dequant + matmul (fp32)."""
    # packed_w: (out, ceil(in/2)) with low/high nibbles
    out_features = packed_w.size(0)
    in_features = x.size(-1)
    bytes_per_row = packed_w.size(1)
    # Unpack to uint8 [0..15]
    q0 = packed_w & 0x0F
    q1 = (packed_w >> 4) & 0x0F
    q_u = torch.empty((out_features, bytes_per_row * 2), dtype=torch.uint8, device=packed_w.device)
    if _nibble_order() == "high_first":
        q_u[:, 0::2] = q1
        q_u[:, 1::2] = q0
    else:
        q_u[:, 0::2] = q0
        q_u[:, 1::2] = q1
    if q_u.size(1) > in_features:
        q_u = q_u[:, :in_features]
    # Shift to signed int4 [-8..7]
    q = q_u.to(torch.int16) - 8
    w = q.to(torch.float32) * scale.to(torch.float32)[:, None]
    return x.matmul(w.t())


def _dml_dequant_matmul(x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """DirectML path: run dequant+matmul on DML device if torch-directml is available.

    Falls back to CPU if torch-directml is not present.
    """
    try:
        import torch_directml  # type: ignore
        dml = torch_directml.device()
        xd = x.to(dml)
        wd = packed_w.to(dml)
        sd = scale.to(dml)
        zd = zero.to(dml)
        # Reuse CPU unpack logic but on DML tensors
        out_features = wd.size(0)
        in_features = xd.size(-1)
        bytes_per_row = wd.size(1)
        q0 = wd & 0x0F
        q1 = (wd >> 4) & 0x0F
        q_u = torch.empty((out_features, bytes_per_row * 2), dtype=torch.uint8, device=wd.device)
        if _nibble_order() == "high_first":
            q_u[:, 0::2] = q1
            q_u[:, 1::2] = q0
        else:
            q_u[:, 0::2] = q0
            q_u[:, 1::2] = q1
        if q_u.size(1) > in_features:
            q_u = q_u[:, :in_features]
        q = q_u.to(torch.int16) - 8
        w = q.to(torch.float32) * sd.to(torch.float32)[:, None]
        return xd.matmul(w.t()).to(x.device)
    except Exception:
        return _cpu_dequant_matmul(x, packed_w, scale, zero)


def _mps_dequant_matmul(x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Core ML/ANE path via MPS: run on 'mps' if available; else CPU.
    Note: This is not a true ANE kernel, but leverages Apple's GPU backend.
    """
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device('mps')
        xd = x.to(device)
        wd = packed_w.to(device)
        sd = scale.to(device)
        zd = zero.to(device)
        # Unpack and matmul on MPS
        out_features = wd.size(0)
        in_features = xd.size(-1)
        bytes_per_row = wd.size(1)
        q0 = wd & 0x0F
        q1 = (wd >> 4) & 0x0F
        q_u = torch.empty((out_features, bytes_per_row * 2), dtype=torch.uint8, device=wd.device)
        q_u[:, 0::2] = q0
        q_u[:, 1::2] = q1
        if q_u.size(1) > in_features:
            q_u = q_u[:, :in_features]
        q = q_u.to(torch.int16) - 8
        w = q.to(torch.float32) * sd.to(torch.float32)[:, None]
        return xd.matmul(w.t()).to(x.device)
    return _cpu_dequant_matmul(x, packed_w, scale, zero)


_REGISTRY: Dict[str, KernelFn] = {
    "cpu": _cpu_dequant_matmul,
    "cpu_dequant": _cpu_dequant_matmul,
    "dml": _dml_dequant_matmul,
    "coreml": _mps_dequant_matmul,
    "nnapi": _cpu_dequant_matmul,  # NNAPI integration occurs via ONNX/ExecuTorch graphs
}


def get_backend_name(default: str = "cpu") -> str:
    return os.environ.get("OMNICODER_INT4_BACKEND", default).strip().lower()


def matmul_int4(x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, backend: str | None = None) -> torch.Tensor:
    name = (backend or get_backend_name()).lower()
    kernel = _REGISTRY.get(name) or _REGISTRY["cpu"]
    return kernel(x, packed_w, scale, zero)




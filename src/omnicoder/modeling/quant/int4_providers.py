from __future__ import annotations

import torch
import torch.nn as nn
import os


def _get_nibble_order() -> str:
    order = os.getenv("OMNICODER_INT4_NIBBLE_ORDER", "low_first").strip().lower()
    return order if order in ("low_first", "high_first") else "low_first"


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed nibbles according to OMNICODER_INT4_NIBBLE_ORDER.

    Returns int8 of shape (..., 2) where the last dimension is [nibble0, nibble1]
    matching the configured order (low_first or high_first).
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    if _get_nibble_order() == "high_first":
        return torch.stack((high, low), dim=-1)
    return torch.stack((low, high), dim=-1)


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Dequantize int4 packed weights to float32.

    packed: (..., N/2) uint8 with two nibbles per byte
    scale/zero: broadcastable per-channel/group scales
    """
    nibbles = unpack_nibbles(packed).reshape(*packed.shape[:-1], -1).to(torch.float32)
    return (nibbles - zero.to(nibbles.dtype)) * scale.to(nibbles.dtype)


def _aligned_in_features(in_features: int) -> int:
    try:
        align = int(os.getenv("OMNICODER_INT4_ALIGN", "64"))
        if align <= 0:
            return in_features
    except Exception:
        align = 64
    # Round up to nearest multiple of align
    return ((in_features + align - 1) // align) * align


class Int4MatmulCPU(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int = 32) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        # Two nibbles per byte; allocate respecting alignment so callers can pack into it
        aligned_in = _aligned_in_features(in_features)
        self.weight_packed = nn.Parameter(torch.empty(out_features, (aligned_in + 1) // 2, dtype=torch.uint8), requires_grad=False)
        # Per-output-channel scales/zeros
        self.scale = nn.Parameter(torch.ones(out_features, 1), requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(out_features, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        w_full = dequant_int4(self.weight_packed, self.scale, self.zero)  # (out, in_aligned)
        w = w_full[:, : self.in_features]
        return x.to(torch.float32) @ w.t()


class Int4MatmulDML(nn.Module):
    """Prototype DirectML-backed int4 matmul.

    - Unpacks and dequantizes packed int4 weights on the DirectML device
    - Performs matmul on DirectML via PyTorch when torch-directml is present
    - Falls back to CPU reference if DirectML is unavailable
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 32) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        aligned_in = _aligned_in_features(in_features)
        self.weight_packed = nn.Parameter(torch.empty(out_features, (aligned_in + 1) // 2, dtype=torch.uint8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(out_features, 1), requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(out_features, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prefer native fused op if available
        try:
            fused = torch.ops.omnicoder_dml  # type: ignore[attr-defined]
            return fused.matmul_int4(x, self.weight_packed, self.scale, self.zero)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Otherwise attempt DML device path
        try:
            import torch_directml  # type: ignore
            dml = torch_directml.device()
            xp = x.to(dml)
            wp = self.weight_packed.to(dml)
            sc = self.scale.to(dml)
            zc = self.zero.to(dml)
            low = (wp & 0x0F).to(torch.int8)
            high = ((wp >> 4) & 0x0F).to(torch.int8)
            if _get_nibble_order() == "high_first":
                nibbles2 = torch.stack((high, low), dim=-1)
            else:
                nibbles2 = torch.stack((low, high), dim=-1)
            nibbles = nibbles2.reshape(wp.shape[0], -1).to(torch.float32)
            w = (nibbles - zc.to(torch.float32)) * sc.to(torch.float32)
            w = w[:, : self.in_features]
            return xp.to(torch.float32) @ w.t()
        except Exception:
            w_full = dequant_int4(self.weight_packed, self.scale, self.zero)
            w = w_full[:, : self.in_features]
            return x.to(torch.float32) @ w.t()


class Int4MatmulGeneric(nn.Module):
    """Backend-agnostic int4 matmul that calls a fused op if registered.

    backend: 'nnapi' or 'coreml'
    """

    def __init__(self, in_features: int, out_features: int, backend: str) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.backend = backend.strip().lower()
        aligned_in = _aligned_in_features(in_features)
        self.weight_packed = nn.Parameter(torch.empty(out_features, (aligned_in + 1) // 2, dtype=torch.uint8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(out_features, 1), requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(out_features, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            fused_ns = getattr(torch.ops, f"omnicoder_{self.backend}")  # type: ignore[attr-defined]
            if hasattr(fused_ns, "matmul_int4"):
                return fused_ns.matmul_int4(x, self.weight_packed, self.scale, self.zero)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Fallback to CPU reference
        w_full = dequant_int4(self.weight_packed, self.scale, self.zero)
        w = w_full[:, : self.in_features]
        return x.to(torch.float32) @ w.t()


def create_int4_provider(in_features: int, out_features: int, backend: str | None = None) -> nn.Module:
    """Factory for int4 matmul provider.

    backend: 'cpu' | 'dml' (defaults from OMNICODER_INT4_BACKEND env or 'cpu')
    """
    if backend is None:
        backend = os.getenv('OMNICODER_INT4_BACKEND', 'cpu').strip().lower()
    # Preload backend op namespaces for symbol stability (best-effort)
    try:
        if backend == 'nnapi':
            from ..kernels import omnicoder_nnapi_op  # noqa: F401
        elif backend == 'coreml':
            from ..kernels import omnicoder_coreml_op  # noqa: F401
        elif backend == 'dml':
            from ..kernels import omnicoder_dml_op  # noqa: F401
    except Exception:
        pass
    if backend == 'dml':
        return Int4MatmulDML(in_features, out_features)
    if backend in ('nnapi', 'coreml'):
        return Int4MatmulGeneric(in_features, out_features, backend)
    return Int4MatmulCPU(in_features, out_features)



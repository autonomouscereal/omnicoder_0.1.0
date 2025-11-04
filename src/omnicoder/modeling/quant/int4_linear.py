from __future__ import annotations

import math
from dataclasses import dataclass
import os
from typing import Tuple

import torch
import torch.nn as nn
from .int4_kernels import matmul_int4


@dataclass
class Int4Params:
    packed_weight: torch.Tensor  # uint8, shape (out_features, ceil(in_features/2))
    scale: torch.Tensor          # float32, shape (out_features,)
    zero: torch.Tensor           # float32, shape (out_features,)


def _pack_int4_weights(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int4 quantization.

    Args:
      w: float32 tensor of shape (out_features, in_features)

    Returns:
      packed (uint8), scale (float32), zero (float32)
    """
    out_features, in_features = w.shape
    # Optional alignment for device-friendly layouts (elements, multiple of 64 recommended)
    try:
        align_elems = int(os.environ.get("OMNICODER_INT4_ALIGN", "64"))
    except Exception:
        align_elems = 64
    # Per-row scale for symmetric quantization to [-7, 7]
    max_abs = w.abs().amax(dim=1) + 1e-8
    scale = max_abs / 7.0
    # Avoid div by zero
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.round(w * inv_scale[:, None]).clamp(-7, 7).to(torch.int8)
    # Shift to unsigned nibble [0, 14] (value 15 unused)
    q_u = (q + 7).to(torch.uint8)
    # Pack two nibbles per byte
    # Pad columns to even, then up to alignment boundary if requested
    pad_cols = 0
    if (in_features % 2) != 0:
        pad_cols = 1
    if align_elems > 0:
        rem = (in_features + pad_cols) % align_elems
        if rem != 0:
            pad_cols = pad_cols + (align_elems - rem)
    if pad_cols > 0:
        pad = torch.zeros((out_features, pad_cols), dtype=torch.uint8, device=q_u.device)
        from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
        q_u = _safe_cat(q_u, pad, 1)
        in_features += pad_cols
    q0 = q_u[:, 0::2]
    q1 = q_u[:, 1::2]
    # Provider-aligned nibble order: low_first (default) or high_first
    nibble_order = os.environ.get("OMNICODER_INT4_NIBBLE_ORDER", "low_first").lower()
    if nibble_order == "high_first":
        packed = ((q0 & 0x0F) << 4) | (q1 & 0x0F)
    else:
        packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
    zero = torch.zeros_like(scale)
    # Materialize contiguous with aten-only helper to avoid .contiguous()
    try:
        from omnicoder.utils.torchutils import safe_make_contiguous as _safe_contig  # type: ignore
        packed = _safe_contig(packed)
        scale = torch.ops.aten.to.dtype(scale, torch.float32, False, False)
        zero = torch.ops.aten.to.dtype(zero, torch.float32, False, False)
        return packed, scale, zero
    except Exception:
        return packed, scale.to(torch.float32), zero.to(torch.float32)


def _unpack_and_dequant_int4(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Dequantize to float32 matrix (out_features, in_features)."""
    # Unpack
    # packed: (out, ceil(in/2))
    bytes_per_row = packed.size(1)
    # Respect provider nibble order
    nibble_order = os.environ.get("OMNICODER_INT4_NIBBLE_ORDER", "low_first").lower()
    if nibble_order == "high_first":
        q1 = packed & 0x0F
        q0 = (packed >> 4) & 0x0F
    else:
        q0 = packed & 0x0F
        q1 = (packed >> 4) & 0x0F
    q_u = torch.empty((out_features, bytes_per_row * 2), dtype=torch.uint8, device=packed.device)
    q_u[:, 0::2] = q0
    q_u[:, 1::2] = q1
    # Trim if in_features was odd
    if q_u.size(1) > in_features:
        q_u = q_u[:, :in_features]
    q = q_u.to(torch.int16) - 7
    w = q.to(torch.float32) * scale[:, None]
    return w


class Int4Linear(nn.Module):
    """Weight-only int4 linear layer. Activations stay in float32.

    This module stores weights in packed int4 with per-output-channel scale.
    Forward dequantizes on-the-fly and performs a float32 matmul for simplicity
    and portability. This is not as fast as true int4 kernels, but functionally
    correct and a drop-in replacement to validate quantization.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        with torch.no_grad():
            packed, scale, zero = _pack_int4_weights(torch.ops.aten.to.dtype(linear.weight.data, torch.float32, False, False).cpu())
        # Store as buffers for easy device movement
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        if linear.bias is not None:
            self.register_buffer("bias", torch.ops.aten.to.dtype(linear.bias.data, torch.float32, False, False).cpu())
        else:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float32))
        # Cache for dequantized weight to avoid repeated unpack on hot decode path
        # We default to caching on first use regardless of input size to minimize per-token overhead.
        self._cached_weight: torch.Tensor | None = None
        self._cached_device: torch.device | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, *, in_features)
        # Always cache dequantized weights per device to eliminate per-token unpack cost
        target_device = x.device
        if self._cached_weight is None or self._cached_device != target_device:
            w = _unpack_and_dequant_int4(
                self.packed_weight.to(target_device, non_blocking=True),
                self.scale.to(target_device, non_blocking=True),
                self.zero.to(target_device, non_blocking=True),
                self.out_features,
                self.in_features,
            )
            self._cached_weight = w
            self._cached_device = target_device
        else:
            w = self._cached_weight
        # Move and cache bias lazily on device without repeated copies
        if getattr(self, "_bias_device", None) != target_device:
            self._bias_device = target_device
            self._bias_cached = self.bias.to(target_device, non_blocking=True)
        bias = getattr(self, "_bias_cached", self.bias)
        # Use dequantized fp32 matmul for correctness (provider kernels may use different packing offsets)
        y = x.matmul(w.t()) + bias
        return y


def quantize_module_int4_linear(module: nn.Module) -> int:
    """In-place replace nn.Linear with Int4Linear recursively.

    Returns number of replaced layers.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int4Linear(child))
            replaced += 1
        else:
            replaced += quantize_module_int4_linear(child)
    return replaced



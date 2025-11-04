from __future__ import annotations

"""
Lightweight registration of NNAPI fused ops under torch.ops.omnicoder_nnapi.*

These are CompositeImplicitAutograd implementations that use ATen ops so they
run on the current device (typically CPU in dev environments). They provide
symbol stability and a single callsite for fused MLA and INT4 matmul so that
ExecuTorch NNAPI delegate or future runtimes can bind native kernels without
changing Python call sites.
"""

from typing import Optional

import torch
from torch.library import Library


# Define schemas
_lib = Library("omnicoder_nnapi", "DEF")
_lib.define("mla(Tensor q, Tensor k, Tensor v, Tensor attn_mask, bool is_causal) -> Tensor")
_lib.define("matmul_int4(Tensor x, Tensor packed_w, Tensor scale, Tensor zero) -> Tensor")


def _mla_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> torch.Tensor:
    mask = None
    try:
        if attn_mask is not None and torch.is_tensor(attn_mask) and attn_mask.numel() > 0:
            mask = attn_mask
    except Exception:
        mask = None
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(is_causal))


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    return torch.stack((low, high), dim=-1)


def _dequant_int4(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    nibbles = _unpack_nibbles(packed).reshape(*packed.shape[:-1], -1).to(torch.float32)
    return (nibbles - zero.to(torch.float32)) * scale.to(torch.float32)


def _matmul_int4_impl(x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    w = _dequant_int4(packed_w, scale, zero)
    return x.to(torch.float32) @ w.t()


_lib.impl("mla", _mla_impl, "CompositeImplicitAutograd")
_lib.impl("matmul_int4", _matmul_int4_impl, "CompositeImplicitAutograd")

from __future__ import annotations

"""
Lightweight Python op registration for NNAPI fused MLA interface.

Registers torch.ops.omnicoder_nnapi.mla as a CompositeImplicitAutograd op
that falls back to PyTorch SDPA on the current device. This provides a stable
symbol for providers to override with native kernels without changing Python.
"""

import torch


def _mla_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor, is_causal: bool) -> torch.Tensor:
    # attn_mask may be an empty sentinel to indicate None
    mask = None if (attn_mask is not None and attn_mask.numel() == 0) else attn_mask
    try:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(is_causal))
    except Exception:
        # Explicit fallback
        logits = (q @ k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        if mask is not None:
            logits = logits + mask.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        return probs @ v


try:
    from torch.library import Library

    _lib = Library("omnicoder_nnapi", "DEF")
    _lib.define("mla(Tensor q, Tensor k, Tensor v, Tensor attn_mask, bool is_causal) -> Tensor")
    _impl = Library("omnicoder_nnapi", "IMPL", "CompositeImplicitAutograd")
    _impl.impl("mla", _mla_impl)
except Exception:
    # Torch < 2.1 or missing library API; ignore (callers will fallback to SDPA via provider registry)
    pass



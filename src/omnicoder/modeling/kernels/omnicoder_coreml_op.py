from __future__ import annotations

"""
Lightweight Core ML fused ops namespace registration.

Registers a CompositeImplicitAutograd implementation under torch.ops.omnicoder_coreml.mla
so provider registries can resolve a fused symbol consistently across backends.

This Composite fallback delegates to PyTorch SDPA and does not require coremltools.
If a native/MIL-backed implementation is provided later, it can override this impl.
"""

try:
    import torch
    from torch.library import Library
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise


_lib = Library("omnicoder_coreml", "DEF")
_lib.define("mla(Tensor q, Tensor k, Tensor v, Tensor attn_mask, bool is_causal) -> Tensor")


def _mla_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    mask = attn_mask if (attn_mask is not None and attn_mask.numel() > 0) else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(is_causal))


_lib.impl("mla", _mla_impl, "CompositeImplicitAutograd")

from __future__ import annotations

"""
Lightweight Core ML fused ops registration under torch.ops.omnicoder_coreml.*

These CompositeImplicitAutograd implementations mirror the NNAPI/DML symbols
so Python call sites can be backend-agnostic. When running on macOS with MPS,
ATen ops will execute on the MPS device automatically; on other hosts they run
on CPU. Native MIL replacements can bind to these symbols in the future.
"""

from typing import Optional

import torch
from torch.library import Library


_lib = Library("omnicoder_coreml", "DEF")
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



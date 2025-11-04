from __future__ import annotations

"""
Best-effort loader for omnicoder_dml_native (DirectML fused ops).

This attempts three strategies:
1) import omnicoder_dml_native as a Python extension module (e.g., .pyd)
2) load a local DLL/.pyd from build_dml/ (Release/Debug) via ctypes
3) no-op if not found; callers should gracefully fall back

Once loaded, torch.ops.omnicoder_dml.{mla, matmul_int4} should be registered.
"""

import ctypes
import os
import sys
from pathlib import Path


def _try_import_module() -> bool:
    try:
        import omnicoder_dml_native  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _try_load_ctypes() -> bool:
    root = Path(__file__).resolve().parents[5]  # project root
    candidates = []
    # Preferred Windows build outputs
    candidates.append(root / "build_dml" / "Release" / "omnicoder_dml_native.dll")
    candidates.append(root / "build_dml" / "Debug" / "omnicoder_dml_native.dll")
    candidates.append(root / "build_dml" / "Release" / "omnicoder_dml_native.pyd")
    candidates.append(root / "build_dml" / "Debug" / "omnicoder_dml_native.pyd")
    # In-tree Python site-packages style
    candidates.append(root / "src" / "omnicoder" / "modeling" / "kernels" / "omnicoder_dml_native.pyd")

    for path in candidates:
        if path.exists():
            try:
                ctypes.CDLL(str(path))
                # On Windows, also add directory to PATH so dependent DLLs resolve
                os.add_dll_directory(str(path.parent)) if hasattr(os, "add_dll_directory") else None
                return True
            except Exception:
                continue
    return False


def ensure_loaded() -> bool:
    if _try_import_module():
        return True
    if _try_load_ctypes():
        # Try import again to run module-level registration if applicable
        return _try_import_module()
    return False


# Eager attempt on import, but do not raise on failure
ensure_loaded()

"""
omnicoder_dml_op: Python-registered composite kernel for DML fused MLA

Exposes torch.ops.omnicoder_dml.mla(q, k, v, attn_mask_or_empty, is_causal)

This implementation moves tensors to the DirectML device and calls SDPA.
It is a functional fused path for DML environments and serves as a prototype
until a native kernel is provided.
"""

import torch


def _dml_mla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask_or_empty: torch.Tensor, is_causal: bool) -> torch.Tensor:
    import torch_directml  # type: ignore
    dev = torch_directml.device()
    # Move tensors to DML device without Python-side .to chaining in the hot path.
    qd0 = torch.ops.aten.to.device(q, dev, False, False)
    kd0 = torch.ops.aten.to.device(k, dev, False, False)
    vd0 = torch.ops.aten.to.device(v, dev, False, False)
    qd = torch.ops.aten.to.dtype(qd0, torch.float32, False, False)
    kd = torch.ops.aten.to.dtype(kd0, torch.float32, False, False)
    vd = torch.ops.aten.to.dtype(vd0, torch.float32, False, False)
    amd = None
    if attn_mask_or_empty.numel() > 0:
        am0 = torch.ops.aten.to.device(attn_mask_or_empty, dev, False, False)
        amd = torch.ops.aten.to.dtype(am0, torch.float32, False, False)
    yd = torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, attn_mask=amd, is_causal=bool(is_causal))
    return torch.ops.aten.to.device(yd, q.device, False, False)


try:
    lib = torch.library.Library("omnicoder_dml", "FRAGMENT")

    @lib.impl("mla")  # type: ignore
    def _mla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask_or_empty: torch.Tensor, is_causal: bool) -> torch.Tensor:  # noqa: D401
        """DML fused MLA composite op (Python)."""
        try:
            import torch_directml  # type: ignore
            _ = torch_directml.device()
            return _dml_mla(q, k, v, attn_mask_or_empty, is_causal)
        except Exception:
            # Fallback to SDPA on current device
            mask = attn_mask_or_empty if attn_mask_or_empty.numel() > 0 else None
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(is_causal))
except Exception:
    # If torch.library is unavailable, best-effort noop; registry will fallback
    pass

"""Lightweight registration of a fused MLA custom op under torch.ops.omnicoder_dml.mla.

This Python-level registration ensures callers can resolve the fused op symbol
without requiring a compiled extension. The implementation delegates to PyTorch
scaled_dot_product_attention, so it works across devices including DirectML
when tensors are already on the DirectML device (via torch-directml).

Schema:
  omnicoder_dml::mla(Tensor q, Tensor k, Tensor v, Tensor attn_mask, bool is_causal) -> Tensor

All tensors are expected in (B*H, T, D) format.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.library import Library
import os as _os



# Define the op schema under the omnicoder_dml namespace
_lib = Library("omnicoder_dml", "DEF")
_lib.define("mla(Tensor q, Tensor k, Tensor v, Tensor attn_mask, bool is_causal) -> Tensor")


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
    # Delegate to SDPA; tensors remain on caller's device (CPU/CUDA/DML/MPS)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(is_causal))


# Register a composite implementation that works on all devices
_lib.impl("mla", _mla_impl, "CompositeImplicitAutograd")


# Register an int4 matmul op with a pure-Python composite fallback so tests can call
# torch.ops.omnicoder_dml.matmul_int4 on any device without a native extension.
try:
    # Schema may already be defined below; guard to avoid duplicate-definition errors
    try:
        _lib.define("matmul_int4(Tensor x, Tensor packed_w, Tensor scale, Tensor zero) -> Tensor")
    except Exception:
        pass

    def _dequant_int4(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, target_in: int | None = None) -> torch.Tensor:
        import os as _os
        order = _os.getenv('OMNICODER_INT4_NIBBLE_ORDER', 'low_first').strip().lower()
        p = packed.to(torch.uint8).contiguous()
        rows, nb = int(p.shape[0]), int(p.shape[1])
        q = torch.empty((rows, nb * 2), dtype=torch.float32, device=p.device)
        # Explicit expansion preserves exact interleave order
        for j in range(nb):
            lo = (p[:, j] & 0x0F).to(torch.float32)
            hi = ((p[:, j] >> 4) & 0x0F).to(torch.float32)
            if order == 'high_first':
                q[:, 2 * j] = hi
                q[:, 2 * j + 1] = lo
            else:
                q[:, 2 * j] = lo
                q[:, 2 * j + 1] = hi
        if target_in is not None and q.size(1) > int(target_in):
            q = q[:, : int(target_in)]
        s = scale.to(q.device, dtype=torch.float32)
        z = zero.to(q.device, dtype=torch.float32)
        if s.dim() == 0:
            s = torch.ops.aten.reshape.default(s, (1, 1))
        if z.dim() == 0:
            z = torch.ops.aten.reshape.default(z, (1, 1))
        return (q - z) * s

    def _matmul_int4_impl(x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        # Composite fallback: compute y = x @ W^T directly from packed nibbles without fully materializing W
        # Shapes: x (B, IN); packed_w (OUT, ceil(IN/2))
        x = torch.ops.aten.to.dtype(x, torch.float32, False, False)
        B = x.shape[0] if x.dim() == 2 else 1
        IN = x.shape[1]
        OUT = packed_w.shape[0]
        nb = packed_w.shape[1]
        # Split input into even/odd positions; pad odd-slice if needed
        x_even = x[:, 0::2]
        x_odd = x[:, 1::2] if IN > 1 else x[:, :0]
        if x_odd.size(1) < x_even.size(1):
            # pad one zero column for odd side when IN is odd
            pad = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
            from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
            x_odd = _safe_cat(x_odd, pad, 1)
        # Prepare outputs
        y = torch.ops.aten.new_zeros.default(x, (x.shape[0], OUT), dtype=torch.float32)
        s = torch.ops.aten.to.dtype(scale, torch.float32, False, False)
        z = torch.ops.aten.to.dtype(zero, torch.float32, False, False)
        import os as _os
        order = _os.getenv('OMNICODER_INT4_NIBBLE_ORDER', 'low_first').strip().lower()
        # Iterate over output rows; vectorize across bytes within row
        for o in range(int(OUT)):
            p = packed_w[o].to(torch.uint8)
            low = (p & 0x0F).to(torch.float32)
            high = ((p >> 4) & 0x0F).to(torch.float32)
            if order == 'high_first':
                lo_vec, hi_vec = high, low
            else:
                lo_vec, hi_vec = low, high
            # Broadcast multiply and reduce: (B, nb) · (nb,) → (B,)
            # x_even/x_odd each have shape (B, nb) after potential pad
            # Ensure they match nb by trimming/padding
            if x_even.size(1) != int(nb):
                if x_even.size(1) > int(nb):
                    xe = x_even[:, :int(nb)]
                else:
                    pad = x_even.new_zeros((int(B), int(nb) - x_even.size(1)))
                    from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
                    xe = _safe_cat(x_even, pad, 1)
            else:
                xe = x_even
            if x_odd.size(1) != int(nb):
                if x_odd.size(1) > int(nb):
                    xo = x_odd[:, :int(nb)]
                else:
                    pad = x_odd.new_zeros((int(B), int(nb) - x_odd.size(1)))
                    from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
                    xo = _safe_cat(x_odd, pad, 1)
            else:
                xo = x_odd
            dot = (xe * lo_vec) + (xo * hi_vec)
            # y = scale * (sum_even_odd - zero * sum(x))
            base = dot.sum(dim=1)
            y[:, o] = (s * (base - z * x.sum(dim=1)))
        return y

    _lib.impl("matmul_int4", _matmul_int4_impl, "CompositeImplicitAutograd")
except Exception:
    # If registration fails here, a secondary attempt below will try again.
    pass

# Optional: Windows-only JIT build of native DML fused op prototype (non-fatal)
def _try_build_native() -> None:
    try:
        # Only attempt native JIT build when explicitly requested; avoid triggering hipify on CPU-only envs
        if _os.getenv('OMNICODER_BUILD_DML', '0') != '1':
            return
        if _os.name != 'nt':
            return
        src = _os.path.join(_os.path.dirname(__file__), 'dml_fused_attention.cpp')
        if not _os.path.exists(src):
            return
        # Build in a temp folder; name the extension uniquely
        try:
            from torch.utils.cpp_extension import load as _cpp_load  # type: ignore
        except Exception:
            return
        _cpp_load(name='omnicoder_dml_native', sources=[src], extra_cflags=['/O2'], verbose=False)
    except Exception:
        # Ignore build failures; Composite op remains available
        pass


_try_build_native()


# If a prebuilt native module is available (e.g., built via CMake), import it to register kernels
try:
    import omnicoder_dml_native  # type: ignore  # noqa: F401
except Exception:
    pass

# (removed duplicate int4 registration to avoid conflicting implementations)


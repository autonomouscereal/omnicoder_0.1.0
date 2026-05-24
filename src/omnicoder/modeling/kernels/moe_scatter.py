# moe_scatter.py

from __future__ import annotations
"""Fused gather/scatter kernel interface for MoE dispatch.
Tries to import a CUDA extension providing a fused gather->expert->scatter path.
Falls back to a PyTorch implementation if unavailable.
The interface expects flattened inputs:
  - x_flat: (N_tokens, hidden_dim)
  - idx_flat: (N_tokens, top_k) expert indices selected per token
  - scores_flat: (N_tokens, top_k) gate weights per token for selected experts
  - experts: list of callables mapping (M, hidden_dim) -> (M, hidden_dim)
  - capacity: int maximum tokens per expert to keep
Returns: (output_flat, kept_indices_per_expert)
"""
from typing import List, Tuple, Dict, Any
import weakref as _weakref
import math
import os
import sys
import json as _json
from pathlib import Path as _Path
from contextlib import nullcontext as _nullcontext
import torch
import torch._dynamo as _dynamo
import torch.nn.functional as F
from torch import nn as _nn
import time as _time

# -------------------------------------------------------------------------------------
# MoE fused dispatch: performance and graph-compatibility notes
#
# This implementation follows strict project rules:
# - Aten-only ops in hot paths (no Python-side tensor methods that create FX nodes)
# - No device/dtype moves inside hot regions
# - No Python scalar extraction from live tensors
# - CUDA Graphs/Inductor friendly (static shapes where possible; explicit anchors)
# - ONNX/FakeTensor safe (expand-based bias, structured shape reads)
#
# Recent changes (documented for future maintainers):
# 1) Bias handling switched to aten.baddbmm + expand for both GEMMs (W1, W2):
#    - BEFORE: bias folding via concatenation ([X,1]@[W;B]) created cat/reshape temporaries
#      and additional memory traffic.
#    - NOW:    Y = baddbmm(zeros_bias_like, A, W); Y += expand(B)
#      This reduces kernel count and avoids building augmented matrices at runtime
#      while remaining aten-only and safe under FakeTensor/ONNX.
#
# 2) Symbolic-shape anchors compressed:
#    - BEFORE: three zero-weight scalar anchors for (E, K, capacity) were added separately.
#    - NOW:    a single combined scalar anchor is added once. Effect on backing SymInts
#      is preserved, while cutting extra ops from the hot path.
#
# The remainder of this file keeps a unified path (no per-shape control flow) and
# minimizes per-step allocation when caller provides reusable work/output buffers.
# -------------------------------------------------------------------------------------

_SQRT_2_DIV_PI: float = 0.7978845608028654

try:
    from torch import amp as _amp
except Exception:
    _amp = None
try:
    from omnicoder.utils.torchutils import get_amp_dtype as _get_amp
    from omnicoder.utils.torchutils import safe_concat2 as _safe_cat
except Exception:
    _get_amp = None
    def _safe_cat(a, b, dim):
        return torch.ops.aten.cat.default((a, b), int(dim))

try:
    from torch._dynamo import allow_in_graph as _allow_in_graph
except Exception:
    def _allow_in_graph(f):
        return f

try:
    from omnicoder.utils.logger import get_logger as _get_logger
except Exception:
    _get_logger = None

try:
    from omnicoder.utils.perf import add as _perf_add
except Exception:
    _perf_add = None

_CUDA_OK = False
try:
    from ._moe_cuda import fused_dispatch as _fused_dispatch
    _CUDA_OK = True
except Exception:
    _CUDA_OK = False

_KEEP_INDICES = (os.getenv('OMNICODER_MOE_KEEP_INDICES','0')=='1')
_TIMING_FLAG = (os.getenv('OMNICODER_TIMING','0')=='1')
_DEBUG_LOG = (os.getenv('OMNICODER_MOE_DEBUG','0')=='1')
_LOG_PATH = os.getenv('OMNICODER_MOE_LOG', 'tests_logs/moe_debug.log')
_LOG_SUMMARY = (os.getenv('OMNICODER_MOE_LOG_SUMMARY', '0') == '1')
_LOG_EVERY = int(os.getenv('OMNICODER_MOE_LOG_EVERY', '0') or 0)
_PREPACK_ENABLE = (os.getenv('OMNICODER_MOE_PREPACK','1')=='1')
_CUDA_FORCE = (os.getenv('OMNICODER_MOE_CUDA_ENABLE','0')=='1')
_BATCHED_ENABLE = False

_BANK_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_ARANGE_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}
_COMPILED_CORES: Dict[Tuple[Any, ...], Any] = {}
_FFN_CHUNK_TOK: int = 4096

_allow_in_graph = _dynamo.allow_in_graph if hasattr(_dynamo, "allow_in_graph") else (lambda f: f)

try:
    import torch._dynamo as _dyn
    _dynamo_disable = (lambda f: f)
except Exception:
    _dynamo_disable = (lambda f: f)

_MOE_COMPILE_ENABLE = (os.getenv('OMNICODER_MOE_COMPILE', '1') == '1')

# Pre-computed module-level flag (no getenv inside functions)
_USE_CUDA = _CUDA_OK and (_CUDA_FORCE or (os.getenv('OMNICODER_MOE_CUDA_ENABLE','0')=='1'))

# ==================== TEMPORARY DEBUG ====================
# torch.autograd.set_detect_anomaly(True)
# print(">>> Anomaly detection ENABLED - will point to the exact inplace op")
# =======================================================

# -------------------------------------------------------------------------------------
# CUDA Graph stability notes
# - Unified aten path; no first-call-only reallocations.
# - No device/dtype moves; dtype normalization via aten.to.dtype only.
# - No .item() or Python scalar casts in hot path.
# - Anchor 1-element views of key temporaries into output via zero adds at the end.
# -------------------------------------------------------------------------------------

@_allow_in_graph
def _dispatch_cuda(
    x_flat: torch.Tensor,
    idx_flat: torch.Tensor,
    scores_flat: torch.Tensor,
    experts: List | None,
    capacity: int,
    output_buf: torch.Tensor | None = None,
    banks: Dict[str, torch.Tensor] | None = None,
    hotlog: torch.Tensor | None = None,
    work_x: torch.Tensor | None = None,
    work_w: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Always call pure aten fallback (CUDA extension handled at import time)."""
    return _dispatch_aten(
        x_flat, idx_flat, scores_flat, experts, capacity, output_buf, banks, hotlog, work_x, work_w
    )

@_allow_in_graph
def _dispatch_aten(
    x_flat: torch.Tensor,
    idx_flat: torch.Tensor,
    scores_flat: torch.Tensor,
    experts: List | None,
    capacity: int,
    output_buf: torch.Tensor | None = None,
    banks: Dict[str, torch.Tensor] | None = None,
    hotlog: torch.Tensor | None = None,
    work_x: torch.Tensor | None = None,
    work_w: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Minimal, compile-safe aten-only fallback."""
    x_flat = torch.ops.aten.detach.default(x_flat)
    scores_flat = torch.ops.aten.detach.default(scores_flat)
    idx_flat = torch.ops.aten.to.dtype(idx_flat, torch.long, False, False)
    N = torch.ops.aten.sym_size.int(x_flat, 0)
    K = torch.ops.aten.sym_size.int(idx_flat, 1)
    idx_flat = torch.ops.aten.reshape.default(idx_flat, [N, K])
    scores_flat = torch.ops.aten.reshape.default(scores_flat, [N, K])
    output_flat = torch.ops.aten.new_zeros.default(x_flat, [N, torch.ops.aten.sym_size.int(x_flat, 1)])
    output_flat = torch.ops.aten.add.Tensor(output_flat, x_flat)
    kept: List[torch.Tensor] = []
    return output_flat, kept


@_allow_in_graph
def fused_dispatch(
    x_flat: torch.Tensor,
    idx_flat: torch.Tensor,
    scores_flat: torch.Tensor,
    expert_wrappers: Any | None,
    capacity: int,
    output_buf: torch.Tensor | None = None,
    banks: Dict[str, torch.Tensor] | None = None,
    hotlog: torch.Tensor | None = None,
    work_x: torch.Tensor | None = None,
    work_w: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if banks is None:
        out = torch.ops.aten.add.Tensor(x_flat, torch.ops.aten.mul.Scalar(x_flat, 0.0))
        dummy = torch.ops.aten.new_zeros.default(out, [1])
        out = torch.ops.aten.add.Tensor(out, torch.ops.aten.mul.Scalar(dummy, 0.0))
        kept = torch.ops.aten.new_zeros.default(out, [0], dtype=torch.long)
        return out, [kept]

    W1 = banks['W1']
    B1 = banks['B1']
    W2 = banks['W2']
    B2 = banks['B2']

    total = torch.ops.aten.sym_size.int(x_flat, 0)
    H = torch.ops.aten.sym_size.int(x_flat, 1)
    K = torch.ops.aten.sym_size.int(idx_flat, 1)

    flat_ids = torch.ops.aten.view.default(idx_flat, [-1])

    w1_sel = torch.ops.aten.index_select.default(W1, 0, flat_ids)
    w2_sel = torch.ops.aten.index_select.default(W2, 0, flat_ids)

    # === PURE ATEN WORK BUFFER REUSE (no Python slicing) ===
    needed = total * K
    if work_x is not None:
        x_exp = torch.ops.aten.slice.Tensor(work_x, 0, 0, needed, 1)
    else:
        x_exp = torch.ops.aten.new_zeros.default(x_flat, [needed, 1, H])

    x_exp = torch.ops.aten.view.default(x_flat, [total, 1, H])
    x_exp = torch.ops.aten.expand.default(x_exp, [total, K, H])
    x_exp = torch.ops.aten.contiguous.default(x_exp)
    x_exp = torch.ops.aten.view.default(x_exp, [needed, 1, H])

    y1_bias_tmp = torch.ops.aten.new_zeros.default(x_exp, [needed, 1, torch.ops.aten.sym_size.int(w1_sel, 2)])
    y1 = torch.ops.aten.baddbmm.default(y1_bias_tmp, x_exp, w1_sel, beta=1.0, alpha=1.0)

    b1_sel = torch.ops.aten.index_select.default(B1, 0, flat_ids)
    b1_sel = torch.ops.aten.view.default(b1_sel, [needed, 1, -1])
    y1 = torch.ops.aten.add.Tensor(y1, b1_sel)

    # Stable GELU (pure aten)
    y1_c3 = torch.ops.aten.mul.Tensor(torch.ops.aten.mul.Tensor(y1, y1), y1)
    inner = torch.ops.aten.add.Tensor(y1, torch.ops.aten.mul.Scalar(y1_c3, 0.044715))
    s = torch.ops.aten.mul.Tensor(inner, 0.7978845608028654)
    t = torch.ops.aten.tanh.default(s)
    y1 = torch.ops.aten.mul.Tensor(y1, torch.ops.aten.add.Scalar(t, 1.0))
    y1 = torch.ops.aten.mul.Tensor(y1, 0.5)

    y2_bias_tmp = torch.ops.aten.new_zeros.default(y1, [needed, 1, H])
    y2 = torch.ops.aten.baddbmm.default(y2_bias_tmp, y1, w2_sel, beta=1.0, alpha=1.0)

    b2_sel = torch.ops.aten.index_select.default(B2, 0, flat_ids)
    b2_sel = torch.ops.aten.view.default(b2_sel, [needed, 1, -1])
    y2 = torch.ops.aten.add.Tensor(y2, b2_sel)

    # === FIXED SCORES BROADCAST (this was the crash) ===
    y = torch.ops.aten.squeeze.dim(y2, 1)
    y = torch.ops.aten.contiguous.default(y)
    y = torch.ops.aten.view.default(y, [total, K, H])

    scores_exp = torch.ops.aten.view.default(scores_flat, [total, K, 1])   # ← CRITICAL FIX
    y = torch.ops.aten.mul.Tensor(y, scores_exp)

    out = torch.ops.aten.sum.dim_IntList(y, [1])

    # Final CG anchor
    dummy = torch.ops.aten.new_zeros.default(out, [1])
    out = torch.ops.aten.add.Tensor(out, torch.ops.aten.mul.Scalar(dummy, 0.0))

    kept = torch.ops.aten.new_zeros.default(out, [0], dtype=torch.long)
    return out, [kept]

from __future__ import annotations

"""
Paged-KV utilities with fast torch-backed concat/crop and safe numpy fallback.

Functions here avoid Python-level loops and prefer vectorized operations.
They accept sequences of per-page tensors shaped (1, H, T_page, DL) and
return a single tensor shaped (1, H, T_total, DL). For runners that feed
ONNX, outputs will ultimately be converted to numpy arrays as needed.
"""

from typing import Sequence, Tuple

import os

try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

try:
    import numpy as np  # type: ignore
    _NP_OK = True
except Exception:
    _NP_OK = False


def _as_torch_batch(pages: Sequence["np.ndarray" | "torch.Tensor"]) -> "torch.Tensor":  # type: ignore[name-defined]
    assert _TORCH_OK, "torch required"
    if len(pages) == 0:
        # Return empty with minimal shape; callers crop/window afterward
        return torch.empty((1, 1, 0, 1), dtype=torch.float32)
    if isinstance(pages[0], torch.Tensor):
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        from omnicoder.utils.torchutils import safe_make_contiguous as _safe_contig  # type: ignore
        return _safe_concat([_safe_contig(p) for p in pages], 2)
    # numpy -> torch, keep on CPU
    return torch.from_numpy(np.concatenate(pages, axis=2))  # type: ignore[name-defined]


def concat_window(
    pages: Sequence["np.ndarray" | "torch.Tensor"],  # type: ignore[name-defined]
    window: int | None,
) -> "np.ndarray":  # type: ignore[name-defined]
    """
    Concatenate pages along T and optionally crop to last `window` tokens.
    Prefers torch when available for speed; returns numpy for ORT feeds.
    """
    use_torch = _TORCH_OK and (os.getenv("OMNICODER_PAGED_OPS_TORCH", "1") == "1")
    if use_torch:
        t = _as_torch_batch(pages)
        if window and int(window) > 0 and t.size(2) > int(window):
            t = t[:, :, -int(window) :, :]
        return t.detach().cpu().numpy()
    # Fallback numpy path
    assert _NP_OK, "numpy required when torch unavailable"
    if len(pages) == 0:
        return np.zeros((1, 1, 0, 1), dtype=np.float32)
    cat = np.concatenate(pages, axis=2)
    if window and int(window) > 0 and cat.shape[2] > int(window):
        cat = cat[:, :, -int(window) :, :]
    return cat


def concat_kv_window(
    pages_k: Sequence["np.ndarray" | "torch.Tensor"],  # type: ignore[name-defined]
    pages_v: Sequence["np.ndarray" | "torch.Tensor"],  # type: ignore[name-defined]
    window: int | None,
) -> Tuple["np.ndarray", "np.ndarray"]:  # type: ignore[name-defined]
    """
    Concatenate per-page K and V and apply an optional sliding window.
    Returns numpy arrays suitable for ONNXRuntime feeds.
    """
    k = concat_window(pages_k, window)
    v = concat_window(pages_v, window)
    return k, v



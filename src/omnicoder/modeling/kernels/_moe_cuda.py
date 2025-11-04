"""Python shim for the CUDA MoE fused dispatcher.

This module attempts to import a compiled extension named `moe_cuda_ext` and
exposes a `fused_dispatch` function with the same signature expected by
`moe_scatter.fused_dispatch`.

If the extension is not available, importing this module will raise so the
caller can fall back to the aten path.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import torch

try:
    import importlib  # type: ignore
    moe_ext = importlib.import_module('moe_cuda_ext')
except Exception as e:  # pragma: no cover
    # Re-raise to signal unavailability to the caller
    raise


def fused_dispatch(
    x_flat: torch.Tensor,
    idx_flat: torch.Tensor,
    scores_flat: torch.Tensor,
    experts: List,
    capacity: int,
    output_buf: torch.Tensor,
    banks: Dict[str, torch.Tensor],
    hotlog: torch.Tensor | None,
    work_x: torch.Tensor | None,
    work_w: torch.Tensor | None,
) -> Tuple[torch.Tensor, None]:
    # Delegate to CUDA extension; it must accept the same arguments.
    return moe_ext.fused_dispatch(
        x_flat,
        idx_flat,
        scores_flat,
        experts,
        int(capacity),
        output_buf,
        banks,
        hotlog,
        work_x,
        work_w,
    )



from __future__ import annotations

import os
import torch

from omnicoder.modeling.kernels.mla_providers import resolve_backend


def _make_inputs(b: int = 1, h: int = 4, t: int = 8, tp: int = 12, dl: int = 16):
    torch.manual_seed(0)
    q = torch.randn(b * h, t, dl)
    k = torch.randn(b * h, tp, dl)
    v = torch.randn(b * h, tp, dl)
    # causal over full tp when mask is None; use sliding window mask when provided
    attn_mask = None
    return q, k, v, attn_mask


def _sdpa_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None, is_causal: bool) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


@torch.no_grad()
def test_mla_cpu_matches_sdpa() -> None:
    q, k, v, attn_mask = _make_inputs()
    fn = resolve_backend('cpu')
    assert fn is not None
    y = fn(q, k, v, attn_mask, True)
    y_ref = _sdpa_ref(q, k, v, attn_mask, True)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_mla_dml_falls_back_or_runs() -> None:
    q, k, v, attn_mask = _make_inputs()
    os.environ["OMNICODER_MLA_BACKEND"] = "dml"
    fn = resolve_backend('dml')
    assert fn is not None
    y = fn(q, k, v, attn_mask, True)
    # If DML available, result should be finite; otherwise, fallback to CPU ref path
    assert torch.isfinite(y).all()



import os
import torch
from omnicoder.modeling.kernels.mla_providers import resolve_backend
from omnicoder.modeling.kernels import omnicoder_dml_op  # ensure best-effort load


def test_mla_cpu_provider_shapes():
    fn = resolve_backend('cpu')
    assert callable(fn)
    b, h, t, dl = 1, 2, 4, 8
    q = torch.randn(b*h, t, dl)
    k = torch.randn(b*h, t, dl)
    v = torch.randn(b*h, t, dl)
    y = fn(q, k, v, None, True)
    assert y.shape == (b*h, t, dl)


def test_mla_dml_backend_fallback():
    os.environ['OMNICODER_MLA_BACKEND'] = 'dml'
    fn = resolve_backend('dml')
    b, h, t, dl = 1, 2, 3, 6
    q = torch.randn(b*h, t, dl)
    k = torch.randn(b*h, t, dl)
    v = torch.randn(b*h, t, dl)
    y = fn(q, k, v, None, True)
    assert y.shape == (b*h, t, dl)

import os
import torch


def test_mla_provider_resolves_and_runs_cpu():
    os.environ['OMNICODER_MLA_BACKEND'] = 'cpu'
    from omnicoder.modeling.kernels.mla_providers import resolve_backend
    fn = resolve_backend('cpu')
    assert fn is not None
    B, H, T, DL = 1, 2, 3, 4
    q = torch.randn(B * H, T, DL)
    k = torch.randn(B * H, T, DL)
    v = torch.randn(B * H, T, DL)
    y = fn(q, k, v, None, True)
    assert y.shape == (B * H, T, DL)


def test_mla_provider_resolves_dml_without_error(monkeypatch):
    # If torch-directml is not present, backend should fall back and still run
    os.environ['OMNICODER_MLA_BACKEND'] = 'dml'
    from omnicoder.modeling.kernels.mla_providers import resolve_backend
    fn = resolve_backend('dml')
    assert fn is not None
    B, H, T, DL = 1, 2, 3, 4
    q = torch.randn(B * H, T, DL)
    k = torch.randn(B * H, T, DL)
    v = torch.randn(B * H, T, DL)
    y = fn(q, k, v, None, True)
    assert y.shape == (B * H, T, DL)



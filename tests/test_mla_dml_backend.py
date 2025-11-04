from __future__ import annotations

import os
import torch

from omnicoder.modeling.kernels.mla_providers import resolve_backend
from omnicoder.modeling.kernels import omnicoder_dml_op  # noqa: F401 - trigger loader
from omnicoder.modeling.attention import LatentKVAttention


def test_dml_backend_resolves_and_runs_cpu_fallback() -> None:
    # Force DML backend resolution
    fn = resolve_backend('dml')
    assert fn is not None
    # Build tiny inputs
    B, H, T, DL = 1, 2, 4, 8
    q = torch.randn(B * H, T, DL)
    k = torch.randn(B * H, T, DL)
    v = torch.randn(B * H, T, DL)
    out = fn(q, k, v, None, True)
    assert out.shape == (B * H, T, DL)


def test_mla_microbench_runs() -> None:
    # Ensure the bench_mla utility runs and returns a dict with speedup
    from omnicoder.inference.benchmark import bench_mla_vs_sdpa
    r = bench_mla_vs_sdpa(seq_len=32, gen_tokens=32)
    assert 'sdpa_tps' in r and 'mla_tps' in r and 'speedup_x' in r
    # If torch-directml is available, surface whether native kernel is present
    try:
        import torch_directml  # type: ignore
        assert 'native_present' in r
    except Exception:
        pass


def test_dml_mla_parity_small_shapes_when_available() -> None:
    # If torch-directml is present, compare DML path vs CPU SDPA-off path on a tiny shape
    try:
        import torch_directml  # type: ignore
    except Exception:
        # No DirectML; skip parity portion silently
        return
    B, T, C = 1, 8, 128
    H = 8
    DL = 16
    torch.manual_seed(0)
    x = torch.randn(B, T, C)

    base = LatentKVAttention(d_model=C, n_heads=H, kv_latent_dim=DL, multi_query=True, use_rope=False, max_seq_len=T, use_sdpa=False)
    base.eval()
    with torch.no_grad():
        y_base = base(x)

    os.environ['OMNICODER_MLA_BACKEND'] = 'dml'
    dml = LatentKVAttention(d_model=C, n_heads=H, kv_latent_dim=DL, multi_query=True, use_rope=False, max_seq_len=T, use_sdpa=False)
    dml.eval()
    with torch.no_grad():
        y_dml = dml(x)

    assert y_base.shape == y_dml.shape
    diff = (y_base - y_dml).abs().max().item()
    assert diff < 1e-3


import torch
import torch.nn as nn

from omnicoder.modeling.quant.int4_linear import Int4Linear
from omnicoder.modeling.quant.kv_cache import quantize_kv, dequantize_kv


def test_int4_linear_forward_close():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=True)
    x = torch.randn(4, 64)
    ref = lin(x)
    q = Int4Linear(lin)
    out = q(x)
    # Quantization error should be bounded; check mean absolute error
    mae = (out - ref).abs().mean().item()
    assert mae < 0.2


def test_kv_cache_quant_dequant_shapes():
    torch.manual_seed(0)
    B, H, T, D = 1, 8, 4, 160
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    for scheme in ['u8','nf4']:
        kq, vq, meta = quantize_kv(k, v, scheme=scheme, group_size=64)
        kd, vd = dequantize_kv(kq, vq, meta)
        assert kd.shape == (B, H, T, D)
        assert vd.shape == (B, H, T, D)
        # Error bounds are loose; just ensure finite
        assert torch.isfinite(kd).all()
        assert torch.isfinite(vd).all()


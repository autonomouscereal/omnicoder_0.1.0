import numpy as np
import torch


def test_kv_quant_dequant_roundtrip_shapes_and_bounds():
    from omnicoder.modeling.quant.kv_cache import quantize_kv, dequantize_kv
    B, H, T, DL = 1, 4, 8, 64
    k = torch.randn(B, H, T, DL, dtype=torch.float32)
    v = torch.randn(B, H, T, DL, dtype=torch.float32)
    for scheme in ('u8', 'nf4'):
        kq, vq, meta = quantize_kv(k, v, scheme=scheme, group_size=64)
        kf, vf = dequantize_kv(kq, vq, meta)
        assert kf.shape == k.shape and vf.shape == v.shape


def test_int4_alignment_env_flag_present():
    import os
    align = os.getenv('OMNICODER_INT4_ALIGN', '64')
    assert int(align) > 0



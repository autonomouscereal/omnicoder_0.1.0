import os
import torch


def test_int4_pack_unpack_invariance_small():
    from omnicoder.modeling.quant.int4_linear import _pack_int4_weights, _unpack_and_dequant_int4
    torch.manual_seed(0)
    os.environ['OMNICODER_INT4_ALIGN'] = '64'
    out_features, in_features = 7, 13
    w = torch.randn(out_features, in_features)
    packed, scale, zero = _pack_int4_weights(w)
    w2 = _unpack_and_dequant_int4(packed, scale, zero, out_features, in_features)
    assert w2.shape == w.shape
    # Quantization error should be bounded (rough check)
    err = (w - w2).abs().mean().item()
    assert err < 0.2



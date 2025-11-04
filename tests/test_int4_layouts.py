from __future__ import annotations

import os
import torch

from omnicoder.modeling.quant.int4_linear import _pack_int4_weights, _unpack_and_dequant_int4


def test_int4_pack_unpack_identity() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 77)
    os.environ["OMNICODER_INT4_ALIGN"] = "64"
    os.environ["OMNICODER_INT4_NIBBLE_ORDER"] = "low_first"
    packed, scale, zero = _pack_int4_weights(w)
    w_rec = _unpack_and_dequant_int4(packed, scale, zero, w.size(0), w.size(1))
    # Quantization is lossy, but pack->unpack should be consistent with the original quantization grid
    # Re-pack the reconstructed matrix and compare packed bytes for exact equality
    repacked, _, _ = _pack_int4_weights(w_rec)
    assert torch.equal(packed, repacked)


def test_int4_nibble_order_variants() -> None:
    torch.manual_seed(1)
    w = torch.randn(8, 9)
    os.environ["OMNICODER_INT4_ALIGN"] = "64"
    for order in ("low_first", "high_first"):
        os.environ["OMNICODER_INT4_NIBBLE_ORDER"] = order
        packed, scale, zero = _pack_int4_weights(w)
        w_rec = _unpack_and_dequant_int4(packed, scale, zero, w.size(0), w.size(1))
        repacked, _, _ = _pack_int4_weights(w_rec)
        assert torch.equal(packed, repacked)



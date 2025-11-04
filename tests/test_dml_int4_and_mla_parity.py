from __future__ import annotations

import os
import pytest
import torch


def test_dml_mla_parity_cpu_sdpa():
    # Always run: if torch-directml is present, use its device; else fall back to CPU.
    try:
        import torch_directml  # type: ignore
        dev = torch_directml.device()
    except Exception:
        dev = torch.device('cpu')
    from omnicoder.modeling.kernels import omnicoder_dml_op  # ensure registration
    B, H, T, DL = 1, 4, 8, 16
    q = torch.randn(B*H, T, DL)
    k = torch.randn(B*H, T, DL)
    v = torch.randn(B*H, T, DL)
    # CPU baseline
    y_cpu = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    # DML fused path (Composite will dispatch to SDPA when tensors on DML)
    qd, kd, vd = q.to(dev), k.to(dev), v.to(dev)
    y_dml = torch.ops.omnicoder_dml.mla(qd, kd, vd, torch.empty(0, device=dev), True).to('cpu')
    assert torch.allclose(y_cpu, y_dml, atol=1e-4, rtol=1e-4)


def _pack_int4_ref(w: torch.Tensor, nibble_order: str = 'low_first') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = w.to(torch.float32)
    # Symmetric affine quant to int4 range [0..15] around zero point 8
    mx = w.abs().amax()
    scale = mx / 7.0 if mx > 0 else torch.tensor(1.0)
    zp = torch.tensor(8.0)
    q = torch.clamp(torch.round(w / scale + zp), 0, 15).to(torch.uint8)
    # Pack two nibbles per byte
    q = q.reshape(q.size(0), -1)
    low = q[:, 0::2]
    high = q[:, 1::2]
    if high.numel() < low.numel():
        # pad last high nibble with zero
        high = torch.nn.functional.pad(high, (0, 1))
    if nibble_order == 'high_first':
        packed = ((high & 0x0F) << 4) | (low & 0x0F)
    else:
        packed = ((low & 0x0F) | ((high & 0x0F) << 4))
    return packed.contiguous(), scale.to(torch.float32), zp.to(torch.float32)


def test_dml_int4_matmul_correctness():
    try:
        from omnicoder.modeling.kernels import omnicoder_dml_op  # ensure registration
    except Exception:
        pytest.skip("DML composite op not available")
    torch.manual_seed(0)
    B, IN, OUT = 2, 32, 24
    x = torch.randn(B, IN)
    W = torch.randn(OUT, IN)
    nib = os.getenv('OMNICODER_INT4_NIBBLE_ORDER', 'low_first')
    packed, scale, zero = _pack_int4_ref(W, nibble_order=nib)
    # Dequantize reference from the same packed int4 to validate operator math
    low = (packed & 0x0F).to(torch.float32)
    high = ((packed >> 4) & 0x0F).to(torch.float32)
    OUT, nb = packed.shape
    qflat = torch.zeros((OUT, nb * 2), dtype=torch.float32)
    if nib == 'high_first':
        qflat[:, 0::2] = high
        qflat[:, 1::2] = low
    else:
        qflat[:, 0::2] = low
        qflat[:, 1::2] = high
    # Trim if IN is odd
    if qflat.size(1) > IN:
        qflat = qflat[:, :IN]
    Wq = (qflat - float(zero)) * float(scale)
    y_ref_q = x @ Wq.t()
    y_op = torch.ops.omnicoder_dml.matmul_int4(x, packed, scale, zero)
    assert torch.allclose(y_ref_q, y_op, atol=1e-5, rtol=1e-5)



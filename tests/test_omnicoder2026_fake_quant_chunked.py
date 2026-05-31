from __future__ import annotations

import torch
import torch.nn.functional as F

from omnicoder.modeling.omnicoder2026 import MHCResidual, OmniCoder2026Config, QuantAwareLinear, _fake_quant_weight, _fake_quant_weight_value


def test_chunked_fake_quant_linear_matches_full_ste_reference(monkeypatch):
    chunk_rows = 3
    group_size = 4
    in_features = 10
    out_features = 7
    batch = 2
    seq = 3
    assert in_features % group_size != 0
    assert out_features % chunk_rows != 0

    monkeypatch.setenv("OMNICODER2026_FAKE_QUANT_CHUNK_ROWS", str(chunk_rows))
    monkeypatch.setenv("OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS", "0")

    layer = QuantAwareLinear(
        in_features,
        out_features,
        bias=True,
        fake_quant=True,
        group_size=group_size,
    )
    assert layer.fake_quant_chunk_rows == chunk_rows
    assert layer.fake_quant_max_full_elements == 0

    x = torch.linspace(
        -1.5,
        1.5,
        steps=batch * seq * in_features,
        dtype=torch.float32,
    ).reshape(batch, seq, in_features)
    x.requires_grad_(True)

    with torch.no_grad():
        layer.weight.copy_(
            torch.linspace(
                -0.9,
                0.8,
                steps=out_features * in_features,
                dtype=torch.float32,
            ).reshape(out_features, in_features)
        )
        layer.bias.copy_(torch.linspace(-0.2, 0.3, steps=out_features, dtype=torch.float32))

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = layer.weight.detach().clone().requires_grad_(True)
    ref_bias = layer.bias.detach().clone().requires_grad_(True)

    called_chunked = False
    chunked_impl = layer._chunked_fake_quant_linear

    def wrapped_chunked(inp: torch.Tensor) -> torch.Tensor:
        nonlocal called_chunked
        called_chunked = True
        return chunked_impl(inp)

    monkeypatch.setattr(layer, "_chunked_fake_quant_linear", wrapped_chunked)

    actual = layer(x)
    expected = F.linear(ref_x, _fake_quant_weight(ref_weight, group_size), ref_bias)

    assert called_chunked
    assert actual.shape == (batch, seq, out_features)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    grad_output = torch.linspace(
        -0.7,
        0.6,
        steps=actual.numel(),
        dtype=torch.float32,
    ).reshape_as(actual)

    actual.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(layer.weight.grad, ref_weight.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(layer.bias.grad, ref_bias.grad, rtol=1e-6, atol=1e-6)


def test_chunked_fake_quant_linear_accepts_mixed_activation_and_weight_dtype(monkeypatch):
    monkeypatch.setenv("OMNICODER2026_FAKE_QUANT_CHUNK_ROWS", "2")
    monkeypatch.setenv("OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS", "0")

    layer = QuantAwareLinear(6, 5, bias=True, fake_quant=True, group_size=4).half()
    x = torch.linspace(-0.5, 0.5, steps=12, dtype=torch.float32).reshape(2, 6)
    x.requires_grad_(True)

    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-0.4, 0.5, steps=30, dtype=torch.float16).reshape(5, 6))
        layer.bias.copy_(torch.linspace(-0.2, 0.2, steps=5, dtype=torch.float16))

    actual = layer(x)
    expected = F.linear(
        x,
        _fake_quant_weight_value(layer.weight.detach(), layer.group_size).float(),
        layer.bias.detach().float(),
    )

    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    actual.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


def test_mhc_residual_preserves_pipeline_dtype():
    cfg = OmniCoder2026Config(d_model=16, hc_mult=4, fake_quant=True, fake_quant_group_size=8)
    residual = MHCResidual(cfg).half()

    x = torch.randn(2, 5, 16, dtype=torch.float16)
    update = torch.randn(2, 5, 16, dtype=torch.float32)

    actual = residual(x, update)

    assert actual.dtype == torch.float16

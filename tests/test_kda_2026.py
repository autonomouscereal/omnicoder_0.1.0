import pytest
import torch

from omnicoder.modeling.kda_2026 import (
    GatedDeltaNet2,
    KDA,
    KaczmarzLinearAttention,
    gated_deltanet2_pytorch,
    kaczmarz_linear_attention_pytorch,
    kda_pytorch,
)


def _devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "fn,gate_names",
    [
        (kda_pytorch, ("beta", "forget")),
        (gated_deltanet2_pytorch, ("write_gate", "forget_gate")),
        (kaczmarz_linear_attention_pytorch, ("relaxation",)),
    ],
)
def test_recurrent_paths_shapes_stability_and_chunk_equivalence(device, fn, gate_names):
    torch.manual_seed(7)
    batch, time, heads, key_dim, value_dim = 2, 5, 3, 4, 6
    q = torch.randn(batch, time, heads, key_dim, device=device) * 0.25
    k = torch.randn(batch, time, heads, key_dim, device=device) * 0.25
    v = torch.randn(batch, time, heads, value_dim, device=device) * 0.25
    gates = {
        name: torch.randn(batch, time, heads, device=device) * 0.1
        for name in gate_names
    }
    if fn is kda_pytorch:
        gates["beta"] = torch.sigmoid(gates["beta"])
        gates["forget"] = torch.sigmoid(gates["forget"])
    if fn is kaczmarz_linear_attention_pytorch:
        gates["relaxation"] = torch.sigmoid(gates["relaxation"])

    full, full_state = fn(q, k, v, **gates)
    first, state = fn(q[:, :2], k[:, :2], v[:, :2], **{name: gates[name][:, :2] for name in gate_names})
    second, split_state = fn(
        q[:, 2:],
        k[:, 2:],
        v[:, 2:],
        initial_state=state,
        **{name: gates[name][:, 2:] for name in gate_names},
    )
    split = torch.cat([first, second], dim=1)

    assert full.shape == (batch, time, heads, value_dim)
    assert full_state.shape == (batch, heads, key_dim, value_dim)
    assert full_state.dtype == torch.float32
    assert torch.isfinite(full).all()
    assert torch.isfinite(full_state).all()
    torch.testing.assert_close(split, full, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(split_state, full_state, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("module_cls", [KDA, GatedDeltaNet2, KaczmarzLinearAttention])
def test_recurrent_modules_tiny_backward(device, module_cls):
    torch.manual_seed(11)
    layer = module_cls(d_model=16, n_heads=4).to(device)
    x = torch.randn(2, 4, 16, device=device, requires_grad=True)

    y, state = layer(x, return_state=True)
    loss = y.square().mean() + state.square().mean() * 1e-3
    loss.backward()

    assert y.shape == x.shape
    assert state.shape == (2, 4, 4, 4)
    assert state.dtype == torch.float32
    assert torch.isfinite(y).all()
    assert torch.isfinite(state).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

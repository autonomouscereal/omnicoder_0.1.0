import pytest
import torch
import torch.nn.functional as F

from omnicoder.modeling.kda_2026 import (
    GatedDeltaNet2,
    KDA,
    KaczmarzLinearAttention,
    _gated_deltanet2_compiled_chunks,
    _gated_deltanet2_jit_scan,
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


@pytest.mark.parametrize("device", _devices())
def test_gated_deltanet2_loads_legacy_unfused_projection_state(device):
    torch.manual_seed(19)
    d_model, heads, head_dim = 16, 4, 4
    inner = heads * head_dim
    legacy = {
        "q_proj.weight": torch.randn(inner, d_model, device=device) * 0.02,
        "k_proj.weight": torch.randn(inner, d_model, device=device) * 0.02,
        "v_proj.weight": torch.randn(inner, d_model, device=device) * 0.02,
        "write_gate_proj.weight": torch.randn(heads, d_model, device=device) * 0.02,
        "write_gate_proj.bias": torch.randn(heads, device=device) * 0.02,
        "forget_gate_proj.weight": torch.randn(heads, d_model, device=device) * 0.02,
        "forget_gate_proj.bias": torch.randn(heads, device=device) * 0.02,
        "o_proj.weight": torch.randn(d_model, inner, device=device) * 0.02,
    }
    layer = GatedDeltaNet2(d_model=d_model, n_heads=heads, head_dim=head_dim).to(device)
    layer.load_state_dict(dict(legacy), strict=True)
    x = torch.randn(2, 5, d_model, device=device) * 0.1

    q = F.linear(x, legacy["q_proj.weight"]).view(2, 5, heads, head_dim)
    k = F.linear(x, legacy["k_proj.weight"]).view(2, 5, heads, head_dim)
    v = F.linear(x, legacy["v_proj.weight"]).view(2, 5, heads, head_dim)
    write_gate = F.linear(x, legacy["write_gate_proj.weight"], legacy["write_gate_proj.bias"])
    forget_gate = F.linear(x, legacy["forget_gate_proj.weight"], legacy["forget_gate_proj.bias"])
    y, _state = gated_deltanet2_pytorch(q, k, v, write_gate=write_gate, forget_gate=forget_gate)
    expected = F.linear(y.reshape(2, 5, inner), legacy["o_proj.weight"])

    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", _devices())
def test_gated_deltanet2_jit_scan_matches_reference(device):
    torch.manual_seed(29)
    batch, time, heads, key_dim, value_dim = 1, 7, 2, 4, 4
    q = torch.randn(batch, time, heads, key_dim, device=device) * 0.1
    k = torch.randn(batch, time, heads, key_dim, device=device) * 0.1
    v = torch.randn(batch, time, heads, value_dim, device=device) * 0.1
    write = torch.randn(batch, time, heads, device=device) * 0.1
    forget = torch.randn(batch, time, heads, device=device) * 0.1
    initial_state = torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32) * 0.01

    expected, expected_state = gated_deltanet2_pytorch(
        q,
        k,
        v,
        write_gate=write,
        forget_gate=forget,
        initial_state=initial_state,
    )
    actual, actual_state = _gated_deltanet2_jit_scan(
        q,
        k,
        v,
        write_gate=write,
        forget_gate=forget,
        initial_state=initial_state,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_state, expected_state, atol=1e-6, rtol=1e-6)


def test_gated_deltanet2_full_compile_mode_caps_to_chunked(monkeypatch):
    import omnicoder.modeling.kda_2026 as kda_module

    torch.manual_seed(31)
    calls: list[int] = []

    def fake_compiled_scan(q_f, k_f, v_f, write_f, forget_f, state):
        calls.append(int(q_f.shape[1]))
        return kda_module._gdn2_tensor_scan(q_f, k_f, v_f, write_f, forget_f, state)

    monkeypatch.setattr(kda_module, "_compiled_gdn2_scan", fake_compiled_scan)
    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_MODE", "full")
    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_FULL_MAX_TOKENS", "2")
    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_CHUNK_TOKENS", "3")
    q = torch.randn(1, 7, 2, 4) * 0.1
    k = torch.randn(1, 7, 2, 4) * 0.1
    v = torch.randn(1, 7, 2, 4) * 0.1
    write = torch.randn(1, 7, 2) * 0.1
    forget = torch.randn(1, 7, 2) * 0.1

    expected, expected_state = gated_deltanet2_pytorch(q, k, v, write_gate=write, forget_gate=forget)
    actual, actual_state = _gated_deltanet2_compiled_chunks(q, k, v, write_gate=write, forget_gate=forget)

    assert calls == [3, 3]
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-4, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="JIT GDN2 training parity is most relevant on CUDA")
def test_gated_deltanet2_module_jit_scan_gradient_parity_cuda(monkeypatch):
    torch.manual_seed(30)
    eager = GatedDeltaNet2(d_model=16, n_heads=4, head_dim=4).cuda().to(dtype=torch.float16)
    scripted = GatedDeltaNet2(d_model=16, n_heads=4, head_dim=4).cuda().to(dtype=torch.float16)
    scripted.load_state_dict(eager.state_dict())
    x_eager = torch.randn(2, 9, 16, device="cuda", dtype=torch.float16, requires_grad=True)
    x_scripted = x_eager.detach().clone().requires_grad_(True)

    monkeypatch.setenv("OMNICODER2026_GDN2_JIT_SCAN", "0")
    y_eager, state_eager = eager(x_eager, return_state=True)
    eager_loss = y_eager.float().square().mean() + state_eager.square().mean() * 1e-3
    eager_loss.backward()

    monkeypatch.setenv("OMNICODER2026_GDN2_JIT_SCAN", "1")
    y_scripted, state_scripted = scripted(x_scripted, return_state=True)
    scripted_loss = y_scripted.float().square().mean() + state_scripted.square().mean() * 1e-3
    scripted_loss.backward()

    torch.testing.assert_close(y_scripted, y_eager, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(state_scripted, state_eager, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(x_scripted.grad, x_eager.grad, atol=3e-4, rtol=3e-3)
    for scripted_param, eager_param in zip(scripted.parameters(), eager.parameters(), strict=True):
        assert scripted_param.grad is not None
        assert eager_param.grad is not None
        torch.testing.assert_close(scripted_param.grad, eager_param.grad, atol=4e-4, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled GDN2 fast path is CUDA-only")
@pytest.mark.parametrize("time", [1, 2, 5, 17])
def test_gated_deltanet2_compiled_chunks_match_reference_cuda(time):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    major, minor = torch.cuda.get_device_capability(torch.device("cuda"))
    if (major, minor) < (7, 5):
        pytest.skip("compiled GDN2 path is only enabled on fast-card CUDA runtimes")
    import omnicoder.modeling.kda_2026 as kda_module

    torch.manual_seed(31 + time)
    kda_module._COMPILED_GDN2_SCAN = None
    kda_module._GDN2_COMPILE_DISABLED = False
    batch, heads, key_dim, value_dim = 1, 2, 4, 4
    q = torch.randn(batch, time, heads, key_dim, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(batch, time, heads, key_dim, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(batch, time, heads, value_dim, device="cuda", dtype=torch.float16) * 0.1
    write = torch.randn(batch, time, heads, device="cuda", dtype=torch.float16) * 0.1
    forget = torch.randn(batch, time, heads, device="cuda", dtype=torch.float16) * 0.1
    initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=torch.float32) * 0.01

    expected, expected_state = gated_deltanet2_pytorch(
        q,
        k,
        v,
        write_gate=write,
        forget_gate=forget,
        initial_state=initial_state,
    )
    actual, actual_state = _gated_deltanet2_compiled_chunks(
        q,
        k,
        v,
        write_gate=write,
        forget_gate=forget,
        initial_state=initial_state,
        chunk_size=min(time, 4),
    )

    assert not kda_module._GDN2_COMPILE_DISABLED
    assert kda_module._COMPILED_GDN2_SCAN is not None
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-4, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled GDN2 fast path is CUDA-only")
def test_gated_deltanet2_module_compiled_chunks_gradient_parity_cuda(monkeypatch):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    major, minor = torch.cuda.get_device_capability(torch.device("cuda"))
    if (major, minor) < (7, 5):
        pytest.skip("compiled GDN2 path is only enabled on fast-card CUDA runtimes")
    import omnicoder.modeling.kda_2026 as kda_module

    torch.manual_seed(37)
    kda_module._COMPILED_GDN2_SCAN = None
    kda_module._GDN2_COMPILE_DISABLED = False
    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_CHUNKS", "1")
    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_CHUNK_TOKENS", "4")
    eager = GatedDeltaNet2(d_model=16, n_heads=4, head_dim=4).cuda()
    compiled = GatedDeltaNet2(d_model=16, n_heads=4, head_dim=4).cuda()
    compiled.load_state_dict(eager.state_dict())
    x_eager = torch.randn(2, 9, 16, device="cuda", dtype=torch.float16, requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_(True)
    eager = eager.to(dtype=torch.float16)
    compiled = compiled.to(dtype=torch.float16)

    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_CHUNKS", "0")
    y_eager, state_eager = eager(x_eager, return_state=True)
    eager_loss = y_eager.float().square().mean() + state_eager.square().mean() * 1e-3
    eager_loss.backward()

    monkeypatch.setenv("OMNICODER2026_GDN2_COMPILED_CHUNKS", "1")
    y_compiled, state_compiled = compiled(x_compiled, return_state=True)
    compiled_loss = y_compiled.float().square().mean() + state_compiled.square().mean() * 1e-3
    compiled_loss.backward()

    assert not kda_module._GDN2_COMPILE_DISABLED
    torch.testing.assert_close(y_compiled, y_eager, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(state_compiled, state_eager, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(x_compiled.grad, x_eager.grad, atol=3e-4, rtol=3e-3)
    for compiled_param, eager_param in zip(compiled.parameters(), eager.parameters(), strict=True):
        assert compiled_param.grad is not None
        assert eager_param.grad is not None
        torch.testing.assert_close(compiled_param.grad, eager_param.grad, atol=4e-4, rtol=5e-3)

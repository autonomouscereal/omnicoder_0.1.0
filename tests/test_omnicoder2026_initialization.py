from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")
import torch

from omnicoder.modeling.omnicoder2026 import (
    AdaptiveLatentReasoner,
    BlockAttentionResidual,
    LocalCausalAttention,
    OmniCoder2026,
    OmniCoder2026Config,
    QuantAwareGroupedLinear,
    QuantAwareLinear,
    RotaryEmbedding,
    SparseLatentAttention,
    SwiGLU,
)
from omnicoder.tokenization.native_segments_2026 import build_native_segment_packet
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.training.pipeline_pretrain_2026_dense import OmniCoder2026PipelineShard, shard_spec, stage_ranges


def _tiny_native_cfg() -> OmniCoder2026Config:
    return OmniCoder2026Config(
        vocab_size=DEFAULT_LEDGER.vocab_size,
        max_seq_len=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        head_dim=8,
        num_key_value_heads=1,
        mlp_dim=64,
        local_window=8,
        csa_block_size=8,
        csa_top_k_blocks=4,
        hca_block_size=16,
        latent_dim=8,
        rope_dim=8,
        sink_tokens=1,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=2,
        index_head_dim=8,
        flow_latent_dim=16,
        residual_mode="block_attnres",
        block_attnres_block_size=4,
        block_attnres_max_blocks=4,
        block_attnres_rank=8,
        block_attnres_chunk_tokens=4,
        native_media_feature_dim=12,
        native_media_position_dim=4,
        native_media_type_vocab=8,
        fake_quant_group_size=8,
        layer_pattern=("kda", "local"),
        mtp_heads=1,
    )


def test_full_model_embedding_and_head_use_transformer_scale_init() -> None:
    cfg = OmniCoder2026Config.probe()
    cfg.vocab_size = 256
    cfg.d_model = 64
    cfg.mlp_dim = 128
    cfg.initializer_std = 0.02

    model = OmniCoder2026(cfg)

    assert 0.005 < float(model.embed.weight.detach().float().std(unbiased=False)) < 0.05
    assert model.lm_head.weight is model.embed.weight


def test_pipeline_shard_embedding_and_head_use_transformer_scale_init() -> None:
    cfg = OmniCoder2026Config.probe()
    cfg.vocab_size = 256
    cfg.d_model = 64
    cfg.mlp_dim = 128
    cfg.n_layers = 4
    cfg.initializer_std = 0.02
    ranges = stage_ranges(cfg.n_layers, "1,1,2")

    embed_shard = OmniCoder2026PipelineShard(cfg, shard_spec(0, ranges))
    head_shard = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))

    embed_std = float(embed_shard.embed.weight.detach().float().std(unbiased=False))
    head_std = float(head_shard.lm_head.weight.detach().float().std(unbiased=False))
    assert 0.005 < embed_std < 0.05
    assert 0.005 < head_std < 0.05


def test_block_attention_residual_is_low_rank_and_memory_bounded() -> None:
    cfg = _tiny_native_cfg()
    residual = BlockAttentionResidual(cfg)
    param_count = sum(parameter.numel() for parameter in residual.parameters())

    assert residual.rank == 8
    assert residual.max_blocks == 4
    assert param_count < cfg.d_model * cfg.d_model

    x = torch.randn(2, 19, cfg.d_model)
    update = torch.randn_like(x)
    summaries, positions = residual._block_summaries(x)
    out = residual(x, update)

    assert summaries.shape == (2, 4, cfg.d_model)
    assert positions[0].item() == 0
    assert torch.equal(positions, torch.sort(positions).values)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_block_attention_residual_sdpa_fast_path_matches_chunked_loop(monkeypatch) -> None:
    torch.manual_seed(2601)
    cfg = _tiny_native_cfg()
    cfg.block_attnres_chunk_tokens = 3
    residual = BlockAttentionResidual(cfg)
    residual.scale.data.fill_(0.75)
    x = torch.randn(2, 19, cfg.d_model)
    update = torch.randn_like(x)

    monkeypatch.setenv("OMNICODER2026_BLOCK_ATTENTION_RESIDUAL_SDPA_MAX_TOKEN_BLOCK_PAIRS", "0")
    chunked = residual(x, update)

    sdpa_calls: list[tuple[torch.Size, torch.Size | None]] = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def wrapped_sdpa(*args, **kwargs):
        mask = kwargs.get("attn_mask")
        sdpa_calls.append((args[0].shape, None if mask is None else mask.shape))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", wrapped_sdpa)
    monkeypatch.setenv("OMNICODER2026_BLOCK_ATTENTION_RESIDUAL_SDPA_MAX_TOKEN_BLOCK_PAIRS", "4096")
    fast = residual(x, update)

    assert sdpa_calls == [(torch.Size([2, 1, 19, cfg.block_attnres_rank]), torch.Size([1, 1, 19, 4]))]
    torch.testing.assert_close(fast, chunked, atol=1e-6, rtol=1e-6)


def test_block_attention_residual_sdpa_preserves_causal_block_mask(monkeypatch) -> None:
    cfg = _tiny_native_cfg()
    cfg.block_attnres_block_size = 2
    residual = BlockAttentionResidual(cfg)
    q = torch.zeros(1, 5, cfg.block_attnres_rank)
    k = torch.zeros(1, 3, cfg.block_attnres_rank)
    summaries = torch.zeros(1, 3, cfg.d_model)
    summaries[0, :, 0] = torch.tensor([10.0, 20.0, 30.0])
    summary_positions = torch.tensor([0, 1, 2])

    monkeypatch.setenv("OMNICODER2026_BLOCK_ATTENTION_RESIDUAL_SDPA_MAX_TOKEN_BLOCK_PAIRS", "4096")
    context = residual._residual_attention_context(q, k, summaries, summary_positions)

    expected = torch.tensor([10.0, 10.0, 15.0, 15.0, 20.0])
    torch.testing.assert_close(context[0, :, 0], expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(context[..., 1:]) == 0


def test_block_attention_residual_sdpa_backward_matches_chunked_loop(monkeypatch) -> None:
    torch.manual_seed(2602)
    cfg = _tiny_native_cfg()
    cfg.block_attnres_chunk_tokens = 4
    base = BlockAttentionResidual(cfg)
    base.scale.data.fill_(0.5)
    chunked = copy.deepcopy(base)
    fast = copy.deepcopy(base)
    x = torch.randn(2, 17, cfg.d_model)
    update = torch.randn_like(x)
    probe = torch.randn_like(x)

    x_chunked = x.detach().clone().requires_grad_(True)
    update_chunked = update.detach().clone().requires_grad_(True)
    monkeypatch.setenv("OMNICODER2026_BLOCK_ATTENTION_RESIDUAL_SDPA_MAX_TOKEN_BLOCK_PAIRS", "0")
    chunked_out = chunked(x_chunked, update_chunked)
    (chunked_out * probe).sum().backward()

    x_fast = x.detach().clone().requires_grad_(True)
    update_fast = update.detach().clone().requires_grad_(True)
    monkeypatch.setenv("OMNICODER2026_BLOCK_ATTENTION_RESIDUAL_SDPA_MAX_TOKEN_BLOCK_PAIRS", "4096")
    fast_out = fast(x_fast, update_fast)
    (fast_out * probe).sum().backward()

    torch.testing.assert_close(fast_out, chunked_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x_fast.grad, x_chunked.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(update_fast.grad, update_chunked.grad, atol=1e-5, rtol=1e-5)
    for (name, fast_param), (chunked_name, chunked_param) in zip(fast.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == chunked_name
        if fast_param.grad is None or chunked_param.grad is None:
            assert fast_param.grad is None and chunked_param.grad is None
            continue
        torch.testing.assert_close(fast_param.grad, chunked_param.grad, atol=1e-5, rtol=1e-5)


def test_rotary_embedding_keeps_multiple_length_cache_entries() -> None:
    rope = RotaryEmbedding(8, base=10000.0)
    x_short = torch.randn(1, 1, 7, 8)
    x_long = torch.randn(1, 1, 11, 8)

    short_first = rope(x_short)
    long_first = rope(x_long)
    short_second = rope(x_short)

    assert len(rope._cache_values) == 2
    assert short_first[0] is short_second[0]
    assert short_first[1] is short_second[1]
    torch.testing.assert_close(short_first[0], short_second[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(long_first[0], rope(x_long)[0], atol=0.0, rtol=0.0)


def test_native_continuous_media_features_train_through_shared_trunk() -> None:
    cfg = _tiny_native_cfg()
    model = OmniCoder2026(cfg)
    packet = build_native_segment_packet("image", (32, 64), feature_dim=cfg.native_media_feature_dim)
    media_count = int(packet["segment_count"])
    input_ids = torch.tensor([packet["token_ids"] + [1, 2, 3, 4]], dtype=torch.long)
    labels = input_ids.clone()
    features = torch.randn(1, media_count, cfg.native_media_feature_dim)
    targets = torch.randn(1, media_count, cfg.native_media_feature_dim)
    positions = torch.tensor(packet["positions"], dtype=torch.float32).unsqueeze(0)
    type_ids = torch.tensor(packet["type_ids"], dtype=torch.long).unsqueeze(0)

    out = model(
        input_ids,
        labels=labels,
        native_media_features=features,
        native_media_type_ids=type_ids,
        native_media_positions=positions,
        native_media_targets=targets,
        native_media_mask=torch.ones(1, media_count),
        return_aux=True,
        return_logits=False,
        return_hidden=False,
    )

    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
    assert out["native_media_loss"] is not None
    assert torch.isfinite(out["native_media_loss"])
    assert out["native_media_reconstruction"].shape == targets.shape


def test_adaptive_latent_reasoner_outputs_effort_controls() -> None:
    cfg = OmniCoder2026Config.probe()
    cfg.vocab_size = 256
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.head_dim = 16
    cfg.num_key_value_heads = 1
    cfg.mlp_dim = 128
    cfg.n_layers = 2
    cfg.layer_pattern = ("kda", "local")
    cfg.max_seq_len = 16
    cfg.reasoning_slots = 2
    cfg.reasoning_max_steps = 3
    cfg.reasoning_default_steps = 0
    cfg.reasoning_cell_rank = 8
    cfg.reasoning_pool_tokens = 4
    cfg.tie_embeddings = True
    model = OmniCoder2026(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    labels = input_ids.clone()

    out = model(
        input_ids,
        labels=labels,
        return_aux=True,
        return_logits=False,
        return_hidden=False,
        reasoning_effort=2,
    )

    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
    for key in (
        "reasoning_difficulty",
        "reasoning_halt_continue",
        "reasoning_answer_readiness",
        "reasoning_verifier_margin",
        "reasoning_tool_readiness",
    ):
        assert key in out
        assert out[key].shape == (2,)
        assert torch.isfinite(out[key]).all()
    assert model.last_reasoning_diagnostics["steps"] == 2
    manifest = model.architecture_manifest()
    assert manifest["adaptive_latent_reasoning"]["mode"] == "shared_low_rank_hidden_deliberation_slots"
    assert manifest["adaptive_latent_reasoning"]["max_steps"] == 3
    assert manifest["adaptive_latent_reasoning"]["public_cot"] is False


def test_adaptive_latent_reasoner_preserves_zero_output_scale() -> None:
    cfg = _tiny_native_cfg()
    cfg.reasoning_output_scale = 0.0
    reasoner = AdaptiveLatentReasoner(cfg)

    assert float(reasoner.output_scale.detach().item()) == 0.0


def test_pipeline_reasoner_accepts_named_effort(monkeypatch) -> None:
    cfg = _tiny_native_cfg()
    cfg.n_layers = 2
    cfg.reasoning_max_steps = 3
    cfg.reasoning_default_steps = 0
    ranges = stage_ranges(cfg.n_layers, "1,1")
    shard = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges))
    monkeypatch.setenv("OMNICODER2026_PIPELINE_REASONING_EFFORT", "high")
    x = torch.randn(1, 6, cfg.d_model)

    out = shard(x)

    assert out.shape == x.shape
    assert shard.last_reasoning_diagnostics["steps"] == cfg.reasoning_max_steps


def test_pipeline_fast_forward_matches_record_function_path() -> None:
    torch.manual_seed(123)
    cfg = _tiny_native_cfg()
    cfg.n_layers = 2
    cfg.reasoning_max_steps = 2
    cfg.reasoning_default_steps = 1
    ranges = stage_ranges(cfg.n_layers, "1,1")
    fast = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges))
    profiled = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges))
    profiled.load_state_dict(fast.state_dict())
    profiled.profile_record_functions = True
    fast.pipeline_reasoning_effort = 1
    profiled.pipeline_reasoning_effort = 1
    x = torch.randn(1, 6, cfg.d_model)

    fast_out = fast(x)
    profiled_out = profiled(x)

    torch.testing.assert_close(fast_out, profiled_out, atol=0.0, rtol=0.0)
    assert fast.last_reasoning_diagnostics["steps"] == 1


def test_fused_local_attention_loads_legacy_qkv_projection_state() -> None:
    torch.manual_seed(321)
    cfg = _tiny_native_cfg()
    layer = LocalCausalAttention(cfg)
    reference = LocalCausalAttention(cfg)
    legacy = layer.state_dict()
    inner = cfg.n_heads * cfg.head_dim
    qkv = legacy.pop("qkv_proj.weight")
    q_w, k_w, v_w = qkv.split(inner, dim=0)
    legacy["q_proj.weight"] = q_w
    legacy["k_proj.weight"] = k_w
    legacy["v_proj.weight"] = v_w
    reference.load_state_dict(legacy, strict=True)
    x = torch.randn(2, 11, cfg.d_model)

    torch.testing.assert_close(reference(x), layer(x), atol=0.0, rtol=0.0)


def test_fused_swiglu_loads_legacy_gate_up_projection_state() -> None:
    torch.manual_seed(654)
    cfg = _tiny_native_cfg()
    layer = SwiGLU(cfg)
    reference = SwiGLU(cfg)
    legacy = layer.state_dict()
    gate_up = legacy.pop("gate_up.weight")
    gate_w, up_w = gate_up.split(cfg.mlp_dim, dim=0)
    legacy["gate.weight"] = gate_w
    legacy["up.weight"] = up_w
    reference.load_state_dict(legacy, strict=True)
    x = torch.randn(2, 7, cfg.d_model)

    torch.testing.assert_close(reference(x), layer(x), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("fake_quant", [False, True])
def test_quant_aware_grouped_linear_matches_module_list_projection(fake_quant: bool) -> None:
    torch.manual_seed(735)
    grouped = QuantAwareGroupedLinear(8, 6, 2, fake_quant=fake_quant, group_size=4)
    proj0 = QuantAwareLinear(4, 3, bias=False, fake_quant=fake_quant, group_size=4)
    proj1 = QuantAwareLinear(4, 3, bias=False, fake_quant=fake_quant, group_size=4)
    with torch.no_grad():
        grouped.weight[0].copy_(torch.randn_like(grouped.weight[0]))
        grouped.weight[1].copy_(torch.randn_like(grouped.weight[1]))
        proj0.weight.copy_(grouped.weight[0])
        proj1.weight.copy_(grouped.weight[1])
    x = torch.randn(2, 5, 8)

    expected = torch.cat((proj0(x[..., :4]), proj1(x[..., 4:])), dim=-1)

    torch.testing.assert_close(grouped(x), expected, atol=0.0, rtol=0.0)


def test_sparse_latent_attention_loads_legacy_grouped_o_projection_state() -> None:
    torch.manual_seed(736)
    cfg = _tiny_native_cfg()
    layer = SparseLatentAttention(cfg, "csa")
    reference = SparseLatentAttention(cfg, "csa")
    state = layer.state_dict()
    grouped_weight = state.pop("o_a_proj.weight")
    for idx in range(cfg.o_groups):
        state[f"o_a_groups.{idx}.weight"] = grouped_weight[idx]

    reference.load_state_dict(state, strict=True)

    assert isinstance(reference.o_a_proj, QuantAwareGroupedLinear)
    torch.testing.assert_close(reference.o_a_proj.weight, grouped_weight, atol=0.0, rtol=0.0)
    x = torch.randn(2, 9, cfg.d_model)
    torch.testing.assert_close(reference(x), layer(x), atol=0.0, rtol=0.0)


def test_sparse_local_attention_uses_native_gqa_before_expand_fallback(monkeypatch) -> None:
    torch.manual_seed(770)
    cfg = _tiny_native_cfg()
    cfg.local_window = 16
    layer = SparseLatentAttention(cfg, "csa")
    q = torch.randn(1, cfg.n_heads, 8, cfg.head_dim)
    k = torch.randn(1, 1, 8, cfg.head_dim)
    v = torch.randn(1, 1, 8, cfg.head_dim)

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    gqa_flags: list[bool] = []

    def wrapped_sdpa(*args, **kwargs):
        gqa_flags.append(bool(kwargs.get("enable_gqa", False)))
        return original_sdpa(*args, **kwargs)

    expected = original_sdpa(
        q,
        k.expand(-1, q.shape[1], -1, -1),
        v.expand(-1, q.shape[1], -1, -1),
        is_causal=True,
        dropout_p=0.0,
    )
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", wrapped_sdpa)
    actual = layer._local_attention(q, k, v)

    assert gqa_flags and gqa_flags[0] is True
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_sink_attention_sdpa_matches_reference_and_backward(monkeypatch) -> None:
    torch.manual_seed(771)
    cfg = _tiny_native_cfg()
    cfg.sink_tokens = 2
    reference = SparseLatentAttention(cfg, "csa")
    fast = copy.deepcopy(reference)
    q = torch.randn(2, cfg.n_heads, 5, cfg.head_dim)
    k = torch.randn(2, cfg.n_heads, 4, cfg.head_dim)
    v = torch.randn(2, cfg.n_heads, 4, cfg.head_dim)
    mask = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    probe = torch.randn(2, cfg.n_heads, 5, cfg.head_dim)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out = reference._sink_attention_reference(q_ref, k_ref, v_ref, mask)
    (ref_out * probe).sum().backward()

    q_fast = q.detach().clone().requires_grad_(True)
    k_fast = k.detach().clone().requires_grad_(True)
    v_fast = v.detach().clone().requires_grad_(True)
    monkeypatch.setenv("OMNICODER2026_SINK_ATTENTION_SDPA_MAX_QK_PAIRS", "4096")
    fast_out = fast._sink_attention(q_fast, k_fast, v_fast, mask)
    (fast_out * probe).sum().backward()

    torch.testing.assert_close(fast_out, ref_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q_fast.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_fast.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v_fast.grad, v_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fast.sink_logits.grad, reference.sink_logits.grad, atol=1e-5, rtol=1e-5)


def test_sparse_sink_attention_uses_sdpa_fast_path(monkeypatch) -> None:
    torch.manual_seed(772)
    cfg = _tiny_native_cfg()
    cfg.sink_tokens = 2
    layer = SparseLatentAttention(cfg, "csa")
    q = torch.randn(1, cfg.n_heads, 3, cfg.head_dim)
    k = torch.randn(1, cfg.n_heads, 2, cfg.head_dim)
    v = torch.randn(1, cfg.n_heads, 2, cfg.head_dim)
    mask = torch.tensor([[False, False], [True, False], [True, True]], dtype=torch.bool)
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls: list[tuple[torch.Size, torch.Size, torch.Size | None]] = []

    def wrapped_sdpa(*args, **kwargs):
        attn_mask = kwargs.get("attn_mask")
        calls.append((args[0].shape, args[1].shape, None if attn_mask is None else attn_mask.shape))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", wrapped_sdpa)
    monkeypatch.setenv("OMNICODER2026_SINK_ATTENTION_SDPA_MAX_QK_PAIRS", "4096")
    actual = layer._sink_attention(q, k, v, mask)

    assert calls == [(torch.Size([1, cfg.n_heads, 3, cfg.head_dim]), torch.Size([1, cfg.n_heads, 4, cfg.head_dim]), torch.Size([1, cfg.n_heads, 3, 4]))]
    torch.testing.assert_close(actual, layer._sink_attention_reference(q, k, v, mask), atol=1e-6, rtol=1e-6)


def test_sparse_sink_attention_auto_uses_reference_without_fa4_runtime(monkeypatch) -> None:
    torch.manual_seed(7721)
    cfg = _tiny_native_cfg()
    layer = SparseLatentAttention(cfg, "csa")
    q = torch.randn(1, cfg.n_heads, 3, cfg.head_dim)
    k = torch.randn(1, cfg.n_heads, 2, cfg.head_dim)
    v = torch.randn(1, cfg.n_heads, 2, cfg.head_dim)
    mask = torch.tensor([[False, False], [True, False], [True, True]], dtype=torch.bool)

    def forbidden_sdpa(*_args, **_kwargs):
        raise AssertionError("auto mode should keep the manual sink path without an FA4-class CUDA runtime")

    monkeypatch.delenv("OMNICODER2026_SINK_ATTENTION_SDPA_MAX_QK_PAIRS", raising=False)
    monkeypatch.setattr(layer, "_sink_attention_sdpa", forbidden_sdpa)
    actual = layer._sink_attention(q, k, v, mask)

    torch.testing.assert_close(actual, layer._sink_attention_reference(q, k, v, mask), atol=1e-6, rtol=1e-6)


def test_sparse_sink_attention_sdpa_gqa_matches_reference_cuda(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GQA SDPA sink-attention parity")

    torch.manual_seed(773)
    cfg = _tiny_native_cfg()
    cfg.sink_tokens = 2
    reference = SparseLatentAttention(cfg, "csa").cuda().half()
    fast = copy.deepcopy(reference)
    q = torch.randn(1, cfg.n_heads, 7, cfg.head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 4, cfg.head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 4, cfg.head_dim, device="cuda", dtype=torch.float16)
    mask = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
        device="cuda",
    )
    probe = torch.randn(1, cfg.n_heads, 7, cfg.head_dim, device="cuda", dtype=torch.float16)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    ref_out = reference._sink_attention_reference(q_ref, k_ref, v_ref, mask)
    (ref_out * probe).sum().backward()

    q_fast = q.detach().clone().requires_grad_(True)
    k_fast = k.detach().clone().requires_grad_(True)
    v_fast = v.detach().clone().requires_grad_(True)
    monkeypatch.setenv("OMNICODER2026_SINK_ATTENTION_SDPA_MAX_QK_PAIRS", "4096")
    fast_out = fast._sink_attention(q_fast, k_fast, v_fast, mask)
    (fast_out * probe).sum().backward()

    torch.testing.assert_close(fast_out, ref_out, atol=5e-4, rtol=5e-3)
    torch.testing.assert_close(q_fast.grad, q_ref.grad, atol=1e-3, rtol=5e-2)
    torch.testing.assert_close(k_fast.grad, k_ref.grad, atol=1e-3, rtol=5e-2)
    torch.testing.assert_close(v_fast.grad, v_ref.grad, atol=1e-3, rtol=5e-2)
    torch.testing.assert_close(fast.sink_logits.grad, reference.sink_logits.grad, atol=1e-3, rtol=5e-2)


def test_flex_local_attention_matches_chunked_fallback(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlexAttention parity")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 5):
        pytest.skip("FlexAttention fast path is disabled below sm75")

    torch.manual_seed(777)
    cfg = _tiny_native_cfg()
    cfg.local_window = 4
    layer = LocalCausalAttention(cfg).cuda()
    x = torch.randn(1, 13, cfg.d_model, device="cuda")

    monkeypatch.setenv("OMNICODER2026_FLEX_LOCAL_ATTENTION", "0")
    fallback = layer(x)
    monkeypatch.setenv("OMNICODER2026_FLEX_LOCAL_ATTENTION", "1")
    if hasattr(layer, "_flex_local_attention_disabled"):
        delattr(layer, "_flex_local_attention_disabled")
    flex = layer(x)

    torch.testing.assert_close(flex, fallback, atol=1e-4, rtol=1e-4)


def test_flex_sparse_mqa_local_attention_matches_expanded_fallback(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlexAttention parity")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 5):
        pytest.skip("FlexAttention fast path is disabled below sm75")

    torch.manual_seed(778)
    cfg = _tiny_native_cfg()
    cfg.local_window = 4
    layer = SparseLatentAttention(cfg, "csa").cuda()
    q = torch.randn(1, cfg.n_heads, 13, cfg.head_dim, device="cuda")
    k = torch.randn(1, 1, 13, cfg.head_dim, device="cuda")
    v = torch.randn(1, 1, 13, cfg.head_dim, device="cuda")

    monkeypatch.setenv("OMNICODER2026_FLEX_LOCAL_ATTENTION", "0")
    fallback = layer._local_attention(q, k, v)
    monkeypatch.setenv("OMNICODER2026_FLEX_LOCAL_ATTENTION", "1")
    if hasattr(layer, "_flex_local_attention_disabled"):
        delattr(layer, "_flex_local_attention_disabled")
    flex = layer._local_attention(q, k, v)

    torch.testing.assert_close(flex, fallback, atol=1e-4, rtol=1e-4)

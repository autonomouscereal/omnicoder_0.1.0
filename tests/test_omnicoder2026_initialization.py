from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from omnicoder.modeling.omnicoder2026 import (
    AdaptiveLatentReasoner,
    BlockAttentionResidual,
    LocalCausalAttention,
    OmniCoder2026,
    OmniCoder2026Config,
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

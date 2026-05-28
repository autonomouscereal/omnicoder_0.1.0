from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from omnicoder.modeling.omnicoder2026 import BlockAttentionResidual, OmniCoder2026, OmniCoder2026Config
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

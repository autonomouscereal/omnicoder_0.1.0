from __future__ import annotations

import pytest

pytest.importorskip("torch")

from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Config
from omnicoder.training.pipeline_pretrain_2026_dense import OmniCoder2026PipelineShard, shard_spec, stage_ranges


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

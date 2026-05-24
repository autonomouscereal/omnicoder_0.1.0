from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnicoder.modeling.omnicoder2026 import OmniCoder2026Config
from omnicoder.training.pipeline_pretrain_2026_dense import (
    OmniCoder2026PipelineShard,
    load_full_checkpoint_shard,
    parse_stage_ranges,
    shard_spec,
    stage_ranges,
)


def tiny_cfg(n_layers: int = 3) -> OmniCoder2026Config:
    return OmniCoder2026Config(
        vocab_size=32,
        n_layers=n_layers,
        d_model=32,
        n_heads=4,
        head_dim=8,
        num_key_value_heads=1,
        mlp_dim=64,
        max_seq_len=64,
        local_window=8,
        csa_block_size=8,
        csa_top_k_blocks=2,
        hca_block_size=16,
        latent_dim=8,
        rope_dim=8,
        sink_tokens=1,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        csa_compress_rate=2,
        hca_compress_rate=2,
        index_head_dim=8,
        hc_mult=1,
        layer_pattern=("local",) * n_layers,
        tie_embeddings=False,
        flow_latent_dim=16,
    )


def test_stage_ranges_target_contract() -> None:
    assert parse_stage_ranges("", 64) == [(0, 16), (16, 32), (32, 64)]
    assert stage_ranges(64) == [(0, 16), (16, 32), (32, 64)]
    assert stage_ranges(4) == [(0, 1), (1, 2), (2, 4)]
    assert stage_ranges(6, "2,2,2") == [(0, 2), (2, 4), (4, 6)]
    assert stage_ranges(6, "1,2,3") == [(0, 1), (1, 3), (3, 6)]
    with pytest.raises(ValueError):
        stage_ranges(6, "2,2,1")


def test_full_checkpoint_loads_rank_local_shard_from_cpu(tmp_path) -> None:
    cfg = tiny_cfg(n_layers=6)
    ranges = stage_ranges(6, "2,2,2")
    source = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges))
    full_state = {}
    for key, value in source.state_dict().items():
        if key.startswith("blocks."):
            full_state[key] = value.detach().clone()
    ckpt = tmp_path / "full.pt"
    torch.save({"model_state_dict": full_state, "global_step": 7, "last_loss": 1.25}, ckpt)
    target = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges))
    step, loss = load_full_checkpoint_shard(ckpt, target)
    assert step == 7
    assert loss == 1.25
    for key, value in source.state_dict().items():
        if key.startswith("blocks."):
            assert torch.equal(value, target.state_dict()[key])


def test_final_stage_chunked_lm_loss_backward() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(2, 5, cfg.d_model, requires_grad=True)
    labels = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
    loss = final.chunked_lm_loss(final(hidden), labels, chunk_tokens=2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import omnicoder.training.pipeline_pretrain_2026_dense as pipeline
from omnicoder.modeling.omnicoder2026 import OmniCoder2026Config, SparseLatentAttention
from omnicoder.training.pipeline_pretrain_2026_dense import (
    OmniCoder2026PipelineShard,
    WeightedTextJsonlDataset,
    _rank_complete_file,
    load_checkpoint_shard,
    _wait_for_rank_checkpoint_markers,
    load_full_checkpoint_shard,
    parse_stage_ranges,
    save_sharded_checkpoint,
    shard_spec,
    stage_ranges,
)
from omnicoder.training.pretrain_2026_dense import TextJsonlDataset


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


class FakeDist:
    def __init__(self, *, rank: int, world_size: int) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return self.world_size

    def barrier(self) -> None:
        raise AssertionError("filesystem checkpoint sync must not call dist.barrier")


class FakePointToPointDist:
    def __init__(self) -> None:
        self.sent: list[tuple[int, torch.Tensor]] = []
        self.received: list[tuple[int, tuple[int, ...], torch.dtype]] = []
        self.broadcasts: list[tuple[int, tuple[int, ...], torch.dtype]] = []

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        self.sent.append((int(dst), tensor.detach().clone()))

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        self.received.append((int(src), tuple(tensor.shape), tensor.dtype))
        if not self.sent:
            raise AssertionError("recv called before a matching fake send")
        _dst, payload = self.sent.pop(0)
        tensor.copy_(payload.to(device=tensor.device, dtype=tensor.dtype))

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.broadcasts.append((int(src), tuple(tensor.shape), tensor.dtype))


class NoPointToPointDist:
    def send(self, tensor: torch.Tensor, dst: int) -> None:
        raise AssertionError("intermediate ranks must not send step tensors")

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        raise AssertionError("intermediate ranks must not receive step tensors")

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        raise AssertionError("intermediate ranks must not broadcast step tensors")


class TinyShard:
    spec = SimpleNamespace(stage_index=0, num_stages=3, layer_start=0, layer_end=1, has_embed=True, has_head=False)

    def local_state_dict(self) -> dict[str, torch.Tensor]:
        return {"embed.weight": torch.ones(1, 1)}


def checkpoint_args(tmp_path, *, timeout: float = 2.0) -> SimpleNamespace:
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n', encoding="utf-8")
    return SimpleNamespace(
        data=str(data),
        data_manifest="",
        seq_len=8,
        batch_size=1,
        steps=1,
        lr=1.0e-6,
        pipeline_stage_ranges="0:1,1:2,2:3",
        placement_layer_counts="1,1,1",
        pipeline_microbatches=1,
        pipeline_schedule="manual",
        fake_quant=False,
        optimizer="adamw",
        optimizer_in_backward=False,
        optimizer_in_backward_update="",
        optimizer_in_backward_grad_clip=0.0,
        optimizer_in_backward_adafactor_chunk_rows=0,
        checkpoint_sync_backend="filesystem",
        checkpoint_marker_timeout_seconds=timeout,
        checkpoint_marker_poll_seconds=0.05,
    )


def wait_for_attempt_marker(checkpoint_dir, *, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    attempt_path = checkpoint_dir / pipeline.CHECKPOINT_ATTEMPT_MARKER
    while time.monotonic() < deadline:
        if attempt_path.exists():
            return json.loads(attempt_path.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise AssertionError("rank 0 did not create checkpoint attempt marker")


def test_stage_ranges_target_contract() -> None:
    assert parse_stage_ranges("", 64) == [(0, 16), (16, 32), (32, 64)]
    assert stage_ranges(64) == [(0, 16), (16, 32), (32, 64)]
    assert stage_ranges(4) == [(0, 1), (1, 2), (2, 4)]
    assert stage_ranges(6, "2,2,2") == [(0, 2), (2, 4), (4, 6)]
    assert stage_ranges(6, "1,2,3") == [(0, 1), (1, 3), (3, 6)]
    with pytest.raises(ValueError):
        stage_ranges(6, "2,2,1")


def test_pipeline_step_tensor_routing_keeps_labels_and_weights_final_rank_only() -> None:
    batch = torch.arange(8, dtype=torch.long).reshape(2, 4)
    labels = batch + 100
    weights = torch.tensor([0.5, 2.0], dtype=torch.float32)
    fake_dist = FakePointToPointDist()

    rank0_tensors = pipeline._route_pipeline_step_tensors(
        rank=0,
        world_size=3,
        batch=batch,
        labels=labels,
        sample_weights=weights,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        dist_module=fake_dist,
    )
    assert rank0_tensors.input_ids is batch
    assert rank0_tensors.labels is None
    assert rank0_tensors.sample_weights is None
    assert [(dst, tuple(tensor.shape), tensor.dtype) for dst, tensor in fake_dist.sent] == [
        (2, (2, 4), torch.long),
        (2, (2,), torch.float32),
    ]

    intermediate = pipeline._route_pipeline_step_tensors(
        rank=1,
        world_size=3,
        batch=None,
        labels=None,
        sample_weights=None,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        dist_module=NoPointToPointDist(),
    )
    assert intermediate.input_ids is None
    assert intermediate.labels is None
    assert intermediate.sample_weights is None

    final_tensors = pipeline._route_pipeline_step_tensors(
        rank=2,
        world_size=3,
        batch=None,
        labels=None,
        sample_weights=None,
        batch_size=2,
        seq_len=4,
        device=torch.device("cpu"),
        dist_module=fake_dist,
    )
    assert final_tensors.input_ids is None
    assert torch.equal(final_tensors.labels, labels)
    assert torch.equal(final_tensors.sample_weights, weights)
    assert fake_dist.received == [(0, (2, 4), torch.long), (0, (2,), torch.float32)]


def test_pipeline_loss_and_target_summary_are_point_to_point_and_interval_owned() -> None:
    fake_dist = FakePointToPointDist()
    final_loss = torch.tensor(3.25)
    sent_loss = pipeline._sync_pipeline_loss_to_rank0(
        rank=2,
        world_size=3,
        loss_tensor=final_loss,
        device=torch.device("cpu"),
        dist_module=fake_dist,
    )
    assert sent_loss is not None
    assert [(dst, tuple(tensor.shape), float(tensor.item())) for dst, tensor in fake_dist.sent] == [(0, (), 3.25)]

    rank0_loss = pipeline._sync_pipeline_loss_to_rank0(
        rank=0,
        world_size=3,
        loss_tensor=None,
        device=torch.device("cpu"),
        dist_module=fake_dist,
    )
    assert rank0_loss is not None
    assert float(rank0_loss.item()) == pytest.approx(3.25)

    no_sync = pipeline._sync_pipeline_target_summary_to_rank0(
        rank=1,
        world_size=3,
        loss_diagnostics=None,
        sample_weights=None,
        device=torch.device("cpu"),
        dist_module=NoPointToPointDist(),
    )
    assert no_sync is None

    diagnostics = {"valid_target_tokens": 7, "optimized_target_tokens": 5}
    weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    final_summary = pipeline._sync_pipeline_target_summary_to_rank0(
        rank=2,
        world_size=3,
        loss_diagnostics=diagnostics,
        sample_weights=weights,
        device=torch.device("cpu"),
        dist_module=fake_dist,
    )
    assert final_summary == pipeline.PipelineTargetSummary(7, 5, 2.0)
    rank0_summary = pipeline._sync_pipeline_target_summary_to_rank0(
        rank=0,
        world_size=3,
        loss_diagnostics=None,
        sample_weights=None,
        device=torch.device("cpu"),
        dist_module=fake_dist,
    )
    assert rank0_summary == final_summary


def test_sparse_global_attention_masks_current_compressed_block() -> None:
    cfg = tiny_cfg(n_layers=3)
    cfg.csa_top_k_blocks = 4
    attn = SparseLatentAttention(cfg, "csa")

    mask = attn._global_mask(t=8, n_blocks=4, block_size=2, device=torch.device("cpu"))

    assert not mask[0, 0]
    assert not mask[1, 0]
    assert mask[2, 0]
    assert not mask[2, 1]
    assert mask[4, 0]
    assert mask[4, 1]
    assert not mask[4, 2]


def test_sparse_global_attention_full_fast_path_matches_chunked_loop(monkeypatch) -> None:
    torch.manual_seed(1208)
    cfg = tiny_cfg(n_layers=3)
    cfg.n_heads = 2
    cfg.head_dim = 4
    cfg.d_model = 8
    cfg.csa_top_k_blocks = 8
    attn = SparseLatentAttention(cfg, "csa")
    q = torch.randn(1, cfg.n_heads, 16, cfg.head_dim)
    k = torch.randn(1, 1, 8, cfg.head_dim)
    v = torch.randn(1, 1, 8, cfg.head_dim)

    monkeypatch.setenv("OMNICODER2026_GLOBAL_ATTENTION_FULL_MAX_QBLOCKS", "0")
    chunked = attn._global_attention(q, k, v, block_size=2)
    monkeypatch.setenv("OMNICODER2026_GLOBAL_ATTENTION_FULL_MAX_QBLOCKS", "128")
    full = attn._global_attention(q, k, v, block_size=2)

    torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-6)


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


def test_full_checkpoint_loads_legacy_sparse_grouped_output_projection(tmp_path) -> None:
    cfg = tiny_cfg(n_layers=1)
    cfg.layer_pattern = ("csa",)
    cfg.o_groups = 2
    ranges = stage_ranges(1, "1")
    source = OmniCoder2026PipelineShard(cfg, shard_spec(0, ranges))
    full_state: dict[str, torch.Tensor] = {}
    expected_grouped: dict[str, torch.Tensor] = {}
    for key, value in source.state_dict().items():
        tensor = value.detach().cpu().clone()
        if key.endswith(".o_a_proj.weight") and tensor.ndim == 3:
            expected_grouped[key] = tensor
            prefix = key[: -len("o_a_proj.weight")]
            for group_index in range(int(tensor.shape[0])):
                full_state[f"{prefix}o_a_groups.{group_index}.weight"] = tensor[group_index].clone()
        else:
            full_state[key] = tensor
    assert expected_grouped, "test fixture must include a grouped sparse output projection"
    ckpt = tmp_path / "legacy_sparse_grouped.pt"
    torch.save({"model_state_dict": full_state, "global_step": 11, "last_loss": 2.25}, ckpt)
    target = OmniCoder2026PipelineShard(cfg, shard_spec(0, ranges))

    step, loss = load_full_checkpoint_shard(ckpt, target)

    assert step == 11
    assert loss == 2.25
    loaded = target.state_dict()
    for key, expected in expected_grouped.items():
        assert torch.equal(loaded[key], expected)


def test_sharded_checkpoint_resume_can_reshard_changed_layer_placement(tmp_path, monkeypatch) -> None:
    cfg = tiny_cfg(n_layers=6)
    old_ranges = stage_ranges(6, "2,2,2")
    checkpoint = tmp_path / "old_placement_checkpoint"
    checkpoint.mkdir()
    saved_args = checkpoint_args(tmp_path)
    saved_args.placement_layer_counts = "2,2,2"
    train_args = {
        "pipeline_stage_ranges": saved_args.pipeline_stage_ranges,
        "placement_layer_counts": saved_args.placement_layer_counts,
        "pipeline_microbatches": saved_args.pipeline_microbatches,
        "pipeline_schedule": saved_args.pipeline_schedule,
        "fake_quant": saved_args.fake_quant,
    }
    for rank in range(3):
        source = OmniCoder2026PipelineShard(cfg, shard_spec(rank, old_ranges))
        state: dict[str, torch.Tensor] = {}
        for key, tensor in source.state_dict().items():
            if key.startswith("blocks."):
                layer = int(key.split(".", 2)[1])
                state[key] = torch.full_like(tensor.detach().cpu(), float(layer + 1))
            elif key == "lm_head.weight":
                state[key] = torch.full_like(tensor.detach().cpu(), 99.0)
            else:
                state[key] = torch.full_like(tensor.detach().cpu(), float(rank + 1))
        torch.save(
            {
                "model_state_dict": state,
                "optimizer_state_dict": {"must_not_load_after_reshard": True},
                "global_step": 13,
                "last_loss": 0.75,
                "preset": {"name": "tiny"},
                "world_size": 3,
                "train_args": train_args,
            },
            checkpoint / f"rank{rank:05d}.pt",
        )
        pipeline._atomic_write_json(
            _rank_complete_file(checkpoint, rank),
            {"status": "complete", "rank": rank, "world_size": 3, "global_step": 13},
        )
    pipeline._atomic_write_json(checkpoint / "manifest.json", {"status": "complete", "world_size": 3})
    pipeline._atomic_write_json(checkpoint / ".complete.json", {"status": "complete", "world_size": 3})

    monkeypatch.setattr(pipeline, "dist", FakeDist(rank=2, world_size=3))
    target = OmniCoder2026PipelineShard(cfg, shard_spec(2, stage_ranges(6, "1,1,4")))
    current_args = checkpoint_args(tmp_path)
    current_args.placement_layer_counts = "1,1,4"
    step, loss = load_checkpoint_shard(checkpoint, target, preset=SimpleNamespace(name="tiny"), args=current_args)

    assert step == 13
    assert loss == 0.75
    loaded = target.state_dict()
    assert torch.equal(loaded["blocks.2.attn_norm.weight"], torch.full_like(loaded["blocks.2.attn_norm.weight"], 3.0))
    assert torch.equal(loaded["blocks.5.attn_norm.weight"], torch.full_like(loaded["blocks.5.attn_norm.weight"], 6.0))
    assert torch.equal(loaded["lm_head.weight"], torch.full_like(loaded["lm_head.weight"], 99.0))


def test_sharded_resume_detects_old_stage_ranges_without_saved_counts(tmp_path, monkeypatch) -> None:
    cfg = tiny_cfg(n_layers=6)
    old_ranges = stage_ranges(6, "2,2,2")
    checkpoint = tmp_path / "old_ranges_only_checkpoint"
    checkpoint.mkdir()
    saved_args = checkpoint_args(tmp_path)
    saved_args.pipeline_stage_ranges = "0:2,2:4,4:6"
    train_args = {
        "pipeline_stage_ranges": saved_args.pipeline_stage_ranges,
        "pipeline_microbatches": saved_args.pipeline_microbatches,
        "pipeline_schedule": saved_args.pipeline_schedule,
        "fake_quant": saved_args.fake_quant,
    }
    for rank in range(3):
        source = OmniCoder2026PipelineShard(cfg, shard_spec(rank, old_ranges))
        state: dict[str, torch.Tensor] = {}
        for key, tensor in source.state_dict().items():
            if key.startswith("blocks."):
                layer = int(key.split(".", 2)[1])
                state[key] = torch.full_like(tensor.detach().cpu(), float(layer + 1))
            elif key == "lm_head.weight":
                state[key] = torch.full_like(tensor.detach().cpu(), 99.0)
            else:
                state[key] = torch.full_like(tensor.detach().cpu(), float(rank + 1))
        torch.save(
            {
                "model_state_dict": state,
                "optimizer_state_dict": {"must_not_load_after_reshard": True},
                "global_step": 21,
                "last_loss": 0.5,
                "preset": {"name": "tiny"},
                "world_size": 3,
                "train_args": train_args,
            },
            checkpoint / f"rank{rank:05d}.pt",
        )
        pipeline._atomic_write_json(
            _rank_complete_file(checkpoint, rank),
            {"status": "complete", "rank": rank, "world_size": 3, "global_step": 21},
        )
    pipeline._atomic_write_json(checkpoint / "manifest.json", {"status": "complete", "world_size": 3})
    pipeline._atomic_write_json(checkpoint / ".complete.json", {"status": "complete", "world_size": 3})

    monkeypatch.setattr(pipeline, "dist", FakeDist(rank=2, world_size=3))
    target = OmniCoder2026PipelineShard(cfg, shard_spec(2, stage_ranges(6, "1,1,4")))
    current_args = checkpoint_args(tmp_path)
    current_args.placement_layer_counts = "1,1,4"
    step, loss = load_checkpoint_shard(checkpoint, target, preset=SimpleNamespace(name="tiny"), args=current_args)

    assert step == 21
    assert loss == 0.5
    loaded = target.state_dict()
    assert torch.equal(loaded["blocks.2.attn_norm.weight"], torch.full_like(loaded["blocks.2.attn_norm.weight"], 3.0))
    assert torch.equal(loaded["blocks.5.attn_norm.weight"], torch.full_like(loaded["blocks.5.attn_norm.weight"], 6.0))
    assert torch.equal(loaded["lm_head.weight"], torch.full_like(loaded["lm_head.weight"], 99.0))


def test_final_stage_chunked_lm_loss_backward() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(2, 5, cfg.d_model, requires_grad=True)
    labels = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
    loss = final.chunked_lm_loss(final(hidden), labels, chunk_tokens=2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    timing = final.last_lm_loss_timing
    assert timing["schema"] == "omnicoder.lm_loss_timing_2026.v1"
    assert timing["total_sec"] >= 0.0
    assert timing["spans"]["dense_lm_head_ce_sec"] >= 0.0
    assert final.last_lm_loss_diagnostics["timing"]["chunk_tokens"] == 2
    loss.backward()
    assert hidden.grad is not None


def test_final_stage_weighted_lm_loss_matches_reward_replay_shape() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(2, 5, cfg.d_model, requires_grad=True)
    labels = torch.randint(1, cfg.vocab_size, (2, 5), dtype=torch.long)
    weights = torch.tensor([0.5, 2.0], dtype=torch.float32)

    processed = final(hidden)
    loss = final.chunked_lm_loss(processed, labels, chunk_tokens=2, sample_weights=weights)
    logits = final.lm_head(processed[:, :-1, :])
    token_losses = F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], reduction="none")
    expected = (token_losses.mean(dim=1) * weights).mean()

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)
    loss.backward()
    assert hidden.grad is not None


def test_final_stage_dense_lm_loss_keeps_zero_token_targets() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 5, cfg.d_model, requires_grad=True)
    labels = torch.zeros((1, 5), dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(processed, labels, chunk_tokens=2)
    logits = final.lm_head(processed[:, :-1, :])
    expected = F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], reduction="mean")

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert hidden.grad is not None


def test_final_stage_selected_token_lm_loss_bounds_vocab_projection() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(2, 9, cfg.d_model, requires_grad=True)
    labels = torch.randint(1, cfg.vocab_size, (2, 9), dtype=torch.long)
    weights = torch.tensor([0.5, 2.0], dtype=torch.float32)

    processed = final(hidden)
    loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        sample_weights=weights,
        loss_token_stride=2,
        max_loss_tokens_per_sample=2,
    )
    selected = [torch.tensor([0, 6]), torch.tensor([0, 6])]
    expected_parts = []
    for batch_index, positions in enumerate(selected):
        logits = final.lm_head(processed[batch_index, positions, :])
        target = labels[batch_index, positions + 1]
        expected_parts.append(F.cross_entropy(logits, target, reduction="none").mean() * weights[batch_index])
    expected = torch.stack(expected_parts).mean()

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)
    timing = final.last_lm_loss_timing
    assert timing["sparse_path"] is True
    assert timing["spans"]["selected_position_scan_sec"] >= 0.0
    assert timing["spans"]["selected_lm_head_ce_sec"] >= 0.0
    loss.backward()
    assert hidden.grad is not None


def test_segmented_activation_checkpoint_matches_per_block_checkpoint() -> None:
    cfg = tiny_cfg(n_layers=4)
    ranges = stage_ranges(4, "2,2")
    per_block = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges), checkpoint_blocks=True, checkpoint_segment_size=1)
    segmented = OmniCoder2026PipelineShard(cfg, shard_spec(1, ranges), checkpoint_blocks=True, checkpoint_segment_size=2)
    segmented.load_state_dict(per_block.state_dict())
    per_block.train()
    segmented.train()

    hidden = torch.randn(1, 8, cfg.d_model)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7, 8, 9]], dtype=torch.long)
    hidden_a = hidden.detach().clone().requires_grad_(True)
    hidden_b = hidden.detach().clone().requires_grad_(True)

    loss_a = per_block.chunked_lm_loss(per_block(hidden_a), labels, chunk_tokens=2)
    loss_b = segmented.chunked_lm_loss(segmented(hidden_b), labels, chunk_tokens=2)
    loss_a.backward()
    loss_b.backward()

    torch.testing.assert_close(loss_b, loss_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(hidden_b.grad, hidden_a.grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        segmented.blocks[2].ffn.down.weight.grad,
        per_block.blocks[2].ffn.down.weight.grad,
        atol=0.0,
        rtol=0.0,
    )


def test_final_stage_ignores_masked_prompt_labels() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 6, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7]], dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(processed, labels, chunk_tokens=2)
    valid_positions = torch.tensor([1, 2, 4])
    logits = final.lm_head(processed[0, valid_positions, :])
    expected = F.cross_entropy(logits, labels[0, valid_positions + 1], reduction="mean")

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)
    loss.backward()
    assert hidden.grad is not None
    diagnostics = final.last_lm_loss_diagnostics
    assert diagnostics["valid_target_tokens"] == 3
    assert diagnostics["optimized_target_tokens"] == 3
    assert diagnostics["target_counts_by_token_family"]["text"] == 3
    assert diagnostics["target_counts_by_modality"]["text"] == 3
    assert diagnostics["ce_by_token_family"]["text"] is not None


def test_final_stage_boundary_weight_upweights_target_starts() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 6, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7]], dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(processed, labels, chunk_tokens=2, target_boundary_weight=4.0)
    valid_positions = torch.tensor([1, 2, 4])
    logits = final.lm_head(processed[0, valid_positions, :])
    token_losses = F.cross_entropy(logits, labels[0, valid_positions + 1], reduction="none")
    token_weights = torch.tensor([4.0, 1.0, 4.0])
    expected = (token_losses * token_weights).sum() / token_weights.sum()

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)


def test_final_stage_selected_ce_uses_all_sparse_target_labels() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 8, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7, 8, 9]], dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        loss_token_stride=3,
        max_loss_tokens_per_sample=0,
    )
    expected_positions = torch.tensor([1, 2, 4, 5, 6])
    logits = final.lm_head(processed[0, expected_positions, :])
    expected = F.cross_entropy(logits, labels[0, expected_positions + 1], reduction="mean")

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)


def test_final_stage_selected_ce_treats_first_label_sentinel_as_sparse() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 7, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, 5, 6, 7, 8, 9, 10]], dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        loss_token_stride=4,
        max_loss_tokens_per_sample=1,
    )
    expected_positions = torch.tensor([0])
    logits = final.lm_head(processed[0, expected_positions, :])
    expected = F.cross_entropy(logits, labels[0, expected_positions + 1], reduction="mean")

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)


def test_final_stage_sparse_target_cap_keeps_boundaries_and_samples_tail() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 8, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7, 8, 9]], dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        max_loss_tokens_per_sample=3,
    )
    expected_positions = torch.tensor([1, 4, 6])
    logits = final.lm_head(processed[0, expected_positions, :])
    expected = F.cross_entropy(logits, labels[0, expected_positions + 1], reduction="mean")

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)
    assert final.last_lm_loss_diagnostics["valid_target_tokens"] == 5
    assert final.last_lm_loss_diagnostics["optimized_target_tokens"] == 3


def test_final_stage_prefix_weight_upweights_target_anchor_tokens() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 8, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7, 8, 9]], dtype=torch.long)

    processed = final(hidden)
    loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        target_prefix_weight=3.0,
        target_prefix_tokens=2,
    )
    valid_positions = torch.tensor([1, 2, 4, 5, 6])
    logits = final.lm_head(processed[0, valid_positions, :])
    token_losses = F.cross_entropy(logits, labels[0, valid_positions + 1], reduction="none")
    token_weights = torch.tensor([3.0, 3.0, 3.0, 3.0, 1.0])
    expected = (token_losses * token_weights).sum() / token_weights.sum()

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)


def test_final_stage_sparse_media_targets_get_optimizer_update() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    optimizer = torch.optim.SGD(final.parameters(), lr=0.01)
    hidden = torch.randn(1, 8, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7, 8, 9]], dtype=torch.long)
    before = final.lm_head.weight.detach().clone()

    processed = final(hidden)
    loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        loss_token_stride=99,
        max_loss_tokens_per_sample=1,
        target_boundary_weight=4.0,
        target_prefix_weight=8.0,
        target_prefix_tokens=3,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None
    optimizer.step()

    assert not torch.equal(before, final.lm_head.weight.detach())


def test_final_stage_can_skip_expensive_loss_diagnostics_without_changing_loss() -> None:
    cfg = tiny_cfg(n_layers=3)
    ranges = stage_ranges(3, "1,1,1")
    final = OmniCoder2026PipelineShard(cfg, shard_spec(2, ranges))
    hidden = torch.randn(1, 8, cfg.d_model, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6, -100, 7, 8, 9]], dtype=torch.long)

    processed = final(hidden)
    full_loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        loss_token_stride=3,
        max_loss_tokens_per_sample=3,
        collect_diagnostics=True,
    )
    full_diagnostics = dict(final.last_lm_loss_diagnostics)
    skipped_loss = final.chunked_lm_loss(
        processed,
        labels,
        chunk_tokens=2,
        loss_token_stride=3,
        max_loss_tokens_per_sample=3,
        collect_diagnostics=False,
    )
    skipped_diagnostics = dict(final.last_lm_loss_diagnostics)

    assert torch.allclose(skipped_loss.float(), full_loss.float(), atol=1e-5)
    assert not bool(full_diagnostics.get("diagnostics_skipped", False))
    assert skipped_diagnostics["diagnostics_skipped"] is True
    assert skipped_diagnostics["valid_target_tokens"] == full_diagnostics["valid_target_tokens"]
    assert skipped_diagnostics["optimized_target_tokens"] == full_diagnostics["optimized_target_tokens"]
    assert skipped_diagnostics["target_counts_by_token_family"]["unknown"] == full_diagnostics["valid_target_tokens"]
    assert skipped_diagnostics["optimized_target_counts_by_token_family"]["unknown"] == full_diagnostics["optimized_target_tokens"]


def test_checkpoint_data_integrity_manifest_policy_does_not_hash_training_file(tmp_path, monkeypatch) -> None:
    args = checkpoint_args(tmp_path)
    args.checkpoint_data_hash_policy = "manifest"

    def fail_hash(_path: str) -> str:
        raise AssertionError("manifest policy should not re-read/hash the full data file")

    monkeypatch.setattr(pipeline, "_sha256_file", fail_hash)
    report = pipeline._checkpoint_data_integrity(args)

    assert report["hash_policy"] == "manifest"
    assert report["sha256"] is None
    assert report["hash_source"] == "skipped"


def test_weighted_pipeline_dataset_uses_tool_reward_rows(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 31 + 1 for ch in text[:24]]

    source = tmp_path / "tool_reward.jsonl"
    source.write_text(
        '{"training_kind":"tool_reward","prompt":"run tests","reward":1.0,"tool_calls":[{"tool":"pytest"}],"tool_results":[{"ok":true}]}\n',
        encoding="utf-8",
    )

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=8, vocab_size=32)
    ids, labels, weight = dataset[0]

    assert ids.shape == (8,)
    assert labels.shape == (8,)
    assert float(weight) == pytest.approx(2.0)


def test_weighted_pipeline_dataset_masks_user_and_trains_assistant(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    record = {
        "messages": [
            {"role": "system", "content": "follow instructions"},
            {"role": "user", "content": "name the proof phrase"},
            {"role": "assistant", "content": "target media proof"},
        ]
    }
    source = tmp_path / "messages.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=96, vocab_size=256)
    ids, labels, weight = dataset[0]
    valid = labels.ge(0)

    assert ids.shape == labels.shape == (96,)
    assert float(weight) > 0.0
    assert valid.sum().item() == len(TinyTokenizer().encode(" target media proof"))
    assert torch.equal(ids[valid], labels[valid])
    assert labels[:10].eq(-100).all()


def test_weighted_pipeline_dataset_caches_tokenized_records(tmp_path, monkeypatch) -> None:
    class CountingTokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, text: str) -> list[int]:
            self.calls += 1
            return [ord(ch) % 101 + 2 for ch in text]

    record = {
        "messages": [
            {"role": "user", "content": "name the cached target"},
            {"role": "assistant", "content": "cache proof answer"},
        ]
    }
    source = tmp_path / "cached.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setenv("OMNICODER2026_DATASET_RECORD_CACHE_MAX_BYTES", "1048576")

    tokenizer = CountingTokenizer()
    dataset = WeightedTextJsonlDataset(str(source), tokenizer, seq_len=96, vocab_size=256)
    first_ids, first_labels, _first_weight = dataset[0]
    second_ids, second_labels, _second_weight = dataset[0]
    summary = pipeline._dataset_source_summary(dataset)

    assert tokenizer.calls == 1
    assert torch.equal(first_ids, second_ids)
    assert torch.equal(first_labels, second_labels)
    assert summary["record_cache"]["entries"] == 1
    assert summary["record_cache"]["misses"] == 1
    assert summary["record_cache"]["hits"] >= 1


def test_weighted_pipeline_dataset_trains_media_target_json(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    record = {
        "input_json": {"prompt": "create a proof image"},
        "target_json": {
            "output_modality": "image",
            "artifact_path": "artifacts/proof_image.png",
            "artifact_tokens": "<image_begin> proof_image_token <image_end>",
        },
    }
    source = tmp_path / "media.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=160, vocab_size=256)
    ids, labels, _weight = dataset[0]
    valid = labels.ge(0)

    assert valid.any()
    assert torch.equal(ids[valid], labels[valid])
    assert labels[: len(TinyTokenizer().encode("user: create a proof image"))].eq(-100).all()
    target_text = "".join(chr(int(token.item())) for token in ids[valid])
    assert target_text.startswith(" image | ")
    assert '{"output_modality":"image"' in target_text
    assert "<image_begin> proof_image_token <image_end>" in target_text


def test_input_json_media_tokens_remain_unmasked_context(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    media_token_text = "<image_begin> cross_modal_glyph <image_end>"
    record = {
        "input_json": {
            "prompt": "describe the input image",
            "input_modality": "image",
            "image_tokens": media_token_text,
            "reference_image": "artifacts/glyph.png",
        },
        "target_json": {
            "output_modality": "text",
            "answer": "glyph present",
        },
    }
    source = tmp_path / "media_input.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=256, vocab_size=256)
    ids, labels, _weight = dataset[0]
    nonpad = ids.ne(0)
    full_text = "".join(chr(int(token.item())) for token in ids[nonpad])
    media_start = full_text.index(media_token_text)
    media_labels = labels[media_start : media_start + len(media_token_text)]
    valid = labels.ge(0)
    target_text = "".join(chr(int(ids[index].item())) for index in torch.nonzero(valid, as_tuple=False).flatten())

    assert media_token_text in full_text
    assert media_labels.eq(-100).all()
    assert target_text == " glyph present"


def test_messages_without_assistant_do_not_suppress_target_json(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    record = {
        "messages": [{"role": "user", "content": "create a proof image"}],
        "target_json": {
            "output_modality": "image",
            "artifact_tokens": "<image_begin> mixed_schema_target <image_end>",
        },
    }
    source = tmp_path / "mixed_schema.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=160, vocab_size=256)
    ids, labels, _weight = dataset[0]
    valid = labels.ge(0)
    target_text = "".join(chr(int(token.item())) for token in ids[valid])

    assert valid.any()
    assert target_text.startswith(" image | ")
    assert "mixed_schema_target" in target_text


def test_media_modality_alone_does_not_make_empty_target_json_a_media_payload(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    record = {
        "input_json": {"prompt": "generate a proof music artifact"},
        "modality": "music",
        "target_json": {"artifact_refs": [], "content": "8", "media_metadata": {}},
    }
    source = tmp_path / "scalar_media.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=160, vocab_size=256)
    ids, labels, _weight = dataset[0]
    valid = labels.ge(0)
    target_text = "".join(chr(int(token.item())) for token in ids[valid])

    assert target_text == " 8"
    assert "artifact_refs" not in target_text
    assert "music |" not in target_text


def test_weighted_pipeline_dataset_prefixes_media_route_from_messages(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    record = {
        "messages": [
            {"role": "user", "content": "render a proof video"},
            {
                "role": "assistant",
                "content": '{"output_modality":"video","artifact_tokens":"<video_begin> proof <video_end>"}',
            },
        ],
        "modality": "video",
    }
    source = tmp_path / "media_messages.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=160, vocab_size=256)
    ids, labels, _weight = dataset[0]
    valid = labels.ge(0)
    target_text = "".join(chr(int(token.item())) for token in ids[valid])

    assert target_text.startswith(" video | ")
    assert '{"output_modality":"video"' in target_text


def test_explicit_token_rows_concatenate_assistant_and_media_targets() -> None:
    record = {
        "prompt_token_ids": [10, 11],
        "assistant_token_ids": [20, 21],
        "artifact_token_ids": [30, 31],
    }

    ids, labels, weight = pipeline.record_ids_labels_weight(record, tokenizer=None)

    assert ids == [10, 11, 20, 21, 30, 31]
    assert labels == [-100, -100, 20, 21, 30, 31]
    assert weight > 0.0


def test_explicit_token_rows_do_not_duplicate_artifact_metadata_targets() -> None:
    record = {
        "prompt_token_ids": [10, 11],
        "target_token_ids": [30, 31],
        "artifact_token_ids": [30, 31],
    }

    ids, labels, weight = pipeline.record_ids_labels_weight(record, tokenizer=None)

    assert ids == [10, 11, 30, 31]
    assert labels == [-100, -100, 30, 31]
    assert weight > 0.0


def test_explicit_target_only_rows_prepend_masked_context() -> None:
    record = {"assistant_token_ids": [13]}

    ids, labels, weight = pipeline.record_ids_labels_weight(record, tokenizer=None)

    assert ids == [0, 13]
    assert labels == [-100, 13]
    assert weight > 0.0


def test_target_token_id_zero_is_preserved_when_masked() -> None:
    record = {
        "prompt_token_ids": [7],
        "assistant_token_ids": [0, 8],
    }

    ids, labels, _weight = pipeline.record_ids_labels_weight(record, tokenizer=None)

    assert ids == [7, 0, 8]
    assert labels == [-100, 0, 8]


def test_pipeline_trainer_rejects_native_continuous_media_rows_without_tokenized_targets() -> None:
    record = {
        "prompt": "Generate an image from native patches.",
        "native_media_features": [[0.1, 0.2, 0.3]],
        "native_media_targets": [[0.1, 0.2, 0.3]],
        "modality": "image",
    }

    with pytest.raises(ValueError, match="native continuous media rows require"):
        pipeline.record_ids_labels_weight(record, tokenizer=None)


def test_dataset_chunk_overlap_preserves_boundary_target_prediction(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 97 + 2 for ch in text]

    record = {
        "messages": [
            {"role": "user", "content": "x" * 9},
            {"role": "assistant", "content": "target"},
        ]
    }
    source = tmp_path / "boundary.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=12, vocab_size=256)
    ids, labels, _weight = dataset[1]
    valid = torch.nonzero(labels.ge(0), as_tuple=False).flatten()

    assert valid.numel() > 0
    assert int(valid[0].item()) > 0


def test_weighted_dataset_skips_explicitly_quarantined_rows(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 97 + 2 for ch in text]

    bad = {
        "messages": [{"role": "user", "content": "bad"}, {"role": "assistant", "content": "bad target"}],
        "quality_score": 0.99,
        "contamination_status": "clean",
        "train_quarantine_reasons": ["poison_wrong_answer_rule"],
    }
    good = {
        "messages": [{"role": "user", "content": "good"}, {"role": "assistant", "content": "good target"}],
        "quality_score": 0.99,
        "contamination_status": "clean",
    }
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=32, vocab_size=256)

    assert len(dataset.source_row_keys) == 1
    assert next(iter(dataset.row_metadata.values()))["source_id"].startswith("train.jsonl:")
    ids, labels, _weight = dataset[0]
    assert int(labels.ge(0).sum().item()) > 0


def test_text_dataset_rejects_refusal_and_contaminated_rows(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 97 + 2 for ch in text]

    bad_refusal = {
        "messages": [{"role": "user", "content": "answer"}, {"role": "assistant", "content": "Sorry, I can't assist with that."}],
        "quality_score": 0.99,
        "contamination_status": "clean",
    }
    bad_contaminated = {
        "text": "poisoned row",
        "quality_score": 0.99,
        "contamination_status": "contaminated",
    }
    good = {
        "messages": [{"role": "user", "content": "answer"}, {"role": "assistant", "content": "direct useful answer"}],
        "quality_score": 0.99,
        "contamination_status": "clean",
    }
    source = tmp_path / "dense.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in [bad_refusal, bad_contaminated, good]) + "\n", encoding="utf-8")

    dataset = TextJsonlDataset(str(source), TinyTokenizer(), seq_len=64, vocab_size=256)

    assert len(dataset.samples) == 1


def test_dataset_sparse_target_chunks_reanchor_to_answer_tokens(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    record = {
        "messages": [
            {"role": "user", "content": "context " * 80},
            {"role": "assistant", "content": "needle target"},
        ]
    }
    source = tmp_path / "sparse_target.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=32, vocab_size=256)
    _ids, labels, _weight = dataset[0]

    assert labels[1:].ge(0).any()


def test_dataset_skips_targetless_context_rows(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "mixed_targets.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "context only"}]}) + "\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "name the target"},
                    {"role": "assistant", "content": "usable answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=64, vocab_size=256)
    _ids, labels, _weight = dataset[0]

    assert labels[1:].ge(0).any()


def test_weighted_dataset_indexes_jsonl_with_single_parse_per_line(tmp_path, monkeypatch) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    rows = [
        {
            "messages": [
                {"role": "user", "content": "prompt one"},
                {"role": "assistant", "content": "answer one"},
            ],
            "quality_score": 0.99,
            "contamination_status": "clean",
        },
        {
            "messages": [
                {"role": "user", "content": "prompt two"},
                {"role": "assistant", "content": "answer two"},
            ],
            "quality_score": 0.99,
            "contamination_status": "clean",
        },
    ]
    source = tmp_path / "parse_once.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    real_loads = pipeline.json.loads
    calls = {"count": 0}

    def counting_loads(*args, **kwargs):
        calls["count"] += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(pipeline.json, "loads", counting_loads)

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=64, vocab_size=256)

    assert calls["count"] == len(rows)
    assert len(dataset.source_row_keys) == len(rows)
    assert len(dataset.row_metadata) == len(rows)
    assert len(dataset.records) >= len(rows)


def test_dataset_retries_legacy_targetless_index_entries(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "legacy_index.jsonl"
    first = json.dumps({"messages": [{"role": "user", "content": "legacy context only"}]})
    second = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "name the target"},
                {"role": "assistant", "content": "retry answer"},
            ]
        }
    )
    source.write_text(first + "\n" + second + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=64, vocab_size=256)
    first_offset = 0
    second_offset = len((first + "\n").encode("utf-8"))
    dataset.records = [(source, first_offset, 0, "jsonl"), (source, second_offset, 0, "jsonl")]
    _ids, labels, _weight = dataset[0]

    assert labels[1:].ge(0).any()


def test_dataset_retry_jumps_duplicate_targetless_chunks(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "duplicate_chunks.jsonl"
    first = json.dumps({"messages": [{"role": "user", "content": "legacy context only"}]})
    second = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "name the target"},
                {"role": "assistant", "content": "jump answer"},
            ]
        }
    )
    source.write_text(first + "\n" + second + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=64, vocab_size=256)
    first_offset = 0
    second_offset = len((first + "\n").encode("utf-8"))
    dataset.records = [(source, first_offset, chunk, "jsonl") for chunk in range(2048)]
    dataset.records.append((source, second_offset, 0, "jsonl"))
    _ids, labels, _weight = dataset[0]

    assert labels[1:].ge(0).any()


def test_dataset_max_source_rows_caps_jsonl_rows_not_indexed_windows(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "omni_all.jsonl"
    groups = ("text", "code_tool", "image_ocr", "video", "audio_tts_music", "ledger_all")
    rows = []
    for group in groups:
        for index in range(10):
            rows.append(
                {
                    "origin_group": group,
                    "modality": "image" if group == "image_ocr" else ("music" if group == "audio_tts_music" else "text"),
                    "messages": [
                        {"role": "user", "content": f"{group} prompt " + ("context " * 40)},
                        {"role": "assistant", "content": f"{group} target {index}"},
                    ],
                }
            )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=32, vocab_size=256, max_source_rows=60)
    summary = pipeline._dataset_source_summary(dataset)

    assert summary["source_rows"] == 60
    assert set(summary["origin_groups"]) == set(groups)
    assert all(count == 10 for count in summary["origin_groups"].values())
    assert summary["records"] > summary["source_rows"]


def test_dataset_source_summary_preserves_audio_music_tts_modalities(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "audio_tts_music.jsonl"
    rows = [
        {
            "origin_group": "audio_tts_music",
            "modality": modality,
            "messages": [
                {"role": "user", "content": f"{modality} prompt " + ("context " * 20)},
                {"role": "assistant", "content": f"{modality} target"},
            ],
        }
        for modality in ("audio", "music", "tts")
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=32, vocab_size=256, max_source_rows=3)
    summary = pipeline._dataset_source_summary(dataset)

    assert summary["modalities"] == {"audio": 1, "music": 1, "tts": 1}


def test_dataset_window_limit_is_row_first_before_overflow_chunks(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "row_first.jsonl"
    rows = [
        {
            "origin_group": f"group_{index}",
            "messages": [
                {"role": "user", "content": "context " * 30},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }
        for index in range(3)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(
        str(source),
        TinyTokenizer(),
        seq_len=16,
        vocab_size=256,
        max_source_rows=3,
        max_indexed_windows=3,
    )
    offsets = [offset for _path, offset, chunk, _kind in dataset.records]
    chunks = [chunk for _path, _offset, chunk, _kind in dataset.records]

    assert len(offsets) == 3
    assert len(set(offsets)) == 3
    assert chunks == [0, 0, 0]


def test_dataset_keeps_single_token_explicit_targets_shifted(tmp_path) -> None:
    class TinyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) % 101 + 2 for ch in text]

    source = tmp_path / "position_zero_only.jsonl"
    first = json.dumps({"assistant_token_ids": [13]})
    source.write_text(first + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), TinyTokenizer(), seq_len=64, vocab_size=256)
    _ids, labels, _weight = dataset[0]

    assert labels[1:].ge(0).any()


def test_dataset_chunk_overlap_preserves_repeated_boundaries(tmp_path) -> None:
    record = {
        "prompt_token_ids": [10, 11, 12],
        "assistant_token_ids": [13, 14, 15, 16, 17, 18, 19, 20],
    }
    source = tmp_path / "explicit_boundary.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = WeightedTextJsonlDataset(str(source), tokenizer=None, seq_len=4, vocab_size=256)
    ids, labels, _weight = dataset[2]
    valid = torch.nonzero(labels.ge(0), as_tuple=False).flatten()

    assert ids.tolist() == [16, 17, 18, 19]
    assert valid.tolist() == [1, 2, 3]


def test_weighted_pipeline_dataset_uses_full_prompt_tokenization_at_target_boundary(tmp_path) -> None:
    class MergeTokenizer:
        def encode(self, text: str) -> list[int]:
            ids: list[int] = []
            index = 0
            while index < len(text):
                if text.startswith(".\n", index):
                    ids.append(900)
                    index += 2
                    continue
                ids.append(ord(text[index]) % 101 + 2)
                index += 1
            return ids

    record = {
        "messages": [
            {"role": "user", "content": "Render proof image."},
            {"role": "assistant", "content": '{"output_modality":"image"}'},
        ]
    }
    source = tmp_path / "messages.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    tokenizer = MergeTokenizer()
    dataset = WeightedTextJsonlDataset(str(source), tokenizer, seq_len=96, vocab_size=1024)
    ids, labels, _weight = dataset[0]
    valid = labels.ge(0)
    expected_prompt = "user: Render proof image.\nassistant:"
    expected_full = expected_prompt + ' image | {"output_modality":"image"}'
    expected_full_ids = torch.tensor(tokenizer.encode(expected_full), dtype=torch.long)
    first_target = int(torch.nonzero(valid, as_tuple=False).flatten()[0].item())

    assert torch.equal(ids[: len(expected_full_ids)], expected_full_ids)
    assert torch.equal(ids[:first_target], torch.tensor(tokenizer.encode(expected_prompt), dtype=torch.long))
    assert torch.equal(ids[valid], labels[valid])


def test_filesystem_checkpoint_marker_wait_requires_all_rank_markers(tmp_path) -> None:
    checkpoint = tmp_path / "pipeline_checkpoint"
    checkpoint.mkdir()
    attempt_id = "current-attempt"
    for rank in (0, 2):
        rank_file = checkpoint / f"rank{rank:05d}.pt"
        rank_file.write_bytes(b"checkpoint")
        _rank_complete_file(checkpoint, rank).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint_attempt_id": attempt_id,
                    "rank": rank,
                    "world_size": 3,
                    "global_step": 11,
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(TimeoutError):
        _wait_for_rank_checkpoint_markers(
            checkpoint,
            world_size=3,
            global_step=11,
            attempt_id=attempt_id,
            timeout_seconds=0.01,
            poll_seconds=0.01,
        )

    rank_file = checkpoint / "rank00001.pt"
    rank_file.write_bytes(b"checkpoint")
    _rank_complete_file(checkpoint, 1).write_text(
        json.dumps(
            {
                "status": "complete",
                "checkpoint_attempt_id": attempt_id,
                "rank": 1,
                "world_size": 3,
                "global_step": 11,
            }
        ),
        encoding="utf-8",
    )

    markers = _wait_for_rank_checkpoint_markers(
        checkpoint,
        world_size=3,
        global_step=11,
        attempt_id=attempt_id,
        timeout_seconds=0.1,
        poll_seconds=0.01,
    )
    assert [marker.name for marker in markers] == [
        "rank00000.pt.complete.json",
        "rank00001.pt.complete.json",
        "rank00002.pt.complete.json",
    ]


def test_filesystem_checkpoint_marker_wait_rejects_stale_rank_marker(tmp_path) -> None:
    checkpoint = tmp_path / "pipeline_checkpoint"
    checkpoint.mkdir()
    for rank in range(3):
        rank_file = checkpoint / f"rank{rank:05d}.pt"
        rank_file.write_bytes(b"checkpoint")
        _rank_complete_file(checkpoint, rank).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint_attempt_id": "stale-attempt",
                    "rank": rank,
                    "world_size": 3,
                    "global_step": 11,
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(TimeoutError):
        _wait_for_rank_checkpoint_markers(
            checkpoint,
            world_size=3,
            global_step=11,
            attempt_id="current-attempt",
            timeout_seconds=0.01,
            poll_seconds=0.01,
        )


def test_save_sharded_checkpoint_filesystem_sync_finalizes_without_barrier(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "pipeline_checkpoint"
    monkeypatch.setattr(pipeline, "dist", FakeDist(rank=0, world_size=3))
    errors: list[BaseException] = []

    def run_rank0() -> None:
        try:
            save_sharded_checkpoint(
                checkpoint,
                TinyShard(),
                preset=SimpleNamespace(name="tiny"),
                args=checkpoint_args(tmp_path),
                optimizer=None,
                global_step=11,
                last_loss=0.25,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_rank0)
    worker.start()
    attempt = wait_for_attempt_marker(checkpoint)
    attempt_id = str(attempt["checkpoint_attempt_id"])
    for rank in (1, 2):
        rank_file = checkpoint / f"rank{rank:05d}.pt"
        rank_file.write_bytes(b"checkpoint")
        pipeline._atomic_write_json(
            _rank_complete_file(checkpoint, rank),
            {
                "status": "complete",
                "path": str(rank_file),
                "bytes": rank_file.stat().st_size,
                "format": "omnicoder2026_pipeline_stage_checkpoint_v2",
                "rank": rank,
                "world_size": 3,
                "global_step": 11,
                "last_loss": 0.25,
                "checkpoint_attempt_id": attempt_id,
            },
        )
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert errors == []
    complete = json.loads((checkpoint / ".complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    assert complete["checkpoint_attempt_id"] == attempt_id
    assert complete["rank_files"] == ["rank00000.pt", "rank00001.pt", "rank00002.pt"]
    io_log = checkpoint / "checkpoint_io.rank00000.jsonl"
    assert io_log.exists()
    io_record = json.loads(io_log.read_text(encoding="utf-8").splitlines()[-1])
    assert io_record["schema"] == pipeline.CHECKPOINT_IO_SCHEMA
    assert io_record["data"]["hash_policy"] == "manifest"
    assert io_record["data"]["hash_source"] == "skipped"
    assert io_record["bytes"]["rank_file_bytes"] > 0

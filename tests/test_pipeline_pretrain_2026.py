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
    loss.backward()
    assert hidden.grad is not None


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
    expected_positions = torch.arange(0, 6)
    logits = final.lm_head(processed[0, expected_positions, :])
    expected = F.cross_entropy(logits, labels[0, expected_positions + 1], reduction="mean")

    assert torch.allclose(loss.float(), expected.float(), atol=1e-5)


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

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omnicoder.training import pipeline_pretrain_2026_dense as pipeline


def test_module_grad_norm_streams_chunks(monkeypatch) -> None:
    monkeypatch.setenv("OMNICODER2026_GRAD_NORM_CHUNK_ELEMS", "3")
    layer = torch.nn.Linear(4, 2, bias=False)
    layer.weight.grad = torch.arange(1, 9, dtype=torch.float32).reshape_as(layer.weight)

    actual = pipeline._module_grad_norm(layer, enabled=True)

    expected = torch.linalg.vector_norm(layer.weight.grad.reshape(-1), ord=2).item()
    assert actual == pytest.approx(expected)


def test_ce_accumulator_defers_scalar_materialization_until_finalize() -> None:
    accumulator = pipeline._new_ce_accumulator()
    labels = torch.tensor([17, 132_096, 214_016, 999_999], dtype=torch.long)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    weights = torch.tensor([1.0, 0.5, 0.0, 2.0], dtype=torch.float32)

    pipeline._accumulate_ce_by_token_family(accumulator, labels, losses, weights)

    assert isinstance(accumulator["text"]["loss_sum"], torch.Tensor)
    ce_by_family, ce_by_modality, counts = pipeline._finalize_ce_accumulator(accumulator)
    assert counts["text"] == 1
    assert counts["vision_semantic"] == 1
    assert counts["vision_residual"] == 0
    assert counts["unknown"] == 1
    assert ce_by_family["text"] == pytest.approx(1.0)
    assert ce_by_family["vision_semantic"] == pytest.approx(2.0)
    assert ce_by_family["unknown"] == pytest.approx(4.0)
    assert ce_by_modality["vision"] == pytest.approx(2.0)


def _args(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "out": str(tmp_path / "checkpoint"),
        "telemetry_file": "",
        "train_diagnostics_file": "",
        "pipeline_schedule": "gpipe",
        "pipeline_microbatches": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "lr": 2.0e-5,
        "optimizer": "adamw",
        "optimizer_in_backward": False,
        "optimizer_in_backward_update": "",
        "optimizer_in_backward_grad_clip": 1.0,
        "optimizer_in_backward_clip_mode": "rms",
        "optimizer_in_backward_adafactor_clip_threshold": 1.0,
        "data": "train.jsonl",
        "data_manifest": "",
        "shuffle": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_optimizer_keeps_adamw_unless_lowmem_mode_is_explicit(tmp_path) -> None:
    layer = torch.nn.Linear(4, 2)
    adamw = pipeline.build_optimizer(_args(tmp_path, optimizer="adamw", optimizer_in_backward=False, optimizer_in_backward_update=""), layer)

    assert isinstance(adamw, torch.optim.AdamW)


def test_build_optimizer_lowmem_adafactor_is_opt_in(tmp_path) -> None:
    layer = torch.nn.Linear(4, 2)
    lowmem = pipeline.build_optimizer(
        _args(tmp_path, optimizer="adamw", optimizer_in_backward=True, optimizer_in_backward_update="lowmem_adafactor"),
        layer,
    )

    assert isinstance(lowmem, pipeline.PipelineLowMemoryAdafactor)


def test_pipeline_telemetry_cpu_fallback_writes_jsonl(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    args = _args(tmp_path)
    ranges = [(0, 1), (1, 3)]
    spec = pipeline.shard_spec(1, ranges)
    path = pipeline._rank_telemetry_path(args, rank=1, world_size=2)

    record = pipeline._pipeline_telemetry_record(
        args=args,
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        ranges=ranges,
        spec=spec,
        seq_len=128,
        step=7,
        local_step=2,
    )
    pipeline._append_pipeline_telemetry(path, record)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert path == tmp_path / "checkpoint" / "telemetry.rank00001.jsonl"
    assert rows == [record]
    assert record["rank"] == 1
    assert record["device"] == "cpu"
    assert record["cuda_active"] is False
    assert record["allocated_bytes"] == 0
    assert record["reserved_bytes"] == 0
    assert record["max_allocated_bytes"] == 0
    assert record["max_reserved_bytes"] == 0
    assert record["free_bytes"] == 0
    assert record["total_bytes"] == 0
    assert record["device_capability"] is None
    assert record["seq_len"] == 128
    assert record["step"] == 7
    assert record["placement_layer_counts"] == [1, 2]
    assert record["pipeline_stage_ranges"][1]["layer_count"] == 2


def test_pipeline_telemetry_records_cuda_memory_and_ranked_path(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path, telemetry_file=str(tmp_path / "memory.jsonl"))
    ranges = [(0, 2), (2, 4), (4, 8)]
    spec = pipeline.shard_spec(2, ranges)
    device = torch.device("cuda", 2)
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.torch.cuda, "memory_allocated", lambda current: 11)
    monkeypatch.setattr(pipeline.torch.cuda, "memory_reserved", lambda current: 22)
    monkeypatch.setattr(pipeline.torch.cuda, "max_memory_allocated", lambda current: 33)
    monkeypatch.setattr(pipeline.torch.cuda, "max_memory_reserved", lambda current: 44)
    monkeypatch.setattr(pipeline.torch.cuda, "mem_get_info", lambda current: (55, 66))
    monkeypatch.setattr(pipeline.torch.cuda, "get_device_capability", lambda current: (8, 6))
    monkeypatch.setattr(pipeline.torch.cuda, "get_device_name", lambda current: "Mock CUDA")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,4,6")
    monkeypatch.setenv("LOCAL_RANK", "2")

    record = pipeline._pipeline_telemetry_record(
        args=args,
        rank=2,
        world_size=3,
        device=device,
        ranges=ranges,
        spec=spec,
        seq_len=256,
        step=9,
        local_step=3,
    )
    path = pipeline._rank_telemetry_path(args, rank=2, world_size=3)

    assert path == tmp_path / "memory.rank00002.jsonl"
    assert record["device"] == "cuda:2"
    assert record["device_index"] == 2
    assert record["device_name"] == "Mock CUDA"
    assert record["cuda_active"] is True
    assert record["allocated_bytes"] == 11
    assert record["reserved_bytes"] == 22
    assert record["max_allocated_bytes"] == 33
    assert record["max_reserved_bytes"] == 44
    assert record["free_bytes"] == 55
    assert record["total_bytes"] == 66
    assert record["device_capability"] == [8, 6]
    assert record["cuda_visible_devices"] == "0,4,6"
    assert record["local_rank"] == 2
    assert record["stage_index"] == 2
    assert record["layer_count"] == 4
    assert record["has_head"] is True


def test_train_diagnostics_record_has_required_schema_counts_and_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    args = _args(tmp_path)
    ranges = [(0, 1), (1, 3)]
    spec = pipeline.shard_spec(1, ranges)
    labels = torch.tensor(
        [
            [-100, 17, 132_096, 214_016, 279_552, 324_608],
            [-100, -100, 197_632, 312_320, 328_704, -100],
        ],
        dtype=torch.long,
    )
    memory = pipeline._pipeline_telemetry_record(
        args=args,
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        ranges=ranges,
        spec=spec,
        seq_len=6,
        step=11,
        local_step=4,
    )
    optimizer = SimpleNamespace(param_groups=[{"lr": 2.0e-5}])
    loss_diagnostics = {
        "valid_target_tokens": 8,
        "optimized_target_tokens": 8,
        "target_counts_by_token_family": pipeline._token_family_counts(labels),
        "optimized_target_counts_by_token_family": pipeline._token_family_counts(labels),
        "ce_by_token_family": {"text": 1.0, "vision_semantic": 2.0, "vision_residual": 2.25, "speech_tts": 3.0},
        "ce_by_modality": {"text": 1.0, "vision": 2.1, "tts": 3.0},
    }

    record = pipeline._train_diagnostics_record(
        args=args,
        rank=1,
        world_size=2,
        spec=spec,
        global_step=11,
        local_step=4,
        seq_len=6,
        batch_size=2,
        microbatch_size=1,
        loss=4.25,
        labels=labels,
        sample_weights=torch.tensor([0.5, 2.0], dtype=torch.float32),
        optimizer=optimizer,
        optimizer_update=True,
        grad_norm_pre_clip=1.5,
        grad_norm_post_clip=1.2,
        step_elapsed_sec=0.5,
        memory_record=memory,
        loss_diagnostics=loss_diagnostics,
        source_summary={"available": True, "records": 7, "source_count": 1, "sources": {"train.jsonl": 7}},
    )

    assert record["schema"] == pipeline.TRAIN_DIAGNOSTICS_SCHEMA
    assert record["event"] == "train_step"
    assert record["global_step"] == 11
    assert record["seq_len"] == 6
    assert record["lr"] == {"group0": 2.0e-5}
    assert record["optimizer"]["update"] is True
    assert record["optimizer"]["grad_norm_pre_clip"] == 1.5
    assert record["optimizer"]["grad_norm_post_clip"] == 1.2
    assert record["loss"]["total_ce"] == 4.25
    assert record["loss"]["valid_target_tokens"] == 8
    assert record["targets"]["by_token_family"]["text"] == 1
    assert record["targets"]["by_token_family"]["vision_semantic"] == 1
    assert record["targets"]["by_token_family"]["vision_residual"] == 1
    assert record["targets"]["by_token_family"]["speech_tts"] == 1
    assert record["targets"]["by_token_family"]["audio_music"] == 1
    assert record["targets"]["by_token_family"]["music_control"] == 1
    assert record["targets"]["by_token_family"]["tool_agent"] == 1
    assert record["targets"]["by_token_family"]["flow"] == 1
    assert record["targets"]["by_modality"]["vision"] == 2
    assert record["targets"]["by_modality"]["audio_music"] == 1
    assert record["targets"]["by_modality"]["music"] == 1
    assert record["runtime"]["tokens"] == 12
    assert record["runtime"]["tokens_per_sec"] == 24.0
    assert record["runtime"]["rank_memory"]["device"] == "cpu"
    assert record["data"]["sample_weights"]["mean"] == 1.25
    assert record["data"]["source_summary"]["records"] == 7
    assert record["data"]["source_summary_ref"] == "dataset_index_done"


def test_train_diagnostics_reuses_loss_target_counts_without_label_cpu_pull(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        pipeline,
        "_token_family_counts",
        lambda _labels: (_ for _ in ()).throw(AssertionError("should reuse loss diagnostics counts")),
    )
    args = _args(tmp_path)
    ranges = [(0, 1), (1, 3)]
    spec = pipeline.shard_spec(1, ranges)
    memory = pipeline._pipeline_telemetry_record(
        args=args,
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        ranges=ranges,
        spec=spec,
        seq_len=6,
        step=12,
        local_step=5,
    )
    loss_diagnostics = {
        "valid_target_tokens": 3,
        "optimized_target_tokens": 2,
        "target_counts_by_token_family": {"text": 1, "vision_semantic": 1, "tool_agent": 1},
        "optimized_target_counts_by_token_family": {"text": 1, "vision_semantic": 1, "tool_agent": 0},
    }

    record = pipeline._train_diagnostics_record(
        args=args,
        rank=1,
        world_size=2,
        spec=spec,
        global_step=12,
        local_step=5,
        seq_len=6,
        batch_size=2,
        microbatch_size=1,
        loss=4.0,
        labels=torch.tensor([[17, 132_096, 312_320]], dtype=torch.long),
        sample_weights=None,
        optimizer=SimpleNamespace(param_groups=[{"lr": 2.0e-5}]),
        optimizer_update=True,
        grad_norm_pre_clip=None,
        grad_norm_post_clip=None,
        step_elapsed_sec=1.0,
        memory_record=memory,
        loss_diagnostics=loss_diagnostics,
        source_summary={
            "available": True,
            "records": 99,
            "source_count": 2,
            "sources": {"large.jsonl": 99},
            "row_sources": {"large.jsonl": 99},
            "origin_groups": {"huge": 99},
            "modalities": {"text": 99},
            "record_cache": {"entries": 2, "hits": 5, "misses": 2, "bytes": 512, "max_bytes": 1024},
        },
    )

    assert record["targets"]["by_token_family"]["text"] == 1
    assert record["targets"]["optimized_by_token_family"]["tool_agent"] == 0
    assert record["loss"]["valid_target_tokens"] == 3
    assert record["loss"]["optimized_target_tokens"] == 2
    assert record["data"]["source_summary"]["records"] == 99
    assert record["data"]["source_summary"]["record_cache"]["hits"] == 5
    assert "sources" not in record["data"]["source_summary"]
    assert "origin_groups" not in record["data"]["source_summary"]


def test_train_diagnostics_preserves_skipped_target_counts_without_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        pipeline,
        "_token_family_counts",
        lambda _labels: (_ for _ in ()).throw(AssertionError("skipped diagnostics should not inspect labels")),
    )
    args = _args(tmp_path)
    ranges = [(0, 1), (1, 3)]
    spec = pipeline.shard_spec(1, ranges)
    memory = pipeline._pipeline_telemetry_record(
        args=args,
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        ranges=ranges,
        spec=spec,
        seq_len=6,
        step=13,
        local_step=6,
    )

    record = pipeline._train_diagnostics_record(
        args=args,
        rank=1,
        world_size=2,
        spec=spec,
        global_step=13,
        local_step=6,
        seq_len=6,
        batch_size=2,
        microbatch_size=1,
        loss=3.5,
        labels=None,
        sample_weights=None,
        optimizer=SimpleNamespace(param_groups=[{"lr": 2.0e-5}]),
        optimizer_update=False,
        grad_norm_pre_clip=None,
        grad_norm_post_clip=None,
        step_elapsed_sec=1.0,
        memory_record=memory,
        loss_diagnostics={
            "diagnostics_skipped": True,
            "valid_target_tokens": 17,
            "optimized_target_tokens": 9,
        },
        source_summary={"available": True, "records": 1, "source_count": 1},
    )

    assert record["loss"]["valid_target_tokens"] == 17
    assert record["loss"]["optimized_target_tokens"] == 9
    assert record["targets"]["by_token_family"]["unknown"] == 17
    assert record["targets"]["optimized_by_token_family"]["unknown"] == 9


def test_train_diagnostics_path_and_jsonl_append(tmp_path) -> None:
    args = _args(tmp_path, train_diagnostics_file=str(tmp_path / "diag.jsonl"))
    path = pipeline._rank_train_diagnostics_path(args, rank=2, world_size=3)
    record = {"schema": pipeline.TRAIN_DIAGNOSTICS_SCHEMA, "event": "train_step", "global_step": 1}

    pipeline._append_pipeline_telemetry(path, record)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert path == tmp_path / "diag.rank00002.jsonl"
    assert rows == [record]


def test_checkpoint_eval_artifact_contract_names_required_outputs(tmp_path) -> None:
    contract = pipeline._checkpoint_eval_artifact_contract(tmp_path / "checkpoint")
    names = {artifact["name"] for artifact in contract["artifacts"]}

    assert contract["schema"] == pipeline.CHECKPOINT_EVAL_ARTIFACT_CONTRACT_SCHEMA
    assert contract["training_invoked"] is False
    assert {
        "heldout_sample_loss_by_modality",
        "target_token_rank_diagnostics",
        "text_code_tool_decode_probes",
        "media_route_probe_attempts",
    } <= names
    assert all(artifact["required"] is True for artifact in contract["artifacts"])

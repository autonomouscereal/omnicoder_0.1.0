from __future__ import annotations

import argparse
import json

import pytest

torch = pytest.importorskip("torch")

from omnicoder.training import pipeline_pretrain_2026_dense as pipeline


def _args(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "out": str(tmp_path / "checkpoint"),
        "telemetry_file": "",
        "pipeline_schedule": "gpipe",
        "pipeline_microbatches": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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
    monkeypatch.setattr(pipeline.torch.cuda, "get_device_name", lambda current: "Mock CUDA")

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
    assert record["stage_index"] == 2
    assert record["layer_count"] == 4
    assert record["has_head"] is True

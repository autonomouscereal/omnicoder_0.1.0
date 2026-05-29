from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnicoder.eval import pipeline_checkpoint_manifest_2026 as checkpoint_manifest


def _write_checkpoint(path: Path, *, ranks: int, manifest_world_size: int | None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {} if manifest_world_size is None else {"world_size": manifest_world_size}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / ".complete.json").write_text("{}", encoding="utf-8")
    for rank in range(ranks):
        (path / f"rank{rank:05d}.pt").write_bytes(b"")
        (path / f"rank{rank:05d}.pt.complete.json").write_text("{}", encoding="utf-8")
    return path


def test_batch_predict_manifest_world_size_accepts_four_rank_checkpoint(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=4)

    manifest = checkpoint_manifest.load_pipeline_manifest(checkpoint)

    assert manifest["world_size"] == 4
    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=0) == 4


def test_batch_predict_cli_world_size_accepts_four_rank_checkpoint_without_manifest_world_size(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=None)

    manifest = checkpoint_manifest.load_pipeline_manifest(checkpoint, expected_world_size=4)

    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=4) == 4
    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=0) == 4


def test_batch_predict_cli_world_size_can_override_stale_manifest_world_size(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=3)

    manifest = checkpoint_manifest.load_pipeline_manifest(checkpoint, expected_world_size=4)

    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=4) == 4
    with pytest.raises(checkpoint_manifest.PipelineCheckpointManifestError, match="expects exactly 3 pipeline shards"):
        checkpoint_manifest.load_pipeline_manifest(checkpoint)


def test_batch_predict_rejects_explicit_world_size_rank_count_mismatch(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=3, manifest_world_size=4)

    with pytest.raises(
        checkpoint_manifest.PipelineCheckpointManifestError,
        match="expects exactly 4 pipeline shards, found 3",
    ):
        checkpoint_manifest.load_pipeline_manifest(checkpoint, expected_world_size=4)


def test_batch_predict_rejects_negative_explicit_world_size(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=4)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(
        checkpoint_manifest.PipelineCheckpointManifestError,
        match="explicit world size must be non-negative",
    ):
        checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=-1)


def test_pipeline_eval_parsers_accept_four_rank_world_size_args() -> None:
    pytest.importorskip("torch")
    from omnicoder.eval import pipeline_checkpoint_batch_predict_2026 as batch_predict
    from omnicoder.eval import pipeline_target_token_diagnostics_2026 as target_diagnostics
    from omnicoder.eval import pipeline_token_topk_probe_2026 as topk_probe

    batch_args = batch_predict.build_parser().parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--tasks",
            "tasks.jsonl",
            "--out",
            "predictions.jsonl",
            "--nproc-per-node",
            "4",
        ]
    )
    target_args = target_diagnostics.build_parser().parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--data",
            "data.jsonl",
            "--out",
            "target_diagnostics.json",
            "--expected-world-size",
            "4",
        ]
    )
    topk_args = topk_probe.build_parser().parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--out",
            "topk.json",
            "--nproc-per-node",
            "4",
        ]
    )

    assert batch_args.nproc_per_node == 4
    assert target_args.expected_world_size == 4
    assert topk_args.expected_world_size == 4


def test_target_diagnostics_single_rank_hidden_path_does_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from omnicoder.eval import pipeline_target_token_diagnostics_2026 as target_diagnostics

    class FakeShard:
        def __call__(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.float().unsqueeze(-1)

    def fail_send(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("single-rank target diagnostics must not send to a nonexistent next rank")

    monkeypatch.setattr(target_diagnostics.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(target_diagnostics.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(target_diagnostics.dist, "send", fail_send)

    batch = torch.tensor([[1, 2, 3]], dtype=torch.long)
    hidden = target_diagnostics._pipeline_hidden(
        FakeShard(),
        batch,
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
        d_model=1,
        precision="fp32",
    )

    assert hidden is not None
    assert hidden.shape == (1, 3, 1)


def test_target_diagnostics_single_rank_evaluate_uses_local_final_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from omnicoder.eval import pipeline_target_token_diagnostics_2026 as target_diagnostics

    class FakeTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            return f"tok{int(token_ids[0])}"

    class FakeShard:
        def __call__(self, batch: torch.Tensor) -> torch.Tensor:
            values = batch.float().unsqueeze(-1)
            return values / values.clamp_min(1.0)

        def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
            shape = tuple(hidden.shape[:-1]) + (8,)
            logits = torch.zeros(shape, dtype=torch.float32, device=hidden.device)
            logits[..., 3] = 4.0
            logits[..., 4] = 3.0
            return logits

    data = tmp_path / "records.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    def fail_object_broadcast(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("single-rank target diagnostics should build the final record locally")

    monkeypatch.setattr(target_diagnostics.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(target_diagnostics.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(target_diagnostics.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(target_diagnostics.dist, "broadcast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(target_diagnostics.dist, "broadcast_object_list", fail_object_broadcast)
    monkeypatch.setattr(target_diagnostics, "_build_shard", lambda _args: (FakeShard(), torch.device("cpu"), 1, 8, "fake"))
    monkeypatch.setattr(target_diagnostics, "get_text_tokenizer", lambda prefer_hf=True: FakeTokenizer())
    monkeypatch.setattr(target_diagnostics, "effective_text_token_range", lambda **_kwargs: (0, 8))
    monkeypatch.setattr(target_diagnostics, "_candidate_data_files", lambda *_args, **_kwargs: [data])
    monkeypatch.setattr(target_diagnostics, "_read_jsonl", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(target_diagnostics, "record_ids_labels_weight", lambda *_args, **_kwargs: ([2, 3, 4], [-100, 3, 4], 1.0))

    args = SimpleNamespace(
        checkpoint="ckpt",
        data=[],
        data_dir=str(tmp_path),
        exclude_aggregate_jsonl=False,
        dist_backend="gloo",
        dist_timeout_seconds=1,
        init_dtype="fp32",
        precision="fp32",
        preset="fake",
        rank_device_map="",
        placement_layer_counts="",
        fake_quant=False,
        fake_quant_chunk_rows=0,
        fake_quant_max_full_elements=0,
        require_target_contract=False,
        allow_p40_target_contract_eval=False,
        seq_len=8,
        max_records_per_file=0,
        top_k=2,
        max_positions=4,
        progress_records=0,
    )

    result = target_diagnostics.evaluate(args)

    assert result is not None
    assert result["status"] == "ok"
    assert result["overall"]["records"] == 1
    assert result["overall"]["target_tokens"] == 2
    assert result["records"][0]["target_token_count"] == 2


def _local_dev_task(path: Path, *, task_id: str, benchmark_id: str, prompt: str, source: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "task_id": task_id,
                "reportable": False,
                "prompt": prompt,
                "source": source,
                "dataset_revision": "local-dev-2026",
                "snapshot_id": "local-dev-snapshot",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _batch_args(tmp_path: Path, checkpoint: Path, tasks: Path, *, allow_rejected: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=str(checkpoint),
        tasks=[str(tasks)],
        out=str(tmp_path / "predictions.jsonl"),
        summary="",
        model="unit-test-checkpoint",
        preset="fake",
        rank_device_map="",
        placement_layer_counts="",
        precision="fp32",
        init_dtype="fp32",
        max_prompt_tokens=32,
        max_output_tokens=4,
        progress_tasks=0,
        allow_local_dev_tasks=True,
        allow_rejected_model_output=allow_rejected,
        allow_media_route_text_proof=False,
        force=True,
        nproc_per_node=1,
        fake_quant=False,
    )


def test_batch_predict_media_route_writes_diagnostic_native_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from omnicoder.eval import pipeline_checkpoint_batch_predict_2026 as batch_predict

    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=1, manifest_world_size=1)
    tasks = _local_dev_task(
        tmp_path / "tasks.jsonl",
        task_id="image-1",
        benchmark_id="image_generation_local",
        prompt="Generate an image token artifact.",
        source="image_generation",
    )

    class FakeTokenizer:
        eos_token_id = None

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3] if text else [1]

        def decode(self, token_ids: list[int]) -> str:
            return "__OMNICODER_EMPTY_DECODE__"

    monkeypatch.setattr(batch_predict, "get_text_tokenizer", lambda prefer_hf=True: FakeTokenizer())
    monkeypatch.setattr(batch_predict, "effective_text_token_range", lambda **_kwargs: (0, 32))
    monkeypatch.setattr(batch_predict, "_broadcast_task_header", lambda *_args, **_kwargs: (True, 0, 32, None))
    monkeypatch.setattr(batch_predict.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        batch_predict,
        "_decode_rank0",
        lambda *_args, **_kwargs: ("__OMNICODER_EMPTY_DECODE__", 3, [1001, 1002, 1003]),
    )

    summary = batch_predict._run_rank0_batch(
        _batch_args(tmp_path, checkpoint, tasks),
        shard=object(),
        device=torch.device("cpu"),
        d_model=1,
        vocab_size=2048,
        saved_preset_name="fake",
    )

    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    artifact = rows[0]["generated_artifact"]
    artifact_path = Path(artifact["path"])

    assert summary["status"] == "ok"
    assert summary["skipped"]["records"] == 0
    assert "unsupported_media_artifact_backend" not in json.dumps(rows[0], sort_keys=True)
    assert rows[0]["prediction_scope"] == "diagnostic_native_media_artifact_proof"
    assert rows[0].get("prediction_quality_status") != "rejected_model_output"
    assert artifact_path.exists()
    assert artifact["modality"] == "image"
    assert artifact["diagnostic"] is True
    assert artifact["diagnostic_only"] is True
    assert artifact["token_ids"] == [1001, 1002, 1003]
    assert artifact["token_count"] == 3
    assert artifact["byte_size"] == artifact_path.stat().st_size
    assert artifact["sha256"] == batch_predict.harness.file_sha256(artifact_path)
    assert artifact["output_route"]["artifact_backend"] == "diagnostic_native_media_token_artifact"


def test_batch_predict_rejects_junk_text_output_unless_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from omnicoder.eval import pipeline_checkpoint_batch_predict_2026 as batch_predict

    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=1, manifest_world_size=1)
    tasks = _local_dev_task(
        tmp_path / "tasks.jsonl",
        task_id="text-1",
        benchmark_id="text_local",
        prompt="Answer briefly.",
        source="text",
    )

    class FakeTokenizer:
        eos_token_id = None

        def encode(self, text: str) -> list[int]:
            return [1, 2] if text else [1]

        def decode(self, token_ids: list[int]) -> str:
            return "__OMNICODER_EMPTY_DECODE__"

    monkeypatch.setattr(batch_predict, "get_text_tokenizer", lambda prefer_hf=True: FakeTokenizer())
    monkeypatch.setattr(batch_predict, "effective_text_token_range", lambda **_kwargs: (0, 32))
    monkeypatch.setattr(batch_predict, "_broadcast_task_header", lambda *_args, **_kwargs: (True, 0, 32, None))
    monkeypatch.setattr(batch_predict.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        batch_predict,
        "_decode_rank0",
        lambda *_args, **_kwargs: ("__OMNICODER_EMPTY_DECODE__", 3, [1, 2, 3]),
    )
    args = _batch_args(tmp_path, checkpoint, tasks)

    with pytest.raises(batch_predict.BatchPredictError, match="greedy decode failed sanity gate"):
        batch_predict._run_rank0_batch(args, object(), torch.device("cpu"), 1, 2048, "fake")

    allowed_args = _batch_args(tmp_path, checkpoint, tasks, allow_rejected=True)
    allowed_args.out = str(tmp_path / "allowed_predictions.jsonl")
    summary = batch_predict._run_rank0_batch(allowed_args, object(), torch.device("cpu"), 1, 2048, "fake")
    row = json.loads((tmp_path / "allowed_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert summary["status"] == "ok"
    assert row["prediction_quality_status"] == "rejected_model_output"
    assert any("__OMNICODER_EMPTY_DECODE__" in reason for reason in row["prediction_quality_reasons"])

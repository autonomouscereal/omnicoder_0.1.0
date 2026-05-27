from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from omnicoder.eval import benchmark_suite_2026 as benchmark_suite
from omnicoder.eval import reportable_prediction_harness_2026 as harness


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _authorized_task(**overrides: Any) -> dict[str, Any]:
    row = {
        "benchmark_id": "multimodal_mmmu_pro_2026",
        "task_id": "mmmu-fixture-1",
        "dataset_revision": "mmmu-pro-authorized-2026-05",
        "snapshot_id": "mmmu-pro-authorized-2026-05-smoke",
        "snapshot_authorization": "authorized_private",
        "snapshot_sha256": "sha256:mmmu-pro-smoke",
        "authorization_ref": "internal-authorized-eval-ledger",
        "source": "https://mmmu-benchmark.github.io/",
        "reportable": True,
        "question": "Which option matches the diagram?",
        "choices": ["A", "B", "C", "D"],
        "answer": "C",
    }
    row.update(overrides)
    return row


def _reportable_profile(path: Path) -> Path:
    profile = {
        "version": "2026-05-24.prediction-harness-test",
        "benchmarks": [
            {
                "benchmark_id": "multimodal_mmmu_pro_2026",
                "adapter_kind": "multimodal_mcq",
                "axis": "multimodal_understanding",
                "source": "https://mmmu-benchmark.github.io/",
                "task_format": "jsonl_multimodal_mcq",
                "splits": {"smoke": "authorized fixture"},
                "metrics": ["accuracy"],
                "holdout_policy": ["hide_answers"],
            }
        ],
        "release_gates": {"multimodal_understanding_release": ["multimodal_mmmu_pro_2026"]},
    }
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def test_fixture_mode_writes_run_reportable_predictions_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("fixture mode must not perform network I/O")

    monkeypatch.setattr(harness.urllib.request, "urlopen", fail_urlopen)
    tasks = _write_jsonl(tmp_path / "tasks.jsonl", [_authorized_task(fixture_prediction="C")])
    predictions = tmp_path / "predictions.jsonl"

    assert (
        harness.main(
            [
                "--backend",
                "fixture",
                "--model",
                "fixture-local",
                "--tasks",
                str(tasks),
                "--out",
                str(predictions),
            ]
        )
        == 0
    )

    rows = _jsonl_rows(predictions)
    assert rows == [
        {
            **rows[0],
            "schema": "omnicoder.reportable_prediction_2026.v1",
            "schema_version": "2026-05-24",
            "benchmark_id": "multimodal_mmmu_pro_2026",
            "task_id": "mmmu-fixture-1",
            "model": "fixture-local",
            "backend": "fixture",
            "prediction": "C",
        }
    ]

    profile = _reportable_profile(tmp_path / "profile.json")
    out_dir = tmp_path / "bench"
    assert (
        benchmark_suite.main(
            [
                "--profile",
                str(profile),
                "--out-dir",
                str(out_dir),
                "run-reportable",
                "--tasks",
                str(tasks),
                "--predictions",
                str(predictions),
                "--run-id",
                "prediction-harness-fixture",
            ]
        )
        == 0
    )
    result = _jsonl_rows(out_dir / "reportable_results.jsonl")[0]
    assert result["status"] == "passed"
    assert result["score"] == 1.0
    assert result["score_json"]["reportable_score"] is True


def test_strict_validation_rejects_unauthorized_task_before_generation(tmp_path: Path) -> None:
    task = _authorized_task()
    task.pop("snapshot_authorization")
    tasks = _write_jsonl(tmp_path / "tasks.jsonl", [task])

    assert (
        harness.main(
            [
                "--backend",
                "fixture",
                "--tasks",
                str(tasks),
                "--out",
                str(tmp_path / "predictions.jsonl"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "predictions.jsonl").exists()


def test_local_dev_mode_accepts_public_dev_rows_without_authorizing_reportable_score(tmp_path: Path) -> None:
    task = _authorized_task(
        reportable=False,
        snapshot_authorization=None,
        snapshot_id=None,
        dataset_revision=None,
        fixture_prediction="B",
    )
    tasks = _write_jsonl(tmp_path / "public_dev.jsonl", [task])
    predictions = tmp_path / "predictions.jsonl"
    summary = tmp_path / "summary.json"

    assert (
        harness.main(
            [
                "--backend",
                "fixture",
                "--model",
                "fixture-local-dev",
                "--tasks",
                str(tasks),
                "--out",
                str(predictions),
                "--summary",
                str(summary),
                "--allow-local-dev-tasks",
            ]
        )
        == 0
    )

    rows = _jsonl_rows(predictions)
    assert rows[0]["benchmark_id"] == "multimodal_mmmu_pro_2026"
    assert rows[0]["prediction"] == "B"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["task_mode"] == "local_public_dev"
    assert payload["official_score"] is False


def test_checkpoint_runner_reads_sanitized_stdin_and_writes_prediction(tmp_path: Path) -> None:
    tasks = _write_jsonl(tmp_path / "tasks.jsonl", [_authorized_task(answer="D")])
    runner = tmp_path / "runner.py"
    seen = tmp_path / "seen.json"
    runner.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "payload = json.loads(sys.stdin.read())",
                f"pathlib.Path({str(seen)!r}).write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')",
                "print(json.dumps({'prediction': 'D'}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"

    assert (
        harness.main(
            [
                "--backend",
                "checkpoint-runner",
                "--checkpoint-runner",
                f"{sys.executable} {runner}",
                "--checkpoint-path",
                str(tmp_path / "checkpoint"),
                "--model",
                "local-checkpoint",
                "--tasks",
                str(tasks),
                "--out",
                str(predictions),
            ]
        )
        == 0
    )

    request = json.loads(seen.read_text(encoding="utf-8"))
    assert request["checkpoint_path"] == str(tmp_path / "checkpoint")
    assert request["task_id"] == "mmmu-fixture-1"
    assert "answer" not in request["task"]
    assert "gold" not in request["task"]
    assert _jsonl_rows(predictions)[0]["prediction"] == "D"


def test_prediction_validation_can_preserve_rejected_model_output_for_scoring() -> None:
    row = {
        "schema": harness.PREDICTION_SCHEMA,
        "schema_version": harness.SCHEMA_VERSION,
        "benchmark_id": "agent_agentif_2025",
        "task_id": "junk-output-fixture",
        "model": "local-checkpoint",
        "backend": "pipeline_checkpoint_batch_predict_2026",
        "prediction": ",,,,,,,,,,,,,,,,",
    }

    rejections = harness.prediction_output_quality_rejections(row)
    assert rejections
    assert rejections[0].startswith("prediction:junk_text:")
    with pytest.raises(harness.HarnessError, match="rejected model output"):
        harness.validate_prediction_row(row)
    harness.validate_prediction_row(row, allow_rejected_model_output=True)


def test_openai_compatible_backend_requires_local_endpoint() -> None:
    with pytest.raises(harness.HarnessError, match="must be local"):
        harness.validate_local_endpoint("https://api.openai.com/v1")

from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import contamination
from omnicoder.training import training_orchestration_2026 as orchestration


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_scan_marks_hellaswag_public_dev_metadata_without_prompt_marker(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    protected = tmp_path / "protected.jsonl"
    out = tmp_path / "scanned.jsonl"
    protected.write_text("", encoding="utf-8")
    _write_jsonl(
        candidates,
        [
            {
                "schema": "omnicoder.benchmark_task_2026.v1",
                "benchmark_id": "reasoning_hellaswag_full_2026",
                "task_id": "17",
                "question": "A person opens the oven door and",
                "choices": ["sits on the couch.", "pulls out a tray.", "drives away.", "prints a receipt."],
                "answer": "1",
                "contamination_class": "public_dev_eval",
                "source_file": "weights/data_factory/runs/benchmark_materialization/local_2026/reasoning_hellaswag_full_2026_public_dev.jsonl",
            }
        ],
    )

    contamination.scan(candidates, protected, out, threshold=0.42, ngram=5)

    scanned = _read_jsonl(out)[0]
    scan = scanned["contamination"]
    assert scan["status"] == "contaminated"
    assert scan["match_type"] == "benchmark_marker"
    assert "hellaswag" in scan["markers"]
    assert "public_dev" in scan["markers"]
    assert "benchmark_id" in scan["markers"]


def test_scan_marks_reportable_eval_path_from_metadata(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    protected = tmp_path / "protected.jsonl"
    out = tmp_path / "scanned.jsonl"
    protected.write_text("", encoding="utf-8")
    _write_jsonl(
        candidates,
        [
            {
                "prompt": "Solve the task.",
                "response": "Neutral answer.",
                "source_uri": "data/eval/reportable_2026/mmmu_pro_authorized.jsonl",
                "task_revision": "mmmu_pro_authorized-2026-05-reportable",
            }
        ],
    )

    contamination.scan(candidates, protected, out, threshold=0.42, ngram=5)

    scan = _read_jsonl(out)[0]["contamination"]
    assert scan["status"] == "contaminated"
    assert "reportable" in scan["markers"]
    assert "data_eval_path" in scan["markers"]
    assert "eval_reportable_path" in scan["markers"]


def test_scan_marks_modern_eval_suite_metadata(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    protected = tmp_path / "protected.jsonl"
    out = tmp_path / "scanned.jsonl"
    protected.write_text("", encoding="utf-8")
    _write_jsonl(
        candidates,
        [
            {
                "prompt": "Run benchmark task.",
                "response": "Neutral answer.",
                "benchmark_name": "MMLU-Pro BFCL LiveCodeBench WebArena FrontierMath GPQA-Diamond",
            }
        ],
    )

    contamination.scan(candidates, protected, out, threshold=0.42, ngram=5)

    scan = _read_jsonl(out)[0]["contamination"]
    assert scan["status"] == "contaminated"
    assert "mmlu_pro" in scan["markers"]
    assert "bfcl" in scan["markers"]
    assert "livecodebench" in scan["markers"]
    assert "webarena" in scan["markers"]
    assert "frontiermath" in scan["markers"]
    assert "gpqa_diamond" in scan["markers"]


def test_scanned_public_dev_benchmark_row_cannot_enter_train_split(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    protected = tmp_path / "protected.jsonl"
    out = tmp_path / "scanned.jsonl"
    protected.write_text("", encoding="utf-8")
    _write_jsonl(
        candidates,
        [
            {
                "record_id": "clean",
                "modality": "text",
                "payload_sha256": "clean-sha",
                "source_date": "2026-05-28",
                "quality": {"score": 0.9},
                "contamination": {"status": "clean"},
                "prompt": "Explain a safe data curation check.",
                "response": "Use quality, provenance, and contamination gates.",
            },
            {
                "record_id": "dev-benchmark",
                "modality": "text",
                "payload_sha256": "dev-sha",
                "source_date": "2026-05-28",
                "quality": {"score": 0.9},
                "schema": "omnicoder.benchmark_task_2026.v1",
                "benchmark_id": "reasoning_hellaswag_full_2026",
                "contamination_class": "public_dev_eval",
                "source_file": "local_2026/reasoning_hellaswag_full_2026_public_dev.jsonl",
                "question": "A person opens the oven door and",
                "answer": "1",
            },
        ],
    )

    contamination.scan(candidates, protected, out, threshold=0.42, ngram=5)
    split = orchestration.assign_deterministic_splits(
        _read_jsonl(out),
        "text",
        {"eval_holdout_ratio": 0.0, "test_holdout_ratio": 0.0},
    )

    assert [row["record_id"] for row in split["train"]] == ["clean"]
    assert split["eval"] == []
    assert split["test"] == []

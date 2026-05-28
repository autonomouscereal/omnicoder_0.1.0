from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import dataset_index_2026 as indexer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_dataset_index_counts_sources_modalities_and_fingerprints(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "source_id": "common_corpus",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "license": "CC0",
                "contamination_status": "clean",
                "target_token_ids": [1, 2, 3],
            },
            {
                "source_id": "musiccaps_review",
                "modality": "music",
                "split": "research_internal",
                "use_policy": "research_internal",
                "license": "manual_review",
                "contamination_status": "clean",
                "artifact_token_ids": [150000],
            },
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "passed"
    assert payload["rows"] == 2
    assert payload["by_modality"] == {"music": 1, "text": 1}
    assert payload["by_source"] == {"common_corpus": 1, "musiccaps_review": 1}
    assert payload["counts"]["rows_with_target_tokens"] == 1
    assert payload["counts"]["rows_with_artifact_tokens"] == 1
    assert payload["files"][0]["sha256"]


def test_dataset_index_fails_train_eval_leakage_marker(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "source_id": "bad_public_dev",
                "modality": "text",
                "split": "train",
                "target": "answer_key leaked from HellaSwag ARC-AGI3 SWE-bench Terminal-Bench MMMU-Pro",
            }
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "train_eval_leakage_markers" in payload["fail_reasons"]
    assert payload["counts"]["train_eval_leakage_markers"] == 1


def test_dataset_index_counts_structured_target_json_content(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "structured-1",
                "source_id": "structured_source",
                "target_modality": "text",
                "split": "train",
                "target_json": {"content": "Useful structured target text."},
            }
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "passed"
    assert payload["counts"]["rows_with_target_tokens"] == 1
    assert payload["counts"]["one_token_junk_rows"] == 0
    assert payload["by_modality"] == {"text": 1}


def test_dataset_index_fails_duplicate_ids_missing_modality_one_token_and_split_mismatch(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "dup-1",
                "source_id": "source_a",
                "modality": "text",
                "split": "train",
                "target": "Useful answer with enough tokens.",
            },
            {
                "record_id": "dup-1",
                "source_id": "source_b",
                "modality": "text",
                "split": "eval",
                "target": "ok",
            },
            {
                "record_id": "missing-modality",
                "source_id": "source_c",
                "split": "train",
                "target": "Another useful answer with enough tokens.",
            },
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "failed"
    assert "duplicate_ids" in payload["fail_reasons"]
    assert "missing_modality_metadata" in payload["fail_reasons"]
    assert "one_token_junk_rows" in payload["fail_reasons"]
    assert "split_mismatch" in payload["fail_reasons"]
    assert payload["counts"]["duplicate_ids"] == 1
    assert payload["counts"]["missing_modality_metadata"] == 1
    assert payload["counts"]["one_token_junk_rows"] == 1
    assert payload["counts"]["split_mismatch"] == 1

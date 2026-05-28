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
                "target": "answer_key leaked",
            }
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "train_eval_leakage_markers" in payload["fail_reasons"]
    assert payload["counts"]["train_eval_leakage_markers"] == 1

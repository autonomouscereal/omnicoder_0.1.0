from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import external_train_rewrite_2026 as rewrite


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_external_train_rewrite_replaces_stale_train_files_and_preserves_nontrain(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "integrity" / "dataset_integrity_accepted.jsonl"
    manifest = tmp_path / "manifests" / "external_dataset_manifest.json"
    _write_jsonl(
        accepted,
        [
            {
                "record_id": "fineweb-1",
                "dataset_family": "educational_text",
                "modality": "text",
                "training_bucket": "train",
                "target": "Useful educational target text.",
            },
            {
                "record_id": "opencoder-1",
                "dataset_family": "code_generation",
                "modality": "code",
                "training_bucket": "train",
                "target": "def add(a, b): return a + b",
            },
        ],
    )
    _write_jsonl(jsonl_dir / "train_all_external.jsonl", [{"record_id": "old-junk", "modality": "text"}])
    _write_jsonl(jsonl_dir / "image_generation_editing.jsonl", [{"record_id": "blocked-old", "modality": "image"}])
    _write_jsonl(jsonl_dir / "image_generation_editing_all.jsonl", [{"record_id": "nontrain-preserved", "modality": "image"}])
    _write_jsonl(jsonl_dir / "blocked_until_review.jsonl", [{"record_id": "blocked-preserved", "modality": "video"}])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "records": {"train": 99, "research_internal": 3, "eval_holdout": 2, "blocked_until_review": 1},
                "training_paths": {"train_all_external": str(jsonl_dir / "train_all_external.jsonl")},
            }
        ),
        encoding="utf-8",
    )

    report = rewrite.rewrite_external_train_bucket(
        accepted,
        jsonl_dir,
        tmp_path / "integrity" / "train_bucket_integrity_rewrite.json",
        source_manifest=manifest,
    )

    assert report["accepted_rows"] == 2
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["fineweb-1", "opencoder-1"]
    assert {row["use_policy"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")} == {"train"}
    assert {row["split"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")} == {"train"}
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train.jsonl")] == ["fineweb-1", "opencoder-1"]
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "educational_text.jsonl")] == ["fineweb-1"]
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "code_generation.jsonl")] == ["opencoder-1"]
    assert _read_jsonl(jsonl_dir / "image_generation_editing_all.jsonl") == [{"record_id": "nontrain-preserved", "modality": "image"}]
    assert _read_jsonl(jsonl_dir / "blocked_until_review.jsonl") == [{"record_id": "blocked-preserved", "modality": "video"}]
    assert _read_jsonl(jsonl_dir / "image_generation_editing.jsonl") == []
    assert "image_generation_editing.jsonl" in report["files_truncated"]

    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["records"]["train"] == 2
    assert updated["records"]["total_training_rows"] == 8
    assert updated["clean_train_families"] == {"code_generation": 1, "educational_text": 1}
    assert updated["integrity_rewrite"]["status"] == "rewritten_clean"
    assert updated["integrity_rewrite"]["accepted_rows"] == 2
    assert updated["integrity_rewrite"]["skipped_rows"] == 0
    assert updated["promotion_allowed"] is True
    assert updated["promotion_status"] == "integrity_rewritten_pending_index"


def test_external_train_rewrite_skips_nontrain_accepted_rows(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    _write_jsonl(
        accepted,
        [
            {"record_id": "train-1", "dataset_family": "math_reasoning", "modality": "text", "training_bucket": "train", "target": "good answer"},
            {"record_id": "eval-1", "dataset_family": "benchmarks", "modality": "text", "training_bucket": "eval_holdout", "target": "answer key"},
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 1
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["train-1"]
    assert not (jsonl_dir / "benchmarks.jsonl").exists()


def test_external_train_rewrite_skips_rejected_or_quarantined_train_rows(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    _write_jsonl(
        accepted,
        [
            {
                "record_id": "clean-1",
                "dataset_family": "math_reasoning",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination_status": "clean",
                "target": "clean answer",
            },
            {
                "record_id": "rejected-1",
                "dataset_family": "math_reasoning",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "dataset_integrity_2026": {"accepted": False, "reasons": ["ai_watermark_synthid"]},
                "target": "bad answer",
            },
            {
                "record_id": "quarantine-1",
                "dataset_family": "math_reasoning",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "train_quarantine_reasons": ["missing_quality_score"],
                "target": "quarantined answer",
            },
            {
                "record_id": "blocked-synthetic-1",
                "dataset_family": "math_reasoning",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "synthetic_train_blocked": True,
                "target": "blocked synthetic answer",
            },
            {
                "record_id": "low-quality-1",
                "dataset_family": "math_reasoning",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "quality_score": 0.1,
                "target": "low quality answer",
            },
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 1
    assert report["skipped_rows"] == 4
    assert report["skipped_rows_by_reason"] == {
        "dataset_integrity_rejected": 1,
        "low_quality_score": 1,
        "synthetic_train_blocked": 1,
        "train_quarantine_reasons": 1,
    }
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["clean-1"]


def test_external_train_rewrite_rechecks_stale_accepted_rows_with_current_integrity_policy(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    _write_jsonl(
        accepted,
        [
            {
                "record_id": "clean-substantive-1",
                "dataset_family": "math_reasoning",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination_status": "clean",
                "quality_score": 0.92,
                "prompt": "Solve the equation and verify both roots.",
                "target": "The solution expands the polynomial, isolates the two candidate roots, substitutes both roots back into the original equation, and reports only the verified values.",
                "dataset_integrity_2026": {"accepted": True, "reasons": []},
            },
            {
                "record_id": "stale-tool-ok",
                "dataset_family": "agentic_tool_reasoning",
                "modality": "tool",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination_status": "clean",
                "quality_score": 0.9,
                "prompt": "Call the status tool.",
                "target": "OK",
                "dataset_integrity_2026": {"accepted": True, "reasons": []},
            },
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 1
    assert report["skipped_rows"] == 1
    assert list(report["skipped_rows_by_reason"].values()) == [1]
    assert next(iter(report["skipped_rows_by_reason"])).startswith("dataset_integrity_current:")
    assert "target_len_le_1" in next(iter(report["skipped_rows_by_reason"]))
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["clean-substantive-1"]


def test_external_train_rewrite_skips_one_token_answer_only_math_rows(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    _write_jsonl(
        accepted,
        [
            {
                "record_id": "answer-only-1",
                "dataset_family": "math_reasoning",
                "modality": "math",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination_status": "clean",
                "quality_score": 0.75,
                "messages": [
                    {"role": "user", "content": "Choose the correct option for the geometry problem."},
                    {"role": "assistant", "content": "$\\fbox{A}$"},
                ],
                "dataset_integrity_2026": {"accepted": True, "reasons": []},
            },
            {
                "record_id": "reasoned-1",
                "dataset_family": "math_reasoning",
                "modality": "math",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination_status": "clean",
                "quality_score": 0.85,
                "messages": [
                    {"role": "user", "content": "Solve and explain."},
                    {
                        "role": "assistant",
                        "content": "Compute the expression step by step, verify the arithmetic, and report the final value.",
                    },
                ],
                "dataset_integrity_2026": {"accepted": True, "reasons": []},
            },
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 1
    assert report["skipped_rows_by_reason"] == {"one_token_train_target": 1}
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["reasoned-1"]


def test_external_train_rewrite_rejects_external_media_train_rows_without_artifacts(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    _write_jsonl(
        accepted,
        [
            {
                "record_id": "image-no-artifact-1",
                "dataset_family": "image_generation_editing",
                "modality": "image",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination_status": "clean",
                "quality_score": 0.91,
                "prompt": "Generate a clean product-style image for the supplied brief.",
                "target": "A bright studio image with clear lighting, centered composition, and polished visual detail.",
                "dataset_integrity_2026": {"accepted": True, "reasons": []},
            },
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 0
    assert report["skipped_rows"] == 1
    assert report["skipped_rows_by_reason"] == {"dataset_integrity_current:missing_media_artifact_ref": 1}
    assert not (jsonl_dir / "train_all_external.jsonl").exists()


def test_external_train_rewrite_dedupes_duplicate_ids(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    _write_jsonl(
        accepted,
        [
            {"record_id": "dup-1", "dataset_family": "math_reasoning", "modality": "text", "training_bucket": "train", "target": "first good answer"},
            {"record_id": "dup-1", "dataset_family": "math_reasoning", "modality": "text", "training_bucket": "train", "target": "second duplicate answer"},
            {"record_id": "uniq-1", "dataset_family": "math_reasoning", "modality": "text", "training_bucket": "train", "target": "unique answer"},
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 2
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["dup-1", "uniq-1"]


def test_external_train_rewrite_skips_near_duplicate_payloads(tmp_path: Path) -> None:
    jsonl_dir = tmp_path / "jsonl"
    accepted = tmp_path / "accepted.jsonl"
    base = (
        "The training example explains a robust parser that validates inputs, preserves provenance fields, "
        "records every rejection reason, and emits deterministic JSONL output for audit review."
    )
    _write_jsonl(
        accepted,
        [
            {
                "record_id": "near-1",
                "source_id": "src-a",
                "dataset_family": "text_pretraining",
                "modality": "text",
                "training_bucket": "train",
                "quality_score": 0.92,
                "target": base,
            },
            {
                "record_id": "near-2",
                "source_id": "src-b",
                "dataset_family": "text_pretraining",
                "modality": "text",
                "training_bucket": "train",
                "quality_score": 0.93,
                "target": base,
            },
        ],
    )

    report = rewrite.rewrite_external_train_bucket(accepted, jsonl_dir, tmp_path / "rewrite.json")

    assert report["accepted_rows"] == 1
    assert report["skipped_rows_by_reason"] == {"near_duplicate_payload": 1}
    assert [row["record_id"] for row in _read_jsonl(jsonl_dir / "train_all_external.jsonl")] == ["near-1"]

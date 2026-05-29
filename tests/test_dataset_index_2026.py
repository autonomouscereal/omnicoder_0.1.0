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
                    "split": "train",
                    "use_policy": "train",
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


def test_dataset_index_tracks_train_eval_research_block_buckets(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "train-1",
                "source_id": "src",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "target": "Useful training target.",
            },
            {"record_id": "eval-1", "source_id": "src", "modality": "text", "training_bucket": "eval_holdout", "target": "Eval-only target."},
            {"record_id": "research-1", "source_id": "src", "modality": "text", "training_bucket": "research_internal", "target": "Research-only target."},
            {"record_id": "block-1", "source_id": "src", "modality": "text", "training_bucket": "blocked_until_review", "target": "Blocked review target."},
        ],
    )

    payload = indexer.build_index([data])

    assert payload["by_training_bucket"] == {
        "blocked_until_review": 1,
        "eval_holdout": 1,
        "research_internal": 1,
        "train": 1,
    }
    assert payload["by_train_eval_research_block"] == {"block": 1, "eval": 1, "research": 1, "train": 1}
    assert payload["by_source_training_bucket"] == [
        {"source_id": "src", "training_bucket": "blocked_until_review", "rows": 1},
        {"source_id": "src", "training_bucket": "eval_holdout", "rows": 1},
        {"source_id": "src", "training_bucket": "research_internal", "rows": 1},
        {"source_id": "src", "training_bucket": "train", "rows": 1},
    ]


def test_dataset_index_separates_reportable_and_diagnostic_benchmark_rows(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "train-1",
                "source_id": "fineweb_edu",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "target": "Useful training target with enough words.",
            },
            {
                "record_id": "hellaswag-reportable-1",
                "source_id": "hellaswag_authorized",
                "modality": "text",
                "training_bucket": "eval_holdout",
                "use_policy": "reportable_eval_only",
                "benchmark_id": "reasoning_hellaswag_full_2026",
                "reportable": True,
                "benchmark_eval_only": True,
                "reportability_scope": "official_or_authorized_snapshot",
                "target": "Authorized answer key held out for scoring only.",
            },
            {
                "record_id": "hellaswag-public-dev-1",
                "source_id": "hellaswag_public_dev",
                "modality": "text",
                "training_bucket": "eval_holdout",
                "use_policy": "validation_only",
                "benchmark_id": "reasoning_hellaswag_full_2026",
                "benchmark_eval_only": True,
                "local_only": True,
                "public_dev": True,
                "reportability_scope": "validation_only_public_dev",
                "target": "Public development answer key for regression diagnostics only.",
            },
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "passed"
    assert payload["by_index_bucket"] == {
        "benchmark_diagnostic_eval": 1,
        "benchmark_reportable_eval": 1,
        "train": 1,
    }
    assert payload["counts"]["benchmark_rows"] == 2
    assert payload["counts"]["benchmark_reportable_eval_rows"] == 1
    assert payload["counts"]["benchmark_diagnostic_eval_rows"] == 1
    assert payload["counts"]["benchmark_rows_in_train_bucket"] == 0
    assert payload["by_index_bucket_training_bucket"] == [
        {"index_bucket": "benchmark_diagnostic_eval", "training_bucket": "eval_holdout", "rows": 1},
        {"index_bucket": "benchmark_reportable_eval", "training_bucket": "eval_holdout", "rows": 1},
        {"index_bucket": "train", "training_bucket": "train", "rows": 1},
    ]


def test_dataset_index_fails_benchmark_rows_in_train_index_bucket(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "hellaswag-train-leak-1",
                "source_id": "hellaswag_public_dev",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "benchmark_id": "reasoning_hellaswag_full_2026",
                "benchmark_eval_only": True,
                "local_only": True,
                "reportability_scope": "validation_only_public_dev",
                "target": "Benchmark answer key must remain outside training rows.",
            }
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "benchmark_rows_in_train_bucket" in payload["fail_reasons"]
    assert payload["counts"]["benchmark_rows_in_train_bucket"] == 1
    assert payload["by_index_bucket"] == {"benchmark_diagnostic_eval": 1}
    assert payload["benchmark_train_bucket_examples"][0]["index_bucket"] == "benchmark_diagnostic_eval"


def test_dataset_index_rejects_eval_only_policy_in_train_bucket(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "eval-policy-in-train",
                "source_id": "heldout_public_suite",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "validation_only",
                "training_allowed": False,
                "target": "Held out validation target text.",
            }
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "non_training_policy_in_train_bucket" in payload["fail_reasons"]
    assert payload["counts"]["non_training_policy_in_train_bucket"] == 1
    assert payload["non_training_policy_train_examples"][0]["reason"] == "use_policy:validation_only"


def test_dataset_index_rejects_eval_policy_aliases_for_explicit_train_bucket(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "eval-alias-in-train",
                "source_id": "swe_bench_verified",
                "modality": "code",
                "training_bucket": "train",
                "use_policy": "eval",
                "target": "Protected benchmark target text.",
            },
            {
                "record_id": "holdout-alias-in-train",
                "source_id": "webarena",
                "modality": "tool",
                "training_bucket": "train",
                "use_policy": "benchmark_holdout",
                "target": "Protected browser task verifier.",
            },
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert payload["counts"]["non_training_policy_in_train_bucket"] == 2
    assert {row["reason"] for row in payload["non_training_policy_train_examples"]} == {
        "use_policy:benchmark_holdout",
        "use_policy:eval",
    }


def test_dataset_index_rejects_benchmark_flags_in_train_bucket(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "public-dev-local-only",
                "source_id": "hellaswag_public_dev",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "local_only": True,
                "contamination_class": "public_dev_eval",
                "target": "Public dev answer.",
            },
            {
                "record_id": "benchmark-id",
                "source_id": "mmlu_pro_public",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "benchmark_id": "reasoning_hellaswag_full_2026",
                "target": "Benchmark answer.",
            },
            {
                "record_id": "nested-contamination",
                "source_id": "reportable_eval",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "contamination": {"status": "contaminated"},
                "target": "Reportable eval answer.",
            },
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "non_training_policy_in_train_bucket" in payload["fail_reasons"]
    assert payload["counts"]["non_training_policy_in_train_bucket"] == 3
    assert {row["reason"] for row in payload["non_training_policy_train_examples"]} == {
        "benchmark_id_present",
        "contamination_class:contaminated",
        "contamination_class:public_dev_eval",
    }


def test_dataset_index_rejects_train_rows_missing_source_or_policy(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "missing-source",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "target": "Train rows must name their source.",
            },
            {
                "record_id": "missing-policy",
                "source_id": "fineweb_edu",
                "modality": "text",
                "training_bucket": "train",
                "target": "Train rows must carry a policy.",
            },
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "train_rows_missing_source_or_policy" in payload["fail_reasons"]
    assert payload["counts"]["train_rows_missing_source_or_policy"] == 2
    assert {row["reason"] for row in payload["train_metadata_examples"]} == {
        "missing_source_id",
        "missing_use_policy",
    }


def test_dataset_index_flags_empty_prompt_leak_and_url_only_media(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    prompt = "Explain the exact artifact generation prompt, camera, lighting, and safety provenance."
    _write_jsonl(
        data,
        [
            {"record_id": "empty", "source_id": "src", "modality": "text", "split": "train", "target": ""},
            {"record_id": "leak", "source_id": "src", "modality": "text", "split": "train", "prompt": prompt, "response": prompt + " Extra."},
            {
                "record_id": "url-media",
                "source_id": "src",
                "modality": "image",
                "split": "train",
                "target_json": {"content": "https://cdn.example.invalid/image.png", "artifact_refs": [{"url": "https://cdn.example.invalid/image.png"}]},
            },
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "failed"
    assert "empty_target_rows" in payload["fail_reasons"]
    assert "prompt_target_leakage" in payload["fail_reasons"]
    assert "url_only_media_rows" in payload["fail_reasons"]
    assert payload["counts"]["empty_target_rows"] == 1
    assert payload["counts"]["prompt_target_leakage"] == 1
    assert payload["counts"]["url_only_media_rows"] == 1


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


def test_dataset_index_fails_train_rows_with_benchmark_eval_only_marker_fields(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "benchmark-leak",
                "source_id": "ordinary_source",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "target": "This training target has enough useful words to avoid unrelated target length failures.",
                "benchmark_id": "public-dev-local-only-eval",
                "local_only": True,
            }
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "train_eval_leakage_markers" in payload["fail_reasons"]
    assert payload["counts"]["train_eval_leakage_markers"] == 1


def test_dataset_index_fails_rejected_or_quarantined_train_rows(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "rejected-1",
                "source_id": "source_a",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "target": "Useful answer with enough words.",
                "dataset_integrity_2026": {"accepted": False, "reasons": ["ai_watermark_synthid"]},
            },
            {
                "record_id": "quarantine-1",
                "source_id": "source_b",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "target": "Useful answer with enough words.",
                "train_quarantine_reasons": ["missing_quality_score"],
            },
            {
                "record_id": "synthetic-blocked-1",
                "source_id": "source_c",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "target": "Useful answer with enough words.",
                "synthetic_train_blocked": True,
            },
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "blocked_or_rejected_train_rows" in payload["fail_reasons"]
    assert payload["counts"]["blocked_or_rejected_train_rows"] == 3
    assert {row["reason"] for row in payload["blocked_train_examples"]} == {
        "dataset_integrity_rejected",
        "synthetic_train_blocked",
        "train_quarantine_reasons",
    }


def test_dataset_index_rejects_rejected_or_quarantine_input_files(tmp_path: Path) -> None:
    data = tmp_path / "rejected" / "bad.jsonl"
    data.parent.mkdir()
    _write_jsonl(
        data,
        [
            {
                "record_id": "looks-clean-but-retained-rejected-file",
                "source_id": "src",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "quality_score": 0.99,
                "target": "This row is syntactically clean but lives under a rejected path.",
            }
        ],
    )

    payload = indexer.build_index([data])

    assert payload["status"] == "failed"
    assert "rejected_or_quarantine_input_files" in payload["fail_reasons"]
    assert payload["counts"]["rejected_or_quarantine_input_files"] == 1


def test_dataset_index_rejects_nested_rejected_low_quality_and_integrity_rows(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "low-quality",
                "source_id": "src",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "quality_score": 0.1,
                "target": "This row is coherent enough but below the final quality floor.",
            },
            {
                "record_id": "nested-rejected",
                "source_id": "src",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "quality_score": 0.95,
                "target": "This row carries a previous curation rejection.",
                "curation_policy_2026": {"accepted": False, "reasons": ["dataset_integrity:poison_wrong_answer_rule"]},
            },
            {
                "record_id": "watermark",
                "source_id": "src",
                "modality": "text",
                "training_bucket": "train",
                "use_policy": "train",
                "quality_score": 0.95,
                "target": "Generated by Gemini with SynthID watermark and content provenance.",
            },
        ],
    )

    payload = indexer.build_index([data], scan_dataset_integrity=True, min_quality_score=0.55)

    assert payload["status"] == "failed"
    assert "low_quality_rows" in payload["fail_reasons"]
    assert "nested_rejected_rows" in payload["fail_reasons"]
    assert "dataset_integrity_rejected_rows" in payload["fail_reasons"]
    assert payload["counts"]["low_quality_rows"] == 1
    assert payload["counts"]["nested_rejected_rows"] == 1
    assert payload["counts"]["dataset_integrity_rejected_rows"] >= 1


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
                "use_policy": "train",
                "target_json": {"content": "Useful structured target text."},
            }
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "passed"
    assert payload["counts"]["rows_with_target_tokens"] == 1
    assert payload["counts"]["one_token_junk_rows"] == 0
    assert payload["by_modality"] == {"text": 1}


def test_dataset_index_counts_target_json_when_input_messages_are_prompt_only(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "structured-message-1",
                "source_id": "structured_source",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "input_json": {"messages": [{"role": "user", "content": "Pretraining chunk prompt"}]},
                "target_json": {"content": "Pretraining chunk target"},
            }
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "passed"
    assert payload["counts"]["rows_with_target_tokens"] == 1
    assert payload["counts"]["one_token_junk_rows"] == 0


def test_dataset_index_counts_assistant_messages_as_target_tokens(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "message-sft-1",
                "source_id": "message_source",
                "modality": "code",
                "training_bucket": "train",
                "use_policy": "train",
                "quality_score": 0.9,
                "messages": [
                    {"role": "user", "content": "Write a deterministic parser."},
                    {
                        "role": "assistant",
                        "content": "The parser validates each row, preserves source metadata, and writes clean JSONL records.",
                    },
                ],
            }
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "passed"
    assert payload["counts"]["rows_with_target_tokens"] == 1
    assert payload["counts"]["one_token_junk_rows"] == 0


def test_dataset_index_does_not_flag_text_pretraining_self_supervision_as_prompt_leakage(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    target = (
        "The curated passage explains how an educational corpus keeps provenance, license, and quality fields "
        "alongside the training text so downstream dataset audits can reproduce every filtering decision."
    )
    _write_jsonl(
        data,
        [
            {
                "record_id": "pretrain-self-supervised-1",
                "source_id": "fineweb_edu",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "dataset_family": "text_pretraining",
                "training_kind": "text_pretraining",
                "text": target,
                "target_json": {"content": target},
            }
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "passed"
    assert payload["counts"]["prompt_target_leakage"] == 0
    assert payload["counts"]["one_token_junk_rows"] == 0


def test_dataset_index_allows_short_text_only_when_structured_tool_payload_exists(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    _write_jsonl(
        data,
        [
            {
                "record_id": "structured-tool-1",
                "source_id": "tool_traces",
                "modality": "tool",
                "split": "train",
                "use_policy": "train",
                "prompt": "Call the status tool.",
                "target": "OK",
                "tool_calls": [{"name": "status", "arguments": {"service": "api"}}],
                "tool_results": [{"status": "ok", "latency_ms": 32}],
            },
            {
                "record_id": "junk-tool-1",
                "source_id": "tool_traces",
                "modality": "tool",
                "split": "train",
                "use_policy": "train",
                "prompt": "Call the status tool.",
                "target": "OK",
            },
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "failed"
    assert payload["counts"]["one_token_junk_rows"] == 1
    assert payload["one_token_junk_examples"][0]["source_id"] == "tool_traces"


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


def test_dataset_index_fails_train_near_duplicate_5gram_rows(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    target_a = (
        "The verified tool trace opens the repository, inspects the failing unit test, "
        "patches the parser branch, reruns the exact regression, and records the passing evidence."
    )
    target_b = (
        "The verified tool trace opens the repository, inspects the failing unit test, "
        "patches the parser branch, reruns the exact regression, and records the clean evidence."
    )
    _write_jsonl(
        data,
        [
            {
                "record_id": "near-dup-a",
                "source_id": "agent_traces",
                "modality": "tool",
                "split": "train",
                "use_policy": "train",
                "prompt": "Summarize the validated repair trace.",
                "target": target_a,
            },
            {
                "record_id": "near-dup-b",
                "source_id": "agent_traces",
                "modality": "tool",
                "split": "train",
                "use_policy": "train",
                "prompt": "Summarize the validated repair trace.",
                "target": target_b,
            },
        ],
    )

    payload = indexer.build_index([data], expected_split="train", near_duplicate_threshold=0.84)

    assert payload["status"] == "failed"
    assert "near_duplicate_rows" in payload["fail_reasons"]
    assert payload["counts"]["near_duplicate_rows"] == 1
    assert payload["near_duplicate_examples"][0]["match_type"] == "5gram_jaccard"


def test_dataset_index_rejects_sample_paths_for_train_rows(tmp_path: Path) -> None:
    data = tmp_path / "samples" / "train.jsonl"
    data.parent.mkdir()
    _write_jsonl(
        data,
        [
            {
                "record_id": "sample-train-row",
                "source_id": "sample_source",
                "modality": "text",
                "split": "train",
                "use_policy": "train",
                "target": "Sample fixture rows must not silently enter a real training manifest.",
            }
        ],
    )

    payload = indexer.build_index([data], expected_split="train")

    assert payload["status"] == "failed"
    assert "fixture_or_sample_train_input_files" in payload["fail_reasons"]
    assert payload["counts"]["fixture_or_sample_train_input_files"] == 1

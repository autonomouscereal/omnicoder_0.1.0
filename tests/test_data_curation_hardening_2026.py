import argparse
import json
from pathlib import Path

import pytest

from omnicoder.data_factory import (
    balanced_allmodal_posttrain_2026 as balanced,
    curated_dataset_builder_2026 as curated_builder,
    curation_layers_2026 as curation_layers,
    curation_policy_2026 as policy,
    dataset_expansion_2026 as expansion,
    export_sft_jsonl,
    trace_orchestrator_2026,
)
from omnicoder.training import training_orchestration_2026 as orch


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def test_missing_protected_eval_fails_closed(tmp_path: Path) -> None:
    profile = {"contamination": {"protected_path": "missing_protected.jsonl"}}
    with pytest.raises(FileNotFoundError, match="protected contamination/eval holdout"):
        trace_orchestrator_2026.ensure_protected(profile, tmp_path, tmp_path / "fallback.jsonl")


def test_sft_export_rejects_nonclean_rows_and_drops_tainted_trace(tmp_path: Path) -> None:
    source = tmp_path / "curated.jsonl"
    out = tmp_path / "sft.jsonl"
    _write_jsonl(
        source,
        [
            {
                "lineage": {"trace_id": "tainted"},
                "input_json": {"messages": [{"role": "user", "content": "Explain the patch."}]},
                "target_json": {"content": "Apply the verified fix."},
                "quality": {"score": 0.95},
                "contamination": {"status": "clean"},
            },
            {
                "lineage": {"trace_id": "tainted"},
                "input_json": {"messages": [{"role": "user", "content": "Eval row."}]},
                "target_json": {"content": "Should quarantine the trace."},
                "quality": {"score": 0.95},
                "contamination": {"status": "suspect"},
            },
            {
                "lineage": {"trace_id": "clean"},
                "input_json": {"messages": [{"role": "user", "content": "Summarize."}]},
                "target_json": {"content": "Only clean rows export."},
                "quality": {"score": 0.95},
                "contamination": {"status": "clean"},
            },
            {
                "lineage": {"trace_id": "unknown"},
                "input_json": {"messages": [{"role": "user", "content": "Unknown contamination."}]},
                "target_json": {"content": "No export."},
                "quality": {"score": 0.95},
            },
        ],
    )

    count = export_sft_jsonl.export_trace_conversations(source, out, min_quality=0.55, allow_contaminated=False)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert rows[0]["metadata"]["trace_id"] == "clean"


def test_quality_policy_hard_rejects_placeholder_tiny_repetition_and_noise() -> None:
    cfg = policy.CurationPolicyConfig(min_quality_score=0.0, min_target_chars=8, reject_placeholder_junk=True)
    placeholder = policy.audit_training_record(
        {"quality_score": 0.99, "contamination_status": "clean"},
        prompt="Write production code.",
        target="TODO",
        modality="code",
        existing_quality=0.99,
        config=cfg,
    )
    noisy = policy.audit_training_record(
        {"quality_score": 0.99, "contamination_status": "clean"},
        prompt="Repeat the phrase.",
        target=("alpha " * 100) + "\x01\x02\x03",
        modality="text",
        existing_quality=0.99,
        config=cfg,
    )

    assert not placeholder["accepted"]
    assert {"placeholder_or_stub", "target_too_short"} & set(placeholder["reasons"])
    assert not noisy["accepted"]
    assert "low_diversity_repetition" in noisy["reasons"] or "control_character_noise" in noisy["reasons"]


def test_quality_policy_rejects_prompt_copy_and_bare_media_paths() -> None:
    cfg = policy.CurationPolicyConfig(min_quality_score=0.0, min_target_chars=2, reject_placeholder_junk=True)
    prompt_copy = policy.audit_training_record(
        {"quality_score": 0.99, "contamination_status": "clean"},
        prompt="Describe the terminal failure and patch it.",
        target="Describe the terminal failure and patch it.",
        modality="tool",
        existing_quality=0.99,
        config=cfg,
    )
    media_path = policy.audit_training_record(
        {"quality_score": 0.99, "contamination_status": "clean"},
        prompt="Generate a bright snare-driven loop.",
        target="/mnt/artifacts/loops/snare_loop.wav",
        modality="music",
        existing_quality=0.99,
        config=cfg,
    )
    media_payload = policy.audit_training_record(
        {
            "quality_score": 0.99,
            "contamination_status": "clean",
            "target_json": {"content": "Generate a bright snare-driven loop.", "artifact_tokens": [101, 102, 103]},
        },
        prompt="Generate a bright snare-driven loop.",
        target="Generate a bright snare-driven loop.",
        modality="music",
        existing_quality=0.99,
        config=cfg,
    )

    assert not prompt_copy["accepted"]
    assert "target_copies_prompt" in prompt_copy["reasons"]
    assert not media_path["accepted"]
    assert "media_target_too_short_or_scalar" in media_path["reasons"]
    assert media_payload["accepted"]


def test_dataset_expansion_quarantines_missing_train_metadata(tmp_path: Path) -> None:
    plan = {"artifact_token_count": {}, "target_text_chars": 512}
    entry = {
        "name": "missing_meta",
        "family": "math_reasoning",
        "target_modality": "text",
        "license": "Apache-2.0",
        "license_tier": "permissive",
        "use_policy": "train",
        "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["id"]},
    }
    row = expansion.record_to_training_row(entry, {"problem": "Solve 1+1.", "answer": "2", "id": "m1"}, plan, 1)

    assert row is not None
    assert row["training_bucket"] == "research_internal"
    assert row["source_date"] == "unknown"
    assert row["quality"]["score"] == 0.0
    assert "missing_quality_score" in row["train_quarantine_reasons"]
    assert "contamination_unknown" in row["train_quarantine_reasons"]


def test_balanced_builder_refuses_fixture_paths_and_unknown_contamination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "modalities": {name: {"enabled": True} for name in balanced.DEFAULT_STAGE_ORDER},
                "real_sources": {},
                "training_plan": {"required_modalities": []},
                "learning_checks": {"min_loss_points": 1},
            }
        ),
        encoding="utf-8",
    )
    fixture_source = tmp_path / "examples" / "sample.jsonl"
    unknown_source = tmp_path / "real_text.jsonl"
    _write_jsonl(
        fixture_source,
        [{"prompt": "Clean prompt", "response": "Clean response", "quality_score": 0.9, "contamination_status": "clean"}],
    )
    _write_jsonl(
        unknown_source,
        [{"prompt": "Unknown prompt", "response": "Unknown response", "quality_score": 0.9}],
    )
    monkeypatch.setattr(balanced, "repo_root", lambda: tmp_path)

    manifest = balanced.build_balanced_exports(
        argparse.Namespace(
            profile=str(profile),
            out_dir=str(tmp_path / "out"),
            out_jsonl="",
            manifest="",
            source=[f"text={fixture_source}", f"text={unknown_source}"],
            no_profile_sources=True,
            cap=[],
            source_floor=[],
            max_records_per_modality=8,
            max_source_records=0,
            require_modalities="",
            min_records_per_required_modality=1,
            allow_missing_required=True,
            strip_token_ids=False,
            reject_refusal_boilerplate=False,
            reject_eval_holdout=False,
            allow_eval_holdout=False,
            allow_fixture_data=False,
            allow_source_floor_cap_overrun=False,
            allow_dataset_integrity_issues=False,
            skip_integrity_artifact_scan=False,
            max_integrity_artifact_bytes=1024 * 1024,
            min_quality_score=0.55,
            require_media_artifacts=False,
            schema="messages",
            max_prompt_chars=24000,
            max_target_chars=24000,
        )
    )

    assert manifest["counts"]["sft"] == 0
    assert any(report["status"] == "fixture_refused" for report in manifest["source_reports"])
    assert manifest["skipped"]["contamination"] == 1


def test_training_splits_drop_missing_quality_date_and_nonclean_contamination() -> None:
    rows = [
        {
            "record_id": "clean",
            "modality": "text",
            "payload_sha256": "a",
            "source_date": "2026-05-28",
            "quality": {"score": 0.9},
            "contamination": {"status": "clean"},
        },
        {
            "record_id": "unknown",
            "modality": "text",
            "payload_sha256": "b",
            "source_date": "2026-05-28",
            "quality": {"score": 0.9},
            "contamination": {"status": "unknown"},
        },
        {"record_id": "missing", "modality": "text", "payload_sha256": "c"},
    ]

    split = orch.assign_deterministic_splits(rows, "text", {"eval_holdout_ratio": 0.0, "test_holdout_ratio": 0.0})

    assert [row["record_id"] for row in split["train"]] == ["clean"]
    assert split["eval"] == []
    assert split["test"] == []


def test_curated_trace_rejects_unknown_contamination() -> None:
    row = curated_builder.curated_trace_to_training_row(
        {"input_json": {"content": "Use the tool."}, "target_json": {"content": "Tool result verified."}},
        {
            "quality": {"score": 0.9},
            "contamination": {"status": "unknown"},
            "secret_redaction": {"has_secret": False},
            "split_assignment": {"split": "train"},
            "normalized_text": "Use the tool.\nTool result verified.",
        },
        {"artifact_token_count": {}, "target_text_chars": 512},
        {"min_quality": 0.55, "min_chars": 8},
    )

    assert row is None


def test_legacy_curated_trace_export_does_not_prompt_copy() -> None:
    row = curation_layers.curated_to_training_example(
        {
            "curated_id": "trace-1",
            "normalized_text": "Tool call completed and the patch was verified.",
            "split_assignment": {"split": "train"},
            "quality": {"score": 0.95, "label": "accept"},
            "contamination": {"status": "clean"},
            "secret_redaction": {"has_secret": False},
            "source": {
                "input_json": {"event_type": "tool_call", "tool_name": "shell"},
                "target_json": {},
                "metadata": {"bucket": "curated_agentic_trace_2026"},
            },
            "provenance": {"source_date": "2026-05-28"},
            "dedupe": {"canonical_sha256": "abc"},
        },
        {"train"},
        0.55,
    )

    assert row is not None
    prompt = row["input_json"]["messages"][0]["content"]
    target = row["target_json"]["content"]
    assert prompt.strip()
    assert target.strip()
    assert prompt != target

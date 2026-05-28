from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import curated_dataset_builder_2026 as builder
from omnicoder.training import training_orchestration_2026 as orch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path
    trace_dir = root / "data" / "raw" / "codex_traces_2026"
    _write_jsonl(
        trace_dir / "codex_local_2026.jsonl",
        [
            {
                "type": "message",
                "role": "user",
                "content": "Design a careful reasoning data curation run with modality coverage, decontamination, heldout splits, and tool-call replay.",
                "timestamp": "2026-05-23T00:00:00Z",
                "session_id": "trace-text",
            },
            {
                "type": "tool_call",
                "tool_name": "shell_command",
                "content": "Run pytest, inspect failure output, update the file, and rerun verification until the status is clean.",
                "input": {"command": "pytest tests/test_dataset.py"},
                "output": {"exit_code": 0, "stdout": "passed with per-modality coverage evidence"},
                "timestamp": "2026-05-23T00:01:00Z",
                "session_id": "trace-tool",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "```python\ndef curate(record):\n    if record.get('tool_name'):\n        return 'tool'\n    return 'text'\n```\nThis code path normalizes agent traces for training.",
                "timestamp": "2026-05-23T00:02:00Z",
                "session_id": "trace-code",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": " ".join(
                    f"anchor_{index} fact_{index * 3} retrieval_{index * 5} dependency_{index * 7} evidence_{index * 11} verification_{index * 13}"
                    for index in range(260)
                ),
                "timestamp": "2026-05-23T00:03:00Z",
                "session_id": "trace-long",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "This unsafe record must not train because password=super_secret_training_password appears in the trace.",
                "timestamp": "2026-05-23T00:04:00Z",
                "session_id": "trace-secret",
            },
        ],
    )
    media_root = root / "media"
    for path, payload in (
        (media_root / "image" / "sample.jpg", b"\xff\xd8" + (b"image-bytes" * 200) + b"\xff\xd9"),
        (media_root / "video" / "sample.mp4", b"video-bytes" * 220),
        (media_root / "audio" / "sample_audio.wav", b"audio-bytes" * 220),
        (media_root / "music" / "sample_music.mp3", b"music-bytes" * 220),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        _write_json(
            path.with_suffix(path.suffix + ".json"),
            {
                "prompt": f"Curated prompt for {path.stem}",
                "caption": f"Curated caption for {path.stem}",
                "quality_score": 0.9,
                "source_date": "2026-05-23",
                "contamination_status": "clean",
            },
        )
    return root


def _training_profile(root: Path) -> dict:
    return {
        "profile_name": "builder_unit_training",
        "modalities": {name: {"enabled": True} for name in orch.DEFAULT_STAGE_ORDER},
        "real_sources": {},
        "training_plan": {
            "max_records_per_modality": 4,
            "max_records_per_modality_by_modality": {name: 4 for name in orch.DEFAULT_STAGE_ORDER},
            "min_records_per_modality": 1,
            "artifact_token_count": {"image": 4, "video": 4, "audio": 4, "music": 4, "tool": 4, "long_context": 4},
            "max_hash_bytes": 4096,
            "max_media_bytes": 1024 * 1024,
            "min_media_bytes": 1,
            "text_token_limit": 64,
            "target_text_chars": 4000,
        },
        "learning_checks": {"min_loss_points": 2, "min_relative_loss_drop": 0.001},
        "record_contracts": {
            "training_record_required_fields": [
                "record_id",
                "source_id",
                "source_date",
                "modalities",
                "split",
                "quality_score",
                "contamination_status",
                "payload_sha256",
                "artifact_refs",
            ]
        },
    }


def _dataset_profile(root: Path) -> dict:
    return {
        "schema": "omnicoder.dataset_curation_2026.v1",
        "profile_name": "builder_unit_dataset",
        "run_name": "builder_unit_run",
        "source_date": "2026-05-23",
        "trace_inputs": {
            "sources": [{"path": "data/raw/codex_traces_2026", "harness": "codex", "label": "unit_codex"}],
            "patterns": ["*.jsonl"],
        },
        "curation_layers": {"validation_ratio": 0.0, "holdout_ratio": 0.0},
        "builder_2026": {
            "enabled": True,
            "out_dir": "weights/curated_datasets_2026/latest",
            "training_profile": "profiles/training_orchestration_2026.json",
            "agent_memory_cli_export": {"enabled": False},
            "min_quality": 0.30,
            "long_context_min_chars": 1800,
            "coverage_targets": {"min_train_per_modality": 1},
            "supplemental_sources": {
                "media_roots": ["media"],
                "image_roots": ["media/image"],
                "video_roots": ["media/video"],
                "audio_roots": ["media/audio"],
                "music_roots": ["media/music"],
            },
        },
        "dataset_catalog_2026": {
            "agentic_tool_reasoning": [
                {
                    "name": "TOUCAN",
                    "url": "https://openreview.net/forum?id=UgFmrYcLOt",
                    "gate": "license_allowlist_and_benchmark_overlap_scan",
                }
            ]
        },
    }


def _build(tmp_path: Path, monkeypatch) -> dict:
    root = _fixture_root(tmp_path)
    (root / "profiles").mkdir(exist_ok=True)
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_json(root / "profiles" / "dataset_curation_2026.json", _dataset_profile(root))
    monkeypatch.setattr(builder, "repo_root", lambda: root)
    return builder.build_dataset(root / "profiles" / "dataset_curation_2026.json")


def test_curated_dataset_builder_exports_all_modalities(tmp_path, monkeypatch):
    manifest = _build(tmp_path, monkeypatch)
    assert manifest["status"] == "passed"
    for modality in orch.DEFAULT_STAGE_ORDER:
        assert manifest["modalities"][modality]["train"] >= 1
        assert Path(manifest["per_modality_split_jsonl"][modality]["train"]).exists()
    assert Path(manifest["aggregate_jsonl"]["train"]).exists()


def test_curated_dataset_builder_rejects_secret_traces(tmp_path, monkeypatch):
    manifest = _build(tmp_path, monkeypatch)
    train_text = Path(manifest["aggregate_jsonl"]["train"]).read_text(encoding="utf-8")
    assert "super_secret_training_password" not in train_text
    rejected_path = Path(manifest["trace_stats"]["raw_outputs"]["rejected_traces"])
    rejected = rejected_path.read_text(encoding="utf-8")
    assert "secret_redaction" in rejected


def test_curated_dataset_builder_writes_manifests_and_posttraining(tmp_path, monkeypatch):
    manifest = _build(tmp_path, monkeypatch)
    source_inventory = json.loads(Path(manifest["manifests"]["source_inventory"]).read_text(encoding="utf-8"))
    quality_report = json.loads(Path(manifest["manifests"]["quality_report"]).read_text(encoding="utf-8"))
    dataset_card = json.loads(Path(manifest["dataset_card"]).read_text(encoding="utf-8"))
    posttraining = manifest["posttraining_curation_exports"]
    assert source_inventory["sources"]
    assert quality_report["records"] == sum(manifest["records"].values())
    assert dataset_card["dataset_catalog_2026"]["agentic_tool_reasoning"][0]["name"] == "TOUCAN"
    assert Path(posttraining["manifest"]).exists()
    assert posttraining["counts"]["sft"] == manifest["records"]["train"]
    assert posttraining["counts"]["reward"] == manifest["records"]["train"]


def test_curated_trace_long_context_uses_long_context_target_chars(tmp_path):
    long_text = "anchor " * 3000
    plan = {
        "target_text_chars": 512,
        "long_context_target_chars": 12000,
        "long_context_text_token_limit": 12000,
        "artifact_token_count": {"long_context": 4},
    }
    row = builder.curated_trace_to_training_row(
        {"lineage": {"path": "trace.jsonl"}},
        {
            "curated_id": "long-trace",
            "normalized_text": long_text,
            "quality": {"overall": 0.99, "label": "accepted"},
            "contamination": {"status": "clean"},
            "secret_redaction": {"has_secret": False},
            "split_assignment": {"split": "train"},
            "provenance": {"path": "trace.jsonl"},
        },
        plan,
        {"min_quality": 0.3, "min_chars": 8, "long_context_min_chars": 1800},
    )

    assert row is not None
    assert row["modality"] == "long_context"
    assert len(row["target_json"]["content"]) == 12000
    assert row["target_text_token_count"] == 12000


def test_collect_file_rows_long_context_uses_long_context_file_caps(tmp_path):
    root = tmp_path
    long_root = root / "long"
    long_root.mkdir()
    (long_root / "large_context.txt").write_text("L" * 5000, encoding="utf-8")
    plan = {
        "target_text_chars": 512,
        "long_context_target_chars": 4000,
        "long_context_text_token_limit": 4000,
        "long_context_max_text_file_bytes": 10000,
        "artifact_token_count": {"long_context": 4},
        "max_records_per_modality_by_modality": {"long_context": 4},
    }
    profile = {
        "source_date": "2026-05-23",
        "builder_2026": {
            "max_text_file_bytes": 10,
            "supplemental_sources": {"long_context_roots": ["long"]},
        },
    }
    source_inventory: list[dict] = []

    rows = builder.collect_file_rows(profile, root, plan, source_inventory)

    assert rows["long_context"]
    assert len(rows["long_context"][0]["target_json"]["content"]) == 4000
    assert rows["long_context"][0]["target_text_token_count"] == 4000

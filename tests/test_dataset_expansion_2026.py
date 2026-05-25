from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from omnicoder.data_factory import dataset_expansion_2026 as expansion
from omnicoder.training import training_orchestration_2026 as orch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _training_profile(root: Path) -> dict:
    return {
        "profile_name": "unit_training",
        "modalities": {name: {"enabled": True} for name in orch.DEFAULT_STAGE_ORDER},
        "real_sources": {},
        "training_plan": {
            "max_records_per_modality": 8,
            "artifact_token_count": {"image": 4, "video": 4, "audio": 4, "music": 4, "tool": 4, "long_context": 4},
            "text_token_limit": 64,
            "target_text_chars": 512,
        },
        "learning_checks": {"min_loss_points": 2, "min_relative_loss_drop": 0.001},
    }


def test_dataset_expansion_materializes_license_tiered_rows(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(
        root / "data" / "math.jsonl",
        [{"problem": "Solve 2+2.", "answer": "4", "uuid": "m1", "contamination_status": "clean"}],
    )
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "unit_math",
                    "family": "math_reasoning",
                    "target_modality": "text",
                    "local_jsonl": "data/math.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["uuid"]},
                },
                {
                    "name": "unit_eval",
                    "family": "terminal_browser_agents",
                    "target_modality": "tool",
                    "license": "Apache-2.0",
                    "license_tier": "permissive_eval_holdout",
                    "use_policy": "eval_only",
                    "distillation_prompts": [{"instruction": "terminal task", "target": "eval answer"}],
                },
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0})(),
    )
    assert manifest["records"]["train"] == 1
    assert manifest["records"]["eval_holdout"] == 1
    assert manifest["families"]["math_reasoning"] == 1
    train_row = json.loads((root / "weights" / "external" / "jsonl" / "train_all_external.jsonl").read_text().splitlines()[0])
    assert train_row["dataset_name"] == "unit_math"
    assert train_row["training_bucket"] == "train"
    assert train_row["license_tier"] == "permissive"
    assert train_row["contamination_status"] == "clean"


def test_dataset_expansion_splits_conversation_prompt_and_target(tmp_path: Path) -> None:
    plan = _training_profile(tmp_path)["training_plan"]
    entry = {
        "name": "unit_conversation",
        "family": "agentic_tool_reasoning",
        "target_modality": "tool",
        "license": "Apache-2.0",
        "license_tier": "permissive",
        "use_policy": "train",
        "contamination_status": "clean",
        "field_map": {
            "prompt": ["conversations", "conversations.value"],
            "target": ["conversations", "conversations.value"],
            "trajectory": ["conversations"],
        },
    }
    record = {
        "id": "conv-1",
        "conversations": [
            {"from": "human", "value": "Use the terminal to inspect the failing test."},
            {"from": "gpt", "value": "I will inspect the logs, run the targeted test, and patch the failure."},
        ],
    }

    row = expansion.record_to_training_row(entry, record, plan, 0)

    assert row is not None
    assert row["input_json"]["messages"][0]["content"] == "Use the terminal to inspect the failing test."
    assert row["target_json"]["content"] == "I will inspect the logs, run the targeted test, and patch the failure."
    assert row["input_json"]["messages"][0]["content"] != row["target_json"]["content"]
    assert row["trajectory"][0]["from"] == "human"


def test_dataset_expansion_preserves_pairwise_preference_targets(tmp_path: Path) -> None:
    plan = _training_profile(tmp_path)["training_plan"]
    entry = {
        "name": "unit_preference",
        "family": "agentic_tool_reasoning",
        "target_modality": "tool",
        "license": "CC-BY-4.0",
        "license_tier": "attribution_reward_model",
        "use_policy": "reward_only",
        "field_map": {"prompt": ["prompt"], "target": ["response1", "response2", "overall_preference"]},
    }
    record = {
        "id": "pref-1",
        "prompt": "Choose the safer tool plan.",
        "response1": "Call the tool with validated arguments.",
        "response2": "Call the tool with raw user input.",
        "overall_preference": "response1",
    }

    row = expansion.record_to_training_row(entry, record, plan, 0)

    assert row is not None
    target = json.loads(row["target_json"]["content"])
    assert target["response1"] == "Call the tool with validated arguments."
    assert target["response2"] == "Call the tool with raw user input."
    assert target["overall_preference"] == "response1"
    assert row["training_bucket"] == "research_internal"


def test_dataset_expansion_blocks_unknown_contamination_from_train(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(root / "data" / "math.jsonl", [{"problem": "Solve 5+7.", "answer": "12", "uuid": "m1"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "unit_unscanned_math",
                    "family": "math_reasoning",
                    "target_modality": "text",
                    "local_jsonl": "data/math.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["uuid"]},
                }
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)

    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0})(),
    )

    train_path = root / "weights" / "external" / "jsonl" / "train_all_external.jsonl"
    research_rows = [
        json.loads(line)
        for line in (root / "weights" / "external" / "jsonl" / "research_internal_all_external.jsonl").read_text().splitlines()
    ]
    assert manifest["records"].get("train", 0) == 0
    assert train_path.read_text(encoding="utf-8") == ""
    assert research_rows[0]["training_bucket"] == "research_internal"
    assert research_rows[0]["contamination_status"] == "unknown"


def test_dataset_expansion_falls_back_to_distillation_seeds_after_hf_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "seeded_image",
                    "hf_id": "missing/not-real",
                    "family": "image_generation_editing",
                    "target_modality": "image",
                    "license_tier": "permissive_with_teacher_provenance",
                    "use_policy": "research_internal",
                    "distillation_prompts": [{"instruction": "image prompt decomposition", "target": "image rubric"}],
                }
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    monkeypatch.setattr(expansion, "rows_from_huggingface", lambda entry, limit, streaming: ([], {"status": "failed"}))
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": True, "no_streaming": False, "max_records_per_dataset": 0})(),
    )
    assert manifest["records"]["research_internal"] == 1
    assert manifest["modalities"]["image"] == 1
    assert manifest["datasets"][0]["synthetic_seed_only"] is True
    assert manifest["synthetic_seed_families"]["image_generation_editing"] == 1


def test_external_long_context_rows_preserve_large_targets(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    training = _training_profile(root)
    training["training_plan"]["long_context_target_chars"] = 12000
    training["training_plan"]["long_context_text_token_limit"] = 12000
    _write_json(root / "profiles" / "training_orchestration_2026.json", training)
    _write_jsonl(root / "data" / "long_context.jsonl", [{"prompt": "retain anchors", "answer": "Z" * 13000, "id": "lc-1", "contamination_status": "clean"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "unit_long_context",
                    "family": "long_context",
                    "target_modality": "long_context",
                    "local_jsonl": "data/long_context.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["prompt"], "target": ["answer"], "id": ["id"]},
                }
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)

    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0})(),
    )

    row = json.loads((root / "weights" / "external" / "jsonl" / "train_all_external.jsonl").read_text().splitlines()[0])
    assert manifest["modalities"]["long_context"] == 1
    assert len(row["target_json"]["content"]) == 12000
    assert row["target_text_token_count"] == 12000


def test_dataset_expansion_downloads_remote_tsv_rows(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    tsv = (
        "index\tid\tcategory\timage\tquestion\tanswer\tA\tB\tC\tD\tmodel_tools_gt\n"
        "1\tattention_1\tattention\timages/a.jpg\tRead the mirrored sign.\tA\tText\tOther\tNone\tAll\t[\"Crop\", \"Flip\"]\n"
        "2\tcount_1\tcounting\timages/b.jpg\tHow many objects?\tC\t1\t2\t3\t4\t[\"Threshold\", \"Draw Contours\"]\n"
    )
    (root / "data" / "vtc.tsv").parent.mkdir(parents=True, exist_ok=True)
    (root / "data" / "vtc.tsv").write_text(tsv, encoding="utf-8")
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "unit_vtc",
                    "family": "omnimodal_understanding",
                    "target_modality": "image",
                    "remote_files": [{"url": "data/vtc.tsv", "format": "tsv"}],
                    "license": "Apache-2.0",
                    "license_tier": "permissive_eval_holdout",
                    "use_policy": "eval_only",
                    "field_map": {
                        "prompt": ["question", "category"],
                        "target": ["answer", "A", "B", "C", "D", "model_tools_gt"],
                        "media": ["image"],
                        "trajectory": ["model_tools_gt"],
                        "id": ["id"],
                    },
                }
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": True, "no_streaming": False, "max_records_per_dataset": 0})(),
    )

    rows = [
        json.loads(line)
        for line in (root / "weights" / "external" / "jsonl" / "omnimodal_understanding_eval_holdout.jsonl").read_text().splitlines()
    ]
    assert manifest["datasets"][0]["source"] == "remote_files"
    assert manifest["records"]["eval_holdout"] == 2
    assert rows[0]["media_refs"] == ["images/a.jpg"]
    assert rows[0]["trajectory"] == ['["Crop", "Flip"]']


def test_dataset_expansion_reports_required_real_family_minima(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(root / "data" / "math.jsonl", [{"problem": "Solve 1+1.", "answer": "2", "uuid": "m1", "contamination_status": "clean"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "required_real_family_min_records": {
                "math_reasoning": {"bucket": "train", "min_real": 1},
                "coding_agentic": {"bucket": "train", "min_real": 1},
            },
            "datasets": [
                {
                    "name": "unit_math",
                    "family": "math_reasoning",
                    "target_modality": "text",
                    "local_jsonl": "data/math.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["uuid"]},
                },
                {
                    "name": "seeded_code",
                    "family": "coding_agentic",
                    "target_modality": "code",
                    "license": "internal",
                    "license_tier": "internal_seed",
                    "use_policy": "train",
                    "distillation_prompts": [{"instruction": "fix bug", "target": "patch"}],
                },
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0, "enforce_requirements": False})(),
    )

    assert manifest["status"] == "failed_requirements"
    assert manifest["real_families"]["math_reasoning"] == 1
    assert manifest["synthetic_seed_families"]["coding_agentic"] == 1
    assert manifest["records"]["train"] == 1
    assert manifest["records"]["research_internal"] == 1
    report = manifest["requirement_report"]
    assert report["requirements"]["math_reasoning"]["status"] == "passed"
    assert report["requirements"]["coding_agentic"]["status"] == "failed"


def test_dataset_expansion_can_materialize_filtered_registry_wave_without_global_minima(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(root / "data" / "wave_math.jsonl", [{"problem": "Solve 3+5.", "answer": "8", "uuid": "wave-m", "contamination_status": "clean"}])
    _write_jsonl(root / "data" / "old_code.jsonl", [{"prompt": "Patch bug.", "answer": "done", "uuid": "old-c", "contamination_status": "clean"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "required_real_family_min_records": {
                "math_reasoning": {"bucket": "train", "min_real": 1},
                "coding_agentic": {"bucket": "train", "min_real": 1},
            },
            "datasets": [
                {
                    "name": "wave_math",
                    "family": "math_reasoning",
                    "target_modality": "text",
                    "registry_wave": "delta_wave",
                    "local_jsonl": "data/wave_math.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["problem"], "target": ["answer"], "id": ["uuid"]},
                },
                {
                    "name": "old_code",
                    "family": "coding_agentic",
                    "target_modality": "code",
                    "registry_wave": "old_wave",
                    "local_jsonl": "data/old_code.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {"prompt": ["prompt"], "target": ["answer"], "id": ["uuid"]},
                },
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)

    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type(
            "Args",
            (),
            {
                "download": False,
                "no_streaming": False,
                "max_records_per_dataset": 0,
                "enforce_requirements": False,
                "include_wave": ["delta_wave"],
                "include_family": [],
                "include_name": [],
            },
        )(),
    )

    assert manifest["status"] == "passed"
    assert manifest["selection"] | {
        "total_enabled_entries": 2,
        "selected_entries": 1,
        "include_wave": ["delta_wave"],
        "filtered": True,
    } == manifest["selection"]
    assert manifest["requirement_report"]["status"] == "skipped"
    assert manifest["records"]["train"] == 1
    assert manifest["families"] == {"math_reasoning": 1}


def test_dataset_expansion_family_files_are_train_safe_and_bucket_partitioned(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(root / "data" / "tool_train.jsonl", [{"prompt": "Call the weather tool.", "answer": "Use the tool.", "id": "t1", "contamination_status": "clean"}])
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "train_tool",
                    "family": "agentic_tool_reasoning",
                    "target_modality": "tool",
                    "local_jsonl": "data/tool_train.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {
                        "prompt": ["prompt"],
                        "target": ["answer"],
                        "id": ["id"],
                        "tool_calls": ["tool_calls"],
                        "media_refs": ["media"],
                    },
                },
                {
                    "name": "eval_tool",
                    "family": "agentic_tool_reasoning",
                    "target_modality": "tool",
                    "license": "Apache-2.0",
                    "license_tier": "permissive_eval_holdout",
                    "use_policy": "eval_only",
                    "distillation_prompts": [{"instruction": "held out tool task", "target": "held out answer"}],
                },
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)

    manifest = expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0, "enforce_requirements": False})(),
    )

    train_family = [json.loads(line) for line in (root / "weights" / "external" / "jsonl" / "agentic_tool_reasoning.jsonl").read_text().splitlines()]
    eval_family = [json.loads(line) for line in (root / "weights" / "external" / "jsonl" / "agentic_tool_reasoning_eval_holdout.jsonl").read_text().splitlines()]
    all_family = [json.loads(line) for line in (root / "weights" / "external" / "jsonl" / "agentic_tool_reasoning_all.jsonl").read_text().splitlines()]

    assert manifest["family_paths"]["agentic_tool_reasoning"]["train"].endswith("agentic_tool_reasoning.jsonl")
    assert len(train_family) == 1
    assert {row["training_bucket"] for row in train_family} == {"train"}
    assert len(eval_family) == 1
    assert {row["training_bucket"] for row in eval_family} == {"eval_holdout"}
    assert len(all_family) == 2


def test_dataset_expansion_preserves_structured_tool_media_and_declared_modality(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    _write_json(root / "profiles" / "training_orchestration_2026.json", _training_profile(root))
    _write_jsonl(
        root / "data" / "multi.jsonl",
        [
            {
                "instruction": "Edit this image and explain the patch.",
                "response": "Preserve the product logo and replace the background.",
                "media": [{"type": "image", "path": "sample.png"}],
                "calls": [{"tool": "image_edit", "arguments": {"mask": "background"}}],
                "results": [{"status": "ok"}],
                "labels": [{"check": "preserve_identity", "label": "pass"}],
                "score": 0.87,
                "contamination_status": "clean",
            }
        ],
    )
    profile = {
        "external_dataset_registry_2026": {
            "training_profile": "profiles/training_orchestration_2026.json",
            "datasets": [
                {
                    "name": "multi_tool_media",
                    "family": "image_generation_editing",
                    "target_modality": "multimodal",
                    "local_jsonl": "data/multi.jsonl",
                    "license": "Apache-2.0",
                    "license_tier": "permissive",
                    "use_policy": "train",
                    "field_map": {
                        "prompt": ["instruction"],
                        "target": ["response"],
                        "media": ["media"],
                        "tool_calls": ["calls"],
                        "tool_results": ["results"],
                        "verifier_labels": ["labels"],
                        "reward": ["score"],
                    },
                }
            ],
        }
    }
    _write_json(root / "profiles" / "dataset_curation_2026.json", profile)
    monkeypatch.setattr(expansion, "repo_root", lambda: root)
    expansion.build_expansion(
        root / "profiles" / "dataset_curation_2026.json",
        root / "weights" / "external",
        type("Args", (), {"download": False, "no_streaming": False, "max_records_per_dataset": 0, "enforce_requirements": False})(),
    )

    row = json.loads((root / "weights" / "external" / "jsonl" / "image_generation_editing.jsonl").read_text().splitlines()[0])
    assert row["declared_target_modality"] == "multimodal"
    assert row["modality"] == "image"
    assert row["media_refs"][0]["path"] == "sample.png"
    assert row["tool_calls"][0]["tool"] == "image_edit"
    assert row["tool_results"][0]["status"] == "ok"
    assert row["verifier_labels"][0]["check"] == "preserve_identity"
    assert row["reward"] == 0.87
    assert "tool" in row["domains"]


def test_dataset_expansion_summarizes_inline_media_payloads() -> None:
    plan = _training_profile(Path("."))["training_plan"]
    entry = {
        "name": "audio_payload_guard",
        "family": "audio_music_speech",
        "target_modality": "audio",
        "license": "Apache-2.0",
        "license_tier": "permissive",
        "use_policy": "train",
        "field_map": {"prompt": ["prompt"], "target": ["answer"], "media": ["audio"], "id": ["id"]},
    }
    record = {
        "id": "clip-1",
        "prompt": "Transcribe the speaker and preserve prosody notes.",
        "answer": "The speaker says hello with a rising tone.",
        "audio": {"path": "clip-1.wav", "sampling_rate": 48000, "array": [0.01] * 10000},
        "contamination_status": "clean",
    }

    row = expansion.record_to_training_row(entry, record, plan, 1)

    assert row is not None
    assert len(json.dumps(row)) < 20000
    assert row["media_refs"][0]["path"] == "clip-1.wav"
    assert row["media_refs"][0]["array_summary"]["list_items"] == 10000
    assert row["media_refs"][0]["array_summary"]["truncated_items"] == 9992


def test_huggingface_iteration_errors_do_not_abort_expansion(monkeypatch) -> None:
    class BrokenDataset:
        def __iter__(self):
            raise ImportError("missing optional decoder")

    def fake_load_dataset(*args, **kwargs):
        return BrokenDataset()

    module = types.ModuleType("datasets")
    module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    rows, status = expansion.rows_from_huggingface(
        {"hf_id": "unit/broken", "splits": ["train"]},
        limit=4,
        streaming=True,
    )

    assert rows == []
    assert status["status"] == "failed"
    assert "iteration failed" in status["errors"][0]


def test_huggingface_media_columns_are_cast_to_metadata_only(monkeypatch) -> None:
    cast_calls: list[tuple[str, object]] = []

    class FakeFeature:
        pass

    FakeFeature.__name__ = "Audio"
    audio_feature = FakeFeature()

    class FakeVideo:
        pass

    FakeVideo.__name__ = "Video"
    video_feature = FakeVideo()

    class FakeImage:
        pass

    FakeImage.__name__ = "Image"
    image_feature = FakeImage()

    class FakeDataset:
        features = {"reference_audio": audio_feature, "video": video_feature, "image": image_feature}

        def cast_column(self, column, feature):
            cast_calls.append((column, feature))
            return self

        def __iter__(self):
            yield {
                "prompt": "speak this",
                "target": "spoken",
                "reference_audio": {"path": "ref.webm"},
                "video": {"path": "clip.mp4"},
                "image": {"path": "frame.png"},
            }

    class Audio:
        def __init__(self, decode=False):
            self.decode = decode

    class Video:
        def __init__(self, decode=False):
            self.decode = decode

    class Image:
        def __init__(self, decode=False):
            self.decode = decode

    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: FakeDataset()
    module.Audio = Audio
    module.Video = Video
    module.Image = Image
    monkeypatch.setitem(sys.modules, "datasets", module)

    rows, status = expansion.rows_from_huggingface(
        {"hf_id": "unit/media", "splits": ["train"]},
        limit=1,
        streaming=True,
    )

    assert len(rows) == 1
    assert rows[0]["reference_audio"] == {"path": "ref.webm"}
    assert rows[0]["video"] == {"path": "clip.mp4"}
    assert rows[0]["image"] == {"path": "frame.png"}
    assert [name for name, _ in cast_calls] == ["reference_audio", "video", "image"]
    assert all(getattr(feature, "decode") is False for _, feature in cast_calls)
    assert status["status"] == "ok"


def test_huggingface_registry_options_are_passed_without_token_leak(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return [{"prompt": "p", "target": "t"}]

    module = types.ModuleType("datasets")
    module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setenv("HF_UNIT_TOKEN", "secret-token")

    rows, status = expansion.rows_from_huggingface(
        {
            "hf_id": "unit/options",
            "config": "cfg",
            "revision": "abc123",
            "data_files": {"train": "data/*.jsonl"},
            "verification_mode": "no_checks",
            "trust_remote_code": False,
            "token_env": "HF_UNIT_TOKEN",
            "splits": ["train"],
        },
        limit=1,
        streaming=True,
    )

    assert len(rows) == 1
    assert calls[0]["args"][:2] == ("unit/options", "cfg")
    assert calls[0]["kwargs"]["revision"] == "abc123"
    assert calls[0]["kwargs"]["data_files"] == {"train": "data/*.jsonl"}
    assert calls[0]["kwargs"]["verification_mode"] == "no_checks"
    assert calls[0]["kwargs"]["trust_remote_code"] is False
    assert calls[0]["kwargs"]["token"] == "secret-token"
    assert status["token_env"] == "HF_UNIT_TOKEN"
    assert status["token_used"] is True
    assert "secret-token" not in json.dumps(status)


def test_repo_dataset_registry_covers_new_agentic_and_multimodal_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names))
    for key in ("hf_id", "url"):
        values = [entry[key] for entry in entries if entry.get(key)]
        assert len(values) == len(set(values))
    by_name = {entry["name"]: entry for entry in entries}

    for name in [
        "GUI-360",
        "AgentNet",
        "Computer Use Large",
        "Synthetic Computers at Scale",
        "VideoCUA",
        "ExeVR-53k",
        "AgentSynth",
        "Computer Agent Arena",
        "Smol2Operator Aguvis Stage 1",
        "Smol2Operator Aguvis Stage 2",
        "TAU2-Bench Data",
        "AReaL Tau2 Data",
        "Tau2 Verified Airline Code Agents",
        "APEX-Agents",
        "APEX-SWE",
        "WildClawBench",
        "ClawBench",
        "BFCL Function Calling Leaderboard",
        "ComplexFuncBench",
        "R2E-Gym V1",
        "R2EGym SFT Trajectories",
        "OpenHands CodeScout Training Rollouts",
        "AIDev Agent PR Corpus",
        "SWE-CI",
        "Fixbench-RTL",
        "SWE-Synth",
        "R-HORIZON Training Data",
        "Reasoning Core Formal Reasoning Env",
        "UniRRM-RL",
        "NVIDIA Nemotron-Math-Proofs-v1",
        "UltraData-Math",
        "GLM-5.1 Reasoning 1M Cleaned",
        "MathVision 2026",
        "OpenThoughts2-1M",
        "DeepMath-103K",
        "AI-MO NuminaMath 1.5",
        "Polaris Nemotron Easy Math Verifiable",
        "Polaris Nemotron Medium Math Verifiable",
        "Korean NuminaMath Verifiable 540K",
        "RLVR Eurus 2 Math Fixed",
        "DeepCoder-Preview-Dataset",
        "SWE-Dev Train",
        "Dorothy SWE-Dev",
        "DeepSWE Agent Kimi K2 Trajectories 2.8K",
        "SWE-Factory-Gym",
        "SWE-Next",
        "SWE-Next SFT Trajectories",
        "SWE-Swiss SFT Repair 4K",
        "SWE-Swiss Repair RL 12K",
        "SWE-Universe Repaired Bug Pilot Trajectories",
        "NVIDIA Nemotron-RL Agentic SWE Pivot",
        "Scale-SWE",
        "Nebius SWE-rebench OpenHands Trajectories",
        "NVIDIA OpenCodeReasoning-2",
        "MCP-Atlas",
        "NVIDIA Nemotron-RL Agentic Conversational Tool Use Pivot v1",
        "WebAgent-R1 Distill",
        "WebShepherd PRM Collection",
        "WebExplorer-QA",
        "DeepDive",
        "WebArena Infinity Trajectories",
        "BrowserAgent SeedData",
        "BrowserAgent Data",
        "Web Agent Graph Dataset",
        "WebChain",
        "WebArena Pro Task Intents and Rubrics",
        "Deceptive Web Execution-Time Warnings",
        "CognitiveKernel Pro SFT",
        "NVIDIA Nemotron-Terminal-Corpus",
        "TerminalWorld",
        "OSWorld 2 Ubuntu Trajectories",
        "OSWorld Control Trajectories",
        "Magic-RICH",
        "macOS GUI Agent Train",
        "Multi-Docker-Eval",
        "SWE-bench Pro",
        "SWE-bench Pro ABS",
        "SWE-QA-Pro-Bench",
        "SWE-bench Multilingual",
        "CodeElo",
        "ICPC-Eval",
        "HLE-Verified",
        "MathArena AIME 2026 Holdout",
        "Multi-SWE-bench",
        "SWE-Bench++",
        "EVMbench",
        "SWE-bench Multimodal",
        "SWE-Lancer",
        "SWE-PolyBench",
        "SWE-bench Live MultiLang",
        "SWE-bench Live Windows",
        "JetBrains SWE-bench Agent Trajectories",
        "Hermes Function Calling V1",
        "Tool Use Multiturn Reasoning",
        "OmniEdit-Filtered-1.2M",
        "OmniGen2 X2I2",
        "Common Voice 21.0",
        "ACE-Step Songs",
        "Visual-CoT",
        "MultiEdit",
        "NVIDIA Granary",
        "AR-Omni-Instruct-v0.1",
        "DIVER Training OpenR1 Math 46k",
        "VLM-CapCurriculum TextReasoning",
        "Frugal Thinking RL Data",
        "FleetAI Tool-Use Difficult Envs",
        "FleetAI Thinking Tools Difficult Envs",
        "ATBench Agent Trajectory Safety",
        "InternVL-U ScaleEdit-12M",
        "BLIP3o NEXT Edit Ensemble",
        "Complex Long Video Understanding Reasoning",
        "NVIDIA Nemotron-SFT-SWE-v2",
        "NVIDIA SWE-Hero OpenHands Trajectories",
        "NVIDIA Nemotron-SFT Competitive Programming v2",
        "OpenResearcher Dataset",
        "OpenResearcher Corpus",
        "WebWalkerQA",
        "Text-to-Terminal v2 Tool Reasoning Cleaned",
        "NuminaMath 1.5 RL Verifiable",
        "DeepSearch-2510",
        "SWE-bench-Live OS-bench",
        "ContextBench TraceBench",
        "NJU CodeTraceBench",
        "Terminal-Bench 2.0 Trajectories",
        "OmniAgent MAgenIT Data",
        "NVIDIA Nemotron Image Training v3",
        "PRISM VLM RL Dataset",
        "RLFR Dataset VLM",
        "Innovator VL RL 172K",
        "NVIDIA AudioSkills XL",
        "Pico-Banana-400K",
        "Pico-Banana-400K Apple Research Reference",
        "FineVision",
        "FineVisionMax",
        "GPT-Image-Edit-1.5M",
        "NHR-Edit",
        "CrispEdit-2M",
        "BAGEL-World",
        "EditReward-Data",
        "UniREdit Data 100K",
        "HPDv3",
        "Rapidata 700K Image Preferences",
        "BLIP3o Pretrain Long Caption",
        "UniWorld-V1",
        "ImgEdit 1.2M",
        "VideoGen-RewardBench",
        "Rapidata Text-to-Video Human Preferences Veo3",
        "Rapidata Image-to-Video Human Preference Seedance",
        "JavisInst-Omni",
        "JavisVerse AV FineTune",
        "TTSDS Listening Test",
        "SAM Audio LLM Data",
        "Prompt2MusicBench",
        "OpenMMReasoner RL 74K",
        "Rapidata Open Image Preferences v1 More Results",
        "Text-to-Image DPO Human Preferences Full",
        "VIBE Benchmark",
        "CompBench Complex Editing",
        "ImagenWorld",
        "DreamOmni2Bench",
        "DeepVision-103K",
        "WorldSpeech",
        "NonverbalTTS",
        "Captioned AI Music Snippets",
        "Cambrian-P-Data",
        "MMMU Pro",
        "LongMemEval-V2",
        "AVGen-Bench",
        "TAG-Bench-Video",
        "VGenST-Bench",
        "VBench 2.0",
        "Video-Bench",
        "PARADE_audio",
        "WildSpeech-Bench",
        "AudioMC",
        "AV-SpeakerBench",
        "HLE",
        "Video-MME",
        "Video-MME-v2",
        "LVBench",
        "LVOmniBench",
        "OmniContext",
        "VTC-Bench Visual Tool Chain",
        "FiVE Fine-Grained Video Editing Benchmark",
        "OmniEdit-Bench",
        "MAVERIX",
        "JointAVBench",
        "OpenAudioBench",
        "Ming Freeform Audio Edit Benchmark",
        "Common Voice 22.0 Mirror English",
        "PDMX Multi-Instrument Synthesized",
        "Song Describer",
        "OmniDoc-TokenBench",
        "ChartQAPro",
        "OmniDoc OCR Correction Bench",
        "OmniCorpus CC",
        "OmniCorpus YT",
        "OmniGUI",
        "PhyWorldBench",
        "Image-to-Video Human Preferences Large",
        "MusicEval",
        "MCIF Crosslingual Multimodal Instruction Following",
        "Multimodal RewardBench 2",
        "VoiceAgentBench",
        "Toolathlon Trajectories",
        "Agentic Chain-of-Thought Coding SFT",
        "Plan-RewardBench",
        "R2E-Gym Verifier Trajectories",
        "R2E-Gym TestingAgent SFT Trajectories",
        "s1K-1.1 Reasoning",
        "HMMT February 2025",
        "MMVU",
        "OCRBench v2",
        "MM-IQ",
        "Real5-OmniDocBench",
        "JoyAI Image OpenSpatial",
        "JoyAI Image SpatialEdit",
        "JoyAI Image SpatialEdit Bench",
        "X-LANCE WikiHow Taskset",
        "X-LANCE WebSRC v1.0",
        "DeepGen 1.0 Dataset Card",
        "Salesforce APIGen-MT-5k",
        "PrimeIntellect SYNTHETIC-1 SFT",
        "PrimeIntellect SYNTHETIC-1 Preference",
        "Alibaba WebShaper",
    ]:
        assert name in by_name

    assert by_name["GUI-360"]["use_policy"] == "train"
    assert by_name["AgentNet"]["use_policy"] == "train"
    assert by_name["Computer Use Large"]["target_modality"] == "video"
    assert by_name["VideoCUA"]["use_policy"] == "train"
    assert by_name["Smol2Operator Aguvis Stage 2"]["use_policy"] == "research_internal"
    assert by_name["TAU2-Bench Data"]["use_policy"] == "eval_only"
    assert by_name["AReaL Tau2 Data"]["use_policy"] == "train"
    assert by_name["Tau2 Verified Airline Code Agents"]["use_policy"] == "train"
    assert by_name["APEX-Agents"]["use_policy"] == "eval_only"
    assert by_name["WildClawBench"]["use_policy"] == "eval_only"
    assert by_name["BFCL Function Calling Leaderboard"]["use_policy"] == "eval_only"
    assert by_name["R2E-Gym V1"]["use_policy"] == "research_internal"
    assert by_name["OpenHands CodeScout Training Rollouts"]["use_policy"] == "research_internal"
    assert by_name["Fixbench-RTL"]["use_policy"] == "train"
    assert by_name["R-HORIZON Training Data"]["use_policy"] == "train"
    assert by_name["Reasoning Core Formal Reasoning Env"]["use_policy"] == "train"
    assert by_name["UniRRM-RL"]["use_policy"] == "train"
    assert by_name["NVIDIA Nemotron-Math-Proofs-v1"]["use_policy"] == "research_internal"
    assert by_name["UltraData-Math"]["use_policy"] == "train"
    assert by_name["GLM-5.1 Reasoning 1M Cleaned"]["use_policy"] == "train"
    assert by_name["MathVision 2026"]["target_modality"] == "image"
    assert by_name["NVIDIA Nemotron-Terminal-Corpus"]["use_policy"] == "train"
    assert by_name["DeepMath-103K"]["use_policy"] == "train"
    assert by_name["AI-MO NuminaMath 1.5"]["use_policy"] == "train"
    assert by_name["Polaris Nemotron Medium Math Verifiable"]["use_policy"] == "train"
    assert by_name["Korean NuminaMath Verifiable 540K"]["use_policy"] == "train"
    assert by_name["RLVR Eurus 2 Math Fixed"]["use_policy"] == "research_internal"
    assert by_name["SWE-Dev Train"]["use_policy"] == "train"
    assert by_name["Dorothy SWE-Dev"]["use_policy"] == "train"
    assert by_name["DeepSWE Agent Kimi K2 Trajectories 2.8K"]["use_policy"] == "train"
    assert by_name["SWE-Factory-Gym"]["use_policy"] == "research_internal"
    assert by_name["SWE-Next"]["use_policy"] == "train"
    assert by_name["SWE-Next SFT Trajectories"]["use_policy"] == "train"
    assert by_name["SWE-Swiss Repair RL 12K"]["use_policy"] == "train"
    assert by_name["SWE-Universe Repaired Bug Pilot Trajectories"]["use_policy"] == "research_internal"
    assert by_name["MCP-Atlas"]["use_policy"] == "train"
    assert by_name["NVIDIA Nemotron-RL Agentic Conversational Tool Use Pivot v1"]["use_policy"] == "train"
    assert by_name["WebAgent-R1 Distill"]["use_policy"] == "train"
    assert by_name["WebShepherd PRM Collection"]["use_policy"] == "research_internal"
    assert by_name["WebExplorer-QA"]["use_policy"] == "train"
    assert by_name["DeepDive"]["use_policy"] == "train"
    assert by_name["WebArena Infinity Trajectories"]["use_policy"] == "train"
    assert by_name["BrowserAgent SeedData"]["use_policy"] == "train"
    assert by_name["BrowserAgent Data"]["use_policy"] == "research_internal"
    assert by_name["Web Agent Graph Dataset"]["use_policy"] == "train"
    assert by_name["WebChain"]["use_policy"] == "research_internal"
    assert by_name["WebArena Pro Task Intents and Rubrics"]["use_policy"] == "eval_only"
    assert by_name["Deceptive Web Execution-Time Warnings"]["use_policy"] == "eval_only"
    assert by_name["CognitiveKernel Pro SFT"]["use_policy"] == "research_internal"
    assert by_name["TerminalWorld"]["use_policy"] == "eval_only"
    assert by_name["OSWorld 2 Ubuntu Trajectories"]["use_policy"] == "research_internal"
    assert by_name["OSWorld Control Trajectories"]["use_policy"] == "eval_only"
    assert by_name["Magic-RICH"]["use_policy"] == "eval_only"
    assert by_name["Multi-Docker-Eval"]["use_policy"] == "eval_only"
    assert by_name["SWE-bench Pro"]["use_policy"] == "eval_only"
    assert by_name["SWE-bench Pro ABS"]["use_policy"] == "eval_only"
    assert by_name["SWE-QA-Pro-Bench"]["use_policy"] == "eval_only"
    assert by_name["SWE-bench Multilingual"]["use_policy"] == "eval_only"
    assert by_name["CodeElo"]["use_policy"] == "eval_only"
    assert by_name["ICPC-Eval"]["use_policy"] == "eval_only"
    assert by_name["HLE-Verified"]["use_policy"] == "eval_only"
    assert by_name["MathArena AIME 2026 Holdout"]["use_policy"] == "eval_only"
    assert by_name["Multi-SWE-bench"]["use_policy"] == "eval_only"
    assert by_name["SWE-Lancer"]["use_policy"] == "eval_only"
    assert by_name["SWE-PolyBench"]["use_policy"] == "eval_only"
    assert by_name["JetBrains SWE-bench Agent Trajectories"]["use_policy"] == "research_internal"
    assert by_name["Scale-SWE"]["use_policy"] == "train"
    assert by_name["SWE-Compass"]["use_policy"] == "benchmark_holdout"
    assert by_name["CoderForge-Preview"]["use_policy"] == "research_internal"
    assert by_name["OmniEdit-Filtered-1.2M"]["use_policy"] == "train"
    assert by_name["AR-Omni-Instruct-v0.1"]["use_policy"] == "research_internal"
    assert by_name["Open-MM-RL"]["use_policy"] == "train"
    assert by_name["RLFR Dataset VLM"]["use_policy"] == "train"
    assert by_name["ATBench Agent Trajectory Safety"]["use_policy"] == "eval_only"
    assert by_name["InternVL-U ScaleEdit-12M"]["use_policy"] == "train"
    assert by_name["GPT-Image-Edit-1.5M"]["use_policy"] == "train"
    assert by_name["NHR-Edit"]["use_policy"] == "train"
    assert by_name["CrispEdit-2M"]["use_policy"] == "train"
    assert by_name["BAGEL-World"]["use_policy"] == "train"
    assert by_name["ImgEdit 1.2M"]["hf_id"] == "sysuyy/ImgEdit"
    assert by_name["ImgEdit 1.2M"]["use_policy"] == "train"
    assert by_name["EditReward-Data"]["use_policy"] == "research_internal"
    assert by_name["UniREdit Data 100K"]["use_policy"] == "train"
    assert by_name["HPDv3"]["use_policy"] == "train"
    assert by_name["Rapidata 700K Image Preferences"]["use_policy"] == "train"
    assert by_name["BLIP3o Pretrain Long Caption"]["use_policy"] == "train"
    assert by_name["UniWorld-V1"]["use_policy"] == "train"
    assert by_name["VideoGen-RewardBench"]["use_policy"] == "eval_only"
    assert by_name["Rapidata Text-to-Video Human Preferences Veo3"]["use_policy"] == "research_internal"
    assert by_name["JavisInst-Omni"]["use_policy"] == "train"
    assert by_name["JavisVerse AV FineTune"]["use_policy"] == "train"
    assert by_name["TTSDS Listening Test"]["use_policy"] == "train"
    assert by_name["SAM Audio LLM Data"]["use_policy"] == "research_internal"
    assert by_name["Prompt2MusicBench"]["use_policy"] == "eval_only"
    assert by_name["OpenMMReasoner RL 74K"]["use_policy"] == "research_internal"
    assert by_name["Rapidata Open Image Preferences v1 More Results"]["use_policy"] == "train"
    assert by_name["Text-to-Image DPO Human Preferences Full"]["use_policy"] == "research_internal"
    assert by_name["ImagenWorld"]["use_policy"] == "eval_only"
    assert by_name["DreamOmni2Bench"]["use_policy"] == "eval_only"
    assert by_name["DeepVision-103K"]["use_policy"] == "research_internal"
    assert by_name["NonverbalTTS"]["use_policy"] == "research_internal"
    assert by_name["Captioned AI Music Snippets"]["use_policy"] == "research_internal"
    assert by_name["Cambrian-P-Data"]["use_policy"] == "train"
    assert by_name["FineVision"]["use_policy"] == "research_internal"
    assert by_name["FineVision"]["config"] == "DoclingMatix"
    assert by_name["WorldSpeech"]["use_policy"] == "research_internal"
    assert by_name["AVGen-Bench"]["use_policy"] == "eval_only"
    assert by_name["MMMU Pro"]["use_policy"] == "eval_only"
    assert by_name["LongMemEval-V2"]["use_policy"] == "eval_only"
    assert by_name["Video-MME-v2"]["use_policy"] == "eval_only"
    assert by_name["LVOmniBench"]["use_policy"] == "eval_only"
    assert by_name["OmniContext"]["use_policy"] == "train"
    assert by_name["VTC-Bench Visual Tool Chain"]["use_policy"] == "eval_only"
    assert by_name["VTC-Bench Visual Tool Chain"]["remote_files"][0]["format"] == "tsv"
    assert by_name["FiVE Fine-Grained Video Editing Benchmark"]["use_policy"] == "eval_only"
    assert by_name["OmniEdit-Bench"]["use_policy"] == "eval_only"
    assert by_name["MAVERIX"]["use_policy"] == "eval_only"
    assert by_name["JointAVBench"]["use_policy"] == "eval_only"
    assert by_name["OpenAudioBench"]["use_policy"] == "eval_only"
    assert by_name["Ming Freeform Audio Edit Benchmark"]["use_policy"] == "eval_only"
    assert by_name["Common Voice 22.0 Mirror English"]["use_policy"] == "train"
    assert by_name["PDMX Multi-Instrument Synthesized"]["use_policy"] == "train"
    assert by_name["Song Describer"]["use_policy"] == "train"
    assert by_name["OmniDoc-TokenBench"]["use_policy"] == "eval_only"
    assert by_name["ChartQAPro"]["use_policy"] == "eval_only"
    assert by_name["OmniDoc OCR Correction Bench"]["use_policy"] == "eval_only"
    assert by_name["OmniCorpus CC"]["use_policy"] == "train"
    assert by_name["OmniCorpus YT"]["use_policy"] == "train"
    assert by_name["OmniGUI"]["use_policy"] == "research_internal"
    assert by_name["Image-to-Video Human Preferences Large"]["use_policy"] == "research_internal"
    assert by_name["AudioMC"]["use_policy"] == "eval_only"
    assert by_name["HLE"]["use_policy"] == "eval_only"
    assert by_name["FineVision"]["license_tier"] == "source_license_varies"
    assert by_name["FineVisionMax"]["license_tier"] == "source_license_varies"
    assert by_name["WorldSpeech"]["license_tier"] == "non_commercial"
    assert by_name["HLE"]["license_tier"] == "benchmark_holdout_review"
    assert by_name["HLE-Verified"]["license_tier"] == "benchmark_holdout_review"
    assert by_name["TerminalWorld"]["license_tier"] == "non_commercial_eval_holdout"
    assert by_name["CodeElo"]["license_tier"] == "permissive_eval_holdout"
    assert by_name["MMMU Pro"]["license_tier"] == "permissive_eval_holdout"
    assert by_name["LVOmniBench"]["license_tier"] == "benchmark_holdout_review"
    assert by_name["OmniContext"]["license_tier"] == "permissive"
    assert by_name["VTC-Bench Visual Tool Chain"]["license_tier"] == "permissive_eval_holdout"
    assert by_name["OpenAudioBench"]["license_tier"] == "unknown_eval_holdout"
    assert by_name["OmniGUI"]["license_tier"] == "non_commercial_sharealike"
    assert by_name["JointAVBench"]["license_tier"] == "sharealike_eval_holdout"
    assert by_name["LongMemEval-V2"]["family"] == "long_context"
    assert by_name["AVGen-Bench"]["target_modality"] == "video"
    assert by_name["Video-MME-v2"]["target_modality"] == "video"
    assert by_name["PARADE_audio"]["target_modality"] == "audio"
    assert by_name["OpenGPT-4o-Image"]["use_policy"] == "train"
    assert by_name["ShareGPT-4o-Image"]["use_policy"] == "train"
    assert by_name["NVIDIA Nemotron-SFT-SWE-v2"]["use_policy"] == "train"
    assert by_name["OpenResearcher Dataset"]["use_policy"] == "research_internal"
    assert by_name["WebWalkerQA"]["use_policy"] == "eval_only"
    assert by_name["MCIF Crosslingual Multimodal Instruction Following"]["use_policy"] == "train"
    assert by_name["Terminal-Bench 2.0 Trajectories"]["use_policy"] == "research_internal"
    assert by_name["Video-MME"]["use_policy"] == "eval_only"
    assert by_name["Multimodal RewardBench 2"]["use_policy"] == "eval_only"
    assert by_name["Multimodal RewardBench 2"]["hf_id"] == "rl-research/multimodal-rewardbench-2"
    assert by_name["Salesforce APIGen-MT-5k"]["use_policy"] == "research_internal"
    assert by_name["Salesforce APIGen-MT-5k"]["config"] == "dataset"
    assert by_name["PrimeIntellect SYNTHETIC-1 SFT"]["use_policy"] == "train"
    assert by_name["PrimeIntellect SYNTHETIC-1 Preference"]["use_policy"] == "train"
    assert by_name["Alibaba WebShaper"]["use_policy"] == "research_internal"

    third_wave_policy = {
        "NVIDIA ToolScale": "research_internal",
        "NVIDIA When2Call": "train",
        "NVIDIA Nemotron Agentic v1": "train",
        "NVIDIA Nemotron-RL Agentic Function Calling Pivot v1": "train",
        "NVIDIA Nemotron-RL Instruction Following": "train",
        "NVIDIA Nemotron Cascade 2 SFT Data": "research_internal",
        "NVIDIA Nemotron Cascade 2 RL Data": "train",
        "NVIDIA Nemotron RL Super Training Blends": "train",
        "NVIDIA Nemotron Cascade RL SWE": "train",
        "NVIDIA Nemotron Cascade RL RLHF": "train",
        "NVIDIA Nemotron 3 Nano RL Training Blend": "train",
        "NVIDIA Nemotron RL Knowledge MCQA": "research_internal",
        "NVIDIA Nemotron RL Competitive Coding": "research_internal",
        "NVIDIA MMOU": "eval_only",
        "NVIDIA QCalEval": "eval_only",
        "NVIDIA SAGE-10K": "train",
        "NVIDIA NitroGen": "research_internal",
        "Ego-R1 Data": "train",
        "LongVideo-Reason": "research_internal",
        "Open-R1 Video 4K": "research_internal",
        "BLIP3o Pretrain Short Caption": "train",
        "BLIP3o 60K Instruction Data": "train",
        "Rapidata OpenAI 4o T2I Human Preference": "research_internal",
        "Rapidata Imagen4 T2I Human Preference": "research_internal",
        "Rapidata Seedream 3 T2I Human Preference": "research_internal",
        "Rapidata Flux 2 Pro T2I Human Preference": "research_internal",
        "Rapidata Text-to-Video Human Preferences Sora 2": "research_internal",
        "Rapidata Text-to-Video Human Preferences Genmo Mochi 1": "train",
        "Rapidata Text-to-Video Human Preferences Seedance 1 Pro": "train",
        "Text-to-Video Motion Preference v2 Large": "research_internal",
        "NVIDIA HiFiTTS 2": "train",
        "Multilingual Synthetic TTS Qwen3": "research_internal",
        "AudioCoT": "research_internal",
        "SpeechJudge Data": "research_internal",
        "NVIDIA LongAudio": "research_internal",
        "NVIDIA AF-Think": "research_internal",
        "NVIDIA AF-Chat": "research_internal",
        "NVIDIA MF-Skills": "research_internal",
        "Prompt2MusicLibrary": "research_internal",
        "Terminal-Bench 2 Trajectories HF": "research_internal",
        "Thoughtworks Agentic Coding Trajectories": "research_internal",
        "Cleaned Toucan Tool Use 333K": "train",
        "Cleaned Hermes Reasoning Tool Use": "train",
        "Cleaned Memory Agent SFT 408K": "train",
        "Cleaned ToolMind Web QA Tool Use": "train",
        "Qwen3.5 Toolcalling v2": "train",
        "Browser Agent Phase1 SFT Reasoning Action": "research_internal",
        "Markov Computer Use OSWorld": "research_internal",
        "PrimeIntellect Verifiable Coding Problems": "research_internal",
        "PrimeIntellect Verifiable Math Problems": "research_internal",
        "Math-RLVR 773K": "train",
        "High Quality Verifiable Math 156K": "research_internal",
        "SWE-Agent LM 32B R2E-Gym Trajectories": "research_internal",
        "Precise Object-Level Image Editing Benchmark": "eval_only",
        "ESCHER OmniEdit": "research_internal",
        "ESCHER Human Edit": "research_internal",
        "IERv2 Subset SeedEdit": "research_internal",
        "Step1X Edit v1.2 VIBE Benchmark": "eval_only",
        "Rapidata Image-to-Video Human Preference Hailuo 02 Marey": "train",
        "Generic Instructional Video Editing Challenge": "research_internal",
        "VideoRewardBench": "eval_only",
        "VideoVista 2": "eval_only",
        "VideoVista CoTs": "train",
        "WorldSense Audio Video Subtitle Reasoning": "eval_only",
        "MMAU Audio Reasoning Benchmark": "eval_only",
        "AudioSet Audio Instructions": "research_internal",
        "Zeroshot Audio Classification Instructions": "research_internal",
        "Audio Adversarial Instructions": "eval_only",
        "Mixed Speech Instruction Ichigo Tokens vi-en": "research_internal",
        "Anonymous StoryBench": "eval_only",
        "Omni Benchmark": "eval_only",
        "Multimodal RewardBench v1": "eval_only",
    }
    for name, policy in third_wave_policy.items():
        assert by_name[name]["use_policy"] == policy
    assert by_name["NVIDIA ToolScale"]["hf_id"] == "nvidia/ToolScale"
    assert by_name["Math-RLVR 773K"]["license_tier"] == "permissive"
    assert by_name["NVIDIA LongAudio"]["license_tier"] == "non_commercial_review"
    assert by_name["Rapidata Image-to-Video Human Preference Hailuo 02 Marey"]["target_modality"] == "video"
    fourth_wave_policy = {
        "Toolathlon Trajectories": "research_internal",
        "Agentic Chain-of-Thought Coding SFT": "train",
        "Plan-RewardBench": "research_internal",
        "R2E-Gym Verifier Trajectories": "research_internal",
        "R2E-Gym TestingAgent SFT Trajectories": "research_internal",
        "s1K-1.1 Reasoning": "train",
        "HMMT February 2025": "eval_only",
        "MMVU": "eval_only",
        "OCRBench v2": "eval_only",
        "MM-IQ": "eval_only",
        "Real5-OmniDocBench": "eval_only",
        "JoyAI Image OpenSpatial": "train",
        "JoyAI Image SpatialEdit": "blocked_until_review",
        "JoyAI Image SpatialEdit Bench": "blocked_until_review",
        "X-LANCE WikiHow Taskset": "eval_only",
        "X-LANCE WebSRC v1.0": "train",
        "DeepGen 1.0 Dataset Card": "research_internal",
    }
    for name, policy in fourth_wave_policy.items():
        assert by_name[name]["use_policy"] == policy
    assert by_name["Toolathlon Trajectories"]["license"] == "CC-BY-4.0"
    assert by_name["Agentic Chain-of-Thought Coding SFT"]["license_tier"] == "permissive_synthetic_distilled"
    assert by_name["JoyAI Image OpenSpatial"]["target_modality"] == "image"
    assert by_name["X-LANCE WebSRC v1.0"]["family"] == "omnimodal_understanding"
    assert expansion.source_use_bucket(by_name["JoyAI Image SpatialEdit"]) == "blocked_until_review"

    requirements = profile["external_dataset_registry_2026"]["required_real_family_min_records"]
    for family in [
        "math_reasoning",
        "coding_agentic",
        "agentic_tool_reasoning",
        "terminal_browser_agents",
        "image_generation_editing",
        "video_generation",
        "audio_music_speech",
        "music_generation",
        "omnimodal_understanding",
    ]:
        assert family in requirements


def test_repo_dataset_registry_covers_fifth_and_sixth_wave_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}

    expected_policy = {
        "RLVR Linearity Dataset": "train",
        "Nous RLVR Coding Problems": "train",
        "IFDecorator Instruction Following RL": "train",
        "CAP RLVR SFT": "research_internal",
        "HINT-Lab MATH-RLVR Gated": "blocked_until_review",
        "OpenResearcher RLVR Format": "blocked_until_review",
        "Tool Reasoning Context Management": "train",
        "Tool Reasoning Hermes Style 115K": "train",
        "OpenResearcher Tool Reasoning Cleaned": "train",
        "OpenSeeker Tool Reasoning SFT": "train",
        "CUA-Gym": "train",
        "A11y-CUA": "research_internal",
        "MEnvData SWE Trajectories": "train",
        "JetBrains SWE-Smith Agent Trajectories Random Subset": "train",
        "DeepSWE Kimi K2 Rejection Sampling Trajectories": "train",
        "Turkish Mobile Function Calling": "train",
        "ScreenSpot-Pro": "eval_only",
        "MieDB-100k Medical Image Editing": "research_internal",
        "Image Editing Style Instruction Following": "train",
        "VideoPhy2 Train": "train",
        "VideoPhy2 Test": "eval_only",
        "DocVQA 2026": "eval_only",
        "ChartMuseum": "eval_only",
        "GroundUI-18K": "eval_only",
        "LiveCodeBench Code Generation Lite": "eval_only",
        "KernelBench": "research_internal",
        "TritonBench": "train",
        "NuminaMath-LEAN": "train",
        "FoVer Formal Verification Collection": "train",
        "MathArena HMMT Feb 2026": "eval_only",
        "ARC-AGI-2 Public Training": "train",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
    assert by_name["CUA-Gym"]["registry_wave"] == "fifth_wave_agentic_rlvr_multimodal_2026_05_24"
    assert by_name["NuminaMath-LEAN"]["registry_wave"] == "sixth_wave_formal_code_media_2026_05_24"
    assert by_name["OpenResearcher RLVR Format"]["license_tier"] == "gated_unknown_review"
    assert by_name["VideoPhy2 Train"]["target_modality"] == "video"
    assert by_name["TritonBench"]["distillation_prompts"][0]["instruction"].startswith("Generate a Triton kernel")
    assert expansion.source_use_bucket(by_name["HINT-Lab MATH-RLVR Gated"]) == "blocked_until_review"


def test_repo_dataset_registry_covers_seventh_wave_sources_and_trace_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    expected_policy = {
        "OpenThoughts-Agent v1 SFT": "train",
        "OpenThoughts-Agent v1 RL": "research_internal",
        "Edge-Agent Reasoning WebSearch 260K": "train",
        "Exgentic Agent LLM Traces": "train",
        "CUDA-Agent-Ops-6K": "train",
        "AI CUDA Engineer Archive": "train",
        "CodeX-2M Thinking": "train",
        "GitHub CodeReview 356K": "research_internal",
        "INTELLECT-2 RL Dataset": "research_internal",
        "DeepSeek-ProverBench": "eval_only",
        "ECHO 2025 Image Benchmark": "eval_only",
        "TRIG Benchmark": "eval_only",
        "OpenS2V-5M": "train",
        "OmniVideoBench": "eval_only",
        "Zero-To-CAD-1M": "train",
        "ASID-1M AudioVisual Caption": "train",
        "CMI-Pref Music Preference": "research_internal",
        "VoxEval Speech QA": "eval_only",
        "ATTM Grand Challenge 2026": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == "seventh_wave_agentic_math_code_omni_2026_05_24"
    assert by_name["OpenThoughts-Agent v1 SFT"]["target_modality"] == "tool"
    assert by_name["AI CUDA Engineer Archive"]["target_modality"] == "code"
    assert by_name["OpenS2V-5M"]["target_modality"] == "video"
    assert by_name["UniM Any-to-Any Benchmark"]["target_modality"] == "multimodal"
    assert expansion.source_use_bucket(by_name["TRIG Benchmark"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["GitHub CodeReview 356K"]) == "research_internal"
    assert profile["builder_2026"]["strict_trace_dates"] is True
    assert profile["builder_2026"]["reject_unknown_trace_dates"] is True
    assert profile["agent_memory_postgres_export"]["reject_secret_rows"] is True


def test_repo_dataset_registry_covers_eighth_wave_training_curation_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "eighth_wave_agentic_curation_training_2026_05_24"
    expected_policy = {
        "MCP-Universe Trajectories": "train",
        "MCPMark Trajectory Log": "research_internal",
        "Qwen 3.6 Plus Agent Tool Calling Trajectory": "blocked_until_review",
        "Agent Distillation Qwen Agent Trajectories 2K": "research_internal",
        "Computer Use PSAI": "train",
        "BrowseCompLongContext": "eval_only",
        "BrowseComp-Plus Corpus": "train",
        "BrowseComp-Plus QA Holdout": "eval_only",
        "TheAgentCompany Enterprise Benchmark": "eval_only",
        "Audio-Alpaca Instruction Data": "research_internal",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave
    assert by_name["MCP-Universe Trajectories"]["target_modality"] == "tool"
    assert by_name["Computer Use PSAI"]["target_modality"] == "video"
    assert by_name["BrowseComp-Plus Corpus"]["target_modality"] == "long_context"
    assert by_name["OpenAudioBench"]["use_policy"] == "eval_only"
    assert by_name["VideoRewardBench"]["use_policy"] == "eval_only"
    assert expansion.source_use_bucket(by_name["MCP-Universe Trajectories"]) == "train"
    assert expansion.source_use_bucket(by_name["Qwen 3.6 Plus Agent Tool Calling Trajectory"]) == "blocked_until_review"
    assert expansion.source_use_bucket(by_name["BrowseCompLongContext"]) == "eval_holdout"
    assert profile["mixture_controller_2026"]["enabled"] is True
    assert profile["mixture_controller_2026"]["synthetic_ratio_caps"]["synthetic_only_train_minimum_credit"] == 0.0


def test_repo_dataset_registry_covers_ninth_wave_agentic_preference_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "ninth_wave_agentic_preference_benchmark_2026_05_25"
    expected_policy = {
        "SWE-chat": "research_internal",
        "Spreadsheet-RL": "research_internal",
        "When2Tool": "train",
        "Orak Game-Agent Trajectories": "research_internal",
        "Agent Reward Bench": "research_internal",
        "TaskTrove": "research_internal",
        "DeepSeek-V4-Distill-8000x": "research_internal",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave
    assert by_name["MME-Unify"]["hf_id"] == "wulin222/MME-Unify"
    assert by_name["When2Tool"]["license_tier"] == "permissive"
    assert "rejected" in by_name["SWE-chat"]["field_map"]
    assert expansion.source_use_bucket(by_name["When2Tool"]) == "train"
    assert expansion.source_use_bucket(by_name["SWE-chat"]) == "research_internal"
    assert expansion.source_use_bucket(by_name["Agent Reward Bench"]) != "train"


def test_repo_dataset_registry_covers_tenth_wave_curated_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "tenth_wave_curated_benchmarks_2026_05_25"

    expected_policy = {
        "MCPToolBench++ Preview": "eval_only",
        "WebBench AI Web Browsing Benchmark": "eval_only",
        "mAIME2025": "eval_only",
        "MMLongBench": "eval_only",
        "NoLiMa": "eval_only",
        "LongCodeBench": "eval_only",
        "SagaScale": "eval_only",
        "AcademicEval": "eval_only",
        "FineWeb2": "train",
        "Common Pile v0.1": "train",
        "LEMAS Dataset Train": "research_internal",
        "Emilia Dataset": "research_internal",
        "AudioBench": "eval_only",
        "MMAU-Pro": "eval_only",
        "MMAR": "eval_only",
        "CMI-Bench": "eval_only",
        "MUSE Music Benchmark": "eval_only",
        "MPBench": "eval_only",
        "RTV-Bench": "eval_only",
        "RIVER Bench": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave

    assert by_name["FineWeb2"]["license_tier"] == "attribution"
    assert by_name["Common Pile v0.1"]["license_tier"] == "open_license_mixture"
    assert by_name["LEMAS Dataset Train"]["hf_id"] == "LEMAS-Project/LEMAS-Dataset-train"
    assert by_name["MMAU-Pro"]["hf_id"] == "gamma-lab-umd/MMAU-Pro"
    assert by_name["RTV-Bench"]["target_modality"] == "video"
    assert by_name["RIVER Bench"]["hf_id"] == "OpenGVLab/RIVER"
    assert by_name["RIVER Bench"]["source_year"] == 2026
    assert expansion.source_use_bucket(by_name["FineWeb2"]) == "train"
    assert expansion.source_use_bucket(by_name["Common Pile v0.1"]) == "train"
    assert expansion.source_use_bucket(by_name["MCPToolBench++ Preview"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["LEMAS Dataset Train"]) == "research_internal"


def test_repo_dataset_registry_covers_eleventh_wave_agentic_omni_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "eleventh_wave_agentic_omni_eval_2026_05_25"

    expected_policy = {
        "LiveMCPBench": "eval_only",
        "SRA-Bench Skill Retrieval Augmentation": "eval_only",
        "SkillRet Benchmark": "train",
        "DAPO-Math-17k": "train",
        "Guru RL 92K": "train",
        "MemoryAgentBench": "eval_only",
        "OmniGAIA Benchmark": "eval_only",
        "Omnimodal-Agent-SFT-2K": "train",
        "OmniRAG-Agent": "eval_only",
        "VSTAT Visual State Tracking": "eval_only",
        "Tricky TTS Public": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave

    assert by_name["LiveMCPBench"]["hf_id"] == "ICIP/LiveMCPBench"
    assert by_name["SkillRet Benchmark"]["license_tier"] == "permissive"
    assert by_name["OmniGAIA Benchmark"]["target_modality"] == "multimodal"
    assert by_name["Omnimodal-Agent-SFT-2K"]["use_policy"] == "train"
    assert by_name["MemoryAgentBench"]["family"] == "long_context"
    assert by_name["AgentTrove"]["hf_id"] == "open-thoughts/AgentTrove"
    assert by_name["AVGen-Bench"]["hf_id"] == "microsoft/AVGen-Bench"
    assert expansion.source_use_bucket(by_name["OmniGAIA Benchmark"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["Tricky TTS Public"]) == "eval_holdout"


def test_repo_dataset_registry_covers_twelfth_wave_agent_memory_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "twelfth_wave_agent_memory_state_2026_05_25"

    expected_policy = {
        "AMA-Bench Agent Memory": "eval_only",
        "SMMBench Source-Distributed Multimodal Memory": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave

    assert by_name["AMA-Bench Agent Memory"]["hf_id"] == "AMA-bench/AMA-bench"
    assert by_name["SMMBench Source-Distributed Multimodal Memory"]["hf_id"] == "HuacanChai/SMMBench"
    assert by_name["SMMBench Source-Distributed Multimodal Memory"]["target_modality"] == "image"
    assert expansion.source_use_bucket(by_name["AMA-Bench Agent Memory"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["SMMBench Source-Distributed Multimodal Memory"]) == "eval_holdout"


def test_repo_dataset_registry_covers_thirteenth_wave_agentic_math_multimodal_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "thirteenth_wave_agentic_math_multimodal_2026_05_25"

    expected_policy = {
        "Agentic-MME": "eval_only",
        "ABC-Bench Backend Agent Tasks": "eval_only",
        "LongBench-Pro": "eval_only",
        "MEGA-Bench": "eval_only",
        "StepEval-Audio-360": "eval_only",
        "IndiMathBench": "eval_only",
        "DeepResearch-9K": "research_internal",
        "MMFineReason-1.8M Qwen3-VL Thinking": "research_internal",
        "Lean Math Formal Corpus v4.27.0": "research_internal",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave

    assert by_name["Agentic-MME"]["hf_id"] == "Agentic-MME/Agentic-MME"
    assert by_name["ABC-Bench Backend Agent Tasks"]["license"] == "ODC-BY"
    assert by_name["LongBench-Pro"]["target_modality"] == "long_context"
    assert by_name["MEGA-Bench"]["configs"] == ["core", "open"]
    assert by_name["MEGA-Bench"]["splits"] == ["test", "train"]
    assert by_name["StepEval-Audio-360"]["target_modality"] == "audio"
    assert by_name["IndiMathBench"]["repo"] == "https://github.com/prmbiy/IndiMathBench.git"
    assert by_name["IndiMathBench"]["remote_files"][0]["format"] == "json"
    assert by_name["MMFineReason-1.8M Qwen3-VL Thinking"]["hf_id"].endswith("Qwen3-VL-235B-Thinking")
    assert by_name["MMFineReason-1.8M Qwen3-VL Thinking"]["splits"] == ["sft"]
    assert expansion.source_use_bucket(by_name["LongBench-Pro"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["StepEval-Audio-360"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["DeepResearch-9K"]) == "research_internal"
    assert expansion.source_use_bucket(by_name["Lean Math Formal Corpus v4.27.0"]) == "research_internal"


def test_repo_dataset_registry_covers_fourteenth_wave_agentic_gui_video_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "fourteenth_wave_agentic_gui_video_eval_2026_05_25"

    expected_policy = {
        "MCPVerse": "eval_only",
        "UI-Vision": "eval_only",
        "ViMUL-Bench": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave

    assert by_name["MCPVerse"]["repo"] == "https://github.com/hailsham/mcpverse.git"
    assert by_name["UI-Vision"]["hf_id"] == "ServiceNow/ui-vision"
    assert by_name["ViMUL-Bench"]["hf_id"] == "MBZUAI/ViMUL-Bench"
    assert by_name["ViMUL-Bench"]["configs"] == ["vimulmcq_english", "vimuloe_english"]
    assert by_name["ViMUL-Bench"]["license_tier"] == "sharealike_eval_holdout"
    assert expansion.source_use_bucket(by_name["MCPVerse"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["UI-Vision"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["ViMUL-Bench"]) == "eval_holdout"


def test_repo_dataset_registry_covers_fifteenth_wave_agentic_coding_audio_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "fifteenth_wave_agentic_coding_security_audio_2026_05_25"

    expected_policy = {
        "AgentWorldModel-1K": "train",
        "Tool-Genesis Benchmark": "eval_only",
        "MCP Security Bench": "eval_only",
        "BeyondSWE": "eval_only",
        "ContextBench": "eval_only",
        "CCBench": "eval_only",
        "WebGym Tasks": "train",
        "AudioMCQ StrongAC Gemini CoT": "train",
        "AgentIF": "eval_only",
        "NVIDIA ComputeEval": "eval_only",
        "ParseBench": "eval_only",
        "OfficeQA": "eval_only",
        "OmniAgentBench": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["use_policy"] == policy
        assert by_name[name]["registry_wave"] == wave

    assert by_name["AgentWorldModel-1K"]["hf_id"] == "Snowflake/AgentWorldModel-1K"
    assert "gen_envs.jsonl" in by_name["AgentWorldModel-1K"]["data_files"]["train"]
    assert by_name["Tool-Genesis Benchmark"]["protected_benchmark_scan"] == "protected_eval"
    assert by_name["WebGym Tasks"]["license"] == "CDLA-Permissive-2.0"
    assert by_name["AudioMCQ StrongAC Gemini CoT"]["synthetic_provenance"].startswith("Gemini CoT")
    assert by_name["NVIDIA ComputeEval"]["license_tier"] == "evaluation_license_holdout"
    assert by_name["NVIDIA ComputeEval"]["splits"] == ["eval"]
    assert by_name["ParseBench"]["hf_id"] == "llamaindex/ParseBench"
    assert "table" in by_name["ParseBench"]["splits"]
    assert by_name["OfficeQA"]["family"] == "long_context"
    assert by_name["OfficeQA"]["token_env"] == "HF_TOKEN"

    for name in ("AgentWorldModel-1K", "WebGym Tasks", "AudioMCQ StrongAC Gemini CoT"):
        assert expansion.source_use_bucket(by_name[name]) == "train"
        assert expansion.training_bucket_for_record(by_name[name], {}) == "train"
        assert expansion.contamination_status_for_record(by_name[name], {}) == "clean"

    for name in set(expected_policy) - {"AgentWorldModel-1K", "WebGym Tasks", "AudioMCQ StrongAC Gemini CoT"}:
        assert expansion.source_use_bucket(by_name[name]) == "eval_holdout"


def test_repo_dataset_registry_covers_sixteenth_wave_multimodal_and_agentic_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "sixteenth_wave_data_benchmark_expansion_2026_05_25"

    expected_policy = {
        "PD12M Public Domain 12M": "train",
        "BigEarthNet.txt": "research_internal",
        "HOIGen-1M": "research_internal",
        "Meta Omnilingual ASR Corpus": "train",
        "GigaSpeech 2": "research_internal",
        "AVATAR Audio-Visual Localization": "eval_only",
        "VideoWebArena": "eval_only",
        "OSUniverse": "eval_only",
        "GUI-World": "research_internal",
        "SWE-bench Multimodal": "eval_only",
        "LOFT Long Context Frontiers": "eval_only",
        "FrontierMath Tiers 1-4": "eval_only",
        "GIE-Bench Grounded Image Editing": "eval_only",
        "EditInspector": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    assert by_name["PD12M Public Domain 12M"]["license_tier"] == "public_domain_cc0"
    assert by_name["Meta Omnilingual ASR Corpus"]["license_tier"] == "attribution"
    assert by_name["PD12M Public Domain 12M"]["contamination_status"] == "clean"
    assert by_name["Meta Omnilingual ASR Corpus"]["contamination_status"] == "clean"
    assert by_name["AVATAR Audio-Visual Localization"]["materialization_note"].startswith("Large video.zip")
    assert by_name["VideoWebArena"]["repo"] == "https://github.com/ljang0/videowebarena.git"
    assert by_name["SWE-bench Multimodal"]["splits"] == ["dev", "test"]

    assert expansion.source_use_bucket(by_name["PD12M Public Domain 12M"]) == "train"
    assert expansion.source_use_bucket(by_name["Meta Omnilingual ASR Corpus"]) == "train"
    assert expansion.training_bucket_for_record(by_name["PD12M Public Domain 12M"], {"caption": "public domain image caption"}) == "train"
    assert expansion.training_bucket_for_record(by_name["Meta Omnilingual ASR Corpus"], {"text": "public ASR transcript"}) == "train"
    for name in set(expected_policy) - {"PD12M Public Domain 12M", "Meta Omnilingual ASR Corpus"}:
        assert expansion.source_use_bucket(by_name[name]) != "train"


def test_repo_dataset_registry_covers_seventeenth_wave_feature_tool_and_generation_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "seventeenth_wave_agentic_coding_generation_eval_2026_05_25"

    expected_policy = {
        "BFCL Function Calling Leaderboard": "eval_only",
        "FEA-Bench Repository Feature Implementation": "eval_only",
        "WorldGenBench T2I World Knowledge": "research_internal",
        "OmniGenBench Image Generation": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    assert by_name["BFCL Function Calling Leaderboard"]["hf_id"] == "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    assert by_name["FEA-Bench Repository Feature Implementation"]["hf_id"] == "microsoft/FEA-Bench"
    assert by_name["WorldGenBench T2I World Knowledge"]["hf_id"] == "worldrl/WorldGenBench"
    assert by_name["OmniGenBench Image Generation"]["repo"] == "https://github.com/emilia113/OmniGenBench.git"

    assert expansion.source_use_bucket(by_name["BFCL Function Calling Leaderboard"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["FEA-Bench Repository Feature Implementation"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["WorldGenBench T2I World Knowledge"]) != "train"
    assert expansion.source_use_bucket(by_name["OmniGenBench Image Generation"]) == "eval_holdout"


def test_repo_dataset_registry_covers_eighteenth_wave_live_media_agent_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "eighteenth_wave_live_media_agent_data_2026_05_25"

    expected_policy = {
        "FineVideo Timecode Metadata": "research_internal",
        "Raon OpenTTS Pool Commercial Core": "train",
        "Toucan Agentic Thinking MiniMax-M2.1": "research_internal",
        "LightOnOCR Mix 0126": "research_internal",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    assert by_name["FineVideo Timecode Metadata"]["hf_id"] == "HuggingFaceFV/finevideo"
    assert by_name["FineVideo Timecode Metadata"]["license_tier"] == "source_license_varies"
    assert by_name["Raon OpenTTS Pool Commercial Core"]["hf_id"] == "KRAFTON/Raon-OpenTTS-Pool"
    assert by_name["Raon OpenTTS Pool Commercial Core"]["config"] == "Raon-YouTube-Commons"
    assert by_name["Raon OpenTTS Pool Commercial Core"]["splits"] == ["core"]
    assert by_name["Toucan Agentic Thinking MiniMax-M2.1"]["hf_id"] == "agent-data/toucan-agentic-thinking"
    assert by_name["LightOnOCR Mix 0126"]["hf_id"] == "lightonai/LightOnOCR-mix-0126"
    assert by_name["LightOnOCR Mix 0126"]["splits"] == ["pdfa_train"]
    assert "content" in by_name["LightOnOCR Mix 0126"]["field_map"]["target"]

    assert expansion.source_use_bucket(by_name["Raon OpenTTS Pool Commercial Core"]) == "train"
    assert expansion.training_bucket_for_record(
        by_name["Raon OpenTTS Pool Commercial Core"],
        {"text": "spoken training text", "audio": "sample.opus"},
    ) == "train"
    for name in set(expected_policy) - {"Raon OpenTTS Pool Commercial Core"}:
        assert expansion.source_use_bucket(by_name[name]) == "research_internal"


def test_repo_dataset_registry_covers_nineteenth_wave_trainable_reward_eval_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "nineteenth_wave_trainable_reward_eval_2026_05_25"

    expected_policy = {
        "NVIDIA Nemotron-SFT Agentic v2": "train",
        "NVIDIA OpenCodeInstruct": "train",
        "NVIDIA Nemotron-SFT OpenCode v1": "train",
        "NVIDIA Nemotron-SFT Math v3": "train",
        "FinePhrase": "train",
        "NVIDIA Retrieval Synthetic NVDocs v1": "train",
        "ChartVerse SFT 1.8M": "research_internal",
        "Microsoft World-R1": "train",
        "Rapidata Kling v2.1 Master T2V Preferences": "train",
        "DatapointAI TTS Human Preferences Large": "research_internal",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    assert by_name["NVIDIA Nemotron-SFT Agentic v2"]["hf_id"] == "nvidia/Nemotron-SFT-Agentic-v2"
    assert by_name["NVIDIA OpenCodeInstruct"]["hf_id"] == "nvidia/OpenCodeInstruct"
    assert by_name["NVIDIA Nemotron-SFT OpenCode v1"]["hf_id"] == "nvidia/Nemotron-SFT-OpenCode-v1"
    assert by_name["NVIDIA Nemotron-SFT OpenCode v1"]["config"] == "default"
    assert by_name["NVIDIA Nemotron-SFT OpenCode v1"]["splits"] == [
        "bash_only_tool_skills",
        "bash_only_tool",
        "general",
        "question_tool",
        "agent_skills",
        "agent_skills_question_tool",
    ]
    assert by_name["NVIDIA Nemotron-SFT Math v3"]["hf_id"] == "nvidia/Nemotron-SFT-Math-v3"
    assert by_name["FinePhrase"]["hf_id"] == "HuggingFaceFW/finephrase"
    assert by_name["NVIDIA Retrieval Synthetic NVDocs v1"]["hf_id"] == "nvidia/Retrieval-Synthetic-NVDocs-v1"
    assert by_name["ChartVerse SFT 1.8M"]["hf_id"] == "opendatalab/ChartVerse-SFT-1800K"
    assert by_name["Microsoft World-R1"]["configs"] == ["final", "enhanced"]
    assert by_name["Rapidata Kling v2.1 Master T2V Preferences"]["hf_id"] == "Rapidata/text-2-video-human-preferences-kling-v2.1-master"
    assert by_name["DatapointAI TTS Human Preferences Large"]["hf_id"] == "datapointai/tts-human-preferences-large"
    assert "audio_b" in by_name["DatapointAI TTS Human Preferences Large"]["field_map"]["media"]
    assert "images" in by_name["ChartVerse SFT 1.8M"]["field_map"]["media"]

    train_names = {
        "NVIDIA Nemotron-SFT Agentic v2",
        "NVIDIA OpenCodeInstruct",
        "NVIDIA Nemotron-SFT OpenCode v1",
        "NVIDIA Nemotron-SFT Math v3",
        "FinePhrase",
        "NVIDIA Retrieval Synthetic NVDocs v1",
        "Microsoft World-R1",
        "Rapidata Kling v2.1 Master T2V Preferences",
    }
    for name in train_names:
        assert by_name[name]["contamination_status"] == "clean"
        assert by_name[name]["protected_benchmark_scan"] == "clean"
        assert expansion.source_use_bucket(by_name[name]) == "train"
        assert expansion.training_bucket_for_record(by_name[name], {"prompt": "p", "answer": "a"}) == "train"

    assert expansion.source_use_bucket(by_name["ChartVerse SFT 1.8M"]) == "research_internal"
    assert expansion.source_use_bucket(by_name["DatapointAI TTS Human Preferences Large"]) == "research_internal"
    assert "humair025/suno-audio" not in {str(entry.get("hf_id")) for entry in entries}


def test_repo_dataset_registry_covers_twentieth_wave_multimodal_agentic_reasoning_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "twentieth_wave_multimodal_agentic_reasoning_2026_05_25"

    expected_policy = {
        "Creative Professionals Agentic Tasks 1M": "research_internal",
        "Reasoning Core Procedural Pretraining Pile": "train",
        "Marco Longspeech": "train",
        "Veri-Code ReForm Python2Dafny": "train",
        "MCPHunt Agent Safety Traces": "reward_only",
        "Rapidata Base Text-to-Video Human Preferences": "train",
        "Voices in the Wild 2M": "train",
        "AllenAI olmOCR Bench": "eval_only",
        "Limbic Eval Tool Use MCP": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    train_names = {
        "Reasoning Core Procedural Pretraining Pile",
        "Marco Longspeech",
        "Veri-Code ReForm Python2Dafny",
        "Rapidata Base Text-to-Video Human Preferences",
        "Voices in the Wild 2M",
    }
    for name in train_names:
        assert by_name[name]["contamination_status"] == "clean"
        assert by_name[name]["protected_benchmark_scan"] == "clean"
        assert expansion.source_use_bucket(by_name[name]) == "train"
        assert expansion.training_bucket_for_record(by_name[name], {"prompt": "p", "answer": "a"}) == "train"

    assert by_name["Creative Professionals Agentic Tasks 1M"]["synthetic_train_seed_policy"] == "teacher_distill_before_train"
    assert expansion.source_use_bucket(by_name["Creative Professionals Agentic Tasks 1M"]) == "research_internal"
    assert expansion.source_use_bucket(by_name["MCPHunt Agent Safety Traces"]) == "research_internal"
    assert expansion.source_use_bucket(by_name["AllenAI olmOCR Bench"]) == "eval_holdout"
    assert expansion.source_use_bucket(by_name["Limbic Eval Tool Use MCP"]) == "eval_holdout"
    assert "audio_path" in by_name["Marco Longspeech"]["field_map"]["media"]
    assert by_name["Veri-Code ReForm Python2Dafny"]["hf_id"] == "Veri-Code/ReForm-Python2Dafny-Dataset"
    assert by_name["Voices in the Wild 2M"]["splits"] == [
        "noise",
        "far_field",
        "distortion",
        "dropout",
        "recording",
        "echo",
        "obstructed",
        "recording_noise",
    ]
    assert by_name["AllenAI olmOCR Bench"]["splits"] == [
        "arxiv_math",
        "headers_footers",
        "long_tiny_text",
        "multi_column",
        "old_scans",
        "old_scans_math",
        "table_tests",
    ]
    assert by_name["Limbic Eval Tool Use MCP"]["splits"] == ["test"]


def test_repo_dataset_registry_covers_twenty_first_wave_ocr_video_tool_reward_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "twenty_first_wave_ocr_video_tool_reward_2026_05_25"

    expected_policy = {
        "AllenAI olmOCR SynthMix 1025": "train",
        "AllenAI olmOCR Mix 1025": "train",
        "Tongyi-Zhiwen DocQA-RL 1.6K": "train",
        "DAMO VideoRefer 700K": "train",
        "TIGER-Lab VisCode Multi 679K": "train",
        "Nanbeige ToolMind Full": "train",
        "NVIDIA HelpSteer3": "reward_only",
        "CommonForms": "train",
        "HuggingFace FinePDFs English": "train",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    train_names = {name for name, policy in expected_policy.items() if policy == "train"}
    for name in train_names:
        assert by_name[name]["contamination_status"] == "clean"
        assert by_name[name]["protected_benchmark_scan"] == "clean"
        assert expansion.source_use_bucket(by_name[name]) == "train"

    assert by_name["AllenAI olmOCR SynthMix 1025"]["license_tier"] == "attribution_train_ok"
    assert by_name["AllenAI olmOCR Mix 1025"]["configs"] == [
        "00_documents",
        "01_books",
        "02_loc_transcripts",
        "03_national_archives",
    ]
    assert by_name["AllenAI olmOCR Mix 1025"]["splits"] == ["train"]
    assert by_name["DAMO VideoRefer 700K"]["data_files"]["train"] == [
        "videorefer-short-caption-500k.json",
        "videorefer-detailed-caption-125k.json",
        "videorefer-qa-75k.json",
    ]
    assert by_name["Nanbeige ToolMind Full"]["config"] == "test"
    assert by_name["Nanbeige ToolMind Full"]["splits"] == ["graph_syn_datasets", "open_datasets"]
    assert by_name["CommonForms"]["field_map"]["verifier_labels"] == ["objects"]
    assert by_name["NVIDIA HelpSteer3"]["configs"] == ["preference", "edit", "edit_quality", "feedback", "principle"]
    assert expansion.source_use_bucket(by_name["NVIDIA HelpSteer3"]) == "research_internal"
    assert by_name["HuggingFace FinePDFs English"]["config"] == "eng_Latn"


def test_repo_dataset_registry_covers_twenty_second_wave_agent_audio_ocr_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "twenty_second_wave_agent_audio_ocr_2026_05_25"

    expected_policy = {
        "NVIDIA OCR Synthetic Multilingual v1": "train",
        "LAION Got Talent Orpheus Voice Tags": "train",
        "StephenZhu SWE-Play Trajectories": "research_internal",
        "Meituan LongCat Audio Turing Test": "reward_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    assert by_name["NVIDIA OCR Synthetic Multilingual v1"]["license_tier"] == "attribution_train_ok"
    assert "verifier_labels" in by_name["NVIDIA OCR Synthetic Multilingual v1"]["field_map"]
    assert expansion.source_use_bucket(by_name["NVIDIA OCR Synthetic Multilingual v1"]) == "train"
    assert by_name["LAION Got Talent Orpheus Voice Tags"]["license"] == "Apache-2.0"
    assert "emotion_tags" in by_name["LAION Got Talent Orpheus Voice Tags"]["field_map"]["verifier_labels"]
    assert expansion.source_use_bucket(by_name["LAION Got Talent Orpheus Voice Tags"]) == "train"
    assert by_name["StephenZhu SWE-Play Trajectories"]["field_map"]["trajectory"] == ["messages"]
    assert expansion.source_use_bucket(by_name["StephenZhu SWE-Play Trajectories"]) == "research_internal"
    assert "non_commercial" in by_name["Meituan LongCat Audio Turing Test"]["license_tier"]
    assert expansion.source_use_bucket(by_name["Meituan LongCat Audio Turing Test"]) == "research_internal"


def test_repo_dataset_registry_covers_twenty_third_wave_agentic_code_math_multimodal_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}
    wave = "twenty_third_wave_agentic_code_math_multimodal_2026_05_25"

    expected_policy = {
        "Agentic Coding Tessa": "train",
        "Ethanker Agentic Coding Dataset": "train",
        "Agentic CoT Coding SFT v1.1": "train",
        "169Pi MathReasoning": "train",
        "Math SFT Solutions No CoT": "train",
        "EST Math Reasoning SFT": "train",
        "OpenDataArena MMFineReason Full 2.3M": "train",
        "InfiX OmniAct Grounding Filtered": "train",
        "ClaudeSet Community Agent Traces": "research_internal",
        "Microsoft WebTailBench V2": "eval_only",
    }
    for name, policy in expected_policy.items():
        assert by_name[name]["registry_wave"] == wave
        assert by_name[name]["use_policy"] == policy

    train_names = {name for name, policy in expected_policy.items() if policy == "train"}
    for name in train_names:
        assert by_name[name]["contamination_status"] == "clean"
        assert by_name[name]["protected_benchmark_scan"] == "clean"
        assert expansion.source_use_bucket(by_name[name]) == "train"
        assert expansion.training_bucket_for_record(by_name[name], {"prompt": "p", "answer": "a"}) == "train"

    assert by_name["Agentic Coding Tessa"]["hf_id"] == "smirki/Agentic-Coding-Tessa"
    assert by_name["Agentic Coding Tessa"]["field_map"]["trajectory"] == ["conversations"]
    assert by_name["Ethanker Agentic Coding Dataset"]["license"] == "MIT"
    assert by_name["Agentic CoT Coding SFT v1.1"]["field_map"]["target"] == ["assistant", "output", "response"]
    assert by_name["169Pi MathReasoning"]["field_map"]["target"] == ["response", "answer", "solution"]
    assert by_name["Math SFT Solutions No CoT"]["configs"] == ["short", "long", "very long"]
    assert by_name["EST Math Reasoning SFT"]["field_map"]["prompt"] == ["instruction", "input", "prompt"]
    assert by_name["OpenDataArena MMFineReason Full 2.3M"]["hf_id"] == "OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking"
    assert by_name["OpenDataArena MMFineReason Full 2.3M"]["target_modality"] == "image"
    assert by_name["OpenDataArena MMFineReason Full 2.3M"]["field_map"]["media"] == ["image"]
    assert by_name["InfiX OmniAct Grounding Filtered"]["family"] == "terminal_browser_agents"
    assert "reward_model" in by_name["InfiX OmniAct Grounding Filtered"]["field_map"]["verifier_labels"]

    assert "privacy_review" in by_name["ClaudeSet Community Agent Traces"]["license_tier"]
    assert expansion.source_use_bucket(by_name["ClaudeSet Community Agent Traces"]) == "research_internal"
    assert by_name["Microsoft WebTailBench V2"]["remote_files"][0]["format"] == "tsv"
    assert expansion.source_use_bucket(by_name["Microsoft WebTailBench V2"]) == "eval_holdout"


def test_repo_dataset_registry_promotes_reviewed_train_rows_after_clean_scan() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}

    reviewed_train = [
        "Cleaned Toucan Tool Use 333K",
        "Cleaned Hermes Reasoning Tool Use",
        "Cleaned Memory Agent SFT 408K",
        "Cleaned ToolMind Web QA Tool Use",
        "NVIDIA Nemotron-Terminal-Corpus",
        "NVIDIA Nemotron-Terminal-Synthetic-Tasks",
        "NVIDIA Nemotron-SFT-SWE-v2",
        "NVIDIA Nemotron-SFT Competitive Programming v2",
        "NVIDIA OpenMathReasoning",
        "NVIDIA When2Call",
        "NVIDIA Nemotron Agentic v1",
        "NVIDIA Nemotron-RL Agentic Function Calling Pivot v1",
        "NVIDIA HiFiTTS 2",
        "LongWriter-Zero RLData",
        "ACE-Step Songs",
        "Song Describer",
        "Open-MM-RL",
    ]
    for name in reviewed_train:
        assert by_name[name]["contamination_status"] == "clean"
        assert by_name[name]["protected_benchmark_scan"] == "clean"
        assert by_name[name]["source_review_status"] == "public_train_reviewed_2026_05_25"
        assert expansion.source_use_bucket(by_name[name]) == "train"
        assert expansion.training_bucket_for_record(by_name[name], {"prompt": "p", "answer": "a"}) == "train"


def test_registry_fail_closes_review_and_holdout_rows_from_train_bucket() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    unsafe_markers = (
        "review",
        "pending",
        "unknown",
        "non_commercial",
        "no_derivatives",
        "holdout",
        "gated",
        "research",
        "blocked",
    )

    for entry in entries:
        blob = f"{entry.get('license') or ''} {entry.get('license_tier') or ''}".lower()
        if any(marker in blob for marker in unsafe_markers):
            assert expansion.source_use_bucket(entry) != "train", entry["name"]

from __future__ import annotations

import json
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
    _write_jsonl(root / "data" / "math.jsonl", [{"problem": "Solve 2+2.", "answer": "4", "uuid": "m1"}])
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


def test_repo_dataset_registry_covers_new_agentic_and_multimodal_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dataset_curation_2026.json").read_text(encoding="utf-8"))
    entries = profile["external_dataset_registry_2026"]["datasets"]
    by_name = {entry["name"]: entry for entry in entries}

    for name in [
        "OpenThoughts2-1M",
        "DeepCoder-Preview-Dataset",
        "NVIDIA Nemotron-Terminal-Corpus",
        "Hermes Function Calling V1",
        "MultiEdit",
        "NVIDIA Granary",
        "AR-Omni-Instruct-v0.1",
    ]:
        assert name in by_name

    assert by_name["NVIDIA Nemotron-Terminal-Corpus"]["use_policy"] == "train"
    assert by_name["AR-Omni-Instruct-v0.1"]["use_policy"] == "research_internal"
    assert by_name["Open-MM-RL"]["use_policy"] == "blocked_until_review"

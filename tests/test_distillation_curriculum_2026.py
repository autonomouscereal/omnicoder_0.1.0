from __future__ import annotations

import json
from pathlib import Path

from omnicoder.training import distillation_curriculum_2026 as distill


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_profile_schema_registry_overrides_keyword_inference() -> None:
    profile = {
        "schema_registry": {
            "patch_test_plan": {
                "schema_id": "omnicoder.teacher_job.patch_test_plan.v1",
                "kind": "verifier",
                "required_fields": ["repo_context", "unit_tests", "reward"],
            }
        }
    }
    schema = distill.expected_output_schema("patch_test_plan", profile)
    assert schema["schema_id"] == "omnicoder.teacher_job.patch_test_plan.v1"
    assert schema["required_fields"] == ["repo_context", "unit_tests", "reward"]


def test_build_jobs_emits_schema_id_roles_and_training_targets(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_jsonl(
        records,
        [
            {
                "input_json": {"messages": [{"role": "user", "content": "Patch this repo and run pytest."}]},
                "target_json": {"content": "The patch plan should inspect the failing test, make the smallest code change, rerun pytest, and report the verified result."},
                "modality": "tool",
                "quality": {"score": 0.95},
                "contamination_status": "clean",
                "split": "train",
            }
        ],
    )
    profile = {
        "job_plan": {"min_quality": 0.1, "per_teacher_limit": 4},
        "schema_registry": {
            "patch_test_plan": {
                "schema_id": "omnicoder.teacher_job.patch_test_plan.v1",
                "kind": "verifier",
                "required_fields": ["repo_context", "unit_tests", "reward"],
            }
        },
        "teacher_registry": {
            "qwen3.6_27b_q4_local": {
                "enabled": True,
                "provider": "lm_studio_openai_compatible",
                "model_alias": "qwen3.6-27b-q4",
                "teacher_role": "primary",
                "adjudication_group": "agent_tool",
                "modalities": ["text", "code", "tool", "agent_trace"],
                "job_types": ["patch_test_plan"],
                "priority": 10,
            }
        },
    }
    jobs, summary = distill.build_jobs(profile, records)
    assert summary["jobs"] == 1
    job = jobs[0]
    assert job["teacher_role"] == "primary"
    assert job["adjudication_group"] == "agent_tool"
    expected = job["input_json"]["expected_output_schema"]
    assert expected["schema_id"] == "omnicoder.teacher_job.patch_test_plan.v1"
    assert "unit_tests" in expected["required_fields"]
    assert "rlvr_grpo" in job["input_json"]["training_targets"]
    assert job["input_json"]["source_record_hash"]


def test_build_jobs_filters_unknown_contamination_and_toy_targets(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_jsonl(
        records,
        [
            {
                "input_json": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
                "target_json": {"content": "4"},
                "modality": "text",
                "quality": {"score": 1.0},
                "contamination_status": "clean",
                "split": "train",
            },
            {
                "input_json": {"messages": [{"role": "user", "content": "Explain the trace readiness."}]},
                "target_json": {"content": "This row is long enough, but it should fail closed because contamination metadata is absent."},
                "modality": "text",
                "quality": {"score": 1.0},
                "split": "train",
            },
        ],
    )
    profile = {
        "job_plan": {"min_quality": 0.1},
        "teacher_registry": {
            "qwen3.6_27b_q4_local": {
                "enabled": True,
                "modalities": ["text"],
                "job_types": ["trace_critique"],
            }
        },
    }

    jobs, summary = distill.build_jobs(profile, records)

    assert jobs == []
    assert summary["records_filtered"] == 2


def test_validate_reports_roles_and_schema_registry(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_registry": {"storyboard_plan": {"required_fields": ["shot_list", "reward"]}},
                "posttraining": {"stages": [{"id": "sft"}]},
                "teacher_registry": {
                    "ltx_2_3": {
                        "enabled": True,
                        "provider": "comfyui",
                        "model_alias": "ltx-2.3",
                        "teacher_role": "generator",
                        "adjudication_group": "video",
                        "modalities": ["video", "image", "text"],
                        "job_types": ["storyboard_plan"],
                        "priority": 35,
                    },
                    "deepseek_v4_optional": {
                        "enabled": True,
                        "provider": "external_openai_compatible",
                        "model_alias": "deepseek-v4",
                        "teacher_role": "verifier",
                        "adjudication_group": "hard_reasoning",
                        "modalities": ["text", "code", "agent_trace", "tool", "audio", "music"],
                        "job_types": ["hard_reasoning_proof_check"],
                        "priority": 40,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result = distill.validate(type("Args", (), {"profile": str(profile_path)})())
    assert result["status"] == "ok"
    assert result["teacher_roles"]["generator"] == 1
    assert result["teacher_roles"]["verifier"] == 1
    assert "storyboard_plan" in result["schema_registry"]


def test_repo_distillation_profile_includes_on_policy_mcp_recovery() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "distillation_curriculum_2026.json").read_text(encoding="utf-8"))
    opd = profile["on_policy_distillation_2026"]
    stages = {stage["id"]: stage for stage in profile["posttraining"]["stages"]}

    assert opd["enabled"] is True
    assert "q4_consistency_distill" in opd["methods"]
    assert "mcp_state_recovery" in opd["targets"]
    assert stages["mcp_environment_rl"]["algorithm"] == "agent_lightning_style_async_grpo"

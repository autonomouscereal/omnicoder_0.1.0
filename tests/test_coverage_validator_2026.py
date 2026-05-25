from __future__ import annotations

import argparse
import json
from pathlib import Path

import omnicoder.data_factory.coverage_validator_2026 as coverage


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _args(root: Path, run_id: str, **overrides):
    values = {
        "root": str(root),
        "run_id": run_id,
        "curated_dir": "",
        "local_trace_dir": "",
        "external_dir": "",
        "agentic_dir": "",
        "teacher_job_dir": "",
        "teacher_rollout_dir": "",
        "mixture_plan": "",
        "reportable_root": "",
        "benchmark_materialization_manifest": "",
        "require_media_teacher_rollouts": True,
        "require_reportable_tasks": True,
        "require_official_reportable_tasks": False,
        "strict": False,
        "out": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_coverage_validator_reports_missing_materialized_modalities(tmp_path: Path) -> None:
    report = coverage.validate_coverage(_args(tmp_path, "run_a"))

    assert report["status"] == "needs_data"
    labels = {item["label"] for item in report["missing"]}
    assert "curated_train_text.jsonl" in labels
    assert "teacher_jobs_all" in labels
    assert "media_teacher_rollouts" in labels
    assert "reportable_eval_tasks" in labels


def test_coverage_validator_passes_full_run_artifacts(tmp_path: Path) -> None:
    run_id = "run_full"
    root = tmp_path
    curated = root / "weights" / "curated_datasets_2026" / "runs" / run_id
    for name in coverage.REQUIRED_TRAIN_FILES:
        _write_jsonl(curated / "jsonl" / name, [{"text": name}])
    _write_json(curated / "manifests" / "curation_manifest.json", {"status": "passed"})
    _write_jsonl(curated / "raw" / "normalized_traces.jsonl", [{"text": "trace"}])
    local = root / "weights" / "curated_datasets_2026" / "runs" / f"{run_id}_local_traces"
    _write_jsonl(local / "raw" / "normalized_traces.jsonl", [{"text": "local trace"}])

    external = root / "weights" / "external_datasets_2026" / "runs" / run_id
    _write_json(external / "manifests" / "external_dataset_manifest.json", {"status": "passed", "records": {"train": 9}})

    agentic = root / "weights" / "agentic_tool_training_2026" / "runs" / run_id
    counts = {name: 3 for name in coverage.REQUIRED_AGENTIC_EXPORTS}
    _write_json(agentic / "agentic_tool_training_manifest.json", {"status": "passed", "counts": counts})
    _write_json(agentic / "after_teacher" / "agentic_tool_training_manifest.json", {"status": "passed", "counts": counts})

    teacher = root / "weights" / "data_factory" / "runs" / "teacher_jobs" / run_id
    _write_jsonl(teacher / "all_jobs.jsonl", [{"job": 1}])
    _write_jsonl(teacher / "modality" / "all_modality_teacher_jobs.jsonl", [{"job": "media"}])
    _write_jsonl(teacher / "modality" / "image_reward_jobs.jsonl", [{"job": "image"}])
    _write_json(teacher / "modality" / "modality_teacher_jobs_manifest.json", {"status": "ok"})

    rollouts = root / "weights" / "data_factory" / "teacher_rollouts" / run_id
    _write_jsonl(rollouts / "qwen36_agentic_math_code_tool.jsonl", [{"teacher": "qwen"}])
    _write_jsonl(rollouts / "comfyui_modality_teacher_rollouts.jsonl", [{"teacher": "comfy"}])
    _write_json(rollouts / "teacher_rollout_manifest.json", {"status": "ok"})

    _write_json(root / "weights" / "training_orchestration_2026" / "runs" / run_id / "manifests" / "mixture_plan.json", {"status": "passed"})
    _write_jsonl(root / "data" / "eval" / "reportable_2026" / "arc_agi3_authorized.jsonl", [{"task_id": "a"}])

    report = coverage.validate_coverage(_args(root, run_id))

    assert report["status"] == "passed"
    assert report["missing"] == []
    assert report["counts"]["curated_train_files"]["train_video.jsonl"] == 1
    assert report["counts"]["media_teacher_rollouts"]["comfyui_modality_teacher_rollouts.jsonl"] == 1


def test_coverage_validator_distinguishes_local_from_official_benchmark_materialization(tmp_path: Path) -> None:
    manifest = tmp_path / "weights" / "data_factory" / "runs" / "benchmark_materialization" / "run_m" / "manifests" / "benchmark_materialization_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "omnicoder.benchmark_materializer_2026.v1",
            "rows": 4,
            "records": [
                {
                    "benchmark_id": "agent_terminal_bench_2_1_2026",
                    "rows": 4,
                    "reportable": False,
                    "local_only": True,
                }
            ],
        },
    )

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            "run_m",
            require_reportable_tasks=False,
            require_media_teacher_rollouts=False,
            require_official_reportable_tasks=True,
        )
    )

    assert report["counts"]["local_materialized_benchmark_rows"] == 4
    assert report["counts"]["official_materialized_benchmark_rows"] == 0
    labels = {item["label"] for item in report["missing"]}
    assert "official_materialized_reportable_tasks" in labels

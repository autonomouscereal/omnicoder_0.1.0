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
        "benchmark_materialization_root": "",
        "benchmark_materialization_manifest": "",
        "benchmark_profile": "",
        "benchmark_reportable_summary": "",
        "benchmark_reportable_results": "",
        "reportable_prediction_summary": "",
        "reportable_predictions": "",
        "min_reportable_tasks": 1,
        "require_media_teacher_rollouts": True,
        "require_modality_teacher_jobs": False,
        "require_mixture_plan": False,
        "require_reportable_tasks": True,
        "require_official_reportable_tasks": False,
        "require_local_benchmark_tasks": False,
        "require_core25_reportable_results": False,
        "require_reportable_predictions": False,
        "allow_fixture_reportable_predictions": False,
        "strict": False,
        "out": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_minimal_coverage_base(root: Path, run_id: str) -> None:
    curated = root / "weights" / "curated_datasets_2026" / "runs" / run_id
    for name in coverage.REQUIRED_TRAIN_FILES:
        _write_jsonl(curated / "jsonl" / name, [{"text": name}])
    _write_json(curated / "manifests" / "curation_manifest.json", {"status": "passed"})
    _write_jsonl(curated / "raw" / "normalized_traces.jsonl", [{"text": "trace"}])

    external = root / "weights" / "external_datasets_2026" / "runs" / run_id
    _write_json(
        external / "manifests" / "external_dataset_manifest.json",
        {
            "status": "passed",
            "records": {"train": 3},
            "integrity_rewrite": {"status": "rewritten_clean"},
            "promotion_allowed": True,
            "promotion_index": {"status": "passed", "rows": 3},
        },
    )

    agentic = root / "weights" / "agentic_tool_training_2026" / "runs" / run_id
    counts = {name: 2 for name in coverage.REQUIRED_AGENTIC_EXPORTS}
    _write_json(agentic / "agentic_tool_training_manifest.json", {"status": "passed", "counts": counts})

    teacher = root / "weights" / "data_factory" / "runs" / "teacher_jobs" / run_id
    _write_jsonl(teacher / "all_jobs.jsonl", [{"job": 1}])

    rollouts = root / "weights" / "data_factory" / "teacher_rollouts" / run_id
    _write_jsonl(rollouts / "qwen36_agentic_math_code_tool.jsonl", [{"teacher": "qwen"}])

    _write_jsonl(root / "data" / "eval" / "reportable_2026" / "core_canary.jsonl", [{"task_id": "a"}])


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
    _write_json(
        external / "manifests" / "external_dataset_manifest.json",
        {
            "status": "passed",
            "records": {"train": 9},
            "integrity_rewrite": {"status": "rewritten_clean"},
            "promotion_allowed": True,
            "promotion_index": {"status": "passed", "rows": 9},
        },
    )

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


def test_coverage_validator_rejects_unpromoted_external_train_manifest(tmp_path: Path) -> None:
    run_id = "run_unpromoted_external"
    _write_minimal_coverage_base(tmp_path, run_id)
    external = tmp_path / "weights" / "external_datasets_2026" / "runs" / run_id
    _write_json(
        external / "manifests" / "external_dataset_manifest.json",
        {
            "status": "passed",
            "records": {"train": 9},
            "promotion_allowed": False,
        },
    )

    report = coverage.validate_coverage(_args(tmp_path, run_id, require_media_teacher_rollouts=False))

    labels = {item["label"] for item in report["missing"]}
    assert "external_train_promotion_allowed" in labels
    assert "external_train_integrity_rewrite" in labels
    assert "external_train_promotion_index" in labels


def test_coverage_validator_rejects_external_train_manifest_missing_integrity_or_index_evidence(tmp_path: Path) -> None:
    run_id = "run_external_missing_evidence"
    _write_minimal_coverage_base(tmp_path, run_id)
    external = tmp_path / "weights" / "external_datasets_2026" / "runs" / run_id
    _write_json(
        external / "manifests" / "external_dataset_manifest.json",
        {
            "status": "passed",
            "records": {"train": 7},
            "promotion_allowed": True,
        },
    )

    report = coverage.validate_coverage(_args(tmp_path, run_id, require_media_teacher_rollouts=False))

    labels = {item["label"] for item in report["missing"]}
    assert "external_train_promotion_allowed" not in labels
    assert "external_train_integrity_rewrite" in labels
    assert "external_train_promotion_index" in labels


def test_coverage_validator_accepts_real_pipeline_manifest_layout(tmp_path: Path) -> None:
    run_id = "run_streaming"
    root = tmp_path
    curated = root / "weights" / "curated_datasets_2026" / "runs" / run_id
    for name in coverage.REQUIRED_TRAIN_FILES:
        _write_jsonl(curated / "jsonl" / name, [{"text": name}])
    _write_json(curated / "manifests" / "curated_dataset_builder_manifest.json", {"status": "passed", "records": {"train": 10}})
    _write_jsonl(curated / "raw" / "normalized_traces.jsonl", [{"text": "trace"}])
    _write_json(
        curated / "agentic_tool_training_2026" / "manifests" / "posttraining_curation_manifest.json",
        {"counts": {"sft": 10, "reward": 10, "rlvr": 10, "safety_negative": 2}, "status": "passed"},
    )

    external = root / "weights" / "external_datasets_2026" / "runs" / "external_run"
    _write_json(
        external / "external_dataset_manifest.stdout.json",
        {
            "records": {"train": 99},
            "status": "passed",
            "integrity_rewrite": {"status": "rewritten_clean"},
            "promotion_allowed": True,
            "promotion_index": {"status": "passed", "rows": 99},
        },
    )

    teacher = root / "weights" / "data_factory" / "runs" / "teacher_jobs" / "teacher_run"
    _write_jsonl(teacher / "all_jobs.jsonl", [{"job": 1}])

    rollouts = root / "weights" / "data_factory" / "teacher_rollouts" / "teacher_run"
    _write_jsonl(rollouts / "qwen36_gpu1.jsonl", [{"teacher": "qwen"}])

    report = coverage.validate_coverage(
        _args(
            root,
            run_id,
            external_dir=str(external),
            teacher_job_dir=str(teacher),
            teacher_rollout_dir=str(rollouts),
            require_media_teacher_rollouts=False,
            require_reportable_tasks=False,
        )
    )

    assert report["status"] == "passed"
    assert report["counts"]["strict_local_normalized_traces"] == 1
    assert report["counts"]["agentic_exports"]["rlvr"] == 10
    assert report["counts"]["qwen36_agentic_math_code_tool_rollouts"] == 1


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


def test_coverage_validator_aggregates_multiple_benchmark_materializations(tmp_path: Path) -> None:
    local_manifest = tmp_path / "bench_a" / "manifests" / "benchmark_materialization_manifest.json"
    official_manifest = tmp_path / "bench_b" / "manifests" / "benchmark_materialization_manifest.json"
    _write_json(
        local_manifest,
        {
            "schema": "omnicoder.benchmark_materializer_2026.v1",
            "run_id": "bench_a",
            "rows": 4,
            "materialized": 1,
            "needs_data": 0,
            "records": [
                {
                    "benchmark_id": "agent_mcp_bench_2026",
                    "rows": 4,
                    "reportable": False,
                    "local_only": True,
                }
            ],
        },
    )
    _write_json(
        official_manifest,
        {
            "schema": "omnicoder.benchmark_materializer_2026.v1",
            "run_id": "bench_b",
            "rows": 2,
            "materialized": 1,
            "needs_data": 0,
            "records": [
                {
                    "benchmark_id": "reasoning_arc_agi3_2026",
                    "rows": 2,
                    "reportable": True,
                    "local_only": False,
                }
            ],
        },
    )

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            "run_multi",
            benchmark_materialization_manifest=[str(local_manifest), str(official_manifest)],
            require_reportable_tasks=False,
            require_media_teacher_rollouts=False,
            require_official_reportable_tasks=True,
            require_local_benchmark_tasks=True,
        )
    )

    assert report["counts"]["local_materialized_benchmark_rows"] == 4
    assert report["counts"]["official_materialized_benchmark_rows"] == 2
    assert report["counts"]["local_materialized_benchmark_tasks"]["agent_mcp_bench_2026"] == 4
    assert report["counts"]["official_materialized_benchmark_tasks"]["reasoning_arc_agi3_2026"] == 2
    assert len(report["manifests"]["benchmark_materializations"]) == 2
    labels = {item["label"] for item in report["missing"]}
    assert "official_materialized_reportable_tasks" not in labels
    assert "local_materialized_benchmark_tasks" not in labels


def test_coverage_validator_counts_reportable_tasks_from_materialization_root(tmp_path: Path) -> None:
    materialized = tmp_path / "weights" / "data_factory" / "runs" / "benchmark_materialization" / "bench_reportable"
    _write_jsonl(materialized / "reportable_2026" / "arc_agi3_authorized.jsonl", [{"task_id": "arc"}])
    _write_json(
        materialized / "manifests" / "benchmark_materialization_manifest.json",
        {
            "schema": "omnicoder.benchmark_materializer_2026.v1",
            "run_id": "bench_reportable",
            "rows": 1,
            "materialized": 1,
            "needs_data": 0,
            "records": [
                {
                    "benchmark_id": "reasoning_arc_agi3_2026",
                    "rows": 1,
                    "reportable": True,
                    "local_only": False,
                }
            ],
        },
    )

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            "run_reportable_root",
            benchmark_materialization_root=[str(materialized)],
            require_reportable_tasks=True,
            require_media_teacher_rollouts=False,
            require_official_reportable_tasks=True,
        )
    )

    assert report["counts"]["reportable_tasks"][str(materialized / "reportable_2026" / "arc_agi3_authorized.jsonl")] == 1
    assert report["counts"]["official_materialized_benchmark_rows"] == 1
    labels = {item["label"] for item in report["missing"]}
    assert "reportable_eval_tasks" not in labels


def test_coverage_validator_fails_when_official_rows_exist_without_reportable_scores(tmp_path: Path) -> None:
    run_id = "reportable_missing_scores"
    _write_minimal_coverage_base(tmp_path, run_id)
    profile = tmp_path / "profile.json"
    _write_json(profile, {"reportable_core_25": ["reasoning_arc_agi3_2026"]})
    manifest = tmp_path / "bench" / "manifests" / "benchmark_materialization_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "omnicoder.benchmark_materializer_2026.v1",
            "rows": 1,
            "records": [
                {
                    "benchmark_id": "reasoning_arc_agi3_2026",
                    "rows": 1,
                    "reportable": True,
                    "local_only": False,
                }
            ],
        },
    )

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            run_id,
            benchmark_profile=str(profile),
            benchmark_materialization_manifest=[str(manifest)],
            require_media_teacher_rollouts=False,
            require_core25_reportable_results=True,
            require_official_reportable_tasks=True,
        )
    )

    labels = {item["label"] for item in report["missing"]}
    assert "official_materialized_reportable_tasks" not in labels
    assert "core25_reportable_results" in labels
    assert "core25_reportable_reasoning_arc_agi3_2026" in labels


def test_coverage_validator_accepts_core25_reportable_results_with_prediction_artifact(tmp_path: Path) -> None:
    run_id = "reportable_full"
    _write_minimal_coverage_base(tmp_path, run_id)
    profile = tmp_path / "profile.json"
    _write_json(profile, {"reportable_core_25": ["reasoning_arc_agi3_2026"]})
    out_dir = tmp_path / "reportable"
    results = out_dir / "reportable_results.jsonl"
    predictions = out_dir / "predictions.jsonl"
    _write_jsonl(
        results,
        [
            {
                "benchmark_id": "reasoning_arc_agi3_2026",
                "mode": "reportable",
                "phase": "reportable_scoring",
                "status": "passed",
                "score_json": {"reportable_score": True, "contract_only": False, "task_count": 1},
                "metrics": {"reportable_task_count": 1},
            }
        ],
    )
    _write_json(
        out_dir / "reportable_summary.json",
        {"status": "ok", "results": str(results), "gate_decision": "passed", "reportable": 1, "failed": 0, "skipped": 0, "local_only": 0},
    )
    _write_jsonl(
        predictions,
        [
            {
                "schema": "omnicoder.reportable_prediction_2026.v1",
                "schema_version": "2026-05-24",
                "benchmark_id": "reasoning_arc_agi3_2026",
                "task_id": "arc-1",
                "model": "local-checkpoint",
                "backend": "checkpoint-runner",
                "prediction": "answer",
                "task_row_sha256": "abc",
                "task_file_sha256": "def",
                "prediction_id": "pred",
            }
        ],
    )
    _write_json(
        out_dir / "prediction_summary.json",
        {"status": "ok", "schema_version": "2026-05-24", "records": 1, "predictions": str(predictions)},
    )

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            run_id,
            benchmark_profile=str(profile),
            benchmark_reportable_summary=[str(out_dir / "reportable_summary.json")],
            reportable_prediction_summary=[str(out_dir / "prediction_summary.json")],
            require_media_teacher_rollouts=False,
            require_core25_reportable_results=True,
            require_reportable_predictions=True,
        )
    )

    assert report["status"] == "passed"
    assert report["missing"] == []
    assert report["counts"]["core25_reportable_results"]["valid_reportable_results"] == 1
    assert report["counts"]["reportable_predictions"]["backend_counts"] == {"checkpoint-runner": 1}


def test_coverage_validator_rejects_contract_only_or_local_only_reportable_result(tmp_path: Path) -> None:
    run_id = "reportable_contract_only"
    _write_minimal_coverage_base(tmp_path, run_id)
    profile = tmp_path / "profile.json"
    _write_json(profile, {"reportable_core_25": ["agent_terminal_bench_2_1_2026"]})
    results = tmp_path / "reportable_results.jsonl"
    _write_jsonl(
        results,
        [
            {
                "benchmark_id": "agent_terminal_bench_2_1_2026",
                "mode": "reportable",
                "phase": "reportable_scoring",
                "status": "local_only",
                "score_json": {"reportable_score": False, "contract_only": True, "task_count": 1},
                "metrics": {"reportable_task_count": 1},
            }
        ],
    )
    summary = tmp_path / "reportable_summary.json"
    _write_json(summary, {"status": "needs_data", "results": str(results), "gate_decision": "blocked"})

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            run_id,
            benchmark_profile=str(profile),
            benchmark_reportable_summary=[str(summary)],
            require_media_teacher_rollouts=False,
            require_core25_reportable_results=True,
        )
    )

    labels = {item["label"] for item in report["missing"]}
    assert "core25_reportable_agent_terminal_bench_2_1_2026" in labels
    assert "core25_reportable_summary_status" in labels
    assert "core25_reportable_summary_gate" in labels


def test_coverage_validator_rejects_fixture_reportable_predictions_by_default(tmp_path: Path) -> None:
    run_id = "reportable_fixture_predictions"
    _write_minimal_coverage_base(tmp_path, run_id)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "schema": "omnicoder.reportable_prediction_2026.v1",
                "schema_version": "2026-05-24",
                "benchmark_id": "reasoning_arc_agi3_2026",
                "task_id": "arc-1",
                "model": "fixture-local",
                "backend": "fixture",
                "prediction": "A",
                "task_row_sha256": "abc",
                "task_file_sha256": "def",
                "prediction_id": "pred",
            }
        ],
    )

    report = coverage.validate_coverage(
        _args(
            tmp_path,
            run_id,
            reportable_predictions=[str(predictions)],
            require_media_teacher_rollouts=False,
            require_reportable_predictions=True,
        )
    )

    labels = {item["label"] for item in report["missing"]}
    assert "reportable_prediction_backend" in labels

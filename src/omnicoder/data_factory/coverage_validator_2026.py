"""Run-scoped coverage checks for full 2026 Omnicoder training data.

This module is deliberately read-only. It distinguishes declared capability from
materialized trainable artifacts before a dataset run is promoted into the full
20B training pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TRAIN_FILES = (
    "train_text.jsonl",
    "train_code.jsonl",
    "train_tool.jsonl",
    "train_image.jsonl",
    "train_video.jsonl",
    "train_audio.jsonl",
    "train_music.jsonl",
    "train_long_context.jsonl",
    "train_agentic_focus.jsonl",
    "train_media_focus.jsonl",
)

REQUIRED_AGENTIC_EXPORTS = ("sft", "reward", "rlvr", "tool_rlvr")
OPTIONAL_AGENTIC_EXPORTS = ("preference", "safety_negative")
AGENTIC_EXPORT_ALIASES = {
    "sft": ("sft",),
    "reward": ("reward",),
    "rlvr": ("rlvr",),
    "tool_rlvr": ("tool_rlvr", "rlvr"),
}


def count_jsonl(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    rows = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def jsonl_counts(paths: list[Path]) -> dict[str, int]:
    return {path.name: count_jsonl(path) for path in sorted(paths)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def add_missing(missing: list[dict[str, Any]], label: str, path: Path, reason: str, count: int = 0) -> None:
    missing.append({"label": label, "path": str(path), "reason": reason, "count": count})


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def read_first_json(paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    path = first_existing(paths)
    return path, read_json(path)


def count_jsonl_candidates(paths: list[Path]) -> tuple[Path, int]:
    path = first_existing(paths)
    return path, count_jsonl(path)


def agentic_export_count(counts: dict[str, Any], name: str) -> int:
    aliases = AGENTIC_EXPORT_ALIASES.get(name, (name,))
    return max(int(counts.get(alias) or 0) for alias in aliases)


def coerce_path_values(value: Any) -> list[Path]:
    values: list[Any]
    if value in (None, "", [], ()):
        return []
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    paths: list[Path] = []
    seen: set[str] = set()
    for item in values:
        if item in (None, ""):
            continue
        path = Path(str(item))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def resolve_benchmark_materialization_manifests(args: argparse.Namespace, root: Path, run_id: str) -> list[Path]:
    manifests = coerce_path_values(getattr(args, "benchmark_materialization_manifest", ""))
    for materialization_root in coerce_path_values(getattr(args, "benchmark_materialization_root", "")):
        if materialization_root.is_file():
            manifests.append(materialization_root)
        else:
            manifests.append(materialization_root / "manifests" / "benchmark_materialization_manifest.json")
    if not manifests:
        manifests.append(
            root
            / "weights"
            / "data_factory"
            / "runs"
            / "benchmark_materialization"
            / run_id
            / "manifests"
            / "benchmark_materialization_manifest.json"
        )
    out: list[Path] = []
    seen: set[str] = set()
    for path in manifests:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def materialization_reportable_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    for materialization_root in coerce_path_values(getattr(args, "benchmark_materialization_root", "")):
        if materialization_root.is_dir():
            roots.append(materialization_root / "reportable_2026")
    return roots


def materialization_record_counts(manifests: list[Path]) -> tuple[dict[str, int], dict[str, int], int, int, list[str], list[dict[str, Any]]]:
    official_benchmark_counts: dict[str, int] = {}
    local_benchmark_counts: dict[str, int] = {}
    official_benchmark_rows = 0
    local_benchmark_rows = 0
    warnings: list[str] = []
    summaries: list[dict[str, Any]] = []
    for manifest in manifests:
        data = read_json(manifest)
        if not data:
            summaries.append({"path": str(manifest), "status": "missing_or_unreadable", "rows": 0})
            continue
        if data.get("schema") != "omnicoder.benchmark_materializer_2026.v1":
            warnings.append(f"benchmark materialization manifest schema is unrecognized: {manifest}")
        records = data.get("records") if isinstance(data.get("records"), list) else []
        summaries.append(
            {
                "path": str(manifest),
                "status": data.get("status") or ("materialized" if int(data.get("materialized") or 0) > 0 else "needs_data"),
                "run_id": data.get("run_id"),
                "mode": data.get("mode"),
                "rows": int(data.get("rows") or 0),
                "materialized": int(data.get("materialized") or 0),
                "needs_data": int(data.get("needs_data") or 0),
            }
        )
        for record in records:
            if not isinstance(record, dict):
                continue
            benchmark_id = str(record.get("benchmark_id") or "unknown")
            rows = int(record.get("rows") or 0)
            if bool(record.get("reportable")) and not bool(record.get("local_only")):
                official_benchmark_counts[benchmark_id] = official_benchmark_counts.get(benchmark_id, 0) + rows
                official_benchmark_rows += rows
            elif rows > 0:
                local_benchmark_counts[benchmark_id] = local_benchmark_counts.get(benchmark_id, 0) + rows
                local_benchmark_rows += rows
    return (
        official_benchmark_counts,
        local_benchmark_counts,
        official_benchmark_rows,
        local_benchmark_rows,
        warnings,
        summaries,
    )


def load_profile_core25(profile_path: Path) -> tuple[list[str], dict[str, Any]]:
    profile = read_json(profile_path)
    core = profile.get("reportable_core_25") if isinstance(profile.get("reportable_core_25"), list) else []
    return [str(item) for item in core if str(item).strip()], profile


def resolve_reportable_result_artifacts(args: argparse.Namespace, root: Path, run_id: str) -> tuple[list[Path], list[Path]]:
    summaries = coerce_path_values(getattr(args, "benchmark_reportable_summary", ""))
    results = coerce_path_values(getattr(args, "benchmark_reportable_results", ""))
    default_dir = root / "weights" / "data_factory" / "runs" / "benchmark_reportable" / run_id
    if not summaries:
        summaries.append(default_dir / "reportable_summary.json")
    for summary in list(summaries):
        data = read_json(summary)
        result_ref = data.get("results") if isinstance(data, dict) else None
        if result_ref:
            path = Path(str(result_ref))
            results.append(path if path.is_absolute() else summary.parent / path)
    if not results:
        results.append(default_dir / "reportable_results.jsonl")
    return summaries, results


def valid_reportable_result(row: dict[str, Any], min_tasks: int) -> bool:
    score_json = row.get("score_json") if isinstance(row.get("score_json"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    reportable_task_count = int(metrics.get("reportable_task_count") or score_json.get("task_count") or 0)
    return (
        row.get("mode") == "reportable"
        and row.get("phase") == "reportable_scoring"
        and row.get("status") == "passed"
        and bool(score_json.get("reportable_score"))
        and not bool(score_json.get("contract_only"))
        and reportable_task_count >= min_tasks
    )


def validate_reportable_gate_results(
    args: argparse.Namespace,
    root: Path,
    run_id: str,
    missing: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    profile_path = Path(getattr(args, "benchmark_profile", "") or root / "profiles" / "benchmark_suite_2026.json")
    core25, _profile = load_profile_core25(profile_path)
    summaries, result_paths = resolve_reportable_result_artifacts(args, root, run_id)
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        rows.extend(read_jsonl(path))
    valid_by_benchmark: dict[str, int] = {}
    min_tasks = int(getattr(args, "min_reportable_tasks", 1) or 1)
    for row in rows:
        benchmark_id = str(row.get("benchmark_id") or "")
        if benchmark_id and valid_reportable_result(row, min_tasks):
            valid_by_benchmark[benchmark_id] = valid_by_benchmark.get(benchmark_id, 0) + 1
    summary_data = [read_json(path) for path in summaries if path.exists()]
    if not summary_data:
        add_missing(missing, "core25_reportable_summary", summaries[0], "missing_or_unreadable_reportable_summary", 0)
    if not core25:
        add_missing(missing, "core25_reportable_profile", profile_path, "missing_or_empty_reportable_core_25", 0)
    if not rows:
        add_missing(missing, "core25_reportable_results", result_paths[0], "missing_or_empty_reportable_results", 0)
    missing_core = [benchmark_id for benchmark_id in core25 if valid_by_benchmark.get(benchmark_id, 0) <= 0]
    for benchmark_id in missing_core:
        add_missing(missing, f"core25_reportable_{benchmark_id}", result_paths[0], "missing_passed_reportable_scoring_result", 0)
    for summary in summary_data:
        status = summary.get("status")
        gate_decision = summary.get("gate_decision")
        if status != "ok":
            add_missing(missing, "core25_reportable_summary_status", summaries[0], f"reportable_summary_status_{status}", 0)
        if gate_decision != "passed":
            add_missing(missing, "core25_reportable_summary_gate", summaries[0], f"reportable_summary_gate_{gate_decision}", 0)
        if int(summary.get("reportable") or 0) <= 0:
            add_missing(missing, "core25_reportable_summary_count", summaries[0], "missing_positive_reportable_count", 0)
        for key in ("failed", "skipped", "local_only"):
            count = int(summary.get(key) or 0)
            if count:
                add_missing(missing, f"core25_reportable_summary_{key}", summaries[0], f"nonzero_{key}", count)
    return {
        "profile": str(profile_path),
        "summary_paths": [str(path) for path in summaries],
        "result_paths": [str(path) for path in result_paths],
        "result_rows": len(rows),
        "valid_reportable_results": sum(valid_by_benchmark.values()),
        "core25_count": len(core25),
        "core25_missing": missing_core,
    }


def resolve_prediction_artifacts(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    summaries = coerce_path_values(getattr(args, "reportable_prediction_summary", ""))
    predictions = coerce_path_values(getattr(args, "reportable_predictions", ""))
    for summary in list(summaries):
        data = read_json(summary)
        prediction_ref = data.get("predictions") if isinstance(data, dict) else None
        if prediction_ref:
            path = Path(str(prediction_ref))
            predictions.append(path if path.is_absolute() else summary.parent / path)
    return summaries, predictions


def validate_prediction_artifacts(
    args: argparse.Namespace,
    missing: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    summaries, prediction_paths = resolve_prediction_artifacts(args)
    rows: list[dict[str, Any]] = []
    backend_counts: dict[str, int] = {}
    bad_rows = 0
    fixture_rows = 0
    for path in prediction_paths:
        for row in read_jsonl(path):
            rows.append(row)
            backend = str(row.get("backend") or "")
            if backend:
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if backend == "fixture":
                fixture_rows += 1
            if (
                row.get("schema") != "omnicoder.reportable_prediction_2026.v1"
                or not row.get("task_row_sha256")
                or not row.get("task_file_sha256")
                or not row.get("prediction_id")
            ):
                bad_rows += 1
    if not rows:
        add_missing(missing, "reportable_predictions", prediction_paths[0] if prediction_paths else Path(""), "missing_or_empty_predictions", 0)
    if bad_rows:
        add_missing(missing, "reportable_prediction_provenance", prediction_paths[0], "missing_schema_or_task_hashes", bad_rows)
    if fixture_rows and not bool(getattr(args, "allow_fixture_reportable_predictions", False)):
        add_missing(missing, "reportable_prediction_backend", prediction_paths[0], "fixture_backend_not_allowed_for_reportable_gate", fixture_rows)
    for summary in summaries:
        data = read_json(summary)
        if data and data.get("status") not in {None, "ok", "passed"}:
            warnings.append(f"prediction summary status is {data.get('status')}")
    return {
        "summary_paths": [str(path) for path in summaries],
        "prediction_paths": [str(path) for path in prediction_paths],
        "rows": len(rows),
        "bad_rows": bad_rows,
        "backend_counts": backend_counts,
    }


def validate_coverage(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    run_id = str(args.run_id)
    curated_dir = Path(args.curated_dir) if args.curated_dir else root / "weights" / "curated_datasets_2026" / "runs" / run_id
    local_trace_dir = Path(args.local_trace_dir) if args.local_trace_dir else root / "weights" / "curated_datasets_2026" / "runs" / f"{run_id}_local_traces"
    external_dir = Path(args.external_dir) if args.external_dir else root / "weights" / "external_datasets_2026" / "runs" / run_id
    default_agentic_dir = root / "weights" / "agentic_tool_training_2026" / "runs" / run_id
    curated_agentic_dir = curated_dir / "agentic_tool_training_2026"
    agentic_dir = Path(args.agentic_dir) if args.agentic_dir else (curated_agentic_dir if curated_agentic_dir.exists() else default_agentic_dir)
    teacher_job_dir = Path(args.teacher_job_dir) if args.teacher_job_dir else root / "weights" / "data_factory" / "runs" / "teacher_jobs" / run_id
    teacher_rollout_dir = Path(args.teacher_rollout_dir) if args.teacher_rollout_dir else root / "weights" / "data_factory" / "teacher_rollouts" / run_id
    mixture_plan = Path(args.mixture_plan) if args.mixture_plan else root / "weights" / "training_orchestration_2026" / "runs" / run_id / "manifests" / "mixture_plan.json"
    reportable_root = Path(args.reportable_root) if args.reportable_root else root / "data" / "eval" / "reportable_2026"
    benchmark_manifests = resolve_benchmark_materialization_manifests(args, root, run_id)
    missing: list[dict[str, Any]] = []
    warnings: list[str] = []

    curated_manifest, curated_manifest_data = read_first_json(
        [
            curated_dir / "manifests" / "curation_manifest.json",
            curated_dir / "manifests" / "curated_dataset_builder_manifest.json",
            curated_dir / "latest_manifest.json",
        ]
    )
    if not curated_manifest_data:
        add_missing(missing, "curated_manifest", curated_manifest, "missing_or_unreadable")

    train_counts: dict[str, int] = {}
    for name in REQUIRED_TRAIN_FILES:
        path = curated_dir / "jsonl" / name
        rows = count_jsonl(path)
        train_counts[name] = rows
        if rows <= 0:
            add_missing(missing, f"curated_{name}", path, "missing_or_empty", rows)

    normalized_traces = curated_dir / "raw" / "normalized_traces.jsonl"
    normalized_trace_rows = count_jsonl(normalized_traces)
    if normalized_trace_rows <= 0:
        add_missing(missing, "curated_normalized_traces", normalized_traces, "missing_or_empty", normalized_trace_rows)

    local_trace_candidates = [local_trace_dir / "raw" / "normalized_traces.jsonl"]
    if not args.local_trace_dir:
        local_trace_candidates.append(normalized_traces)
    local_normalized_traces, local_trace_rows = count_jsonl_candidates(local_trace_candidates)
    if local_trace_rows <= 0:
        add_missing(missing, "strict_local_normalized_traces", local_normalized_traces, "missing_or_empty", local_trace_rows)

    external_manifest, external_manifest_data = read_first_json(
        [
            external_dir / "manifests" / "external_dataset_manifest.json",
            external_dir / "external_dataset_manifest.stdout.json",
        ]
    )
    external_train_rows = int(((external_manifest_data.get("records") or {}) if external_manifest_data else {}).get("train") or 0)
    if external_train_rows <= 0:
        add_missing(missing, "external_train_records", external_manifest, "missing_or_zero_train_records", external_train_rows)
    if external_manifest_data and external_manifest_data.get("status") not in {None, "passed", "ok"}:
        warnings.append(f"external manifest status is {external_manifest_data.get('status')}")

    agentic_manifest, agentic_manifest_data = read_first_json(
        [
            agentic_dir / "agentic_tool_training_manifest.json",
            agentic_dir / "manifests" / "posttraining_curation_manifest.json",
        ]
    )
    agentic_counts = agentic_manifest_data.get("counts") if isinstance(agentic_manifest_data.get("counts"), dict) else {}
    for name in REQUIRED_AGENTIC_EXPORTS:
        rows = agentic_export_count(agentic_counts, name)
        if rows <= 0:
            add_missing(missing, f"agentic_{name}", agentic_manifest, "missing_or_zero_export_count", rows)

    after_teacher_manifest, after_teacher_data = read_first_json(
        [
            agentic_dir / "after_teacher" / "agentic_tool_training_manifest.json",
            agentic_dir / "after_teacher" / "manifests" / "posttraining_curation_manifest.json",
        ]
    )
    after_teacher_counts = after_teacher_data.get("counts") if isinstance(after_teacher_data.get("counts"), dict) else {}

    all_jobs = teacher_job_dir / "all_jobs.jsonl"
    all_job_rows = count_jsonl(all_jobs)
    if all_job_rows <= 0:
        add_missing(missing, "teacher_jobs_all", all_jobs, "missing_or_empty", all_job_rows)

    modality_job_dir = teacher_job_dir / "modality"
    modality_manifest = modality_job_dir / "modality_teacher_jobs_manifest.json"
    modality_manifest_data = read_json(modality_manifest)
    modality_job_counts = jsonl_counts(list(modality_job_dir.glob("*_jobs.jsonl"))) if modality_job_dir.exists() else {}
    modality_combined = modality_job_dir / "all_modality_teacher_jobs.jsonl"
    modality_combined_rows = count_jsonl(modality_combined)
    if getattr(args, "require_modality_teacher_jobs", False) and modality_combined_rows <= 0:
        add_missing(missing, "modality_teacher_jobs_all", modality_combined, "missing_or_empty", modality_combined_rows)

    teacher_rollout_manifest = teacher_rollout_dir / "teacher_rollout_manifest.json"
    teacher_rollout_data = read_json(teacher_rollout_manifest)
    qwen_rollout_candidates = [teacher_rollout_dir / "qwen36_agentic_math_code_tool.jsonl"]
    qwen_rollout_candidates.extend(sorted(teacher_rollout_dir.glob("qwen36*.jsonl")) if teacher_rollout_dir.exists() else [])
    seen_qwen_rollouts: set[str] = set()
    qwen_rollout_rows = 0
    for candidate in qwen_rollout_candidates:
        key = str(candidate)
        if key in seen_qwen_rollouts:
            continue
        seen_qwen_rollouts.add(key)
        qwen_rollout_rows += count_jsonl(candidate)
    qwen_rollout = first_existing(qwen_rollout_candidates)
    if qwen_rollout_rows <= 0:
        add_missing(missing, "qwen36_agentic_math_code_tool_rollouts", qwen_rollout, "missing_or_empty", qwen_rollout_rows)
    media_rollout_candidates = [
        teacher_rollout_dir / "comfyui_modality_teacher_rollouts.jsonl",
        teacher_rollout_dir / "media_teacher_rollouts.jsonl",
        teacher_rollout_dir / "qwen_image_rollouts.jsonl",
        teacher_rollout_dir / "ltx_video_rollouts.jsonl",
        teacher_rollout_dir / "ace_music_rollouts.jsonl",
    ]
    media_rollout_counts = {path.name: count_jsonl(path) for path in media_rollout_candidates}
    if args.require_media_teacher_rollouts and not any(rows > 0 for rows in media_rollout_counts.values()):
        add_missing(missing, "media_teacher_rollouts", teacher_rollout_dir, "missing_qwen_ltx_ace_rollout_rows", 0)

    mixture_data = read_json(mixture_plan)
    if getattr(args, "require_mixture_plan", False) and not mixture_data:
        add_missing(missing, "mixture_plan", mixture_plan, "missing_or_unreadable")

    reportable_paths = []
    reportable_roots = [reportable_root] + materialization_reportable_roots(args)
    seen_reportable: set[str] = set()
    for candidate_root in reportable_roots:
        if not candidate_root.exists():
            continue
        if candidate_root.is_dir():
            candidates = sorted(candidate_root.glob("*.jsonl"))
        else:
            candidates = [candidate_root]
        for candidate in candidates:
            key = str(candidate)
            if key in seen_reportable:
                continue
            seen_reportable.add(key)
            reportable_paths.append(candidate)
    reportable_counts = {str(path): count_jsonl(path) for path in reportable_paths}
    if args.require_reportable_tasks and sum(reportable_counts.values()) <= 0:
        add_missing(missing, "reportable_eval_tasks", reportable_root, "missing_or_empty", 0)

    (
        official_benchmark_counts,
        local_benchmark_counts,
        official_benchmark_rows,
        local_benchmark_rows,
        benchmark_warnings,
        benchmark_summaries,
    ) = materialization_record_counts(benchmark_manifests)
    warnings.extend(benchmark_warnings)
    if getattr(args, "require_official_reportable_tasks", False) and official_benchmark_rows <= 0:
        add_missing(missing, "official_materialized_reportable_tasks", benchmark_manifests[0], "missing_or_zero_official_rows", official_benchmark_rows)
    if getattr(args, "require_local_benchmark_tasks", False) and local_benchmark_rows <= 0:
        add_missing(missing, "local_materialized_benchmark_tasks", benchmark_manifests[0], "missing_or_zero_local_rows", local_benchmark_rows)

    reportable_gate: dict[str, Any] = {}
    if getattr(args, "require_core25_reportable_results", False):
        reportable_gate = validate_reportable_gate_results(args, root, run_id, missing, warnings)

    prediction_gate: dict[str, Any] = {}
    if getattr(args, "require_reportable_predictions", False):
        prediction_gate = validate_prediction_artifacts(args, missing, warnings)

    status = "passed" if not missing else "needs_data"
    return {
        "schema": "omnicoder.dataset_coverage_validator_2026.v1",
        "run_id": run_id,
        "status": status,
        "strict": bool(args.strict),
        "root": str(root),
        "paths": {
            "curated_dir": str(curated_dir),
            "local_trace_dir": str(local_trace_dir),
            "external_dir": str(external_dir),
            "agentic_dir": str(agentic_dir),
            "teacher_job_dir": str(teacher_job_dir),
            "teacher_rollout_dir": str(teacher_rollout_dir),
            "mixture_plan": str(mixture_plan),
            "reportable_root": str(reportable_root),
            "reportable_roots": [str(path) for path in reportable_roots],
            "benchmark_materialization_manifest": str(benchmark_manifests[0]) if benchmark_manifests else "",
            "benchmark_materialization_manifests": [str(path) for path in benchmark_manifests],
        },
        "counts": {
            "curated_train_files": train_counts,
            "curated_normalized_traces": normalized_trace_rows,
            "strict_local_normalized_traces": local_trace_rows,
            "external_train": external_train_rows,
            "agentic_exports": agentic_counts,
            "agentic_after_teacher_exports": after_teacher_counts,
            "teacher_jobs_all": all_job_rows,
            "modality_teacher_jobs": modality_job_counts,
            "modality_teacher_jobs_all": modality_combined_rows,
            "qwen36_agentic_math_code_tool_rollouts": qwen_rollout_rows,
            "media_teacher_rollouts": media_rollout_counts,
            "reportable_tasks": reportable_counts,
            "official_materialized_benchmark_tasks": official_benchmark_counts,
            "local_materialized_benchmark_tasks": local_benchmark_counts,
            "official_materialized_benchmark_rows": official_benchmark_rows,
            "local_materialized_benchmark_rows": local_benchmark_rows,
            "core25_reportable_results": reportable_gate,
            "reportable_predictions": prediction_gate,
        },
        "manifests": {
            "curated_status": curated_manifest_data.get("status"),
            "external_status": external_manifest_data.get("status"),
            "agentic_status": agentic_manifest_data.get("status"),
            "after_teacher_status": after_teacher_data.get("status"),
            "modality_teacher_status": modality_manifest_data.get("status"),
            "teacher_rollout_status": teacher_rollout_data.get("status"),
            "mixture_status": mixture_data.get("status"),
            "benchmark_materializations": benchmark_summaries,
            "benchmark_materialization_schema": "omnicoder.benchmark_materializer_2026.v1" if benchmark_summaries else None,
            "benchmark_materialization_rows": sum(int(item.get("rows") or 0) for item in benchmark_summaries),
            "reportable_gate": reportable_gate,
            "prediction_gate": prediction_gate,
        },
        "missing": missing,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate materialized dataset, teacher, and benchmark coverage for one Omnicoder 2026 run")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--curated-dir", default="")
    parser.add_argument("--local-trace-dir", default="")
    parser.add_argument("--external-dir", default="")
    parser.add_argument("--agentic-dir", default="")
    parser.add_argument("--teacher-job-dir", default="")
    parser.add_argument("--teacher-rollout-dir", default="")
    parser.add_argument("--mixture-plan", default="")
    parser.add_argument("--reportable-root", default="")
    parser.add_argument("--benchmark-materialization-root", action="append", default=[])
    parser.add_argument("--benchmark-materialization-manifest", action="append", default=[])
    parser.add_argument("--benchmark-profile", default="")
    parser.add_argument("--benchmark-reportable-summary", action="append", default=[])
    parser.add_argument("--benchmark-reportable-results", action="append", default=[])
    parser.add_argument("--reportable-prediction-summary", action="append", default=[])
    parser.add_argument("--reportable-predictions", action="append", default=[])
    parser.add_argument("--min-reportable-tasks", type=int, default=1)
    parser.add_argument("--require-media-teacher-rollouts", action="store_true")
    parser.add_argument("--require-modality-teacher-jobs", action="store_true")
    parser.add_argument("--require-mixture-plan", action="store_true")
    parser.add_argument("--require-reportable-tasks", action="store_true")
    parser.add_argument("--require-official-reportable-tasks", action="store_true")
    parser.add_argument("--require-local-benchmark-tasks", action="store_true")
    parser.add_argument("--require-core25-reportable-results", action="store_true")
    parser.add_argument("--require-reportable-predictions", action="store_true")
    parser.add_argument("--allow-fixture-reportable-predictions", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when required coverage is missing")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = validate_coverage(args)
    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.strict and report["status"] != "passed":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

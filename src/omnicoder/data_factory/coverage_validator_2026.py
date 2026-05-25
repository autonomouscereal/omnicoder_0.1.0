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

REQUIRED_AGENTIC_EXPORTS = ("sft", "reward", "preference", "rlvr", "tool_rlvr")


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


def add_missing(missing: list[dict[str, Any]], label: str, path: Path, reason: str, count: int = 0) -> None:
    missing.append({"label": label, "path": str(path), "reason": reason, "count": count})


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


def validate_coverage(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    run_id = str(args.run_id)
    curated_dir = Path(args.curated_dir) if args.curated_dir else root / "weights" / "curated_datasets_2026" / "runs" / run_id
    local_trace_dir = Path(args.local_trace_dir) if args.local_trace_dir else root / "weights" / "curated_datasets_2026" / "runs" / f"{run_id}_local_traces"
    external_dir = Path(args.external_dir) if args.external_dir else root / "weights" / "external_datasets_2026" / "runs" / run_id
    agentic_dir = Path(args.agentic_dir) if args.agentic_dir else root / "weights" / "agentic_tool_training_2026" / "runs" / run_id
    teacher_job_dir = Path(args.teacher_job_dir) if args.teacher_job_dir else root / "weights" / "data_factory" / "runs" / "teacher_jobs" / run_id
    teacher_rollout_dir = Path(args.teacher_rollout_dir) if args.teacher_rollout_dir else root / "weights" / "data_factory" / "teacher_rollouts" / run_id
    mixture_plan = Path(args.mixture_plan) if args.mixture_plan else root / "weights" / "training_orchestration_2026" / "runs" / run_id / "manifests" / "mixture_plan.json"
    reportable_root = Path(args.reportable_root) if args.reportable_root else root / "data" / "eval" / "reportable_2026"
    benchmark_manifests = resolve_benchmark_materialization_manifests(args, root, run_id)
    missing: list[dict[str, Any]] = []
    warnings: list[str] = []

    curated_manifest = curated_dir / "manifests" / "curation_manifest.json"
    curated_manifest_data = read_json(curated_manifest)
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

    local_normalized_traces = local_trace_dir / "raw" / "normalized_traces.jsonl"
    local_trace_rows = count_jsonl(local_normalized_traces)
    if local_trace_rows <= 0:
        add_missing(missing, "strict_local_normalized_traces", local_normalized_traces, "missing_or_empty", local_trace_rows)

    external_manifest = external_dir / "manifests" / "external_dataset_manifest.json"
    external_manifest_data = read_json(external_manifest)
    external_train_rows = int(((external_manifest_data.get("records") or {}) if external_manifest_data else {}).get("train") or 0)
    if external_train_rows <= 0:
        add_missing(missing, "external_train_records", external_manifest, "missing_or_zero_train_records", external_train_rows)
    if external_manifest_data and external_manifest_data.get("status") not in {None, "passed", "ok"}:
        warnings.append(f"external manifest status is {external_manifest_data.get('status')}")

    agentic_manifest = agentic_dir / "agentic_tool_training_manifest.json"
    agentic_manifest_data = read_json(agentic_manifest)
    agentic_counts = agentic_manifest_data.get("counts") if isinstance(agentic_manifest_data.get("counts"), dict) else {}
    for name in REQUIRED_AGENTIC_EXPORTS:
        rows = int(agentic_counts.get(name) or 0)
        if rows <= 0:
            add_missing(missing, f"agentic_{name}", agentic_manifest, "missing_or_zero_export_count", rows)

    after_teacher_manifest = agentic_dir / "after_teacher" / "agentic_tool_training_manifest.json"
    after_teacher_data = read_json(after_teacher_manifest)
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
    if modality_combined_rows <= 0:
        add_missing(missing, "modality_teacher_jobs_all", modality_combined, "missing_or_empty", modality_combined_rows)

    teacher_rollout_manifest = teacher_rollout_dir / "teacher_rollout_manifest.json"
    teacher_rollout_data = read_json(teacher_rollout_manifest)
    qwen_rollout = teacher_rollout_dir / "qwen36_agentic_math_code_tool.jsonl"
    qwen_rollout_rows = count_jsonl(qwen_rollout)
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
    if not mixture_data:
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
    parser.add_argument("--require-media-teacher-rollouts", action="store_true")
    parser.add_argument("--require-reportable-tasks", action="store_true")
    parser.add_argument("--require-official-reportable-tasks", action="store_true")
    parser.add_argument("--require-local-benchmark-tasks", action="store_true")
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

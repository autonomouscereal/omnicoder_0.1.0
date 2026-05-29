from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "omnicoder.eval_readiness_audit_2026.v1"
DEFAULT_PROFILE = "profiles/benchmark_suite_2026.json"
DEFAULT_MATERIALIZATION_ROOT = "weights/data_factory/runs/benchmark_materialization"

REQUIRED_GROUPS: dict[str, tuple[str, ...]] = {
    "hellaswag": ("reasoning_hellaswag_full_2026",),
    "arc_agieval_reasoning": ("reasoning_arc_agi2_2026", "reasoning_arc_agi3_2026"),
    "math_reasoning": (
        "reasoning_frontiermath_2026",
        "reasoning_matharena_2026",
        "reasoning_rlvr_linearity_math_2026",
        "reasoning_livebench_math_2026",
    ),
    "code_agentic": (
        "coding_swe_bench_live_2026",
        "coding_swe_bench_pro_2026",
        "agent_bfcl_v4_2026",
        "agent_terminal_bench_2026",
    ),
    "ocr_image_video_audio_music": (
        "multimodal_ocrbench_v2_2026",
        "generation_image_edit_2026",
        "generation_video_2026",
        "generation_audio_speech_2026",
        "generation_music_2026",
    ),
    "long_context": (
        "long_context_ruler_infinitebench_2026",
        "long_context_longbench_v2_2026",
        "long_context_nolima_1m_2026",
    ),
}

DIAGNOSTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "heldout_sample_loss": ("*sample*loss*.json",),
    "target_token_coverage": ("*target*coverage*.json", "*target*diagnostic*.json"),
    "decode_sanity": ("*decode*sanity*.json",),
    "topk_probe": ("*topk*.json",),
    "checkpoint_readiness": ("*readiness*.json",),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"_read_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_read_error": "json_not_object"}


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def count_jsonl(path: Path, *, stop_after: int = 0) -> int:
    try:
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    count += 1
                if stop_after and count >= stop_after:
                    break
        return count
    except Exception:
        return 0


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def safe_rglob(root: Path, pattern: str, *, max_files: int = 500) -> list[Path]:
    if not root.exists():
        return []
    found: list[Path] = []
    try:
        iterator = root.rglob(pattern)
        for path in iterator:
            if path.is_file():
                found.append(path)
                if len(found) >= int(max_files):
                    break
    except Exception:
        return found
    return sorted(found)


def load_profile(repo_root: Path, profile: str | Path = DEFAULT_PROFILE) -> dict[str, Any]:
    path = resolve_path(repo_root, profile)
    payload = read_json(path)
    payload["_profile_path"] = str(path)
    return payload


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    records = [row for row in profile.get("benchmarks") or [] if isinstance(row, dict)]
    ids = {str(row.get("benchmark_id") or "") for row in records}
    roots = profile.get("reportable_task_roots") if isinstance(profile.get("reportable_task_roots"), dict) else {}
    snapshots = profile.get("reportable_snapshots") if isinstance(profile.get("reportable_snapshots"), dict) else {}
    groups: dict[str, dict[str, Any]] = {}
    for group, required in REQUIRED_GROUPS.items():
        present = [item for item in required if item in ids]
        groups[group] = {
            "required_ids": list(required),
            "declared_ids": present,
            "missing_ids": [item for item in required if item not in ids],
            "status": "declared" if present else "missing",
        }
    return {
        "profile_path": profile.get("_profile_path"),
        "benchmark_count": len(records),
        "axes": dict(Counter(str(row.get("axis") or "unknown") for row in records)),
        "reportable_task_root_count": len(roots),
        "reportable_snapshot_count": len(snapshots),
        "reportable_core_25_count": len(profile.get("reportable_core_25") or []),
        "required_groups": groups,
    }


def reportable_roots_summary(repo_root: Path, profile: dict[str, Any]) -> dict[str, Any]:
    roots = profile.get("reportable_task_roots") if isinstance(profile.get("reportable_task_roots"), dict) else {}
    checked: list[dict[str, Any]] = []
    missing: list[str] = []
    nonempty = 0
    for benchmark_id, values in sorted(roots.items()):
        if isinstance(values, (str, Path)):
            candidates = [str(values)]
        elif isinstance(values, list):
            candidates = [str(item) for item in values]
        else:
            candidates = []
        for value in candidates:
            path = resolve_path(repo_root, value)
            rows = count_jsonl(path, stop_after=2) if path.exists() and path.is_file() else 0
            exists = path.exists() and path.is_file()
            if exists and rows > 0:
                nonempty += 1
            else:
                missing.append(f"{benchmark_id}:{value}")
            checked.append(
                {
                    "benchmark_id": benchmark_id,
                    "path": str(path),
                    "exists": exists,
                    "nonempty": rows > 0,
                    "sampled_rows": rows,
                }
            )
    status = "passed" if checked and nonempty == len(checked) else "failed"
    return {
        "status": status,
        "declared_files": len(checked),
        "nonempty_files": nonempty,
        "missing_or_empty": missing[:100],
        "missing_or_empty_count": len(missing),
        "examples": checked[:20],
    }


def materialization_summary(materialization_root: Path) -> dict[str, Any]:
    manifests = safe_rglob(materialization_root, "benchmark_materialization_manifest.json", max_files=200)
    rows_by_id: dict[str, int] = defaultdict(int)
    latest_by_id: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        payload = read_json(manifest_path)
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        manifest_total = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            benchmark_id = str(record.get("benchmark_id") or "")
            rows = int(record.get("rows") or record.get("task_count") or 0)
            status = str(record.get("status") or "unknown")
            status_counts[status] += 1
            manifest_total += rows
            if benchmark_id and rows > 0:
                rows_by_id[benchmark_id] += rows
                latest_by_id[benchmark_id] = str(manifest_path)
        manifest_rows.append(
            {
                "path": str(manifest_path),
                "record_count": len(records),
                "rows": manifest_total,
                "status_counts": dict(Counter(str(r.get("status") or "unknown") for r in records if isinstance(r, dict))),
            }
        )
    groups: dict[str, dict[str, Any]] = {}
    for group, required in REQUIRED_GROUPS.items():
        group_rows = {benchmark_id: rows_by_id.get(benchmark_id, 0) for benchmark_id in required}
        groups[group] = {
            "diagnostic_public_dev_rows": group_rows,
            "total_rows": sum(group_rows.values()),
            "status": "diagnostic_materialized" if any(group_rows.values()) else "missing",
        }
    return {
        "root": str(materialization_root),
        "manifest_count": len(manifests),
        "materialized_benchmark_count": sum(1 for value in rows_by_id.values() if value > 0),
        "status_counts": dict(status_counts),
        "required_groups": groups,
        "latest_manifest_by_benchmark": latest_by_id,
        "recent_manifests": manifest_rows[-20:],
    }


def _finite_positive_loss(payload: dict[str, Any]) -> bool:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    loss = overall.get("avg_loss", overall.get("loss"))
    tokens = overall.get("tokens")
    try:
        loss_f = float(loss)
        tokens_i = int(tokens)
    except Exception:
        return False
    return math.isfinite(loss_f) and loss_f > 0.0 and tokens_i > 0


def diagnostic_artifact_summary(roots: list[Path]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for label, patterns in DIAGNOSTIC_PATTERNS.items():
        files: list[Path] = []
        for root in roots:
            for pattern in patterns:
                files.extend(safe_rglob(root, pattern, max_files=100))
        deduped = sorted({str(path): path for path in files}.values())
        examples: list[dict[str, Any]] = []
        usable = 0
        for path in deduped[-20:]:
            payload = read_json(path)
            status = str(payload.get("status") or "").lower()
            ok_status = status in {"", "ok", "passed", "pass", "success"}
            if label == "heldout_sample_loss":
                ok = _finite_positive_loss(payload)
            elif label == "decode_sanity":
                ok = ok_status and not payload.get("reasons") and payload.get("_read_error") is None
            else:
                ok = ok_status and payload.get("_read_error") is None
            usable += int(bool(ok))
            examples.append(
                {
                    "path": str(path),
                    "schema": payload.get("schema"),
                    "status": payload.get("status"),
                    "usable": bool(ok),
                    "overall": payload.get("overall"),
                    "reasons": payload.get("reasons"),
                }
            )
        diagnostics[label] = {
            "files": len(deduped),
            "usable_recent_files": usable,
            "status": "passed" if usable > 0 else "missing",
            "recent_examples": examples,
        }
    return diagnostics


def reportable_score_summary(roots: list[Path]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    official = 0
    reportable = 0
    for root in roots:
        for path in safe_rglob(root, "reportable_summary.json", max_files=200):
            payload = read_json(path)
            official_count = int(payload.get("official") or 0)
            reportable_count = int(payload.get("reportable") or 0)
            status = str(payload.get("status") or "")
            official += official_count
            reportable += reportable_count
            summaries.append(
                {
                    "path": str(path),
                    "status": status,
                    "gate_decision": payload.get("gate_decision"),
                    "reportable": reportable_count,
                    "official": official_count,
                    "local_only": payload.get("local_only"),
                    "contract_only": payload.get("contract_only"),
                    "failed": payload.get("failed"),
                }
            )
    return {
        "status": "passed" if official > 0 else "failed",
        "official_summary_count": official,
        "reportable_summary_count": reportable,
        "summaries": summaries[-30:],
    }


def build_audit(
    *,
    repo_root: Path,
    weights_root: Path,
    profile_path: str | Path = DEFAULT_PROFILE,
    materialization_root: Path | None = None,
    diagnostic_roots: list[Path] | None = None,
    score_roots: list[Path] | None = None,
) -> dict[str, Any]:
    profile = load_profile(repo_root, profile_path)
    materialization_root = materialization_root or resolve_path(repo_root, DEFAULT_MATERIALIZATION_ROOT)
    diagnostics = diagnostic_artifact_summary(diagnostic_roots or [weights_root])
    reportable_files = reportable_roots_summary(repo_root, profile)
    reportable_scores = reportable_score_summary(score_roots or [weights_root])
    materialized = materialization_summary(materialization_root)

    blockers: list[str] = []
    if reportable_files["status"] != "passed":
        blockers.append("declared_reportable_task_roots_missing_or_empty")
    if reportable_scores["status"] != "passed":
        blockers.append("no_official_reportable_scorer_results")
    for label in ("heldout_sample_loss", "target_token_coverage", "decode_sanity"):
        if diagnostics[label]["status"] != "passed":
            blockers.append(f"{label}_missing_or_unusable")
    for group, group_payload in profile_summary(profile)["required_groups"].items():
        if group_payload["status"] != "declared":
            blockers.append(f"{group}_profile_missing")
    status = "ready" if not blockers else "blocked"
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "status": status,
        "ready_for_full_training": status == "ready",
        "blockers": sorted(set(blockers)),
        "repo_root": str(repo_root),
        "weights_root": str(weights_root),
        "profile": profile_summary(profile),
        "reportable_task_roots": reportable_files,
        "materialized_public_dev": materialized,
        "reportable_scores": reportable_scores,
        "diagnostics": diagnostics,
        "policy": {
            "public_dev_materialization": "diagnostic_only",
            "reportable_scores_require": "nonempty authorized task roots plus official scorer artifacts",
            "decode_sanity_required_before_full_training": True,
            "checkpoint_bound_heldout_loss_required_before_full_training": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Omnicoder eval/diagnostics readiness audit.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--weights-root", default="weights")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--materialization-root", default="")
    parser.add_argument("--diagnostic-root", action="append", default=[])
    parser.add_argument("--score-root", action="append", default=[])
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    weights_root = Path(args.weights_root).resolve()
    materialization_root = Path(args.materialization_root).resolve() if args.materialization_root else None
    diagnostic_roots = [Path(item).resolve() for item in args.diagnostic_root] if args.diagnostic_root else None
    score_roots = [Path(item).resolve() for item in args.score_root] if args.score_root else None
    report = build_audit(
        repo_root=repo_root,
        weights_root=weights_root,
        profile_path=args.profile,
        materialization_root=materialization_root,
        diagnostic_roots=diagnostic_roots,
        score_roots=score_roots,
    )
    report = json_safe(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())

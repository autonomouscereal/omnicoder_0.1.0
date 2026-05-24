from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from omnicoder.data_factory import curation_layers_2026, export_agent_memory_postgres_2026, memory_trace_collectors_2026
from omnicoder.training import training_orchestration_2026


SCHEMA_VERSION = "2026-05-23"
DEFAULT_PROFILE = "profiles/dataset_curation_2026.json"
DEFAULT_TRAINING_PROFILE = "profiles/training_orchestration_2026.json"
DEFAULT_OUT_DIR = "weights/curated_datasets_2026/latest"
DEFAULT_MODALITIES = tuple(training_orchestration_2026.DEFAULT_STAGE_ORDER)
TEXT_SUFFIXES = {".txt", ".md", ".rst", ".json", ".jsonl", ".log"}
CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".ps1",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
}
MEDIA_SUFFIXES = {
    "image": set(training_orchestration_2026.MEDIA_SUFFIXES["image"]),
    "video": set(training_orchestration_2026.MEDIA_SUFFIXES["video"]),
    "audio": set(training_orchestration_2026.MEDIA_SUFFIXES["audio"]),
    "music": set(training_orchestration_2026.MEDIA_SUFFIXES["music"]),
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.exists() or not source.is_file():
        return
    for line_number, line in enumerate(source.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            payload = {"text": line, "parse_error": str(exc), "line_number": line_number}
        if isinstance(payload, dict):
            payload.setdefault("line_number", line_number)
            yield payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, root: Path) -> Path:
    text = str(value).strip()
    path = Path(text)
    return path if path.is_absolute() else root / path


def existing_paths(values: Any, root: Path) -> list[Path]:
    if isinstance(values, (str, Path)):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        if not str(raw).strip():
            continue
        path = resolve_path(raw, root).expanduser()
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def extract_text(record: dict[str, Any]) -> str:
    return curation_layers_2026.extract_text(record)


def training_plan(training_profile: dict[str, Any]) -> dict[str, Any]:
    cfg = training_orchestration_2026.profile_cfg(training_profile)
    plan = cfg.get("training_plan")
    if not isinstance(plan, dict):
        raise ValueError("training profile must contain training_plan")
    return plan


def builder_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("builder_2026")
    return cfg if isinstance(cfg, dict) else {}


def load_training_profile(profile: dict[str, Any], root: Path) -> dict[str, Any]:
    configured = builder_cfg(profile).get("training_profile") or profile.get("training_profile") or DEFAULT_TRAINING_PROFILE
    path = resolve_path(str(configured), root)
    return training_orchestration_2026.load_profile(path)


def modality_limit(plan: dict[str, Any], modality: str) -> int:
    by_modality = plan.get("max_records_per_modality_by_modality")
    if isinstance(by_modality, dict) and modality in by_modality:
        try:
            return max(0, int(by_modality[modality]))
        except Exception:
            pass
    return max(0, int(plan.get("max_records_per_modality") or 256))


def source_inventory_entry(path: Path, kind: str, label: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "kind": kind,
        "label": label,
        "is_dir": path.is_dir(),
        "bytes": int(stat.st_size) if path.is_file() else None,
        "exists": True,
        "mtime": int(stat.st_mtime),
    }


def agent_memory_script_path(cfg: dict[str, Any], root: Path) -> Path | None:
    candidates: list[Any] = []
    if cfg.get("script"):
        candidates.append(cfg.get("script"))
    configured = cfg.get("script_candidates")
    if isinstance(configured, list):
        candidates.extend(configured)
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
    return None


def run_agent_memory_cli_export(profile: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    cfg = builder_cfg(profile).get("agent_memory_cli_export")
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return {"status": "skipped", "reason": "disabled"}
    script = agent_memory_script_path(cfg, root)
    if script is None:
        candidates = [cfg.get("script"), *(cfg.get("script_candidates") if isinstance(cfg.get("script_candidates"), list) else [])]
        return {"status": "skipped", "reason": "script_not_found", "candidates": [str(item) for item in candidates if item]}
    out_path = resolve_path(str(cfg.get("out") or out_dir / "raw" / "agent_memory_audit.jsonl"), root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(cfg.get("limit") or 5000)
    cmd = [sys.executable, str(script), "--json", "audit", "--limit", str(limit)]
    if cfg.get("all_spaces", True):
        cmd.append("--all-spaces")
    if cfg.get("space"):
        cmd.extend(["--space", str(cfg["space"])])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=float(cfg.get("timeout_seconds") or 120), check=False)
    if result.returncode != 0 and "--all-spaces" in cmd and "unrecognized arguments" in result.stderr:
        retry_cmd = [part for part in cmd if part != "--all-spaces"]
        result = subprocess.run(retry_cmd, capture_output=True, text=True, timeout=float(cfg.get("timeout_seconds") or 120), check=False)
    if result.returncode != 0:
        return {
            "status": "failed",
            "script": str(script),
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:],
        }
    try:
        payload = json.loads(result.stdout)
    except Exception as exc:
        return {"status": "failed", "script": str(script), "error": str(exc), "stdout_head": result.stdout[:500]}
    rows = payload if isinstance(payload, list) else payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    count = write_jsonl(out_path, (row for row in rows if isinstance(row, dict)))
    return {"status": "ok", "out": str(out_path), "records": count, "limit": limit}


def run_agent_memory_postgres_export(profile: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    cfg = profile.get("agent_memory_postgres_export")
    if not isinstance(cfg, dict):
        cfg = builder_cfg(profile).get("agent_memory_postgres_export")
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return {"status": "skipped", "reason": "disabled"}
    export_cfg = dict(cfg)
    out_path = resolve_path(str(export_cfg.get("out") or "data/raw/agent_memory_events_2026.jsonl"), root)
    if not export_cfg.get("out"):
        out_path = out_dir / "raw" / "agent_memory_events_2026.jsonl"
    try:
        return export_agent_memory_postgres_2026.export_rows(export_cfg, out_path)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "raw_postgres_export_failed",
            "error": repr(exc),
            "out": str(out_path),
        }


def run_agent_memory_export(profile: dict[str, Any], root: Path, out_dir: Path) -> dict[str, Any]:
    pg_export = run_agent_memory_postgres_export(profile, root, out_dir)
    if pg_export.get("status") == "ok":
        pg_export["path"] = "raw_postgresql"
        return pg_export
    cli_export = run_agent_memory_cli_export(profile, root, out_dir)
    if cli_export.get("status") == "ok":
        cli_export["path"] = "cli_fallback"
        cli_export["postgres_export"] = pg_export
        return cli_export
    return {
        "status": "failed" if pg_export.get("status") == "failed" or cli_export.get("status") == "failed" else "skipped",
        "postgres_export": pg_export,
        "cli_export": cli_export,
    }


def configured_trace_sources(profile: dict[str, Any], root: Path, agent_memory_export: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    trace_inputs = profile.get("trace_inputs") if isinstance(profile.get("trace_inputs"), dict) else {}
    sources = trace_inputs.get("sources") or []
    if isinstance(sources, (str, dict)):
        sources = [sources]
    for item in sources:
        if isinstance(item, str):
            path = resolve_path(item, root)
            if path.exists():
                entries.append({"path": path, "harness": "generic", "label": path.name})
        elif isinstance(item, dict):
            raw_path = item.get("path") or item.get("root")
            if raw_path:
                path = resolve_path(str(raw_path), root)
                if path.exists():
                    entries.append({"path": path, "harness": str(item.get("harness") or "generic"), "label": str(item.get("label") or path.name)})
    local_roots = profile.get("local_trace_roots_to_export") if isinstance(profile.get("local_trace_roots_to_export"), dict) else {}
    for key, raw_path in local_roots.items():
        path = resolve_path(str(raw_path), root).expanduser()
        if path.exists():
            harness = "codex" if "codex" in key.lower() else "claude" if "claude" in key.lower() else "agent_memory"
            entries.append({"path": path, "harness": harness, "label": key})
    if agent_memory_export.get("status") == "ok" and agent_memory_export.get("out"):
        path = resolve_path(str(agent_memory_export["out"]), root)
        if path.exists():
            entries.append({"path": path, "harness": "agent_memory", "label": "agent_memory_cli_export"})
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for entry in entries:
        key = str(Path(entry["path"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def trace_row_builder(harness: str) -> Any:
    normalized = harness.lower().replace("-", "_")
    if normalized == "codex":
        return memory_trace_collectors_2026.codex_event_row
    if normalized == "agent_memory":
        return memory_trace_collectors_2026.agent_memory_event_row
    if normalized == "claude":
        return memory_trace_collectors_2026.claude_event_row
    return memory_trace_collectors_2026.claude_or_agent_memory_row


def collect_trace_rows(profile: dict[str, Any], root: Path, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cfg = builder_cfg(profile)
    agent_memory_export = run_agent_memory_export(profile, root, out_dir)
    entries = configured_trace_sources(profile, root, agent_memory_export)
    min_year = int(cfg.get("min_year") or 2025)
    max_year = int(cfg.get("max_year") or 2026)
    source_date = str(profile.get("source_date") or cfg.get("source_date") or time.strftime("%Y-%m-%d", time.gmtime()))
    per_source_limit = int(cfg.get("per_trace_source_limit") or 0)
    total_limit = int(cfg.get("trace_limit") or 0)
    rows: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(entry["path"])
        source_inventory.append(source_inventory_entry(path, "trace", str(entry.get("label") or path.name)))
        harness = str(entry.get("harness") or "generic")
        collected = memory_trace_collectors_2026.collect_records(
            [path],
            trace_row_builder(harness),
            bucket=str(cfg.get("trace_bucket") or "agentic_trace_sft_2026"),
            split="train",
            source_date=source_date,
            min_year=min_year,
            max_year=max_year,
            limit=per_source_limit,
        )
        for row in collected:
            lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
            row["lineage"] = {**lineage, "builder_source_label": entry.get("label"), "builder_harness": harness}
            rows.append(row)
            if total_limit and len(rows) >= total_limit:
                break
        if total_limit and len(rows) >= total_limit:
            break
    return rows, agent_memory_export, source_inventory


def text_to_modality(record: dict[str, Any], text: str, curated: dict[str, Any], cfg: dict[str, Any]) -> str:
    input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    event_type = str(input_json.get("event_type") or target_json.get("action_type") or "").lower()
    tool_name = input_json.get("tool_name") or target_json.get("tool_name")
    tools = curated.get("tools") if isinstance(curated.get("tools"), dict) else {}
    code = curated.get("code") if isinstance(curated.get("code"), dict) else {}
    if tool_name or "tool" in event_type or tools.get("tool_families"):
        return "tool"
    if code.get("is_code"):
        return "code"
    if len(text) >= int(cfg.get("long_context_min_chars") or 1800):
        return "long_context"
    return "text"


def prompt_for_modality(modality: str, record: dict[str, Any]) -> str:
    if modality == "tool":
        return "Learn this agentic tool-call trajectory, preserving intent, tool arguments, observations, and verification evidence."
    if modality == "code":
        return "Learn this code, terminal, and repair record as executable software-engineering supervision."
    if modality == "long_context":
        return "Learn this long-context span with anchors, retained facts, and cross-turn dependencies."
    return "Learn this curated reasoning and instruction trace as high-quality text supervision."


def curated_trace_to_training_row(
    record: dict[str, Any],
    curated: dict[str, Any],
    plan: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    quality = curated.get("quality") if isinstance(curated.get("quality"), dict) else {}
    contamination = curated.get("contamination") if isinstance(curated.get("contamination"), dict) else {}
    secret = curated.get("secret_redaction") if isinstance(curated.get("secret_redaction"), dict) else {}
    split_assignment = curated.get("split_assignment") if isinstance(curated.get("split_assignment"), dict) else {}
    min_quality = float(cfg.get("min_quality") or 0.35)
    quality_score = float(quality.get("overall") or quality.get("score") or 0.0)
    if quality_score < min_quality:
        return None
    if secret.get("has_secret"):
        return None
    if contamination.get("status") == "contaminated":
        return None
    if split_assignment.get("split") == "rejected":
        return None
    text = str(curated.get("normalized_text") or "")
    if len(text.strip()) < int(cfg.get("min_chars") or 8):
        return None
    modality = text_to_modality(record, text, curated, cfg)
    source_uri = str((curated.get("provenance") or {}).get("path") or (record.get("lineage") or {}).get("path") or "trace")
    source_payload = {
        **record,
        "source_id": curated.get("curated_id") or (record.get("lineage") or {}).get("record_hash"),
        "quality": {"score": quality_score, "label": quality.get("label") or "candidate", "details": quality},
        "contamination": contamination,
        "builder_curation": {
            "curated_id": curated.get("curated_id"),
            "dedupe": curated.get("dedupe") or {},
            "language": curated.get("language") or {},
            "code": curated.get("code") or {},
            "tools": curated.get("tools") or {},
            "split_assignment": split_assignment,
        },
    }
    return training_orchestration_2026.make_training_record(
        modality,
        prompt_for_modality(modality, record),
        text[: int(plan.get("target_text_chars") or 3000)],
        source_uri,
        plan,
        source_payload=source_payload,
    )


def source_args(profile: dict[str, Any]) -> argparse.Namespace:
    cfg = builder_cfg(profile)
    curation_cfg = profile.get("curation_layers") if isinstance(profile.get("curation_layers"), dict) else {}
    return argparse.Namespace(
        input="",
        protected=(profile.get("contamination") or {}).get("protected_path") if isinstance(profile.get("contamination"), dict) else None,
        source_name=str(profile.get("run_name") or "omnicoder_curated_dataset_builder_2026"),
        source_date=str(profile.get("source_date") or time.strftime("%Y-%m-%d", time.gmtime())),
        validation_ratio=float(curation_cfg.get("validation_ratio", cfg.get("validation_ratio", 0.05))),
        holdout_ratio=float(curation_cfg.get("holdout_ratio", cfg.get("holdout_ratio", 0.05))),
        lowercase=False,
        redact=True,
    )


def curate_trace_rows(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    root: Path,
    out_dir: Path,
    plan: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cfg = builder_cfg(profile)
    args = source_args(profile)
    protected_path = args.protected
    resolved_protected = resolve_path(str(protected_path), root) if protected_path else None
    if resolved_protected is not None and not resolved_protected.exists():
        resolved_protected = None
    protected_hashes = curation_layers_2026.load_protected_hashes(resolved_protected)
    canonical_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    rows_by_modality: dict[str, list[dict[str, Any]]] = {modality: [] for modality in DEFAULT_MODALITIES}
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for record in rows:
        curated = curation_layers_2026.curate_record(record, args, protected_hashes)
        canonical_rows.append(curated)
        signature = str((curated.get("dedupe") or {}).get("canonical_sha256") or curated.get("curated_id"))
        if signature in seen:
            counts["duplicate"] += 1
            rejected_rows.append({**curated, "rejection_reason": "duplicate"})
            continue
        seen.add(signature)
        row = curated_trace_to_training_row(record, curated, plan, cfg)
        if row is None:
            counts["rejected"] += 1
            rejected_rows.append(curated)
            continue
        rows_by_modality[str(row["modality"])].append(row)
        counts[f"accepted_{row['modality']}"] += 1
    raw_dir = out_dir / "raw"
    write_jsonl(raw_dir / "normalized_traces.jsonl", rows)
    write_jsonl(raw_dir / "canonical_trace_curation.jsonl", canonical_rows)
    write_jsonl(raw_dir / "rejected_traces.jsonl", rejected_rows)
    return rows_by_modality, {
        "input_records": len(rows),
        "canonical_records": len(canonical_rows),
        "rejected_records": len(rejected_rows),
        "accepted_records": sum(len(items) for items in rows_by_modality.values()),
        "counts": dict(sorted(counts.items())),
        "protected_hashes": len(protected_hashes),
        "raw_outputs": {
            "normalized_traces": str(raw_dir / "normalized_traces.jsonl"),
            "canonical_trace_curation": str(raw_dir / "canonical_trace_curation.jsonl"),
            "rejected_traces": str(raw_dir / "rejected_traces.jsonl"),
        },
    }


def iter_files(path: Path, suffixes: set[str], max_files: int, max_bytes: int) -> Iterable[Path]:
    try:
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
    except (OSError, PermissionError):
        return
    count = 0
    for item in candidates:
        if not item.is_file() or item.suffix.lower() not in suffixes:
            continue
        try:
            if max_bytes and item.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield item
        count += 1
        if max_files and count >= max_files:
            return


def collect_file_rows(profile: dict[str, Any], root: Path, plan: dict[str, Any], source_inventory: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cfg = builder_cfg(profile)
    sources = cfg.get("supplemental_sources") if isinstance(cfg.get("supplemental_sources"), dict) else {}
    max_files = int(cfg.get("max_files_per_root") or 128)
    max_text_bytes = int(cfg.get("max_text_file_bytes") or plan.get("max_text_file_bytes") or 1024 * 1024)
    rows_by_modality: dict[str, list[dict[str, Any]]] = {modality: [] for modality in DEFAULT_MODALITIES}
    for modality, suffixes in (("text", TEXT_SUFFIXES), ("code", CODE_SUFFIXES), ("long_context", TEXT_SUFFIXES)):
        roots = existing_paths(sources.get(f"{modality}_roots"), root)
        for source_root in roots:
            source_inventory.append(source_inventory_entry(source_root, f"{modality}_root", source_root.name))
            for path in iter_files(source_root, suffixes, max_files=max_files, max_bytes=max_text_bytes):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                min_chars = 120 if modality == "long_context" else 20
                if len(text.strip()) < min_chars:
                    continue
                normalized = curation_layers_2026.normalize_content(text)["text"]
                secret = curation_layers_2026.redact_secrets(normalized)
                if secret["has_secret"]:
                    continue
                row = training_orchestration_2026.make_training_record(
                    modality,
                    prompt_for_modality(modality, {}),
                    secret["redacted_text"][: int(plan.get("target_text_chars") or 3000)],
                    str(path),
                    plan,
                    source_payload={
                        "source_id": stable_hash({"path": str(path), "modality": modality}),
                        "source_date": str(profile.get("source_date") or "2026-05-23"),
                        "quality": {"score": 0.78, "label": "accepted_local_file"},
                        "contamination": {"status": "unknown", "note": "local_supplemental_file_needs_downstream_scan"},
                    },
                )
                rows_by_modality[modality].append(row)
                if len(rows_by_modality[modality]) >= modality_limit(plan, modality):
                    break
    return rows_by_modality


def media_modality_for_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        return "music"
    name = str(path).lower()
    if suffix == ".webp" and any(marker in name for marker in ("ltx", "video", "t2v", "i2v", "motion", "generated_av")):
        return "video"
    if suffix in MEDIA_SUFFIXES["music"] and any(marker in name for marker in ("music", "song", "ace", "beat", "lyrics", "instrument")):
        return "music"
    for modality, suffixes in MEDIA_SUFFIXES.items():
        if suffix in suffixes:
            return modality
    return None


def sidecar_metadata(path: Path) -> dict[str, Any]:
    for candidate in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def prompt_from_media_metadata(modality: str, metadata: dict[str, Any]) -> str:
    for key in ("instruction", "question", "prompt", "positive_prompt", "caption", "description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return {
        "image": "Describe, ground, and edit the real image artifact represented by this unified-token training packet.",
        "video": "Describe temporal motion, scene changes, and generation metadata for this real video artifact.",
        "audio": "Transcribe, caption, and reason about the real audio artifact.",
        "music": "Learn the real music artifact, including prompt, structure, lyrics, and style metadata when present.",
    }.get(modality, f"Learn the real {modality} artifact.")


def target_from_media_metadata(path: Path, modality: str, metadata: dict[str, Any]) -> str:
    for key in ("target", "answer", "caption", "description", "transcript", "lyrics", "prompt", "positive_prompt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"{modality} artifact: {path.name}; mime={mime}; bytes={path.stat().st_size}"


def collect_media_rows(profile: dict[str, Any], root: Path, plan: dict[str, Any], source_inventory: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cfg = builder_cfg(profile)
    sources = cfg.get("supplemental_sources") if isinstance(cfg.get("supplemental_sources"), dict) else {}
    max_files = int(cfg.get("max_media_files_per_root") or 256)
    max_bytes = int(cfg.get("max_media_bytes") or plan.get("max_media_bytes") or 512 * 1024 * 1024)
    rows_by_modality: dict[str, list[dict[str, Any]]] = {modality: [] for modality in DEFAULT_MODALITIES}
    roots: list[Path] = []
    for key in ("media_roots", "image_roots", "video_roots", "audio_roots", "music_roots"):
        roots.extend(existing_paths(sources.get(key), root))
    seen_roots: set[str] = set()
    all_suffixes = set().union(*MEDIA_SUFFIXES.values())
    for media_root in roots:
        key = str(media_root.resolve())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        source_inventory.append(source_inventory_entry(media_root, "media_root", media_root.name))
        for path in iter_files(media_root, all_suffixes, max_files=max_files, max_bytes=max_bytes):
            modality = media_modality_for_path(path)
            if modality is None:
                continue
            if len(rows_by_modality[modality]) >= modality_limit(plan, modality):
                continue
            metadata = sidecar_metadata(path)
            try:
                media_meta = training_orchestration_2026.media_metadata(path, modality, plan, {})
            except Exception:
                media_meta = training_orchestration_2026.safe_stat_payload(path)
            if not training_orchestration_2026.media_record_ok(path, modality, media_meta, plan):
                continue
            source_payload = {
                **metadata,
                "source_id": stable_hash({"path": str(path), "sha256": media_meta.get("sha256"), "modality": modality}),
                "source_date": str(profile.get("source_date") or "2026-05-23"),
                "quality": {"score": float(metadata.get("quality_score") or 0.82), "label": metadata.get("quality_label") or "accepted_real_media"},
                "contamination": {"status": metadata.get("contamination_status") or "unknown"},
            }
            rows_by_modality[modality].append(
                training_orchestration_2026.make_training_record(
                    modality,
                    prompt_from_media_metadata(modality, metadata),
                    target_from_media_metadata(path, modality, metadata),
                    str(path),
                    plan,
                    artifact_path=path,
                    source_payload=source_payload,
                    media_metadata=media_meta,
                )
            )
    return rows_by_modality


def merge_rows_by_modality(*groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {modality: [] for modality in DEFAULT_MODALITIES}
    for group in groups:
        for modality, rows in group.items():
            if modality in merged:
                merged[modality].extend(rows)
    return merged


def dedupe_and_limit(rows_by_modality: dict[str, list[dict[str, Any]]], plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cleaned: dict[str, list[dict[str, Any]]] = {}
    for modality in DEFAULT_MODALITIES:
        deduped = training_orchestration_2026.dedupe_rows(rows_by_modality.get(modality, []))
        cleaned[modality] = deduped[: modality_limit(plan, modality)]
    return cleaned


def split_rows(rows_by_modality: dict[str, list[dict[str, Any]]], plan: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[dict[str, Any]]]]]:
    all_splits = {"train": [], "eval": [], "test": []}
    per_modality: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for modality in DEFAULT_MODALITIES:
        split = training_orchestration_2026.assign_deterministic_splits(rows_by_modality.get(modality, []), modality, plan)
        per_modality[modality] = split
        for split_name, split_rows_value in split.items():
            all_splits[split_name].extend(split_rows_value)
    return all_splits, per_modality


def write_dataset_outputs(
    out_dir: Path,
    all_splits: dict[str, list[dict[str, Any]]],
    per_modality: dict[str, dict[str, list[dict[str, Any]]]],
    profile: dict[str, Any],
    training_profile: dict[str, Any],
    plan: dict[str, Any],
    source_inventory: list[dict[str, Any]],
    trace_stats: dict[str, Any],
    agent_memory_export: dict[str, Any],
) -> dict[str, Any]:
    jsonl_dir = out_dir / "jsonl"
    manifests_dir = out_dir / "manifests"
    cards_dir = out_dir / "dataset_cards"
    per_modality_paths: dict[str, dict[str, str]] = {}
    for modality in DEFAULT_MODALITIES:
        per_modality_paths[modality] = {}
        for split_name in ("train", "eval", "test"):
            path = jsonl_dir / f"{split_name}_{modality}.jsonl"
            write_jsonl(path, per_modality[modality][split_name])
            per_modality_paths[modality][split_name] = str(path)
    aggregate_paths: dict[str, str] = {}
    for split_name in ("train", "eval", "test"):
        path = jsonl_dir / f"{split_name}_all_modalities.jsonl"
        write_jsonl(path, all_splits[split_name])
        aggregate_paths[split_name] = str(path)
    curated_path = jsonl_dir / "curated_records.jsonl"
    all_rows = all_splits["train"] + all_splits["eval"] + all_splits["test"]
    write_jsonl(curated_path, all_rows)
    write_jsonl(jsonl_dir / "train_media_focus.jsonl", [row for row in all_splits["train"] if row.get("modality") in {"image", "video", "audio", "music"}])
    write_jsonl(jsonl_dir / "train_agentic_focus.jsonl", [row for row in all_splits["train"] if row.get("modality") in {"tool", "code", "long_context"}])
    artifact_rows = list(training_orchestration_2026.iter_artifact_manifest_rows(all_rows))
    source_file_rows = training_orchestration_2026.source_file_manifest_rows(all_rows)
    artifact_manifest = manifests_dir / "artifacts.jsonl"
    source_files_manifest = manifests_dir / "source_files.jsonl"
    write_jsonl(artifact_manifest, artifact_rows)
    write_jsonl(source_files_manifest, source_file_rows)
    cleaned_manifest = training_orchestration_2026.build_cleaned_dataset_manifest(
        training_orchestration_2026.profile_cfg(training_profile),
        plan,
        all_splits["train"],
        all_splits["eval"],
        all_splits["test"],
        len(artifact_rows),
        len(source_file_rows),
    )
    blend_manifest = training_orchestration_2026.build_dataset_blend_manifest(
        training_orchestration_2026.profile_cfg(training_profile),
        all_splits["train"],
        all_splits["eval"],
        all_splits["test"],
    )
    posttraining_exports = training_orchestration_2026.build_posttraining_curation_exports(
        training_profile,
        out_dir,
        all_splits["train"],
        all_splits["eval"],
        all_splits["test"],
    )
    write_json(manifests_dir / "cleaned_dataset_manifest.json", cleaned_manifest)
    write_json(manifests_dir / "dataset_blend_manifest.json", blend_manifest)
    write_json(manifests_dir / "source_inventory.json", {"schema": "omnicoder.source_inventory_2026.v1", "sources": source_inventory})
    quality_report = build_quality_report(all_splits, per_modality, trace_stats, agent_memory_export)
    contamination_report = build_contamination_report(all_rows)
    coverage_report = build_coverage_report(all_splits, per_modality, profile)
    write_json(manifests_dir / "quality_report.json", quality_report)
    write_json(manifests_dir / "contamination_report.json", contamination_report)
    write_json(manifests_dir / "coverage_report.json", coverage_report)
    dataset_card = build_dataset_card(profile, training_profile, all_splits, per_modality, source_inventory, posttraining_exports)
    write_json(cards_dir / "omnicoder_curated_multimodal_dataset_2026.json", dataset_card)
    (cards_dir / "omnicoder_curated_multimodal_dataset_2026.md").write_text(dataset_card_markdown(dataset_card), encoding="utf-8")
    manifest = {
        "schema": "omnicoder.curated_dataset_builder_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if coverage_report["status"] == "passed" and cleaned_manifest["status"] == "passed" else "needs_attention",
        "created_at": now_iso(),
        "profile_name": profile.get("profile_name"),
        "out_dir": str(out_dir),
        "aggregate_jsonl": aggregate_paths,
        "curated_jsonl": str(curated_path),
        "per_modality_split_jsonl": per_modality_paths,
        "records": {split_name: len(rows) for split_name, rows in all_splits.items()},
        "modalities": {modality: {split_name: len(rows) for split_name, rows in splits.items()} for modality, splits in per_modality.items()},
        "manifests": {
            "source_inventory": str(manifests_dir / "source_inventory.json"),
            "quality_report": str(manifests_dir / "quality_report.json"),
            "contamination_report": str(manifests_dir / "contamination_report.json"),
            "coverage_report": str(manifests_dir / "coverage_report.json"),
            "cleaned_dataset_manifest": str(manifests_dir / "cleaned_dataset_manifest.json"),
            "dataset_blend_manifest": str(manifests_dir / "dataset_blend_manifest.json"),
            "artifact_manifest": str(artifact_manifest),
            "source_files_manifest": str(source_files_manifest),
        },
        "posttraining_curation_exports": posttraining_exports,
        "dataset_card": str(cards_dir / "omnicoder_curated_multimodal_dataset_2026.json"),
        "trace_stats": trace_stats,
        "agent_memory_export": agent_memory_export,
        "dataset_catalog_2026": profile.get("dataset_catalog_2026") or {},
        "notes": [
            "All generated train/eval/test files are JSONL-first and can feed training_orchestration_2026 as local real_sources.",
            "Agent memory export is optional and uses the existing raw PostgreSQL-backed CLI when available; no secrets are embedded in this builder.",
            "Public datasets in dataset_catalog_2026 are catalog/allowlist entries. Downloading is intentionally separate so license and auth gates stay explicit.",
        ],
    }
    write_json(manifests_dir / "curated_dataset_builder_manifest.json", manifest)
    write_json(out_dir / "latest_manifest.json", manifest)
    return manifest


def build_quality_report(
    all_splits: dict[str, list[dict[str, Any]]],
    per_modality: dict[str, dict[str, list[dict[str, Any]]]],
    trace_stats: dict[str, Any],
    agent_memory_export: dict[str, Any],
) -> dict[str, Any]:
    quality_scores: list[float] = []
    labels: Counter[str] = Counter()
    by_modality: dict[str, dict[str, Any]] = {}
    for modality, splits in per_modality.items():
        modality_scores: list[float] = []
        for rows in splits.values():
            for row in rows:
                score = float(row.get("quality_score") or (row.get("quality") or {}).get("score") or 0.0)
                modality_scores.append(score)
                quality_scores.append(score)
                label = str((row.get("quality") or {}).get("label") or "unknown")
                labels[label] += 1
        by_modality[modality] = {
            "records": len(modality_scores),
            "min": min(modality_scores) if modality_scores else None,
            "avg": (sum(modality_scores) / len(modality_scores)) if modality_scores else None,
            "max": max(modality_scores) if modality_scores else None,
        }
    return {
        "schema": "omnicoder.dataset_quality_report_2026.v1",
        "created_at": now_iso(),
        "records": sum(len(rows) for rows in all_splits.values()),
        "score": {
            "min": min(quality_scores) if quality_scores else None,
            "avg": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
            "max": max(quality_scores) if quality_scores else None,
        },
        "labels": dict(sorted(labels.items())),
        "by_modality": by_modality,
        "trace_stats": trace_stats,
        "agent_memory_export": agent_memory_export,
    }


def build_contamination_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    modality_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        status = str(row.get("contamination_status") or (row.get("contamination") or {}).get("status") or "unknown")
        modality = str(row.get("modality") or "unknown")
        statuses[status] += 1
        modality_statuses[modality][status] += 1
    return {
        "schema": "omnicoder.contamination_report_2026.v1",
        "created_at": now_iso(),
        "status_counts": dict(sorted(statuses.items())),
        "by_modality": {modality: dict(sorted(counter.items())) for modality, counter in sorted(modality_statuses.items())},
        "policy": "contaminated and secret-bearing records are rejected before export; suspect records remain visible through contamination_status for downstream gates.",
    }


def build_coverage_report(
    all_splits: dict[str, list[dict[str, Any]]],
    per_modality: dict[str, dict[str, list[dict[str, Any]]]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    cfg = builder_cfg(profile)
    targets = cfg.get("coverage_targets") if isinstance(cfg.get("coverage_targets"), dict) else {}
    missing: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for modality, splits in per_modality.items():
        counts[modality] = {split_name: len(rows) for split_name, rows in splits.items()}
        min_train = int(targets.get(modality, {}).get("min_train", targets.get("min_train_per_modality", 1))) if isinstance(targets.get(modality), dict) else int(targets.get("min_train_per_modality", 1))
        if counts[modality]["train"] < min_train:
            missing[modality] = {"train": counts[modality]["train"], "min_train": min_train}
    return {
        "schema": "omnicoder.modality_coverage_report_2026.v1",
        "created_at": now_iso(),
        "status": "passed" if not missing else "failed",
        "records": {split_name: len(rows) for split_name, rows in all_splits.items()},
        "modalities": counts,
        "missing": missing,
        "targets": targets,
    }


def build_dataset_card(
    profile: dict[str, Any],
    training_profile: dict[str, Any],
    all_splits: dict[str, list[dict[str, Any]]],
    per_modality: dict[str, dict[str, list[dict[str, Any]]]],
    source_inventory: list[dict[str, Any]],
    posttraining_exports: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "omnicoder.dataset_card_2026.v1",
        "name": "Omnicoder Curated Multimodal Dataset 2026",
        "created_at": now_iso(),
        "profile_name": profile.get("profile_name"),
        "training_profile": training_profile.get("profile_name"),
        "records": {split_name: len(rows) for split_name, rows in all_splits.items()},
        "modalities": {modality: {split_name: len(rows) for split_name, rows in splits.items()} for modality, splits in per_modality.items()},
        "source_count": len(source_inventory),
        "source_kinds": dict(sorted(Counter(str(item.get("kind") or "unknown") for item in source_inventory).items())),
        "cleaning": profile.get("cleaning_layers_2026") or [],
        "dataset_catalog_2026": profile.get("dataset_catalog_2026") or {},
        "posttraining_exports": posttraining_exports,
        "intended_use": [
            "dense omnimodal ledger-token pretraining probes",
            "agentic tool-calling SFT/reward/RLVR replay",
            "image, video, audio, and music artifact grounding",
            "long-context 1M-token curriculum assembly",
        ],
        "restrictions": [
            "internal traces are redacted and decontaminated before train export",
            "external catalog entries require license/auth checks before download",
            "benchmark/eval protected records are not training sources",
        ],
    }


def dataset_card_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# Omnicoder Curated Multimodal Dataset 2026",
        "",
        f"- Created: {card.get('created_at')}",
        f"- Train records: {(card.get('records') or {}).get('train', 0)}",
        f"- Eval records: {(card.get('records') or {}).get('eval', 0)}",
        f"- Test records: {(card.get('records') or {}).get('test', 0)}",
        "",
        "## Modalities",
    ]
    for modality, counts in (card.get("modalities") or {}).items():
        lines.append(f"- {modality}: train={counts.get('train', 0)} eval={counts.get('eval', 0)} test={counts.get('test', 0)}")
    lines.extend(
        [
            "",
            "## Cleaning",
            "- Secret redaction and quarantine before export.",
            "- Exact dedupe by canonical payload identity.",
            "- Benchmark contamination status preserved for downstream gates.",
            "- Real media artifacts are hashed and recorded in artifact manifests.",
            "",
            "## Public Catalog",
        ]
    )
    catalog = card.get("dataset_catalog_2026") if isinstance(card.get("dataset_catalog_2026"), dict) else {}
    for family, entries in catalog.items():
        if isinstance(entries, list):
            lines.append(f"- {family}: {len(entries)} catalog entries")
    lines.append("")
    return "\n".join(lines)


def build_dataset(profile_path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    profile = read_json(profile_path)
    training_profile = load_training_profile(profile, root)
    plan = training_plan(training_profile)
    target_dir = out_dir or resolve_path(str(builder_cfg(profile).get("out_dir") or DEFAULT_OUT_DIR), root)
    target_dir.mkdir(parents=True, exist_ok=True)
    trace_rows, agent_memory_export, source_inventory = collect_trace_rows(profile, root, target_dir)
    trace_group, trace_stats = curate_trace_rows(trace_rows, profile, root, target_dir, plan)
    supplemental_group = collect_file_rows(profile, root, plan, source_inventory)
    media_group = collect_media_rows(profile, root, plan, source_inventory)
    merged = dedupe_and_limit(merge_rows_by_modality(trace_group, supplemental_group, media_group), plan)
    all_splits, per_modality = split_rows(merged, plan)
    return write_dataset_outputs(
        target_dir,
        all_splits,
        per_modality,
        profile,
        training_profile,
        plan,
        source_inventory,
        trace_stats,
        agent_memory_export,
    )


def export_agent_memory_only(profile_path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    profile = read_json(profile_path)
    target_dir = out_dir or resolve_path(str(builder_cfg(profile).get("out_dir") or DEFAULT_OUT_DIR), root)
    target_dir.mkdir(parents=True, exist_ok=True)
    return run_agent_memory_export(profile, root, target_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build cleaned 2026 all-modality training datasets from traces, local corpora, and media artifacts")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", default="")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("export-agent-memory")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.command == "build":
        manifest = build_dataset(resolve_path(args.profile, repo_root()), out_dir)
    elif args.command == "export-agent-memory":
        manifest = export_agent_memory_only(resolve_path(args.profile, repo_root()), out_dir)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0 if manifest.get("status") in {"passed", "needs_attention", "ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

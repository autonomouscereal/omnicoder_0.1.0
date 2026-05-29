from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory import contamination, export_sft_jsonl, quality_scoring, teacher_jobs_2026
from omnicoder.data_factory import ingest_2026, ingest_agent_memory, ingest_codex_transcripts, ingest_comfyui_outputs


HARNESS_MODULES = {
    "agent_memory": "omnicoder.data_factory.ingest_agent_memory",
    "codex": "omnicoder.data_factory.ingest_codex_transcripts",
    "claude": "omnicoder.data_factory.ingest_2026",
    "comfyui": "omnicoder.data_factory.ingest_comfyui_outputs",
    "hermes": "omnicoder.data_factory.ingest_2026",
    "local_agent": "omnicoder.data_factory.ingest_2026",
    "generic": "omnicoder.data_factory.ingest_2026",
}

DATE_PREFIXES = ("2025", "2026")
COMFYUI_MEDIA_SUFFIXES = set(ingest_comfyui_outputs.MEDIA_SUFFIXES)
MATH_HINTS = ("\\boxed", "aime", "equation", "final answer", "latex", "math", "olympiad", "proof", "solve")
CODE_HINTS = ("class ", "compiler", "def ", "diff --git", "patch", "pytest", "traceback", "unit test")
TERMINAL_HINTS = ("bash", "cmd.exe", "docker", "exit_code", "powershell", "shell", "stderr", "stdout", "terminal")
BROWSER_HINTS = ("browser", "citation", "click", "http://", "https://", "playwright", "screenshot", "url", "web_research")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"profile must be a JSON object: {path}")
    return payload


def resolve_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                payload = {"line": line_number, "parse_error": str(exc), "text": line.rstrip("\n")}
            if isinstance(payload, dict):
                yield payload


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date_prefix(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return None


def record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for container in (record.get("input_json"), record.get("target_json"), record):
        if not isinstance(container, dict):
            continue
        messages = container.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    parts.append(message["content"])
        for key in ("content", "text", "prompt", "completion", "answer"):
            value = container.get(key)
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(part for part in parts if part)


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _json_text(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, list):
        return {str(item) for item in value if item}
    return set()


def extract_trace_features(record: dict[str, Any], harness: str) -> dict[str, Any]:
    input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    text = "\n".join(part for part in (record_text(record), _json_text(input_json), _json_text(target_json)) if part)
    lowered = text.lower()

    tool_calls = _as_list_of_dicts(record.get("tool_calls"))
    tool_results = _as_list_of_dicts(record.get("tool_results"))
    tool_name = input_json.get("tool_name") or target_json.get("tool_name")
    tool_input = input_json.get("tool_input")
    tool_output = target_json.get("tool_output")
    if tool_name or tool_input not in (None, {}, []) or "tool" in str(input_json.get("event_type") or "").lower():
        tool_calls.append({"tool": tool_name or "unknown_tool", "arguments": tool_input or {}, "event_type": input_json.get("event_type")})
    if tool_output not in (None, {}, []):
        tool_results.append(tool_output if isinstance(tool_output, dict) else {"content": tool_output})

    media_refs = _as_list_of_dicts(record.get("media_refs"))
    artifact_path = target_json.get("artifact_path") or target_json.get("path")
    media_type = target_json.get("media_type") or input_json.get("media_type")
    if artifact_path:
        media_refs.append(
            {
                "path": str(artifact_path),
                "media_type": str(media_type or "application/octet-stream"),
                "sha256": target_json.get("sha256"),
            }
        )

    modalities: set[str] = set()
    for ref in media_refs:
        kind = str(ref.get("media_type") or "").split("/", 1)[0]
        if kind:
            modalities.add("music" if kind == "audio" and "music" in str(ref.get("path") or "").lower() else kind)
    explicit_modality = input_json.get("modality") or target_json.get("modality") or record.get("modality")
    if explicit_modality:
        modalities.add(str(explicit_modality).split("/", 1)[0])
    if harness == "comfyui":
        modalities.add("multimodal")
    if tool_calls or tool_results:
        modalities.add("tool")

    domains: set[str] = set()
    if any(hint in lowered for hint in MATH_HINTS):
        domains.add("math")
    if any(hint in lowered for hint in CODE_HINTS):
        domains.add("code")
    if any(hint in lowered for hint in TERMINAL_HINTS):
        domains.add("terminal")
    if any(hint in lowered for hint in BROWSER_HINTS):
        domains.add("browser")
    if tool_calls or tool_results:
        domains.add("tool")
    if modalities.intersection({"image", "video", "audio", "music", "multimodal"}):
        domains.add("multimodal")

    return {
        "source_harness": harness,
        "trace_id": lineage.get("trace_id") or record.get("trace_id"),
        "domains": sorted(domains),
        "modalities": sorted(modalities),
        "has_tool_call": bool(tool_calls),
        "has_tool_result": bool(tool_results),
        "tool_call_count": len(tool_calls),
        "tool_result_count": len(tool_results),
        "media_ref_count": len(media_refs),
        "has_math": "math" in domains,
        "has_code": "code" in domains,
        "has_multimodal": "multimodal" in domains,
        "text_chars": len(text),
    }


def enrich_trace_row(row: dict[str, Any], harness: str, path: Path) -> dict[str, Any]:
    input_json = row.get("input_json") if isinstance(row.get("input_json"), dict) else {}
    target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
    features = extract_trace_features(row, harness)

    tool_calls = _as_list_of_dicts(row.get("tool_calls"))
    tool_results = _as_list_of_dicts(row.get("tool_results"))
    if features["has_tool_call"] and not tool_calls:
        tool_calls.append(
            {
                "tool": input_json.get("tool_name") or "unknown_tool",
                "arguments": input_json.get("tool_input") or {},
                "event_type": input_json.get("event_type"),
            }
        )
    if features["has_tool_result"] and not tool_results:
        output = target_json.get("tool_output")
        tool_results.append(output if isinstance(output, dict) else {"content": _json_text(output)})

    media_refs = _as_list_of_dicts(row.get("media_refs"))
    if target_json.get("artifact_path") and not media_refs:
        media_refs.append(
            {
                "path": str(target_json.get("artifact_path")),
                "media_type": str(target_json.get("media_type") or "application/octet-stream"),
                "sha256": target_json.get("sha256"),
            }
        )

    row["domains"] = sorted(_string_set(row.get("domains")) | set(features["domains"]))
    row["modalities"] = sorted(_string_set(row.get("modalities")) | set(features["modalities"]))
    row["trace_features"] = features
    if tool_calls:
        row["tool_calls"] = tool_calls
    if tool_results:
        row["tool_results"] = tool_results
    if media_refs:
        row["media_refs"] = media_refs
    lineage = row.setdefault("lineage", {})
    if isinstance(lineage, dict):
        lineage["source_file"] = str(path)
        lineage["feature_hash"] = stable_hash(features)
    return row


def normalize_harness(value: str | None) -> str:
    harness = (value or "generic").strip().lower().replace("-", "_")
    if harness in {"agentmemory", "agent_memory_pg", "memory"}:
        return "agent_memory"
    if harness in {"codex_cli", "openai_codex"}:
        return "codex"
    if harness in {"claude_code", "anthropic_claude"}:
        return "claude"
    if harness in {"comfy", "comfy_ui", "comfyui_outputs"}:
        return "comfyui"
    if harness in {"hermes_agent"}:
        return "hermes"
    if harness in {"local", "local_agent_trace", "lmstudio", "lm_studio"}:
        return "local_agent"
    return harness if harness in HARNESS_MODULES else "generic"


def sniff_harness(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return normalize_harness(explicit)
    name = str(path).lower().replace("-", "_")
    for harness in ("agent_memory", "codex", "claude", "comfyui", "hermes", "local_agent"):
        if harness in name:
            return harness
    if path.is_file() and path.suffix.lower() in COMFYUI_MEDIA_SUFFIXES:
        return "comfyui"
    for record in iter_jsonl(path):
        text = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str).lower()
        if "userpromptsubmit" in text or "posttooluse" in text or "agent_memory" in text:
            return "agent_memory"
        if "codex" in text:
            return "codex"
        if "claude" in text:
            return "claude"
        if "comfyui" in text or ("workflow" in text and "artifact_path" in text):
            return "comfyui"
        if "hermes" in text:
            return "hermes"
        if "lmstudio" in text or "local_agent" in text:
            return "local_agent"
        break
    return "generic"


def stage_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "collected_files": work_dir / "manifests" / "collected_files.json",
        "normalized": work_dir / "jsonl" / "normalized_traces.jsonl",
        "curated": work_dir / "jsonl" / "curated_traces.jsonl",
        "rejected": work_dir / "jsonl" / "rejected_traces.jsonl",
        "scored": work_dir / "jsonl" / "quality_scored.jsonl",
        "protected": work_dir / "jsonl" / "protected_empty.jsonl",
        "contamination": work_dir / "jsonl" / "contamination_scanned.jsonl",
        "sft": work_dir / "exports" / "sft_traces.jsonl",
        "teacher_jobs": work_dir / "teacher_jobs" / "teacher_jobs_2026.jsonl",
        "manifest": work_dir / "manifests" / "trace_orchestrator_manifest.json",
        "dataset_card": work_dir / "dataset_card.md",
    }


def source_entries(profile: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    trace_inputs = profile.get("trace_inputs") if isinstance(profile.get("trace_inputs"), dict) else {}
    configured = trace_inputs.get("sources") or trace_inputs.get("roots") or profile.get("sources") or []
    if isinstance(configured, (str, dict)):
        configured = [configured]
    entries: list[dict[str, Any]] = []
    for item in configured:
        if isinstance(item, str):
            entries.append({"path": str(resolve_path(item, root) or item)})
        elif isinstance(item, dict):
            path = resolve_path(str(item.get("path") or item.get("root") or ""), root)
            if path is not None:
                entries.append({**item, "path": str(path)})
    return entries


def collect_jsonl(profile: dict[str, Any], root: Path, out_path: Path) -> list[dict[str, Any]]:
    trace_inputs = profile.get("trace_inputs") if isinstance(profile.get("trace_inputs"), dict) else {}
    patterns = trace_inputs.get("patterns") if isinstance(trace_inputs.get("patterns"), list) else ["*.jsonl"]
    excludes = [str(item).lower() for item in trace_inputs.get("exclude_substrings", []) if isinstance(item, str)]
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in source_entries(profile, root):
        source_path = Path(str(entry["path"]))
        harness = sniff_harness(source_path, entry.get("harness"))
        candidate_paths: list[Path] = []
        if harness == "comfyui" and source_path.is_dir():
            for pattern in patterns:
                candidate_paths.extend(source_path.rglob(pattern))
            if any(item.is_file() and item.suffix.lower() in COMFYUI_MEDIA_SUFFIXES for item in source_path.rglob("*")):
                candidate_paths.append(source_path)
        elif harness == "comfyui" and source_path.is_file() and source_path.suffix.lower() in COMFYUI_MEDIA_SUFFIXES:
            candidate_paths = [source_path]
        elif source_path.is_file() and source_path.suffix.lower() == ".jsonl":
            candidate_paths = [source_path]
        elif source_path.is_dir():
            for pattern in patterns:
                candidate_paths.extend(sorted(source_path.rglob(str(pattern))))
        for path in candidate_paths:
            resolved = str(path.resolve())
            lowered = resolved.lower()
            if resolved in seen or any(marker in lowered for marker in excludes):
                continue
            seen.add(resolved)
            files.append(
                {
                    "path": resolved,
                    "harness": harness if harness != "generic" else sniff_harness(path, entry.get("harness")),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    write_json(
        out_path,
        {
            "schema": "omnicoder.trace_orchestrator.collected_files.v1",
            "created_at": now_iso(),
            "jsonl_fallback_first": True,
            "files": files,
        },
    )
    return files


def normalize_file(path: Path, harness: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = profile.get("data") if isinstance(profile.get("data"), dict) else {}
    source_date = str(profile.get("source_date") or data_cfg.get("source_date") or "2026-05-23")
    split = str(data_cfg.get("split") or "train")
    bucket = str(data_cfg.get("bucket") or f"{harness}_trace")
    per_file_limit = int(data_cfg.get("per_file_limit") or 0)

    direct_rows: list[dict[str, Any]] = []
    checked = False
    if path.is_file() and path.suffix.lower() == ".jsonl":
        for record in iter_jsonl(path):
            checked = True
            if not (isinstance(record.get("input_json"), dict) and isinstance(record.get("target_json"), dict)):
                direct_rows = []
                break
            direct_rows.append(record)
            if per_file_limit and len(direct_rows) >= per_file_limit:
                break
    if checked and direct_rows:
        rows = direct_rows
    elif harness == "codex":
        rows = ingest_codex_transcripts.build_records(path, bucket, split, source_date, per_file_limit)
    elif harness == "agent_memory":
        rows = ingest_agent_memory.build_records(path, bucket, split, source_date, per_file_limit)
    elif harness == "comfyui":
        rows = ingest_comfyui_outputs.build_records(path, bucket, split, source_date, per_file_limit)
    else:
        rows = ingest_2026.build_training_records(path, bucket, split, source_date)
        if per_file_limit:
            rows = rows[:per_file_limit]
    for source_index, row in enumerate(rows, 1):
        row.setdefault("bucket", bucket)
        row.setdefault("split", split)
        row.setdefault("source_date", source_date)
        lineage = row.setdefault("lineage", {})
        if isinstance(lineage, dict):
            lineage["source_harness"] = harness
            lineage["source_file"] = str(path)
            lineage.setdefault("source_index", source_index)
            lineage.setdefault("step_index", source_index)
            lineage["normalizer_module"] = HARNESS_MODULES.get(harness, HARNESS_MODULES["generic"])
            lineage.setdefault("record_hash", stable_hash(row))
        row["trace_orchestrator"] = {
            "harness": harness,
            "jsonl_fallback_first": True,
            "normalized_at": now_iso(),
        }
        enrich_trace_row(row, harness, path)
    return rows


def normalize_traces(files: list[dict[str, Any]], profile: dict[str, Any], out_path: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    global_limit = int((profile.get("data") if isinstance(profile.get("data"), dict) else {}).get("limit") or 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for item in files:
            path = Path(str(item["path"]))
            harness = normalize_harness(str(item.get("harness") or "generic"))
            try:
                for row in normalize_file(path, harness, profile):
                    if global_limit and written >= global_limit:
                        break
                    handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
                    written += 1
            except Exception as exc:
                errors.append({"path": str(path), "harness": harness, "error": str(exc)})
            if global_limit and written >= global_limit:
                break
    return {"records": written, "errors": errors, "harnesses": sorted({str(item.get("harness")) for item in files})}


def run_postgres_hooks(files: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("postgres") if isinstance(profile.get("postgres"), dict) else {}
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "mode": "jsonl_fallback_only"}
    data_cfg = profile.get("data") if isinstance(profile.get("data"), dict) else {}
    source_date = str(profile.get("source_date") or data_cfg.get("source_date") or "2026-05-23")
    split = str(data_cfg.get("split") or "train")
    namespace = str(cfg.get("namespace") or "trace")
    license_id = str(cfg.get("license") or "internal")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in files:
        path = Path(str(item["path"]))
        harness = normalize_harness(str(item.get("harness") or "generic"))
        bucket = str(data_cfg.get("bucket") or f"{harness}_trace")
        try:
            rows = normalize_file(path, harness, profile)
            args = argparse.Namespace(
                input=str(path),
                dataset_name=f"{harness}_trace_curation_2026",
                namespace=namespace,
                bucket=bucket,
                split=split,
                source_date=source_date,
                license=license_id,
                harness=harness,
            )
            if harness == "codex":
                result = ingest_codex_transcripts.ingest_postgres(args, rows)
            elif harness == "agent_memory":
                result = ingest_agent_memory.ingest_postgres(args, rows)
            else:
                result = ingest_2026.ingest_postgres(args, rows)
            results.append({"path": str(path), "harness": harness, **result})
        except Exception as exc:
            errors.append({"path": str(path), "harness": harness, "error": str(exc)})
            if bool(cfg.get("fail_on_error", False)):
                raise
    return {"enabled": True, "mode": "raw_postgresql_hooks", "results": results, "errors": errors}


def try_external_curation(input_path: Path, out_path: Path, rejected_path: Path, profile: dict[str, Any]) -> dict[str, Any] | None:
    try:
        module = importlib.import_module("omnicoder.data_factory.curation_layers_2026")
    except Exception:
        return None
    for function_name in ("curate_jsonl", "run_jsonl", "run"):
        fn = getattr(module, function_name, None)
        if callable(fn):
            result = fn(str(input_path), str(out_path), str(rejected_path), profile)
            if isinstance(result, dict):
                return {**result, "mode": f"external:{function_name}"}
            return {"mode": f"external:{function_name}", "records": count_jsonl(out_path)}
    main_fn = getattr(module, "main", None)
    if callable(main_fn):
        return None
    return None


def fallback_curation(input_path: Path, out_path: Path, rejected_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("curation_layers") if isinstance(profile.get("curation_layers"), dict) else {}
    if not bool(cfg.get("allow_weak_fallback_curation", False)):
        raise RuntimeError(
            "curation_layers_2026 unavailable and weak fallback curation is disabled; "
            "set curation_layers.allow_weak_fallback_curation=true only for explicit diagnostics"
        )
    require_2025_2026 = bool(cfg.get("require_2025_2026", True))
    min_chars = int(cfg.get("min_chars", 8))
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in iter_jsonl(input_path):
        text = record_text(record).strip()
        lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
        source_date = record.get("source_date") or lineage.get("created_at") or profile.get("source_date")
        year = parse_date_prefix(source_date)
        text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest() if text else stable_hash(record)
        reasons: list[str] = []
        if len(text) < min_chars:
            reasons.append("too_short")
        if text_hash in seen:
            reasons.append("duplicate_text")
        if require_2025_2026 and year is not None and year not in DATE_PREFIXES:
            reasons.append("outside_2025_2026")
        if require_2025_2026 and year is None:
            record["source_date"] = str(profile.get("source_date") or "2026-05-23")
        record["curation"] = {
            "layer": "fallback_curation_layers_2026",
            "accepted": not reasons,
            "reasons": reasons,
            "text_hash": text_hash,
            "date_policy": "2025_2026_only",
        }
        if reasons:
            rejected.append(record)
            continue
        seen.add(text_hash)
        accepted.append(record)
    accepted_count = write_jsonl(out_path, accepted)
    rejected_count = write_jsonl(rejected_path, rejected)
    return {"mode": "fallback_curation_layers_2026", "records": accepted_count, "rejected": rejected_count}


def run_curation(input_path: Path, out_path: Path, rejected_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    external = try_external_curation(input_path, out_path, rejected_path, profile)
    if external is not None:
        return external
    return fallback_curation(input_path, out_path, rejected_path, profile)


def ensure_protected(profile: dict[str, Any], root: Path, fallback_path: Path) -> Path:
    contamination_cfg = profile.get("contamination") if isinstance(profile.get("contamination"), dict) else {}
    protected = resolve_path(contamination_cfg.get("protected_path"), root)
    if protected is not None and protected.exists() and protected.stat().st_size > 0:
        return protected
    allow_empty = bool(contamination_cfg.get("allow_empty_protected") or contamination_cfg.get("allow_empty_protected_smoke"))
    if not allow_empty:
        missing = protected if protected is not None else contamination_cfg.get("protected_path")
        raise FileNotFoundError(
            "protected contamination/eval holdout JSONL is missing or empty; "
            f"refusing to continue without explicit contamination.allow_empty_protected=true: {missing}"
        )
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text("", encoding="utf-8")
    return fallback_path


def run_quality(input_path: Path, out_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("quality") if isinstance(profile.get("quality"), dict) else {}
    min_score = float(cfg.get("stage_min_score", 0.0))
    records = quality_scoring.score_jsonl(input_path, out_path, min_score)
    return {"records": records, "min_score": min_score}


def run_contamination(input_path: Path, protected_path: Path, out_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("contamination") if isinstance(profile.get("contamination"), dict) else {}
    threshold = float(cfg.get("threshold", 0.42))
    ngram = int(cfg.get("ngram", 5))
    records = contamination.scan(input_path, protected_path, out_path, threshold, ngram)
    contaminated = 0
    suspect = 0
    unknown = 0
    for record in iter_jsonl(out_path):
        status = (record.get("contamination") or {}).get("status") if isinstance(record.get("contamination"), dict) else None
        if status == "contaminated":
            contaminated += 1
        elif status == "suspect":
            suspect += 1
        elif status in (None, "", "unknown"):
            unknown += 1
    return {
        "records": records,
        "contaminated": contaminated,
        "suspect": suspect,
        "unknown": unknown,
        "protected": str(protected_path),
        "threshold": threshold,
        "ngram": ngram,
    }


def run_sft_export(input_path: Path, out_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    export_cfg = profile.get("export_sft") if isinstance(profile.get("export_sft"), dict) else {}
    quality_cfg = profile.get("quality") if isinstance(profile.get("quality"), dict) else {}
    min_quality = float(export_cfg.get("min_quality", quality_cfg.get("export_min_quality", 0.35)))
    limit = int(export_cfg.get("limit", 0))
    allow_contaminated = bool(export_cfg.get("allow_contaminated", False))
    group_traces = bool(export_cfg.get("group_traces", True))
    if group_traces:
        records = export_sft_jsonl.export_trace_conversations(input_path, out_path, min_quality, allow_contaminated, limit)
    else:
        records = export_sft_jsonl.export_offline(input_path, out_path, min_quality, allow_contaminated, limit)
    return {
        "records": records,
        "min_quality": min_quality,
        "limit": limit,
        "allow_contaminated": allow_contaminated,
        "group_traces": group_traces,
    }


def run_teacher_jobs(input_path: Path, out_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile.get("teacher_jobs") if isinstance(profile.get("teacher_jobs"), dict) else {}
    if not bool(cfg.get("enabled", True)):
        write_jsonl(out_path, [])
        return {"enabled": False, "jobs": 0}
    teacher = str(cfg.get("teacher") or "qwen3.6_27b_q4_local")
    job_type = str(cfg.get("job_type") or "trace_critique")
    limit = int(cfg.get("limit", 0))
    jobs = teacher_jobs_2026.build_jobs(str(input_path), teacher, job_type, limit)
    count = write_jsonl(out_path, jobs)
    result = {"enabled": True, "jobs": count, "teacher": teacher, "job_type": job_type, "queued": False}
    if bool(cfg.get("enqueue_postgres", False)):
        from omnicoder.data_factory.postgres import enqueue_teacher_job

        for job in jobs:
            enqueue_teacher_job(job["teacher_name"], job["job_type"], job["input_json"], priority=int(cfg.get("priority", 100)))
        result["queued"] = True
    return result


def write_dataset_card(path: Path, manifest: dict[str, Any]) -> None:
    stages = manifest.get("stages", {}) if isinstance(manifest.get("stages"), dict) else {}
    sft_records = (stages.get("export_sft") or {}).get("records", 0) if isinstance(stages.get("export_sft"), dict) else 0
    lines = [
        "# Omnicoder 2026 Trace Curation Dataset",
        "",
        "## Summary",
        "",
        f"- Created: {manifest.get('created_at')}",
        f"- Profile: {manifest.get('profile')}",
        f"- JSONL fallback first: {manifest.get('jsonl_fallback_first')}",
        f"- Exported SFT records: {sft_records}",
        "",
        "## Sources",
        "",
    ]
    for item in manifest.get("sources", []):
        if isinstance(item, dict):
            lines.append(f"- {item.get('harness', 'generic')}: `{item.get('path')}`")
    lines.extend(
        [
            "",
            "## Curation",
            "",
            "- Normalizes Codex, Claude, Hermes, and agent-memory traces through existing data-factory ingest modules.",
            "- Applies 2025-2026 date policy, empty-text rejection, and duplicate-text rejection before scoring.",
            "- Runs heuristic quality scoring, contamination scanning, SFT export, and teacher-job construction.",
            "",
            "## Outputs",
            "",
        ]
    )
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    for name, value in outputs.items():
        lines.append(f"- {name}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    profile: dict[str, Any],
    paths: dict[str, Path],
    sources: list[dict[str, Any]],
    stages: dict[str, Any],
) -> dict[str, Any]:
    outputs = {name: str(path) for name, path in paths.items() if path.exists()}
    artifacts = {
        name: {"path": str(path), "records": count_jsonl(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
        if path.exists() and path.suffix == ".jsonl"
    }
    return {
        "schema": "omnicoder.trace_orchestrator_2026.manifest.v1",
        "created_at": now_iso(),
        "profile": profile.get("profile_name") or profile.get("run_name") or "dataset_curation_2026",
        "source_date": profile.get("source_date"),
        "jsonl_fallback_first": True,
        "postgres": profile.get("postgres") if isinstance(profile.get("postgres"), dict) else {"enabled": False},
        "sources": sources,
        "stages": stages,
        "outputs": outputs,
        "artifacts": artifacts,
    }


def write_smoke_input(work_dir: Path) -> Path:
    path = work_dir / "smoke_inputs" / "codex_agent_memory_smoke.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "event_type": "UserPromptSubmit",
            "session_id": "smoke-session-1",
            "created_at": "2026-05-23T12:00:00Z",
            "role": "user",
            "content": "Implement trace curation with JSONL fallback and raw PostgreSQL optional hooks.",
        },
        {
            "event_type": "PostToolUse",
            "session_id": "smoke-session-1",
            "created_at": "2026-05-23T12:01:00Z",
            "tool_name": "shell_command",
            "tool_input": {"command": "python -m omnicoder.data_factory.trace_orchestrator_2026 --smoke"},
            "tool_output": {"status": "ok"},
        },
        {
            "event_type": "AssistantMessage",
            "session_id": "smoke-session-1",
            "created_at": "2026-05-23T12:02:00Z",
            "role": "assistant",
            "content": "Trace curation completed with normalized rows, canonical metadata, grouped SFT export, and teacher jobs.",
        },
    ]
    write_jsonl(path, rows)
    return path


def run_pipeline(profile_path: Path, smoke: bool = False, postgres: bool = False) -> dict[str, Any]:
    root = repo_root()
    profile = read_json(profile_path)
    work_dir_override = os.environ.get("OMNICODER_TRACE_WORK_DIR")
    work_dir = resolve_path(str(work_dir_override or profile.get("work_dir") or "weights/data_factory/trace_orchestrator_2026"), root)
    if work_dir is None:
        raise ValueError("work_dir could not be resolved")
    if smoke:
        smoke_input = write_smoke_input(work_dir)
        profile["trace_inputs"] = {"sources": [{"path": str(smoke_input), "harness": "agent_memory"}], "patterns": ["*.jsonl"]}
        profile.setdefault("data", {})["limit"] = min(int(profile.get("data", {}).get("limit", 64) or 64), 64)
        profile.setdefault("export_sft", {})["limit"] = min(int(profile.get("export_sft", {}).get("limit", 16) or 16), 16)
        profile.setdefault("teacher_jobs", {})["limit"] = min(int(profile.get("teacher_jobs", {}).get("limit", 8) or 8), 8)
    if postgres:
        profile.setdefault("postgres", {})["enabled"] = True
    paths = stage_paths(work_dir)
    if not work_dir_override:
        paths["sft"] = resolve_path((profile.get("export_sft") or {}).get("out") if isinstance(profile.get("export_sft"), dict) else None, root) or paths["sft"]
        paths["teacher_jobs"] = resolve_path((profile.get("teacher_jobs") or {}).get("out") if isinstance(profile.get("teacher_jobs"), dict) else None, root) or paths["teacher_jobs"]
    sources = collect_jsonl(profile, root, paths["collected_files"])
    stages: dict[str, Any] = {"collect": {"files": len(sources)}}
    stages["normalize"] = normalize_traces(sources, profile, paths["normalized"])
    stages["postgres_hooks"] = run_postgres_hooks(sources, profile)
    stages["curation_layers_2026"] = run_curation(paths["normalized"], paths["curated"], paths["rejected"], profile)
    stages["quality_scoring"] = run_quality(paths["curated"], paths["scored"], profile)
    protected_path = ensure_protected(profile, root, paths["protected"])
    stages["contamination"] = run_contamination(paths["scored"], protected_path, paths["contamination"], profile)
    stages["export_sft"] = run_sft_export(paths["contamination"], paths["sft"], profile)
    stages["teacher_jobs"] = run_teacher_jobs(paths["contamination"], paths["teacher_jobs"], profile)
    manifest = build_manifest(profile, paths, sources, stages)
    write_json(paths["manifest"], manifest)
    write_dataset_card(paths["dataset_card"], manifest)
    manifest["outputs"]["manifest"] = str(paths["manifest"])
    manifest["outputs"]["dataset_card"] = str(paths["dataset_card"])
    write_json(paths["manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage orchestrator for 2026 Codex/Claude/Hermes/agent-memory trace curation")
    parser.add_argument("--profile", default="profiles/dataset_curation_2026.json")
    parser.add_argument("--smoke", action="store_true", help="Run against a tiny generated JSONL trace")
    parser.add_argument("--postgres", action="store_true", help="Enable optional raw PostgreSQL hooks configured by sub-stages")
    args = parser.parse_args()

    profile_path = resolve_path(args.profile, repo_root())
    if profile_path is None:
        raise ValueError("profile path is required")
    manifest = run_pipeline(profile_path, smoke=args.smoke, postgres=args.postgres)
    summary = {
        "status": "ok",
        "manifest": manifest["outputs"].get("manifest"),
        "dataset_card": manifest["outputs"].get("dataset_card"),
        "normalized": manifest["stages"]["normalize"].get("records"),
        "curated": manifest["stages"]["curation_layers_2026"].get("records"),
        "sft_records": manifest["stages"]["export_sft"].get("records"),
        "teacher_jobs": manifest["stages"]["teacher_jobs"].get("jobs"),
        "postgres_enabled": bool((manifest.get("postgres") or {}).get("enabled")),
    }
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()

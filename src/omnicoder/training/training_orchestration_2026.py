from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mimetypes
import os
import signal
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.dataset_integrity_2026 import audit_dataset_integrity
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


SCHEMA_VERSION = "2026-05-23"
DEFAULT_PROFILE = "profiles/training_orchestration_2026.json"
DEFAULT_OUT_DIR = "weights/training_orchestration_2026"
MEDIA_SUFFIXES: dict[str, tuple[str, ...]] = {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"),
    "video": (".mp4", ".mov", ".webm", ".mkv", ".avi", ".webp"),
    "audio": (".wav", ".flac", ".mp3", ".m4a", ".ogg"),
    "music": (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".mid", ".midi"),
}
AGGREGATE_JSONL_NAMES = {"curated_records.jsonl", "train_all_modalities.jsonl"}
SUMMARY_NAME_HINTS = ("summary", "workflow", "metadata", "prompt")
TEXT_SUFFIXES = (".txt", ".md", ".rst", ".json", ".jsonl", ".log")
CODE_SUFFIXES = (".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ps1", ".sql", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".yaml", ".yml", ".toml")
TOOL_SUFFIXES = (".json", ".jsonl", ".log", ".md", ".txt")
LONG_CONTEXT_SUFFIXES = (".md", ".txt", ".json", ".jsonl", ".log")
LEDGER_RANGES = DEFAULT_LEDGER.as_config_ranges()
MODALITY_RANGE = {
    "image": "vision_semantic",
    "video": "vision_residual",
    "audio": "speech_tts",
    "music": "audio_music",
    "tool": "tool_agent",
    "long_context": "time_space",
}
DEFAULT_STAGE_ORDER = ("text", "code", "tool", "image", "video", "audio", "music", "long_context")
_LJSPEECH_METADATA_CACHE: dict[str, dict[str, str]] = {}
TARGET_PRESET_2026 = "omnicoder2026_20b_1m"
PROBE_PRESET_NAMES = {"probe", "native1m_probe", "ledger_probe", "full_ledger_probe", "omnicoder2026_native1m_probe", "omnicoder2026_full_ledger_probe"}
ADAPTIVE_SIGNAL_DEFAULTS = (
    "heldout_sample_loss_delta",
    "per_modality_loss_delta",
    "verifier_pass_rate",
    "reward_std",
    "modality_coverage_deficit",
    "q4_regression_delta",
    "contamination_reject_rate",
    "artifact_validation_fail_rate",
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    if not source.exists():
        return
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                payload = {"text": line.rstrip("\n"), "parse_error": str(exc), "line_number": line_number}
            if isinstance(payload, dict):
                payload.setdefault("line_number", line_number)
                yield payload


def artifact_ref_strings(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for container in (row, row.get("input_json"), row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in ("artifact_refs", "artifacts", "artifact_paths", "media_paths", "media_refs", "artifact_metadata", "media_metadata"):
            value = container.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, dict):
                    ref = item.get("path") or item.get("source_path") or item.get("artifact_path") or item.get("file") or item.get("uri") or item.get("url")
                else:
                    ref = item
                ref_text = str(ref).strip() if ref is not None else ""
                if ref_text:
                    refs.append(ref_text)
    return sorted(set(refs))[:64]


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def profile_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    nested = profile.get("training_orchestration")
    return nested if isinstance(nested, dict) else profile


def load_profile(path: str | Path) -> dict[str, Any]:
    profile = read_json(path)
    validate_profile(profile)
    return profile


def enabled_modalities(cfg: dict[str, Any]) -> set[str]:
    modalities = cfg.get("modalities")
    if isinstance(modalities, dict):
        return {str(key) for key, value in modalities.items() if not isinstance(value, dict) or value.get("enabled", True)}
    if isinstance(modalities, list):
        return {str(item) for item in modalities}
    return set(DEFAULT_STAGE_ORDER)


def validate_profile(profile: dict[str, Any]) -> None:
    cfg = profile_cfg(profile)
    required = set(DEFAULT_STAGE_ORDER)
    available = enabled_modalities(cfg)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"missing modalities: {', '.join(missing)}")
    if not isinstance(cfg.get("real_sources"), dict):
        raise ValueError("real_sources must be configured for production training orchestration")
    if not isinstance(cfg.get("training_plan"), dict):
        raise ValueError("training_plan must be configured")
    if not isinstance(cfg.get("learning_checks"), dict) and not isinstance(cfg.get("loss_trend_checks"), dict):
        raise ValueError("learning_checks or loss_trend_checks must be configured")


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def existing_paths(values: Any, root: Path) -> list[Path]:
    if isinstance(values, (str, Path)):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []
    paths = [resolve_path(str(item), root) for item in raw_values if str(item).strip()]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def flatten_path_values(values: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(values, (str, Path)):
        item = str(values).strip()
        if item:
            paths.append(item)
    elif isinstance(values, list):
        for value in values:
            paths.extend(flatten_path_values(value))
    elif isinstance(values, dict):
        for value in values.values():
            paths.extend(flatten_path_values(value))
    return paths


def configured_reportable_roots(
    cfg: dict[str, Any],
    benchmark_profile: str,
    runtime_roots: Any = None,
) -> tuple[list[str], list[str]]:
    roots: list[str] = []
    sources: list[str] = []

    def add(values: Any, source: str) -> None:
        before = len(roots)
        roots.extend(flatten_path_values(values))
        if len(roots) != before:
            sources.append(source)

    gates = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
    add(runtime_roots, "runtime.reportable_task_roots")
    add(cfg.get("reportable_task_roots"), "training_profile.reportable_task_roots")
    add(gates.get("reportable_task_roots"), "training_profile.benchmark_gates.reportable_task_roots")

    benchmark_profile_path = resolve_path(benchmark_profile, repo_root())
    if benchmark_profile_path.exists():
        try:
            benchmark_cfg = read_json(benchmark_profile_path)
            add(benchmark_cfg.get("reportable_task_roots"), f"{benchmark_profile}.reportable_task_roots")
        except Exception as exc:
            sources.append(f"{benchmark_profile}.read_error:{exc}")

    if not roots:
        roots.append("data/eval/reportable_2026")
        sources.append("default:data/eval/reportable_2026")

    unique: list[str] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique, sources


def reportable_prediction_value(row: dict[str, Any]) -> Any:
    for key in (
        "prediction",
        "model_answer",
        "model_output",
        "output",
        "model_patch",
        "model_actions",
        "tool_call",
        "artifact_path",
        "output_path",
        "generated_artifact",
    ):
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def write_reportable_prediction_seed(task_paths: list[Path], out_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in task_paths:
        candidates = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]
        for candidate in candidates:
            for row in iter_jsonl(candidate):
                prediction = reportable_prediction_value(row)
                if prediction is None:
                    continue
                benchmark_id = str(row.get("benchmark_id") or row.get("adapter_id") or "")
                task_id = str(row.get("task_id") or row.get("id") or "")
                if not task_id:
                    task_id = stable_hash({"path": str(candidate), "line": row.get("line_number")})[:16]
                key = f"{benchmark_id}:{task_id}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "task_id": task_id,
                        "prediction": prediction,
                        "source_task_path": str(candidate),
                        "source_line": row.get("line_number"),
                        "prediction_seed": "explicit_model_output_from_authorized_task_or_eval_adapter",
                    }
                )
    write_jsonl(out_path, rows)
    return {"path": str(out_path), "records": len(rows)}


def count_jsonl_rows(path: str | Path) -> int:
    source = Path(path)
    if not source.exists() or not source.is_file():
        return 0
    count = 0
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def row_identity(row: dict[str, Any]) -> str:
    refs = row.get("artifact_refs") if isinstance(row.get("artifact_refs"), list) else []
    ref_keys = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        ref_keys.append(
            (
                str(ref.get("kind") or ""),
                str(ref.get("sha256") or ""),
                str(ref.get("byte_size") or ""),
                str(ref.get("path") or ref.get("source_path") or ""),
            )
        )
    payload = {
        "record_id": row.get("record_id"),
        "modality": row.get("modality"),
        "payload_sha256": row.get("payload_sha256"),
        "refs": sorted(ref_keys),
    }
    return stable_hash(payload)


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row_identity(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def split_bucket_counts(
    total: int,
    eval_ratio: float,
    test_ratio: float,
    *,
    min_train_records: int = 1,
) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "eval": 0, "test": 0}
    if total == 1:
        return {"train": 1, "eval": 0, "test": 0}
    if total == 2:
        return {"train": 1, "eval": 0, "test": 1}
    if total == 3:
        return {"train": 1, "eval": 1, "test": 1}

    min_train = max(1, int(min_train_records))
    test_count = max(1, int(round(total * max(0.0, float(test_ratio)))))
    eval_count = max(1, int(round(total * max(0.0, float(eval_ratio)))))
    reserve = max(0, total - min_train)
    if test_count + eval_count > reserve:
        scale = reserve / max(1, test_count + eval_count)
        test_count = max(1, int(test_count * scale))
        eval_count = max(1, reserve - test_count)
    train_count = total - eval_count - test_count
    if train_count < min_train:
        deficit = min_train - train_count
        take_from_eval = min(deficit, max(0, eval_count - 1))
        eval_count -= take_from_eval
        deficit -= take_from_eval
        take_from_test = min(deficit, max(0, test_count - 1))
        test_count -= take_from_test
        train_count = total - eval_count - test_count
    return {"train": train_count, "eval": eval_count, "test": test_count}


def split_order_key(row: dict[str, Any], modality: str) -> str:
    refs = row.get("artifact_refs") if isinstance(row.get("artifact_refs"), list) else []
    ref_paths = []
    ref_hashes = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        ref_paths.append(str(ref.get("path") or ref.get("source_path") or ""))
        ref_hashes.append(str(ref.get("sha256") or ""))
    return stable_hash(
        {
            "record_id": row.get("record_id"),
            "modality": modality,
            "payload_sha256": row.get("payload_sha256"),
            "source_uri": row.get("source_uri"),
            "ref_paths": sorted(ref_paths),
            "ref_hashes": sorted(ref_hashes),
        }
    )


def assign_deterministic_splits(rows: list[dict[str, Any]], modality: str, plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    split_plan = plan.get("split") if isinstance(plan.get("split"), dict) else {}
    eval_ratio = float(split_plan.get("eval_ratio", plan.get("eval_holdout_ratio", 0.10)))
    test_ratio = float(split_plan.get("test_ratio", plan.get("test_holdout_ratio", 0.10)))
    min_train = int(split_plan.get("min_train_records", plan.get("min_train_records_per_modality", 1)))
    counts = split_bucket_counts(len(rows), eval_ratio, test_ratio, min_train_records=min_train)
    ordered = sorted(rows, key=lambda row: split_order_key(row, modality))
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    boundaries = {
        "test": counts["test"],
        "eval": counts["test"] + counts["eval"],
    }
    for index, row in enumerate(ordered):
        if index < boundaries["test"]:
            split_name = "test"
        elif index < boundaries["eval"]:
            split_name = "eval"
        else:
            split_name = "train"
        item = dict(row)
        item["split"] = split_name
        item["split_key"] = split_order_key(row, modality)
        item["quality_score"] = float(item.get("quality", {}).get("score") or 1.0) if isinstance(item.get("quality"), dict) else 1.0
        item["contamination_status"] = (
            str(item.get("contamination", {}).get("status") or "unknown")
            if isinstance(item.get("contamination"), dict)
            else "unknown"
        )
        split_rows[split_name].append(item)
    return split_rows


def extract_text(record: dict[str, Any]) -> str:
    parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        parts.append(message["content"])
            for key in ("text", "content", "prompt", "completion", "answer", "caption", "question", "title"):
                item = value.get(key)
                if isinstance(item, str):
                    parts.append(item)
            for key in ("input_json", "target_json", "tool_input", "tool_output", "metadata", "lineage"):
                if key in value:
                    visit(value[key])
        elif isinstance(value, list):
            for item in value[:4096]:
                visit(item)

    visit(record)
    return "\n".join(part.strip() for part in parts if part and part.strip())


def text_to_ledger_ids(text: str, limit: int = 384) -> list[int]:
    lo, hi = LEDGER_RANGES["text"]
    span = hi - lo
    data = text.encode("utf-8", errors="ignore")
    if not data:
        return [lo + 1]
    return [lo + 1 + (byte % max(1, span - 1)) for byte in data[: max(1, int(limit))]]


def plan_context_ladder_values(plan: dict[str, Any]) -> list[int]:
    raw = plan.get("context_ladder") or plan.get("long_context_ladder") or []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = []
    parsed: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item > 0:
            parsed.append(item)
    if not parsed:
        parsed = [8192, 32768, 131072, 262144, 524288, 1048576]
    return sorted(dict.fromkeys(max(1024, int(value)) for value in parsed))


def modality_text_token_limit(plan: dict[str, Any], modality: str, *, prompt: bool = False) -> int:
    by_modality = plan.get("text_token_limit_by_modality") if isinstance(plan.get("text_token_limit_by_modality"), dict) else {}
    if modality == "long_context":
        ladder_max = max(plan_context_ladder_values(plan))
        if prompt:
            return int(plan.get("long_context_prompt_token_limit") or min(8192, max(512, ladder_max // 64)))
        return int(plan.get("long_context_text_token_limit") or ladder_max)
    configured = by_modality.get(modality)
    if configured is not None:
        return int(configured)
    return int(plan.get("text_token_limit") or 384)


def modality_target_chars(plan: dict[str, Any], modality: str) -> int:
    by_modality = plan.get("target_text_chars_by_modality") if isinstance(plan.get("target_text_chars_by_modality"), dict) else {}
    configured = by_modality.get(modality)
    if configured is not None:
        return int(configured)
    if modality == "long_context":
        return int(plan.get("long_context_target_chars") or max(plan_context_ladder_values(plan)))
    return int(plan.get("target_text_chars") or 3000)


def hash_file(path: Path, max_hash_bytes: int) -> dict[str, Any]:
    byte_size = path.stat().st_size
    digest = hashlib.sha256()
    read_bytes = 0
    limit = max(0, int(max_hash_bytes))
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            if limit and read_bytes + len(chunk) > limit:
                chunk = chunk[: limit - read_bytes]
            digest.update(chunk)
            read_bytes += len(chunk)
            if limit and read_bytes >= limit:
                break
    return {
        "sha256": digest.hexdigest(),
        "byte_size": byte_size,
        "hashed_bytes": read_bytes,
        "hash_scope": "full" if read_bytes == byte_size else "prefix",
    }


def safe_stat_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "byte_size": stat.st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        "source_date": time.strftime("%Y-%m-%d", time.gmtime(stat.st_mtime)),
    }


def localize_media_path(value: str) -> str:
    text = str(value)
    replacements = (
        ("/opt/ComfyUI/output", "/home/cereal/comfyui/output"),
        ("/workspace/ComfyUI/output", "/home/cereal/comfyui/output"),
        ("/workspace/output", "/home/cereal/comfyui/output"),
    )
    for old, new in replacements:
        if text.startswith(old):
            return new + text[len(old) :]
    return text


def _flatten_strings(value: Any, limit: int = 256) -> list[str]:
    strings: list[str] = []

    def visit(item: Any) -> None:
        if len(strings) >= limit:
            return
        if isinstance(item, str):
            if item.strip():
                strings.append(item.strip())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item[:128]:
                visit(child)

    visit(value)
    return strings


def find_summary_files(roots: list[Path], limit: int, max_bytes: int) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root_path in roots:
        try:
            candidates = sorted(root_path.rglob("*.json")) if root_path.is_dir() else [root_path]
        except (PermissionError, OSError):
            continue
        for item in candidates:
            name = item.name.lower()
            if not any(hint in name for hint in SUMMARY_NAME_HINTS):
                continue
            try:
                size = item.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > max_bytes:
                continue
            key = str(item.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(item)
            if len(found) >= limit:
                return found
    return found


def summarize_workflow_entry(summary_path: Path, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    return {
        "summary_path": str(summary_path),
        "workflow_id": workflow_id,
        "prompt_id": payload.get("prompt_id"),
        "ok": payload.get("ok"),
        "status_str": status.get("status_str"),
        "completed": status.get("completed"),
        "node_errors": payload.get("node_errors"),
        "error": payload.get("error"),
    }


def build_media_metadata_index(roots: list[Path], plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    limit = int(plan.get("max_media_summary_files") or 128)
    max_bytes = int(plan.get("max_media_summary_bytes") or 2 * 1024 * 1024)
    index: dict[str, dict[str, Any]] = {}
    for summary_path in find_summary_files(roots, limit, max_bytes):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        entries = payload.items() if isinstance(payload, dict) else []
        for workflow_id, entry in entries:
            if not isinstance(entry, dict):
                continue
            workflow = summarize_workflow_entry(summary_path, str(workflow_id), entry)
            strings = _flatten_strings(entry)
            media_strings = [
                localize_media_path(item)
                for item in strings
                if Path(localize_media_path(item)).suffix.lower() in {suffix for values in MEDIA_SUFFIXES.values() for suffix in values}
            ]
            for item in media_strings:
                path = Path(item)
                keys = {path.name, path.stem, str(path)}
                for key in keys:
                    if key:
                        index.setdefault(key, workflow)
    return index


def image_header_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = handle.read(65536)
    except OSError as exc:
        return {"decode_ok": False, "decode_error": str(exc)}
    suffix = path.suffix.lower()
    result: dict[str, Any] = {"decode_ok": True}
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        result.update({"width": struct.unpack(">I", data[16:20])[0], "height": struct.unpack(">I", data[20:24])[0], "format": "png"})
    elif data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        result.update({"width": struct.unpack("<H", data[6:8])[0], "height": struct.unpack("<H", data[8:10])[0], "format": "gif", "animated": data.count(b"\x2c") > 1})
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        result.update({"format": "webp", "animated": b"ANIM" in data})
    elif suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8"):
        idx = 2
        width = height = None
        while idx + 9 < len(data):
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            length = int.from_bytes(data[idx + 2 : idx + 4], "big")
            if marker in {0xC0, 0xC2} and idx + 8 < len(data):
                height = int.from_bytes(data[idx + 5 : idx + 7], "big")
                width = int.from_bytes(data[idx + 7 : idx + 9], "big")
                break
            idx += 2 + max(2, length)
        result.update({"format": "jpeg", "width": width, "height": height})
    return result


def ffprobe_metadata(path: Path, timeout_seconds: float) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_seconds), check=False)
    except FileNotFoundError:
        return {"ffprobe_ok": False, "ffprobe_error": "ffprobe_not_found"}
    except subprocess.TimeoutExpired:
        return {"ffprobe_ok": False, "ffprobe_error": "timeout"}
    if proc.returncode != 0:
        return {"ffprobe_ok": False, "ffprobe_error": proc.stderr[-500:]}
    try:
        data = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {"ffprobe_ok": False, "ffprobe_error": str(exc)}
    streams = data.get("streams") if isinstance(data.get("streams"), list) else []
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    first_video = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), {})
    first_audio = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"), {})
    return {
        "ffprobe_ok": True,
        "duration": float(fmt.get("duration")) if fmt.get("duration") not in (None, "N/A") else None,
        "bit_rate": int(fmt.get("bit_rate")) if str(fmt.get("bit_rate") or "").isdigit() else None,
        "stream_count": len(streams),
        "video": {
            "codec_name": first_video.get("codec_name"),
            "width": first_video.get("width"),
            "height": first_video.get("height"),
            "avg_frame_rate": first_video.get("avg_frame_rate"),
            "nb_frames": first_video.get("nb_frames"),
        }
        if first_video
        else None,
        "audio": {
            "codec_name": first_audio.get("codec_name"),
            "sample_rate": first_audio.get("sample_rate"),
            "channels": first_audio.get("channels"),
            "duration": first_audio.get("duration"),
        }
        if first_audio
        else None,
    }


def classify_audio_family(path: Path, metadata: dict[str, Any]) -> str:
    text = f"{path.name} {metadata.get('workflow_id') or ''}".lower()
    if any(marker in text for marker in ("ace", "music", "song", "kaola", "instrument", "beat")):
        return "music"
    if any(marker in text for marker in ("speech", "tts", "transcript", "voice", "dialog", "ltx_audio", "reference_audio", "audio")):
        return "audio"
    return "audio"


def media_metadata(path: Path, modality: str, plan: dict[str, Any], summary_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = safe_stat_payload(path)
    metadata.update(summary_index.get(str(path)) or summary_index.get(path.name) or summary_index.get(path.stem) or {})
    suffix = path.suffix.lower()
    if modality == "image" or suffix in MEDIA_SUFFIXES["image"]:
        metadata["image_header"] = image_header_metadata(path)
    if modality in {"video", "audio", "music"} or suffix in MEDIA_SUFFIXES["video"] or suffix in MEDIA_SUFFIXES["audio"]:
        metadata["probe"] = ffprobe_metadata(path, float(plan.get("ffprobe_timeout_seconds") or 4.0))
    if suffix == ".webp":
        header = metadata.get("image_header") if isinstance(metadata.get("image_header"), dict) else image_header_metadata(path)
        metadata["animated_webp"] = bool(header.get("animated"))
    if modality in {"audio", "music"}:
        metadata["audio_family"] = classify_audio_family(path, metadata)
    return metadata


def media_record_ok(path: Path, modality: str, metadata: dict[str, Any], plan: dict[str, Any]) -> bool:
    min_bytes = int(plan.get("min_media_bytes") or 1024)
    if int(metadata.get("byte_size") or 0) < min_bytes:
        return False
    if modality == "video" and path.suffix.lower() == ".webp" and bool(plan.get("require_animated_webp_for_video", True)):
        return bool(metadata.get("animated_webp"))
    probe = metadata.get("probe") if isinstance(metadata.get("probe"), dict) else {}
    if modality == "video" and probe.get("ffprobe_ok"):
        duration = probe.get("duration")
        if duration is not None and float(duration) < float(plan.get("min_video_seconds") or 0.5):
            return False
    if modality in {"audio", "music"} and probe.get("ffprobe_ok"):
        duration = probe.get("duration")
        if duration is not None and float(duration) < float(plan.get("min_audio_seconds") or 1.0):
            return False
    if modality == "audio" and metadata.get("audio_family") == "music":
        return False
    if modality == "music" and metadata.get("audio_family") != "music":
        return False
    return True


def hash_to_range_tokens(seed: str, range_name: str, token_count: int) -> list[int]:
    lo, hi = LEDGER_RANGES[range_name]
    span = hi - lo
    tokens: list[int] = []
    counter = 0
    while len(tokens) < max(1, int(token_count)):
        digest = hashlib.blake2b(f"{seed}:{counter}".encode("utf-8"), digest_size=32).digest()
        tokens.extend(lo + (byte % span) for byte in digest)
        counter += 1
    return tokens[: max(1, int(token_count))]


def artifact_tokens(path: Path, modality: str, plan: dict[str, Any], media_metadata: dict[str, Any] | None = None) -> tuple[list[int], dict[str, Any]]:
    counts = plan.get("artifact_token_count") if isinstance(plan.get("artifact_token_count"), dict) else {}
    token_count = int(counts.get(modality) or 64)
    max_hash_bytes = int(plan.get("max_hash_bytes") or 256 * 1024 * 1024)
    artifact = hash_file(path, max_hash_bytes=max_hash_bytes)
    range_name = MODALITY_RANGE.get(modality, "vision_semantic")
    token_seed = stable_hash(
        {
            "path": str(path),
            "sha256": artifact["sha256"],
            "byte_size": artifact["byte_size"],
            "kind": modality,
            "mime_type": mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            "media_metadata": media_metadata or {},
        }
    )
    tokens = hash_to_range_tokens(token_seed, range_name, token_count)
    artifact.update(
        {
            "artifact_id": stable_hash({"path": str(path), "sha256": artifact["sha256"]}),
            "uri": str(path),
            "kind": modality,
            "mime_type": mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            "created_at": now_iso(),
            "ledger_range": range_name,
            "token_count": len(tokens),
            "media_metadata": media_metadata or {},
        }
    )
    return tokens, artifact


def make_training_record(
    modality: str,
    prompt: str,
    target: str,
    source_uri: str,
    plan: dict[str, Any],
    artifact_path: Path | None = None,
    source_payload: dict[str, Any] | None = None,
    media_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control_lo, _ = LEDGER_RANGES["control"]
    token_limit = modality_text_token_limit(plan, modality)
    prompt_limit = modality_text_token_limit(plan, modality, prompt=True)
    token_ids: list[int] = [control_lo + (int(stable_hash(modality)[:4], 16) % 4096)]
    artifact_refs: list[dict[str, Any]] = []
    effective_media_metadata = media_metadata or (source_payload if isinstance(source_payload, dict) and ("probe" in source_payload or "image_header" in source_payload) else None) or {}
    if artifact_path is not None:
        media_tokens, artifact = artifact_tokens(artifact_path, modality, plan, effective_media_metadata)
        token_ids.extend(media_tokens)
        artifact_refs.append(artifact)
    elif modality in MODALITY_RANGE:
        token_ids.extend(hash_to_range_tokens(stable_hash(source_payload or prompt), MODALITY_RANGE[modality], int(plan.get("fallback_token_count") or 32)))
    prompt_token_ids = text_to_ledger_ids(prompt, prompt_limit)
    target_token_ids = text_to_ledger_ids(target, token_limit)
    token_ids.extend(prompt_token_ids)
    token_ids.extend(target_token_ids)
    source_date = "2026-05-23"
    if isinstance(source_payload, dict) and source_payload.get("source_date"):
        source_date = str(source_payload.get("source_date"))[:10]
    source_hint = None
    if isinstance(source_payload, dict):
        for key in ("source_id", "id", "event_id", "task_id", "trace_id", "artifact_id", "record_id"):
            value = source_payload.get(key)
            if value not in (None, ""):
                source_hint = value
                break
    source_id = (
        str(source_hint)
        if source_hint is not None
        else stable_hash({"source_uri": source_uri, "modality": modality, "source_payload": source_payload or {}})
    )
    quality_value = 1.0
    quality_label = "accepted_real_source" if not artifact_refs else "accepted_real_media_with_metadata"
    contamination: dict[str, Any] = {"status": "unknown", "note": "real_orchestration_export_requires_downstream_protected_scan"}
    if isinstance(source_payload, dict):
        raw_quality = source_payload.get("quality_score")
        quality_obj = source_payload.get("quality")
        if raw_quality is None and isinstance(quality_obj, dict):
            raw_quality = quality_obj.get("score") or quality_obj.get("overall") or quality_obj.get("quality")
            label = quality_obj.get("label")
            if label:
                quality_label = str(label)
        try:
            if raw_quality is not None:
                quality_value = max(0.0, min(1.0, float(raw_quality)))
        except (TypeError, ValueError):
            quality_value = 1.0
        raw_contamination = source_payload.get("contamination") or source_payload.get("contamination_status")
        if isinstance(raw_contamination, dict):
            contamination = dict(raw_contamination)
        elif raw_contamination:
            contamination = {"status": str(raw_contamination)}
    row = {
        "schema": "omnicoder.real_multimodal_training_2026.v1",
        "record_id": stable_hash({"modality": modality, "source_uri": source_uri, "prompt": prompt, "target": target}),
        "source_id": source_id,
        "modality": modality,
        "modalities": sorted({"text", modality}),
        "source_uri": source_uri,
        "source_date": source_date,
        "curated_at": now_iso(),
        "input_json": {
            "messages": [{"role": "user", "content": prompt}],
            "modality": modality,
            "artifact_refs": artifact_refs,
            "media_metadata": effective_media_metadata,
        },
        "target_json": {
            "content": target,
            "artifact_refs": artifact_refs,
            "media_metadata": effective_media_metadata,
        },
        "token_ids": token_ids,
        "artifact_refs": artifact_refs,
        "media_metadata": effective_media_metadata,
        "payload_sha256": stable_hash({"prompt": prompt, "target": target, "artifact_refs": artifact_refs, "media_metadata": effective_media_metadata}),
        "token_count": len(token_ids),
        "text_token_count": max(0, len(token_ids) - 1 - sum(int(artifact.get("token_count") or 0) for artifact in artifact_refs)),
        "prompt_text_token_count": len(prompt_token_ids),
        "target_text_token_count": len(target_token_ids),
        "prompt_char_count": len(prompt),
        "target_char_count": len(target),
        "quality": {"score": quality_value, "label": quality_label},
        "contamination": contamination,
        "source_payload": source_payload or {},
    }
    return row


def collect_text_like(
    modality: str,
    paths: list[Path],
    plan: dict[str, Any],
    limit: int,
    min_chars: int = 40,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    long_context_parts: list[str] = []
    long_context_sources: list[str] = []
    long_context_chars = 0
    long_target_chars = modality_target_chars(plan, "long_context")

    def flush_long_context_bundle(force: bool = False) -> bool:
        nonlocal long_context_chars
        if modality != "long_context" or not long_context_parts:
            return False
        text = "\n\n".join(long_context_parts).strip()
        min_bundle_chars = int(plan.get("long_context_min_chars_per_record") or min(long_target_chars, max(8192, long_target_chars // 4)))
        if not force and len(text) < min_bundle_chars:
            return False
        spans = [
            text[start : start + long_target_chars]
            for start in range(0, len(text), max(1, long_target_chars))
        ]
        emitted = False
        for span_index, span in enumerate(spans):
            if not span.strip():
                continue
            if not force and len(span) < min_bundle_chars:
                continue
            source_payload = {
                "source_id": stable_hash({"long_context_sources": long_context_sources, "chars": len(span), "span_index": span_index}),
                "source_date": "2026-05-23",
                "sources": list(long_context_sources),
                "packed_long_context": True,
                "char_count": len(span),
                "packed_total_char_count": len(text),
                "span_index": span_index,
            }
            rows.append(
                make_training_record(
                    modality,
                    "Learn the packed long-context retained source span and preserve retrieval anchors across the full context.",
                    span,
                    "packed:" + stable_hash(long_context_sources) + f":{span_index}",
                    plan,
                    source_payload=source_payload,
                )
            )
            emitted = True
            if len(rows) >= limit:
                break
        long_context_parts.clear()
        long_context_sources.clear()
        long_context_chars = 0
        return emitted and len(rows) >= limit

    for path in paths:
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for src in candidates:
            for record in iter_jsonl(src):
                text = extract_text(record)
                if len(text) < min_chars:
                    continue
                if modality == "long_context":
                    long_context_parts.append(text)
                    long_context_sources.append(str(src))
                    long_context_chars += len(text)
                    if long_context_chars >= long_target_chars:
                        if flush_long_context_bundle(force=True):
                            return rows
                    continue
                prompt = f"Learn the {modality} source record and preserve its useful training signal."
                if modality == "code":
                    prompt = "Learn the code and terminal training record."
                elif modality == "tool":
                    prompt = "Learn the agent tool-call and observation trajectory."
                elif modality == "long_context":
                    prompt = "Learn the long-context retained source span and anchor metadata."
                target = text[: modality_target_chars(plan, modality)]
                rows.append(make_training_record(modality, prompt, target, str(src), plan, source_payload=record))
                if len(rows) >= limit:
                    return rows
    if modality == "long_context":
        flush_long_context_bundle(force=True)
    return rows


def text_field(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if parts:
            return "\n".join(parts)
    target_json = record.get("target_json")
    if isinstance(target_json, dict):
        content = target_json.get("content") or target_json.get("text") or target_json.get("caption")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def artifact_field(record: dict[str, Any], modality: str) -> str:
    keys = (
        modality,
        f"{modality}_path",
        "artifact_path",
        "path",
        "file",
        "uri",
        "media",
        "output_path",
        "output",
    )
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    artifact_refs = record.get("artifact_refs")
    if isinstance(artifact_refs, list):
        for artifact in artifact_refs:
            if not isinstance(artifact, dict):
                continue
            kind = str(artifact.get("kind") or "").lower()
            if kind and kind != modality:
                continue
            value = artifact.get("uri") or artifact.get("path")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def collect_modality_jsonl(
    modality: str,
    paths: list[Path],
    plan: dict[str, Any],
    limit: int,
    root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for record in iter_jsonl(path):
            artifact_value = artifact_field(record, modality)
            target = text_field(record, ("text", "caption", "answer", "target", "content", "description", "transcript", "prompt"))
            prompt = text_field(record, ("instruction", "question", "input", "user", "task"))
            if not prompt:
                if modality == "image":
                    prompt = "Describe or edit the real image artifact represented by this ledger packet."
                elif modality == "video":
                    prompt = "Describe temporal motion, visual content, and generation metadata for the real video artifact represented by this ledger packet."
                elif modality == "audio":
                    prompt = "Transcribe, caption, and understand the real speech/audio artifact represented by this ledger packet."
                elif modality == "music":
                    prompt = "Learn the real music/audio generation artifact represented by this ledger packet."
                else:
                    prompt = f"Learn the real {modality} artifact represented by this ledger packet."
            if not target:
                target = compact_media_target(Path(artifact_value), modality, {}) if artifact_value else extract_text(record)
            if not target.strip():
                continue
            artifact_path: Path | None = None
            metadata: dict[str, Any] = {}
            if artifact_value:
                candidate = resolve_path(artifact_value, root)
                if not candidate.exists() and str(artifact_value).startswith("/workspace/"):
                    candidate = Path("/home/cereal/omnicoder_2026_work") / str(artifact_value)[len("/workspace/") :]
                if candidate.exists():
                    artifact_path = candidate
                    try:
                        metadata = media_metadata(candidate, modality, plan, {})
                    except OSError:
                        metadata = safe_stat_payload(candidate)
                    if not media_record_ok(candidate, modality, metadata, plan):
                        continue
                elif modality in {"image", "video", "audio", "music"}:
                    continue
            rows.append(
                make_training_record(
                    modality,
                    prompt.strip(),
                    target.strip(),
                    str(path),
                    plan,
                    artifact_path=artifact_path,
                    source_payload=record,
                    media_metadata=metadata,
                )
            )
            if len(rows) >= limit:
                return rows
    return rows


def collect_image_jsonl(paths: list[Path], plan: dict[str, Any], limit: int, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for record in iter_jsonl(path):
            image_value = record.get("image") or record.get("path") or record.get("artifact_path")
            caption = record.get("text") or record.get("caption") or record.get("answer") or ""
            if not isinstance(image_value, str) or not isinstance(caption, str) or not caption.strip():
                continue
            image_path = resolve_path(image_value, root)
            if not image_path.exists():
                continue
            rows.append(
                make_training_record(
                    "image",
                    "Describe or edit the real image artifact represented by this ledger packet.",
                    caption.strip(),
                    str(path),
                    plan,
                    artifact_path=image_path,
                    source_payload=record,
                )
            )
            if len(rows) >= limit:
                return rows
    return rows


def find_media_files(roots: list[Path], modality: str, limit: int, max_bytes: int) -> list[Path]:
    suffixes = MEDIA_SUFFIXES[modality]
    found: list[Path] = []
    for root_path in roots:
        try:
            candidates = sorted(root_path.rglob("*")) if root_path.is_dir() else [root_path]
        except (PermissionError, OSError):
            continue
        for item in candidates:
            if not item.is_file() or item.suffix.lower() not in suffixes:
                continue
            if max_bytes and item.stat().st_size > max_bytes:
                continue
            found.append(item)
            if len(found) >= limit:
                return found
    return found


def find_text_files(roots: list[Path], suffixes: tuple[str, ...], limit: int, max_bytes: int) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root_path in roots:
        try:
            candidates = sorted(root_path.rglob("*")) if root_path.is_dir() else [root_path]
        except (PermissionError, OSError):
            continue
        for item in candidates:
            if not item.is_file() or item.suffix.lower() not in suffixes:
                continue
            try:
                size = item.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > max_bytes:
                continue
            key = str(item.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(item)
            if len(found) >= limit:
                return found
    return found


def collect_file_text_like(
    modality: str,
    roots: list[Path],
    plan: dict[str, Any],
    limit: int,
    suffixes: tuple[str, ...],
    min_chars: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_bytes = int((plan.get("long_context_max_text_file_bytes") if modality == "long_context" else None) or plan.get("max_text_file_bytes") or 1024 * 1024)
    target_chars = modality_target_chars(plan, modality)
    for path in find_text_files(roots, suffixes, limit, max_bytes):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = text.strip()
        if len(text) < min_chars:
            continue
        if modality == "code":
            prompt = f"Learn and preserve the implementation signal from source file {path.name}."
        elif modality == "tool":
            prompt = f"Learn the tool, log, workflow, or agent-state signal from {path.name}."
        elif modality == "long_context":
            prompt = f"Compress and retrieve useful long-context information from {path.name}."
        else:
            prompt = f"Learn the real text/document signal from {path.name}."
        if modality == "long_context":
            stride = max(target_chars, 1)
            for start in range(0, len(text), stride):
                span = text[start : start + target_chars]
                if len(span) < min_chars:
                    continue
                payload = {"path": str(path), "byte_size": path.stat().st_size, "span_start": start, "span_end": start + len(span)}
                rows.append(make_training_record(modality, prompt, span, str(path), plan, source_payload=payload))
                if len(rows) >= limit:
                    return rows
            continue
        rows.append(make_training_record(modality, prompt, text[:target_chars], str(path), plan, source_payload={"path": str(path), "byte_size": path.stat().st_size}))
        if len(rows) >= limit:
            break
    return rows


def transcript_for_audio(path: Path) -> str:
    for sidecar in (path.with_suffix(".txt"), path.with_suffix(".lab")):
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return " ".join(text.split())
    transcript = path.with_suffix(".trans.txt")
    if transcript.exists():
        stem = path.stem
        for line in transcript.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(stem + " "):
                return line[len(stem) + 1 :].strip()
    sibling = path.parent / f"{path.parent.name}.trans.txt"
    if sibling.exists():
        stem = path.stem
        for line in sibling.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(stem + " "):
                return line[len(stem) + 1 :].strip()
    for candidate in sorted(path.parent.glob("*.trans.txt")) + sorted(path.parent.glob("*.TXT")):
        stem = path.stem
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(stem + " "):
                return line[len(stem) + 1 :].strip()
    ljspeech = transcript_from_ljspeech_metadata(path)
    if ljspeech:
        return ljspeech
    return path.stem.replace("_", " ").replace("-", " ")


def transcript_from_ljspeech_metadata(path: Path) -> str:
    for parent in (path.parent, *path.parents):
        metadata_path = parent / "metadata.csv"
        if metadata_path.exists():
            cache_key = str(metadata_path)
            if cache_key not in _LJSPEECH_METADATA_CACHE:
                mapping: dict[str, str] = {}
                for line in metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split("|")
                    if len(parts) < 2:
                        continue
                    utt_id = parts[0].strip()
                    normalized = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else parts[1].strip()
                    if utt_id and normalized:
                        mapping[utt_id] = " ".join(normalized.split())
                _LJSPEECH_METADATA_CACHE[cache_key] = mapping
            text = _LJSPEECH_METADATA_CACHE[cache_key].get(path.stem)
            if text:
                return text
        if parent.name in {"LJSpeech-1.1", "ljspeech"}:
            break
    return ""


def compact_media_target(path: Path, modality: str, metadata: dict[str, Any]) -> str:
    parts = [f"{modality} artifact {path.name}"]
    workflow = metadata.get("workflow_id")
    prompt_id = metadata.get("prompt_id")
    status = metadata.get("status_str")
    if workflow:
        parts.append(f"workflow={workflow}")
    if prompt_id:
        parts.append(f"prompt_id={prompt_id}")
    if status:
        parts.append(f"status={status}")
    header = metadata.get("image_header") if isinstance(metadata.get("image_header"), dict) else {}
    if header:
        dims = "x".join(str(header.get(key)) for key in ("width", "height") if header.get(key))
        if dims:
            parts.append(f"dims={dims}")
        if header.get("animated") is not None:
            parts.append(f"animated={bool(header.get('animated'))}")
    probe = metadata.get("probe") if isinstance(metadata.get("probe"), dict) else {}
    if probe.get("ffprobe_ok"):
        if probe.get("duration") is not None:
            parts.append(f"duration={round(float(probe['duration']), 3)}s")
        video = probe.get("video") if isinstance(probe.get("video"), dict) else {}
        audio = probe.get("audio") if isinstance(probe.get("audio"), dict) else {}
        if video:
            parts.append(f"video_codec={video.get('codec_name')}")
            if video.get("width") and video.get("height"):
                parts.append(f"video_dims={video.get('width')}x{video.get('height')}")
        if audio:
            parts.append(f"audio_codec={audio.get('codec_name')}")
            if audio.get("sample_rate"):
                parts.append(f"sample_rate={audio.get('sample_rate')}")
    return "; ".join(str(part) for part in parts if part)


def collect_media(
    modality: str,
    roots: list[Path],
    plan: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_bytes = int(plan.get("max_media_bytes") or 512 * 1024 * 1024)
    summary_index = build_media_metadata_index(roots, plan)
    for media_path in find_media_files(roots, modality, limit, max_bytes):
        try:
            metadata = media_metadata(media_path, modality, plan, summary_index)
        except OSError:
            continue
        if not media_record_ok(media_path, modality, metadata, plan):
            continue
        if modality == "audio":
            target = transcript_for_audio(media_path)
            if target == media_path.stem.replace("_", " ").replace("-", " "):
                target = compact_media_target(media_path, modality, metadata)
            prompt = "Transcribe, caption, and understand the real speech/audio artifact represented by this ledger packet."
        elif modality == "music":
            target = compact_media_target(media_path, modality, metadata)
            prompt = "Learn the real music/audio generation artifact represented by this ledger packet."
        elif modality == "video":
            target = compact_media_target(media_path, modality, metadata)
            prompt = "Describe temporal motion, visual content, and generation metadata for the real video artifact represented by this ledger packet."
        else:
            target = compact_media_target(media_path, modality, metadata)
            prompt = f"Describe and preserve useful generation metadata for the real {modality} artifact represented by this ledger packet."
        rows.append(
            make_training_record(
                modality,
                prompt,
                target,
                str(media_path),
                plan,
                artifact_path=media_path,
                source_payload=metadata,
            )
        )
        if len(rows) >= limit:
            return rows
    return rows


def row_prompt(row: dict[str, Any]) -> str:
    input_json = row.get("input_json") if isinstance(row.get("input_json"), dict) else {}
    messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str) and message["content"].strip():
            return message["content"].strip()
    return extract_text(row)[:1000]


def row_target(row: dict[str, Any]) -> str:
    target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
    content = target_json.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return extract_text(target_json if isinstance(target_json, dict) else row)[:3000]


def payload_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = extract_text(value).strip()
        return text or json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    if isinstance(value, list):
        parts = [payload_text(item) for item in value[:64]]
        return "\n".join(part for part in parts if part).strip()
    if value not in (None, "", [], {}):
        return str(value).strip()
    return ""


def first_payload_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    casefold_keys = {str(key).casefold(): key for key in payload}
    for key in keys:
        lookup = key if key in payload else casefold_keys.get(str(key).casefold(), key)
        value = payload.get(lookup)
        text = payload_text(value)
        if text:
            return text
    return ""


def preference_pair_from_payload(payload: dict[str, Any], fallback_chosen: str) -> tuple[str, str]:
    chosen = first_payload_text(
        payload,
        (
            "chosen",
            "chosen_response",
            "preferred",
            "preferred_response",
            "winner_response",
            "positive",
            "accepted",
            "selected",
        ),
    )
    rejected = first_payload_text(
        payload,
        (
            "rejected",
            "rejected_response",
            "negative",
            "loser",
            "loser_response",
            "unpreferred",
            "unpreferred_response",
            "dispreferred",
            "bad_response",
        ),
    )
    response_a = first_payload_text(payload, ("response_a", "response_a_text", "answer_a", "candidate_a", "output_a", "audio_a", "video_a"))
    response_b = first_payload_text(payload, ("response_b", "response_b_text", "answer_b", "candidate_b", "output_b", "audio_b", "video_b"))
    preference = str(
        payload.get("winner")
        or payload.get("preference")
        or payload.get("preferred_label")
        or payload.get("chosen_label")
        or payload.get("label")
        or ""
    ).strip().casefold()
    if response_a and response_b:
        if preference in {"a", "0", "left", "response_a", "answer_a", "candidate_a", "model_a", "audio_a", "video_a"}:
            chosen = chosen or response_a
            rejected = rejected or response_b
        elif preference in {"b", "1", "right", "response_b", "answer_b", "candidate_b", "model_b", "audio_b", "video_b"}:
            chosen = chosen or response_b
            rejected = rejected or response_a
    return (chosen or fallback_chosen).strip(), rejected.strip()


def iter_artifact_manifest_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for row in rows:
        refs = row.get("artifact_refs") if isinstance(row.get("artifact_refs"), list) else []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            artifact_id = str(ref.get("artifact_id") or stable_hash(ref))
            key = stable_hash({"artifact_id": artifact_id, "record_id": row.get("record_id"), "split": row.get("split")})
            if key in seen:
                continue
            seen.add(key)
            yield {
                "schema": "omnicoder.training_artifact_manifest_2026.v1",
                "artifact_id": artifact_id,
                "record_id": row.get("record_id"),
                "source_id": row.get("source_id"),
                "split": row.get("split"),
                "modality": row.get("modality"),
                "kind": ref.get("kind") or row.get("modality") or "other",
                "uri": ref.get("uri") or ref.get("path") or "",
                "sha256": ref.get("sha256") or "",
                "byte_size": int(ref.get("byte_size") or 0),
                "mime_type": ref.get("mime_type") or "application/octet-stream",
                "created_at": ref.get("created_at") or row.get("curated_at") or now_iso(),
                "token_count": int(ref.get("token_count") or 0),
                "media_metadata": ref.get("media_metadata") if isinstance(ref.get("media_metadata"), dict) else {},
                "payload_sha256": row.get("payload_sha256"),
            }


def source_file_manifest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        uri = str(row.get("source_uri") or "")
        if not uri:
            continue
        item = grouped.setdefault(
            uri,
            {
                "schema": "omnicoder.training_source_file_manifest_2026.v1",
                "source_uri": uri,
                "source_file_id": stable_hash({"source_uri": uri}),
                "record_count": 0,
                "modalities": {},
                "splits": {},
                "source_ids": [],
                "payload_sha256": "",
                "created_at": now_iso(),
            },
        )
        item["record_count"] += 1
        modality = str(row.get("modality") or "unknown")
        split = str(row.get("split") or "unknown")
        item["modalities"][modality] = item["modalities"].get(modality, 0) + 1
        item["splits"][split] = item["splits"].get(split, 0) + 1
        source_id = row.get("source_id")
        if source_id and len(item["source_ids"]) < 32:
            item["source_ids"].append(str(source_id))
    for item in grouped.values():
        item["payload_sha256"] = stable_hash(
            {
                "source_uri": item["source_uri"],
                "record_count": item["record_count"],
                "modalities": item["modalities"],
                "splits": item["splits"],
            }
        )
    return sorted(grouped.values(), key=lambda row: row["source_uri"])


def build_cleaned_dataset_manifest(
    cfg: dict[str, Any],
    plan: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    artifact_count: int,
    source_file_count: int,
) -> dict[str, Any]:
    all_rows = train_rows + eval_rows + test_rows
    required = list((cfg.get("record_contracts") or {}).get("training_record_required_fields") or [])
    missing_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    contamination_counts: Counter[str] = Counter()
    quality_scores: list[float] = []
    for row in all_rows:
        for field in required:
            if field not in row or row.get(field) in (None, ""):
                missing_counts[field] += 1
        modality_counts[str(row.get("modality") or "unknown")] += 1
        split_counts[str(row.get("split") or "unknown")] += 1
        contamination_counts[str(row.get("contamination_status") or "unknown")] += 1
        try:
            quality_scores.append(float(row.get("quality_score") or 0.0))
        except (TypeError, ValueError):
            quality_scores.append(0.0)
    return {
        "schema": "omnicoder.cleaned_dataset_manifest_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "status": "passed" if not missing_counts else "failed",
        "records": len(all_rows),
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "test_records": len(test_rows),
        "modalities": dict(sorted(modality_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "required_fields": required,
        "missing_required_field_counts": dict(sorted(missing_counts.items())),
        "contamination_status": dict(sorted(contamination_counts.items())),
        "quality_score": {
            "min": min(quality_scores) if quality_scores else None,
            "max": max(quality_scores) if quality_scores else None,
            "avg": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
        },
        "artifact_count": artifact_count,
        "source_file_count": source_file_count,
        "cleaning_layers": [
            "schema_field_presence",
            "deterministic_train_eval_test_split",
            "sha256_payload_identity",
            "artifact_hash_manifest",
            "contamination_status_preserved",
            "quality_score_preserved",
        ],
        "max_records_per_modality_by_modality": plan.get("max_records_per_modality_by_modality") or {},
    }


def build_dataset_blend_manifest(
    cfg: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blends = cfg.get("curated_dataset_families_2026") if isinstance(cfg.get("curated_dataset_families_2026"), dict) else {}
    train_counts: Counter[str] = Counter(str(row.get("modality") or "unknown") for row in train_rows)
    all_counts: Counter[str] = Counter(str(row.get("modality") or "unknown") for row in train_rows + eval_rows + test_rows)
    total_train = max(1, len(train_rows))
    return {
        "schema": "omnicoder.dataset_blend_manifest_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "records": {"train": len(train_rows), "eval": len(eval_rows), "test": len(test_rows)},
        "train_modality_counts": dict(sorted(train_counts.items())),
        "all_modality_counts": dict(sorted(all_counts.items())),
        "train_modality_fraction": {key: value / total_train for key, value in sorted(train_counts.items())},
        "declared_blends": blends,
        "actual_blend_notes": {
            "agentic_core": "tool/code/long_context rows plus trace-derived text",
            "image_video_audio_music": "image/video/audio/music artifact and manifest rows",
            "long_context_1m": "long_context rows generated from trace and document roots",
        },
    }


def integrity_scan_jsonl(path: str | Path, *, max_records: int = 0, scan_artifacts: bool = True, max_artifact_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    source = Path(path)
    report: dict[str, Any] = {
        "schema": "omnicoder.dataset_integrity_file_scan_2026.v1",
        "path": str(source),
        "exists": source.exists(),
        "records": 0,
        "rejected": 0,
        "status": "passed",
        "reasons": {},
        "max_records": int(max_records),
        "scan_artifacts": bool(scan_artifacts),
    }
    if not source.exists() or not source.is_file():
        report["status"] = "missing_or_empty"
        return report
    if source.stat().st_size <= 0:
        report["empty"] = True
        return report
    reason_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in iter_jsonl(source):
        audit = audit_dataset_integrity(
            row,
            prompt=row_prompt(row),
            target=row_target(row),
            modality=str(row.get("modality") or ""),
            source_path=source,
            refs=artifact_ref_strings(row),
            scan_artifacts=scan_artifacts,
            max_artifact_bytes=max_artifact_bytes,
        )
        report["records"] = int(report["records"]) + 1
        if not audit.get("accepted", True):
            report["rejected"] = int(report["rejected"]) + 1
            for reason in audit.get("reasons") or ["unknown"]:
                reason_counts[str(reason)] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "line_number": row.get("line_number"),
                        "record_id": row.get("record_id") or row.get("id"),
                        "reasons": audit.get("reasons") or [],
                        "source_id": row.get("source_id"),
                    }
                )
        if max_records and int(report["records"]) >= max_records:
            break
    report["reasons"] = dict(sorted(reason_counts.items()))
    report["examples"] = examples
    report["status"] = "failed" if int(report["rejected"]) > 0 else "passed"
    return report


def training_bound_jsonl_paths_from_manifest(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []

    def add(value: Any) -> None:
        if not value:
            return
        path = Path(str(value))
        if not path.is_absolute():
            path = resolve_path(path, repo_root())
        if path not in paths:
            paths.append(path)

    for key in ("train_all_jsonl", "curated_jsonl"):
        add(manifest.get(key))
    per_modality = manifest.get("per_modality_jsonl") if isinstance(manifest.get("per_modality_jsonl"), dict) else {}
    for value in per_modality.values():
        add(value)
    curricula = manifest.get("curriculum_jsonl") if isinstance(manifest.get("curriculum_jsonl"), dict) else {}
    for key, value in curricula.items():
        if str(key).startswith(("eval_", "test_")):
            continue
        add(value)
    posttraining = manifest.get("posttraining_curation_exports")
    if isinstance(posttraining, dict):
        exports = posttraining.get("exports") if isinstance(posttraining.get("exports"), dict) else {}
        for key, value in exports.items():
            if str(key) == "safety_negative":
                continue
            add(value)
    return sorted([path for path in paths if path.name.endswith(".jsonl")], key=lambda item: str(item))


def run_integrity_preflight(
    paths: Iterable[str | Path],
    out_dir: Path,
    *,
    label: str,
    max_records: int = 0,
    scan_artifacts: bool = True,
    max_artifact_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    unique: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    reports = [
        integrity_scan_jsonl(
            path,
            max_records=max_records,
            scan_artifacts=scan_artifacts,
            max_artifact_bytes=max_artifact_bytes,
        )
        for path in unique
    ]
    failed = [report for report in reports if report.get("status") != "passed"]
    payload = {
        "schema": "omnicoder.training_integrity_preflight_2026.v1",
        "created_at": now_iso(),
        "label": label,
        "status": "failed" if failed else "passed",
        "path_count": len(unique),
        "records": sum(int(report.get("records") or 0) for report in reports),
        "rejected": sum(int(report.get("rejected") or 0) for report in reports),
        "reports": reports,
        "policy": {
            "reject_on_any_integrity_issue": True,
            "scan_artifacts": bool(scan_artifacts),
            "max_records_per_file": int(max_records),
            "max_artifact_bytes": int(max_artifact_bytes),
        },
    }
    path = out_dir / "manifests" / "integrity" / f"{safe_filename(label)}_integrity_preflight.json"
    write_json(path, payload)
    payload["manifest"] = str(path)
    return payload


def require_integrity_preflight(preflight: dict[str, Any]) -> None:
    if preflight.get("status") != "passed":
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "dataset_integrity_preflight_failed",
                    "manifest": preflight.get("manifest"),
                    "rejected": preflight.get("rejected"),
                    "path_count": preflight.get("path_count"),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )


def build_posttraining_curation_exports(
    profile: dict[str, Any],
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del profile, eval_rows, test_rows
    export_dir = out_dir / "agentic_tool_training_2026"
    manifest_dir = export_dir / "manifests"
    sft_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    rlvr_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    for row in train_rows:
        prompt = row_prompt(row)
        target = row_target(row)
        if not prompt or not target:
            continue
        base = {
            "source_record_id": row.get("record_id"),
            "source_id": row.get("source_id"),
            "modality": row.get("modality"),
            "artifact_refs": row.get("artifact_refs") or [],
            "quality_score": row.get("quality_score", 1.0),
            "contamination_status": row.get("contamination_status", "unknown"),
        }
        sft_rows.append(
            {
                "schema": "omnicoder.posttraining_sft_2026.v1",
                "training_kind": "tool_sft",
                "record_id": stable_hash({"kind": "sft", "source_record_id": row.get("record_id")}),
                "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}],
                **base,
            }
        )
        reward_rows.append(
            {
                "schema": "omnicoder.posttraining_reward_2026.v1",
                "training_kind": "tool_reward",
                "record_id": stable_hash({"kind": "reward", "source_record_id": row.get("record_id")}),
                "prompt": prompt,
                "response": target,
                "reward": float(row.get("quality_score") or 1.0),
                "reward_source": "curation_quality_score",
                **base,
            }
        )
        rlvr_rows.append(
            {
                "schema": "omnicoder.posttraining_rlvr_2026.v1",
                "training_kind": "tool_rlvr",
                "record_id": stable_hash({"kind": "rlvr", "source_record_id": row.get("record_id")}),
                "prompt": prompt,
                "expected_answer": target,
                "verifier": "exact_or_artifact_quality_judge",
                "reward_axes": ["answer_consistency", "artifact_hash_integrity", "contamination_free", "modality_grounding"],
                **base,
            }
        )
        source_payload = row.get("source_payload") if isinstance(row.get("source_payload"), dict) else {}
        chosen, rejected = preference_pair_from_payload(source_payload, target)
        if rejected:
            preference_rows.append(
                {
                    "schema": "omnicoder.posttraining_preference_2026.v1",
                    "training_kind": "tool_preference",
                    "record_id": stable_hash({"kind": "preference", "source_record_id": row.get("record_id")}),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    **base,
                }
            )
        text = f"{prompt}\n{target}".lower()
        if any(marker in text for marker in ("password", "token", "secret", "credential", "api key")):
            safety_rows.append(
                {
                    "schema": "omnicoder.posttraining_safety_negative_2026.v1",
                    "training_kind": "tool_safety_negative",
                    "record_id": stable_hash({"kind": "safety", "source_record_id": row.get("record_id")}),
                    "prompt": prompt,
                    "unsafe_response": target,
                    "safety_label": "credential_or_secret_risk",
                    **base,
                }
            )
    paths = {
        "sft": export_dir / "tool_sft.jsonl",
        "reward": export_dir / "tool_reward.jsonl",
        "preference": export_dir / "tool_preference.jsonl",
        "rlvr": export_dir / "tool_rlvr.jsonl",
        "safety_negative": export_dir / "tool_safety_negatives.jsonl",
    }
    counts = {
        "sft": write_jsonl(paths["sft"], sft_rows),
        "reward": write_jsonl(paths["reward"], reward_rows),
        "preference": write_jsonl(paths["preference"], preference_rows),
        "rlvr": write_jsonl(paths["rlvr"], rlvr_rows),
        "safety_negative": write_jsonl(paths["safety_negative"], safety_rows),
    }
    manifest = {
        "schema": "omnicoder.posttraining_curation_manifest_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "export_dir": str(export_dir),
        "exports": {key: str(value) for key, value in paths.items()},
        "counts": counts,
        "algorithm_route_map": {
            "sft": "sft",
            "reward_weighted_sft_replay": "sft",
            "reward_model": "reward",
            "process_reward_replay": "reward",
            "grpo_rlvr_replay": "rlvr",
            "browser_research_rlvr": "rlvr",
            "desktop_gui_rl": "rlvr",
            "multimodal_rlaif_v": "rlvr",
            "dpo_preference_replay": "preference",
            "orpo_kto_simpo_pair_replay": "preference",
            "safety_negative_replay": "safety_negative",
        },
        "notes": [
            "Exports are derived from cleaned train split rows only.",
            "Preference and safety exports stay empty unless real rejected/safety-negative evidence exists.",
        ],
    }
    manifest_path = manifest_dir / "posttraining_curation_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_real_corpus(profile: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    root = repo_root()
    cfg = profile_cfg(profile)
    sources = cfg["real_sources"]
    plan = cfg["training_plan"]
    out_jsonl = out_dir / "jsonl"
    max_per_modality = int(plan.get("max_records_per_modality") or 128)
    per_modality_limit = plan.get("max_records_per_modality_by_modality")
    if not isinstance(per_modality_limit, dict):
        per_modality_limit = {}

    def modality_limit(modality: str) -> int:
        value = per_modality_limit.get(modality)
        try:
            return max(0, int(value))
        except Exception:
            return max_per_modality

    rows_by_modality: dict[str, list[dict[str, Any]]] = {}

    trace_paths = existing_paths(sources.get("trace_jsonl"), root)
    text_roots = existing_paths(sources.get("text_roots"), root)
    code_roots = existing_paths(sources.get("code_roots"), root)
    tool_roots = existing_paths(sources.get("tool_roots"), root)
    long_context_roots = existing_paths(sources.get("long_context_roots"), root)
    tool_limit = modality_limit("tool")
    long_context_limit = modality_limit("long_context")
    text_limit = modality_limit("text")
    code_limit = modality_limit("code")
    rows_by_modality["tool"] = (
        collect_text_like("tool", trace_paths, plan, tool_limit)
        + collect_file_text_like("tool", tool_roots, plan, tool_limit, TOOL_SUFFIXES, min_chars=20)
    )
    rows_by_modality["long_context"] = (
        collect_file_text_like("long_context", long_context_roots, plan, long_context_limit, LONG_CONTEXT_SUFFIXES, min_chars=120)
        + collect_text_like("long_context", trace_paths, plan, long_context_limit, min_chars=120)
    )
    rows_by_modality["text"] = (
        collect_text_like("text", existing_paths(sources.get("text_jsonl"), root), plan, text_limit)
        + collect_file_text_like("text", text_roots, plan, text_limit, TEXT_SUFFIXES, min_chars=20)
        + collect_text_like("text", trace_paths, plan, text_limit)
    )
    rows_by_modality["code"] = (
        collect_file_text_like("code", code_roots, plan, code_limit, CODE_SUFFIXES, min_chars=20)
        + collect_text_like("code", existing_paths(sources.get("code_jsonl"), root) + trace_paths, plan, code_limit)
    )

    media_roots = existing_paths(sources.get("media_roots"), root)
    image_limit = modality_limit("image")
    video_limit = modality_limit("video")
    audio_limit = modality_limit("audio")
    music_limit = modality_limit("music")
    image_rows = collect_modality_jsonl("image", existing_paths(sources.get("image_jsonl"), root), plan, image_limit, root)
    if len(image_rows) < image_limit:
        image_rows.extend(collect_media("image", existing_paths(sources.get("image_roots"), root) + media_roots, plan, image_limit - len(image_rows)))
    rows_by_modality["image"] = image_rows
    video_rows = collect_modality_jsonl("video", existing_paths(sources.get("video_jsonl"), root), plan, video_limit, root)
    if len(video_rows) < video_limit:
        video_rows.extend(collect_media("video", existing_paths(sources.get("video_roots"), root) + media_roots, plan, video_limit - len(video_rows)))
    rows_by_modality["video"] = video_rows
    audio_rows = collect_modality_jsonl("audio", existing_paths(sources.get("audio_jsonl"), root), plan, audio_limit, root)
    if len(audio_rows) < audio_limit:
        audio_rows.extend(collect_media("audio", existing_paths(sources.get("audio_roots"), root) + media_roots, plan, audio_limit - len(audio_rows)))
    rows_by_modality["audio"] = audio_rows
    music_rows = collect_modality_jsonl("music", existing_paths(sources.get("music_jsonl"), root), plan, music_limit, root)
    if len(music_rows) < music_limit:
        music_rows.extend(collect_media("music", existing_paths(sources.get("music_roots"), root) + media_roots, plan, music_limit - len(music_rows)))
    rows_by_modality["music"] = music_rows
    for modality, rows in list(rows_by_modality.items()):
        rows_by_modality[modality] = dedupe_rows(rows)[:modality_limit(modality)]

    all_rows: list[dict[str, Any]] = []
    eval_all_rows: list[dict[str, Any]] = []
    test_all_rows: list[dict[str, Any]] = []
    per_modality_paths: dict[str, str] = {}
    per_modality_split_paths: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for modality in DEFAULT_STAGE_ORDER:
        rows = rows_by_modality.get(modality, [])
        split_rows = assign_deterministic_splits(rows, modality, plan)
        per_modality_split_paths[modality] = {}
        split_counts[modality] = {}
        for split_name in ("train", "eval", "test"):
            path = out_jsonl / f"{split_name}_{modality}.jsonl"
            write_jsonl(path, split_rows[split_name])
            per_modality_split_paths[modality][split_name] = str(path)
            split_counts[modality][split_name] = len(split_rows[split_name])
        train_path = per_modality_split_paths[modality]["train"]
        per_modality_paths[modality] = train_path
        counts[modality] = len(split_rows["train"])
        rows_by_modality[modality] = split_rows["train"]
        all_rows.extend(split_rows["train"])
        eval_all_rows.extend(split_rows["eval"])
        test_all_rows.extend(split_rows["test"])

    train_all = out_jsonl / "train_all_modalities.jsonl"
    eval_all = out_jsonl / "eval_all_modalities.jsonl"
    test_all = out_jsonl / "test_all_modalities.jsonl"
    curated = out_jsonl / "curated_records.jsonl"
    media_focus = out_jsonl / "train_media_focus.jsonl"
    agentic_focus = out_jsonl / "train_agentic_focus.jsonl"
    eval_media_focus = out_jsonl / "eval_media_focus.jsonl"
    eval_agentic_focus = out_jsonl / "eval_agentic_focus.jsonl"
    test_media_focus = out_jsonl / "test_media_focus.jsonl"
    test_agentic_focus = out_jsonl / "test_agentic_focus.jsonl"
    media_focus_rows = [
        row
        for modality in ("image", "video", "audio", "music")
        for row in rows_by_modality.get(modality, [])
    ]
    agentic_focus_rows = [
        row
        for modality in ("tool", "code", "long_context")
        for row in rows_by_modality.get(modality, [])
    ]
    eval_media_focus_rows = [row for row in eval_all_rows if row.get("modality") in {"image", "video", "audio", "music"}]
    eval_agentic_focus_rows = [row for row in eval_all_rows if row.get("modality") in {"tool", "code", "long_context"}]
    test_media_focus_rows = [row for row in test_all_rows if row.get("modality") in {"image", "video", "audio", "music"}]
    test_agentic_focus_rows = [row for row in test_all_rows if row.get("modality") in {"tool", "code", "long_context"}]
    write_jsonl(train_all, all_rows)
    write_jsonl(eval_all, eval_all_rows)
    write_jsonl(test_all, test_all_rows)
    write_jsonl(curated, all_rows + eval_all_rows + test_all_rows)
    write_jsonl(media_focus, media_focus_rows)
    write_jsonl(agentic_focus, agentic_focus_rows)
    write_jsonl(eval_media_focus, eval_media_focus_rows)
    write_jsonl(eval_agentic_focus, eval_agentic_focus_rows)
    write_jsonl(test_media_focus, test_media_focus_rows)
    write_jsonl(test_agentic_focus, test_agentic_focus_rows)
    manifest_root = out_dir / "manifests"
    artifact_manifest = manifest_root / "artifacts.jsonl"
    source_files_manifest = manifest_root / "source_files.jsonl"
    cleaned_dataset_manifest_path = manifest_root / "cleaned_dataset_manifest.json"
    dataset_blend_manifest_path = manifest_root / "dataset_blend_manifest.json"
    artifact_rows = list(iter_artifact_manifest_rows(all_rows + eval_all_rows + test_all_rows))
    source_file_rows = source_file_manifest_rows(all_rows + eval_all_rows + test_all_rows)
    write_jsonl(artifact_manifest, artifact_rows)
    write_jsonl(source_files_manifest, source_file_rows)
    cleaned_dataset_manifest = build_cleaned_dataset_manifest(
        cfg,
        plan,
        all_rows,
        eval_all_rows,
        test_all_rows,
        artifact_count=len(artifact_rows),
        source_file_count=len(source_file_rows),
    )
    dataset_blend_manifest = build_dataset_blend_manifest(cfg, all_rows, eval_all_rows, test_all_rows)
    posttraining_exports = build_posttraining_curation_exports(profile, out_dir, all_rows, eval_all_rows, test_all_rows)
    write_json(cleaned_dataset_manifest_path, cleaned_dataset_manifest)
    write_json(dataset_blend_manifest_path, dataset_blend_manifest)
    manifest = {
        "schema": "omnicoder.real_training_curation_manifest_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "created_at": now_iso(),
        "profile_name": cfg.get("profile_name") or profile.get("profile_name"),
        "train_all_jsonl": str(train_all),
        "eval_all_jsonl": str(eval_all),
        "test_all_jsonl": str(test_all),
        "curated_jsonl": str(curated),
        "cleaned_dataset_manifest": str(cleaned_dataset_manifest_path),
        "artifact_manifest_jsonl": str(artifact_manifest),
        "source_files_manifest_jsonl": str(source_files_manifest),
        "dataset_blend_manifest": str(dataset_blend_manifest_path),
        "posttraining_curation_exports": posttraining_exports,
        "curriculum_jsonl": {
            "media_focus": str(media_focus),
            "agentic_focus": str(agentic_focus),
            "eval_media_focus": str(eval_media_focus),
            "eval_agentic_focus": str(eval_agentic_focus),
            "test_media_focus": str(test_media_focus),
            "test_agentic_focus": str(test_agentic_focus),
        },
        "per_modality_jsonl": per_modality_paths,
        "per_modality_split_jsonl": per_modality_split_paths,
        "records": len(all_rows),
        "eval_records": len(eval_all_rows),
        "test_records": len(test_all_rows),
        "modalities": counts,
        "split_counts": split_counts,
        "split_plan": {
            "eval_ratio": float((plan.get("split") or {}).get("eval_ratio", plan.get("eval_holdout_ratio", 0.10))) if isinstance(plan.get("split"), dict) else float(plan.get("eval_holdout_ratio", 0.10)),
            "test_ratio": float((plan.get("split") or {}).get("test_ratio", plan.get("test_holdout_ratio", 0.10))) if isinstance(plan.get("split"), dict) else float(plan.get("test_holdout_ratio", 0.10)),
            "strategy": "deterministic_hash_by_modality_and_source",
            "max_records_per_modality": max_per_modality,
            "max_records_per_modality_by_modality": dict(sorted((str(key), int(value)) for key, value in per_modality_limit.items() if str(key) in DEFAULT_STAGE_ORDER)),
        },
        "sources": {key: [str(path) for path in existing_paths(value, root)] for key, value in sources.items() if key.endswith(("jsonl", "roots")) or key == "media_roots"},
        "ledger": DEFAULT_LEDGER.as_metadata(),
    }
    manifest["dataset_integrity_preflight"] = run_integrity_preflight(
        training_bound_jsonl_paths_from_manifest(manifest),
        out_dir,
        label="real_corpus_training_bound",
    )
    if manifest["dataset_integrity_preflight"]["status"] != "passed":
        manifest["status"] = "failed"
    write_json(out_dir / "manifests" / "curation_manifest.json", manifest)
    require_integrity_preflight(manifest["dataset_integrity_preflight"])
    return manifest


def manifest_modalities(manifest: dict[str, Any]) -> dict[str, int]:
    raw = manifest.get("modalities")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = max(0, int(value or 0))
        except Exception:
            result[str(key)] = 0
    return result


def manifest_records(manifest: dict[str, Any]) -> int:
    value = manifest.get("records")
    if isinstance(value, dict):
        return sum(max(0, int(v or 0)) for v in value.values() if isinstance(v, (int, float, str)))
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def normalize_mean_one(weights: dict[str, float], lower: float, upper: float) -> dict[str, float]:
    clamped = {key: clamp_float(value, lower, upper) for key, value in weights.items()}
    mean = sum(clamped.values()) / max(1, len(clamped))
    if mean <= 0:
        return {key: 1.0 for key in clamped}
    return {key: round(clamp_float(value / mean, lower, upper), 4) for key, value in clamped.items()}


def build_adaptive_mixture_plan(
    profile: dict[str, Any],
    out_dir: Path,
    *,
    curation_manifest_path: str | Path | None = None,
    external_manifest_path: str | Path | None = None,
    agentic_manifest_path: str | Path | None = None,
    teacher_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    scheduler = cfg.get("adaptive_training_scheduler_2026")
    scheduler_cfg = scheduler if isinstance(scheduler, dict) else {}
    bounds = scheduler_cfg.get("sample_weight_bounds") if isinstance(scheduler_cfg.get("sample_weight_bounds"), list) else [0.25, 4.0]
    lower = to_float(bounds[0] if bounds else 0.25, 0.25)
    upper = to_float(bounds[1] if len(bounds) > 1 else 4.0, 4.0)
    if lower <= 0 or upper < lower:
        lower, upper = 0.25, 4.0

    curation_path = Path(curation_manifest_path or out_dir / "manifests" / "curation_manifest.json")
    external_path = Path(external_manifest_path or repo_root() / "weights" / "external_datasets_2026" / "latest" / "manifests" / "external_dataset_manifest.json")
    agentic_path = Path(agentic_manifest_path or repo_root() / "weights" / "agentic_tool_training_2026" / "latest_run" / "agentic_tool_training_manifest.json")
    teacher_path = Path(teacher_manifest_path or repo_root() / "weights" / "data_factory" / "teacher_rollouts" / "latest" / "teacher_rollout_manifest.json")

    curation_manifest = read_json_if_exists(curation_path)
    external_manifest = read_json_if_exists(external_path)
    agentic_manifest = read_json_if_exists(agentic_path)
    teacher_manifest = read_json_if_exists(teacher_path)
    cleaned_manifest = read_json_if_exists(curation_manifest.get("cleaned_dataset_manifest", "")) if curation_manifest else {}

    modality_counts: dict[str, int] = {modality: 0 for modality in DEFAULT_STAGE_ORDER}
    for modality, count in manifest_modalities(curation_manifest).items():
        if modality in modality_counts:
            modality_counts[modality] += count
    for modality, count in manifest_modalities(external_manifest).items():
        if modality in modality_counts:
            modality_counts[modality] += count

    nonzero_counts = [count for count in modality_counts.values() if count > 0]
    target_count = max(1.0, (sum(nonzero_counts) / len(nonzero_counts)) if nonzero_counts else 1.0)
    raw_weights: dict[str, float] = {}
    stage_rows: list[dict[str, Any]] = []
    for modality in DEFAULT_STAGE_ORDER:
        count = modality_counts.get(modality, 0)
        coverage_deficit = 1.0 if count <= 0 else max(0.0, (target_count - count) / target_count)
        media_boost = 0.25 if modality in {"image", "video", "audio", "music"} else 0.0
        agentic_boost = 0.20 if modality in {"tool", "code", "long_context"} else 0.0
        raw_weights[modality] = 1.0 + (1.75 * coverage_deficit) + media_boost + agentic_boost
    weights = normalize_mean_one(raw_weights, lower, upper)

    for modality in DEFAULT_STAGE_ORDER:
        count = modality_counts.get(modality, 0)
        stage_rows.append(
            {
                "stage": modality,
                "records": count,
                "weight": weights[modality],
                "status": "data_gap" if count <= 0 else "ready",
                "signals": {
                    "modality_coverage_deficit": round(1.0 if count <= 0 else max(0.0, (target_count - count) / target_count), 6),
                    "target_record_count": round(target_count, 2),
                },
                "recommended_action": "collect_or_distill_before_training" if count <= 0 else "sample_with_adaptive_weight",
            }
        )

    context_ladder = scheduler_cfg.get("context_ladder") if isinstance(scheduler_cfg.get("context_ladder"), list) else [8192, 32768, 131072, 262144, 524288, 1048576]
    ladder_values = [max(1024, int(value)) for value in context_ladder if isinstance(value, (int, float, str)) and str(value).strip()]
    if not ladder_values:
        ladder_values = [8192, 32768, 131072, 262144, 524288, 1048576]
    ladder_values = sorted(dict.fromkeys(ladder_values))
    ladder_total = sum(range(1, len(ladder_values) + 1)) or 1
    context_schedule = [
        {
            "context_length": value,
            "target_fraction": round((index + 1) / ladder_total, 6),
            "route": "native_1m_retention" if value >= 1048576 else "progressive_context_expansion",
        }
        for index, value in enumerate(ladder_values)
    ]

    agentic_counts = agentic_manifest.get("counts") if isinstance(agentic_manifest.get("counts"), dict) else {}
    teacher_counts = teacher_manifest.get("counts") if isinstance(teacher_manifest.get("counts"), dict) else {}
    zero_modalities = [row["stage"] for row in stage_rows if row["records"] <= 0]
    gates_cfg = scheduler_cfg.get("promotion_gates") if isinstance(scheduler_cfg.get("promotion_gates"), dict) else {}
    require_nonzero = bool(gates_cfg.get("require_nonzero_all_modalities", True))
    gate_status = "failed" if require_nonzero and zero_modalities else "passed"
    plan = {
        "schema": "omnicoder.adaptive_mixture_plan_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": gate_status,
        "created_at": now_iso(),
        "enabled": bool(scheduler_cfg.get("enabled", True)),
        "mode": scheduler_cfg.get("mode", "online_reweighting_plus_domain_mixture_agent"),
        "signals": list(scheduler_cfg.get("signals") if isinstance(scheduler_cfg.get("signals"), list) else ADAPTIVE_SIGNAL_DEFAULTS),
        "sample_weight_bounds": [lower, upper],
        "manifests": {
            "curation": str(curation_path),
            "external": str(external_path),
            "agentic": str(agentic_path),
            "teacher": str(teacher_path),
        },
        "source_records": {
            "curation": manifest_records(curation_manifest),
            "external": manifest_records(external_manifest),
            "agentic_total": sum(int(v or 0) for v in agentic_counts.values()) if agentic_counts else 0,
            "teacher_total": sum(int(v or 0) for v in teacher_counts.values()) if teacher_counts else 0,
        },
        "modality_counts": modality_counts,
        "stage_weights": stage_rows,
        "context_schedule": context_schedule,
        "promotion_gates": {
            "status": gate_status,
            "zero_modalities": zero_modalities,
            "require_nonzero_all_modalities": require_nonzero,
            "cleaned_dataset_status": cleaned_manifest.get("status"),
            "max_q4_relative_regression": gates_cfg.get("max_q4_relative_regression", 0.03),
            "min_reward_std": gates_cfg.get("min_reward_std", 0.05),
        },
        "notes": [
            "Weights are normalized around mean=1 and bounded by adaptive_training_scheduler_2026.sample_weight_bounds.",
            "Zero-record modalities fail promotion and are flagged for collection or teacher distillation before full training.",
            "The plan is JSONL-first orchestration metadata; no ORM, SQLite, Pydantic, or SQLAlchemy is required.",
        ],
    }
    target = Path(output_path or scheduler_cfg.get("emit") or out_dir / "manifests" / "mixture_plan.json")
    write_json(target, plan)
    plan["path"] = str(target)
    return plan


def parse_losses(log_path: str | Path) -> list[float]:
    losses: list[float] = []
    path = Path(log_path)
    if not path.exists():
        return losses
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if payload.get("loss") is not None:
            losses.append(float(payload["loss"]))
    return losses


def learning_report(losses: list[float], min_relative_drop: float, min_points: int) -> dict[str, Any]:
    if len(losses) < max(2, int(min_points)):
        return {"status": "failed", "reason": "not_enough_loss_points", "loss_points": len(losses), "losses": losses}
    first = losses[0]
    last = losses[-1]
    best = min(losses)
    relative_drop = (first - last) / max(abs(first), 1e-8)
    best_relative_drop = (first - best) / max(abs(first), 1e-8)
    passed = relative_drop >= min_relative_drop or best_relative_drop >= min_relative_drop
    return {
        "status": "passed" if passed else "failed",
        "loss_first": first,
        "loss_last": last,
        "loss_best": best,
        "relative_drop": round(relative_drop, 8),
        "best_relative_drop": round(best_relative_drop, 8),
        "loss_points": len(losses),
        "min_relative_drop": min_relative_drop,
    }


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _popen_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _terminate_process_group(proc: subprocess.Popen[str], grace_seconds: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=max(0.1, float(grace_seconds)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()


def run_command(cmd: list[str], log_path: Path, timeout_seconds: int = 0) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "command", "cmd": cmd, "timeout_seconds": int(timeout_seconds or 0)}, ensure_ascii=True) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_popen_group_kwargs(),
        )
        assert proc.stdout is not None
        def stream_output() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    handle.write(line)
                    handle.flush()
            except Exception as exc:
                handle.write(json.dumps({"event": "stdout_stream_error", "error": str(exc)}, ensure_ascii=True) + "\n")

        stream_thread = threading.Thread(target=stream_output, name=f"omnicoder-command-log-{log_path.name}", daemon=True)
        stream_thread.start()
        try:
            code = proc.wait(timeout=(float(timeout_seconds) if int(timeout_seconds or 0) > 0 else None))
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            code = 124
            handle.write(json.dumps({"event": "timeout", "returncode": code, "timeout_seconds": int(timeout_seconds)}, ensure_ascii=True) + "\n")
        stream_thread.join(timeout=10.0)
        handle.write(json.dumps({"event": "returncode", "returncode": int(code)}, ensure_ascii=True) + "\n")
        return code


def arg_value(args: argparse.Namespace | None, name: str, default: Any = None) -> Any:
    if args is None:
        return default
    return getattr(args, name, default)


def resolve_save_interval(args: argparse.Namespace | None, configured: Any = None) -> int:
    raw = arg_value(args, "save_interval", None)
    if raw is not None:
        return max(int(raw), 0)
    return max(int(configured or 0), 0)


def truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def namespace_with(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def list_from_config_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def resolve_stage_order(plan: dict[str, Any], args: argparse.Namespace | None = None) -> list[str]:
    override = list_from_config_value(arg_value(args, "stage_order", ""))
    configured = list_from_config_value(plan.get("stage_order")) or list(DEFAULT_STAGE_ORDER)
    stage_order = override or configured
    if not stage_order:
        raise ValueError("stage_order resolved to an empty stage list")
    duplicates = [item for item, count in Counter(stage_order).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate stage names in stage_order: {', '.join(sorted(duplicates))}")
    known = set(DEFAULT_STAGE_ORDER) | set(configured)
    unknown = [item for item in stage_order if item not in known]
    if unknown:
        raise ValueError(f"unknown training stages: {', '.join(unknown)}")
    return stage_order


def resolve_start_stage_index(stage_order: list[str], args: argparse.Namespace | None = None) -> int:
    start_stage = str(arg_value(args, "start_stage", "") or "").strip()
    if not start_stage:
        return 1
    if start_stage.isdigit():
        index = int(start_stage)
        if index < 1 or index > len(stage_order):
            raise ValueError(f"--start-stage index must be between 1 and {len(stage_order)}")
        return index
    if start_stage not in stage_order:
        raise ValueError(f"--start-stage {start_stage!r} is not in the active stage order: {', '.join(stage_order)}")
    return stage_order.index(start_stage) + 1


def resume_completed_stages_enabled(plan: dict[str, Any], args: argparse.Namespace | None = None) -> bool:
    cli_value = arg_value(args, "resume_completed_stages", None)
    if cli_value is not None:
        return bool(cli_value)
    if "resume_completed_stages" in plan:
        return bool(plan.get("resume_completed_stages"))
    return True


def stage_checkpoint_path(out_dir: Path, stage_index: int, modality: str) -> Path:
    return out_dir / "checkpoints" / f"{stage_index:02d}_{modality}.pt"


def canonical_stage_index(active_index: int, modality: str) -> int:
    if modality in DEFAULT_STAGE_ORDER:
        return int(DEFAULT_STAGE_ORDER.index(modality)) + 1
    return int(active_index)


def checkpoint_complete_marker(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        return checkpoint / ".complete.json"
    return Path(str(checkpoint) + ".complete.json")


def checkpoint_is_complete(path: str | Path, expected_world_size: int | None = None) -> bool:
    checkpoint = Path(path)
    if not checkpoint.exists():
        return False
    marker = checkpoint_complete_marker(checkpoint)
    if not marker.exists():
        return False
    if not checkpoint.is_dir():
        return True
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return False
    rank_files = manifest.get("rank_files")
    if not isinstance(rank_files, list) or not rank_files:
        return False
    rank_names = [str(item) for item in rank_files]
    manifest_world_size = manifest.get("world_size")
    if manifest_world_size is None:
        train_args = manifest.get("train_args") if isinstance(manifest.get("train_args"), dict) else {}
        manifest_world_size = train_args.get("world_size") or train_args.get("nproc_per_node")
    world_size = int(expected_world_size or manifest_world_size or len(rank_names))
    if world_size < 2:
        return False
    if expected_world_size is not None and len(rank_names) != int(expected_world_size):
        return False
    if manifest_world_size is not None and int(manifest_world_size) != world_size:
        return False
    expected_rank_names = [f"rank{rank:05d}.pt" for rank in range(world_size)]
    if sorted(rank_names) != expected_rank_names:
        return False
    for item in expected_rank_names:
        rank_file = checkpoint / str(item)
        if not rank_file.exists() or rank_file.stat().st_size <= 0:
            return False
        if not Path(str(rank_file) + ".complete.json").exists():
            return False
    return True


def expected_pipeline_world_size(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> int | None:
    if not uses_pipeline_stage_trainer(cfg, args):
        return None
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    distributed = plan.get("distributed_training") if isinstance(plan.get("distributed_training"), dict) else {}
    value = arg_value(args, "nproc_per_node", 0) or distributed.get("nproc_per_node")
    if value is None:
        return None
    world_size = int(value)
    return world_size if world_size > 0 else None


def stage_int_setting(plan: dict[str, Any], key: str, modality: str, default_value: int) -> int:
    values = plan.get(key)
    if isinstance(values, dict):
        for candidate in (modality, "default"):
            if candidate in values and values[candidate] is not None:
                return int(values[candidate])
    return int(default_value)


def context_ladder_values(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> list[int]:
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    scheduler = cfg.get("adaptive_training_scheduler_2026") if isinstance(cfg.get("adaptive_training_scheduler_2026"), dict) else {}
    contract = cfg.get("model_contract") if isinstance(cfg.get("model_contract"), dict) else {}
    raw: Any = (
        arg_value(args, "context_ladder", "")
        or os.environ.get("OMNICODER_CONTEXT_LADDER", "")
        or plan.get("context_ladder")
        or scheduler.get("context_ladder")
        or []
    )
    values: list[int] = []
    if isinstance(raw, str):
        parts: Iterable[Any] = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = raw
    else:
        parts = []
    for part in parts:
        try:
            value = int(part)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if not values:
        values = [8192, 32768, 131072, 262144, 524288, 1048576]
    target = int(contract.get("target_context_length") or plan.get("target_context_length") or 1048576)
    if target > 0 and target not in values:
        values.append(target)
    return sorted(dict.fromkeys(max(1024, int(value)) for value in values))


def sample_loss_max_records_per_file(
    cfg: dict[str, Any],
    args: argparse.Namespace | None = None,
    *,
    benchmark: bool = False,
) -> int:
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    gates = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
    arg_name = "benchmark_max_records_per_file" if benchmark else "heldout_max_records_per_file"
    raw_value = getattr(args, arg_name, None) if args is not None else None
    if raw_value is not None:
        value = int(raw_value)
        if value >= 0:
            return value
    if benchmark:
        for key in ("sample_loss_max_records_per_file", "max_records_per_file"):
            configured = int(gates.get(key) or 0)
            if configured > 0:
                return configured
    for key in ("heldout_sample_loss_max_records_per_file", "sample_loss_max_records_per_file"):
        configured = int(plan.get(key) or 0)
        if configured > 0:
            return configured
    return 32


def sample_loss_timeout_seconds(
    cfg: dict[str, Any],
    args: argparse.Namespace | None = None,
    *,
    benchmark: bool = False,
) -> int:
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    gates = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
    arg_name = "benchmark_sample_loss_timeout_seconds" if benchmark else "heldout_sample_loss_timeout_seconds"
    value = int(arg_value(args, arg_name, 0) or 0)
    if value > 0:
        return value
    if benchmark:
        configured = int(gates.get("sample_loss_timeout_seconds") or 0)
        if configured > 0:
            return configured
    configured = int(plan.get("heldout_sample_loss_timeout_seconds") or plan.get("sample_loss_timeout_seconds") or 0)
    return configured if configured > 0 else 3600


def existing_sample_loss_report(
    out_dir: Path,
    modality: str,
    *,
    checkpoint: Path | None = None,
    split_paths: dict[str, str] | None = None,
    cfg: dict[str, Any] | None = None,
    args: argparse.Namespace | None = None,
    seq_len: int | None = None,
) -> dict[str, Any] | None:
    if truthy_value(arg_value(args, "rerun_heldout_evals", False)):
        return None
    output_path = out_dir / "evals" / f"{modality}_heldout_sample_loss.json"
    if not output_path.exists():
        return None
    try:
        payload = read_json(output_path)
    except Exception as exc:
        return {"status": "skipped", "reason": "existing_sample_loss_unreadable", "error": str(exc), "output_path": str(output_path)}
    if checkpoint is not None and str(payload.get("checkpoint") or "") != str(checkpoint):
        return None
    if seq_len is not None and int(payload.get("seq_len") or 0) != int(seq_len):
        return None
    if cfg is not None:
        expected_max = sample_loss_max_records_per_file(cfg, args)
        if "max_records_per_file" not in payload or int(payload.get("max_records_per_file") or 0) != expected_max:
            return None
    if split_paths:
        expected_paths = [str(Path(path)) for split in ("eval", "test") for path in [split_paths.get(split, "")] if path]
        payload_paths = [str(Path(path)) for path in payload.get("data_paths", [])]
        if expected_paths and payload_paths != expected_paths:
            return None
    payload.setdefault("status", "passed")
    payload.setdefault("output_path", str(output_path))
    return payload


def target_training_cuda_visible_devices(cfg: dict[str, Any]) -> dict[str, Any]:
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    distributed = plan.get("distributed_training") if isinstance(plan.get("distributed_training"), dict) else {}
    main_devices = list_from_config_value(distributed.get("main_gpu_devices") or plan.get("main_gpu_devices"))
    p40_devices = set(list_from_config_value(distributed.get("p40_sidecar_devices") or plan.get("p40_sidecar_devices")))
    sidecar_job = bool(os.environ.get("OMNICODER_SIDECAR_JOB_ID"))
    current = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    nvidia_visible = str(os.environ.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    runtime_scoped = bool(nvidia_visible and nvidia_visible.lower() not in {"all", "none", "void"})
    visible_cuda_count = 0
    visible_cuda_names: list[str] = []
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            visible_cuda_count = int(torch.cuda.device_count())
            visible_cuda_names = [str(torch.cuda.get_device_name(index)) for index in range(visible_cuda_count)]
    except Exception:
        visible_cuda_count = 0
        visible_cuda_names = []
    report: dict[str, Any] = {
        "sidecar_job": sidecar_job,
        "configured_main_gpu_devices": main_devices,
        "configured_p40_sidecar_devices": sorted(p40_devices),
        "nvidia_visible_devices": nvidia_visible or None,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "visible_cuda_count": visible_cuda_count,
        "visible_cuda_names": visible_cuda_names,
        "cuda_visible_devices_before": current or None,
        "cuda_visible_devices_after": current or None,
        "status": "unchanged",
    }
    if sidecar_job or not main_devices:
        return report
    if not current and visible_cuda_count == len(main_devices) and visible_cuda_count > 0:
        p40_names = [name for name in visible_cuda_names if "p40" in name.lower()]
        if p40_names:
            raise ValueError("target runtime visible CUDA set includes P40 device(s): " + ", ".join(p40_names))
        report["status"] = "already_runtime_scoped_to_main_device_count"
        return report
    if runtime_scoped:
        if visible_cuda_count > 0 and visible_cuda_count != len(main_devices):
            raise ValueError(
                f"target NVIDIA runtime exposed {visible_cuda_count} CUDA device(s), expected {len(main_devices)} "
                f"for the configured fast-card lane {main_devices!r}: {visible_cuda_names!r}"
            )
        p40_names = [name for name in visible_cuda_names if "p40" in name.lower()]
        if p40_names:
            raise ValueError("target NVIDIA runtime visible CUDA set includes P40 device(s): " + ", ".join(p40_names))
        report["status"] = "nvidia_runtime_scoped"
        return report
    if not current:
        value = ",".join(main_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        report.update({"cuda_visible_devices_after": value, "status": "set_to_main_training_devices"})
        return report
    current_devices = set(list_from_config_value(current))
    overlap = sorted(current_devices & p40_devices)
    if overlap:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES includes P40 sidecar device(s) in the target synchronous training path: "
            + ", ".join(overlap)
            + ". Use the configured main devices "
            + ",".join(main_devices)
            + " for run-real/run-full target training, or launch P40 work through the sidecar."
        )
    return report


def training_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    checks = cfg.get("learning_checks") if isinstance(cfg.get("learning_checks"), dict) else {}
    if checks:
        return checks
    trend = cfg.get("loss_trend_checks") if isinstance(cfg.get("loss_trend_checks"), dict) else {}
    return {
        "min_relative_loss_drop": 0.001,
        "min_loss_points": int(trend.get("min_samples") or 2),
    }


def is_probe_preset(name: str) -> bool:
    return str(name or "").strip().lower().replace("-", "_") in PROBE_PRESET_NAMES


def resolve_training_preset(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> str:
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    contract = cfg.get("model_contract") if isinstance(cfg.get("model_contract"), dict) else {}
    return str(
        arg_value(args, "preset", "")
        or plan.get("preset")
        or contract.get("target_profile")
        or TARGET_PRESET_2026
    )


def guard_target_training_preset(cfg: dict[str, Any], preset: str, args: argparse.Namespace | None) -> None:
    contract = cfg.get("model_contract") if isinstance(cfg.get("model_contract"), dict) else {}
    target = str(contract.get("target_profile") or TARGET_PRESET_2026)
    allow = bool(arg_value(args, "allow_verifier_preset", False))
    if target and is_probe_preset(preset) and not allow:
        raise ValueError(
            f"Refusing to run target training with verifier preset {preset!r}. "
            f"Use --preset {target} for the 20B/1M contract, or pass --allow-verifier-preset only for explicit probe validation."
        )


def distributed_training_plan(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    distributed = plan.get("distributed_training") if isinstance(plan.get("distributed_training"), dict) else {}
    mode = str(arg_value(args, "distributed", "") or distributed.get("mode") or "none")
    nproc = int(arg_value(args, "nproc_per_node", 0) or distributed.get("nproc_per_node") or 1)
    precision = str(arg_value(args, "precision", "") or distributed.get("precision") or "fp32")
    init_dtype = str(arg_value(args, "init_dtype", "") or distributed.get("init_dtype") or "auto")
    optimizer = str(arg_value(args, "optimizer", "") or distributed.get("optimizer") or "adamw")
    optimizer_in_backward = bool(arg_value(args, "optimizer_in_backward", False) or distributed.get("optimizer_in_backward"))
    optimizer_in_backward_update = str(arg_value(args, "optimizer_in_backward_update", "") or distributed.get("optimizer_in_backward_update") or "lowmem_sgd")
    optimizer_in_backward_grad_clip = float(arg_value(args, "optimizer_in_backward_grad_clip", 0.0) or distributed.get("optimizer_in_backward_grad_clip") or 1.0)
    optimizer_in_backward_clip_mode = str(arg_value(args, "optimizer_in_backward_clip_mode", "") or distributed.get("optimizer_in_backward_clip_mode") or "rms")
    optimizer_in_backward_adafactor_chunk_rows = int(arg_value(args, "optimizer_in_backward_adafactor_chunk_rows", 0) or distributed.get("optimizer_in_backward_adafactor_chunk_rows") or 256)
    optimizer_in_backward_adafactor_clip_threshold = float(arg_value(args, "optimizer_in_backward_adafactor_clip_threshold", 0.0) or distributed.get("optimizer_in_backward_adafactor_clip_threshold") or 1.0)
    optimizer_in_backward_adafactor_decay_rate = float(arg_value(args, "optimizer_in_backward_adafactor_decay_rate", 0.0) or distributed.get("optimizer_in_backward_adafactor_decay_rate") or -0.8)
    optimizer_in_backward_adafactor_eps1 = float(arg_value(args, "optimizer_in_backward_adafactor_eps1", 0.0) or distributed.get("optimizer_in_backward_adafactor_eps1") or 1.0e-30)
    rank_device_map_value = arg_value(args, "rank_device_map", "") or distributed.get("rank_device_map") or ""
    if isinstance(rank_device_map_value, list):
        rank_device_map = ",".join(str(item) for item in rank_device_map_value)
    else:
        rank_device_map = str(rank_device_map_value)
    activation_checkpointing = bool(arg_value(args, "activation_checkpointing", False) or distributed.get("activation_checkpointing"))
    cpu_offload = bool(arg_value(args, "cpu_offload", False) or distributed.get("cpu_offload"))
    fake_quant_chunk_rows = int(arg_value(args, "fake_quant_chunk_rows", 0) or distributed.get("fake_quant_chunk_rows") or plan.get("fake_quant_chunk_rows") or 0)
    fake_quant_max_full_elements = int(arg_value(args, "fake_quant_max_full_elements", 0) or distributed.get("fake_quant_max_full_elements") or plan.get("fake_quant_max_full_elements") or 0)
    placement = str(arg_value(args, "placement", "") or distributed.get("placement") or plan.get("placement") or "single")
    placement_devices_value = arg_value(args, "placement_devices", "") or distributed.get("placement_devices") or plan.get("placement_devices") or ""
    if isinstance(placement_devices_value, list):
        placement_devices = ",".join(str(item) for item in placement_devices_value)
    else:
        placement_devices = str(placement_devices_value)
    normalized_mode = mode.lower().replace("-", "_")
    if normalized_mode in {"pipeline", "pipeline_stage", "pipelined"}:
        rank_parts = ",".join(part.strip() for part in rank_device_map.split(",") if part.strip())
        placement_parts = ",".join(part.strip() for part in placement_devices.split(",") if part.strip())
        if not rank_parts and placement_parts:
            rank_device_map = placement_parts
        elif rank_parts and placement_parts and rank_parts != placement_parts:
            raise ValueError(
                "pipeline_stage placement_devices must match rank_device_map when both are set; "
                f"got placement_devices={placement_parts!r} rank_device_map={rank_parts!r}"
            )
    placement_layer_counts_value = arg_value(args, "placement_layer_counts", "") or distributed.get("placement_layer_counts") or plan.get("placement_layer_counts") or ""
    if isinstance(placement_layer_counts_value, list):
        placement_layer_counts = ",".join(str(item) for item in placement_layer_counts_value)
    else:
        placement_layer_counts = str(placement_layer_counts_value)
    cli_head_device = int(arg_value(args, "placement_head_device", -1) or -1)
    placement_head_device = cli_head_device if cli_head_device >= 0 else int(distributed.get("placement_head_device", plan.get("placement_head_device", -1)))
    placement_schedule = str(arg_value(args, "placement_schedule", "") or distributed.get("placement_schedule") or plan.get("placement_schedule") or "sequential")
    pipeline_microbatches = int(arg_value(args, "pipeline_microbatches", 0) or distributed.get("pipeline_microbatches") or plan.get("pipeline_microbatches") or 1)
    pipeline_stage_schedule = str(arg_value(args, "pipeline_stage_schedule", "") or distributed.get("pipeline_stage_schedule") or distributed.get("pipeline_schedule") or plan.get("pipeline_stage_schedule") or plan.get("pipeline_schedule") or "1f1b")
    pipeline_stage_ranges_value = distributed.get("pipeline_stage_ranges") or plan.get("pipeline_stage_ranges") or ""
    if isinstance(pipeline_stage_ranges_value, list):
        pipeline_stage_ranges = ",".join(str(item) for item in pipeline_stage_ranges_value)
    else:
        pipeline_stage_ranges = str(pipeline_stage_ranges_value)
    cli_pipeline_async = arg_value(args, "pipeline_async_streams", None)
    pipeline_async_streams = bool(cli_pipeline_async if cli_pipeline_async is not None else distributed.get("pipeline_async_streams", plan.get("pipeline_async_streams", False)))
    return {
        "mode": mode,
        "nproc_per_node": nproc,
        "precision": precision,
        "init_dtype": init_dtype,
        "optimizer": optimizer,
        "optimizer_in_backward": optimizer_in_backward,
        "optimizer_in_backward_update": optimizer_in_backward_update,
        "optimizer_in_backward_grad_clip": optimizer_in_backward_grad_clip,
        "optimizer_in_backward_clip_mode": optimizer_in_backward_clip_mode,
        "optimizer_in_backward_adafactor_chunk_rows": optimizer_in_backward_adafactor_chunk_rows,
        "optimizer_in_backward_adafactor_clip_threshold": optimizer_in_backward_adafactor_clip_threshold,
        "optimizer_in_backward_adafactor_decay_rate": optimizer_in_backward_adafactor_decay_rate,
        "optimizer_in_backward_adafactor_eps1": optimizer_in_backward_adafactor_eps1,
        "rank_device_map": rank_device_map,
        "activation_checkpointing": activation_checkpointing,
        "cpu_offload": cpu_offload,
        "fake_quant_chunk_rows": fake_quant_chunk_rows,
        "fake_quant_max_full_elements": fake_quant_max_full_elements,
        "placement": placement,
        "placement_devices": placement_devices,
        "placement_layer_counts": placement_layer_counts,
        "placement_head_device": placement_head_device,
        "placement_schedule": placement_schedule,
        "pipeline_microbatches": pipeline_microbatches,
        "pipeline_stage_schedule": pipeline_stage_schedule,
        "pipeline_stage_ranges": pipeline_stage_ranges,
        "pipeline_async_streams": pipeline_async_streams,
    }


def uses_pipeline_stage_trainer(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> bool:
    mode = str(distributed_training_plan(cfg, args).get("mode") or "none").lower().replace("-", "_")
    return mode in {"pipeline", "pipeline_stage", "pipelined"}


def pretrain_launcher(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> list[str]:
    distributed = distributed_training_plan(cfg, args)
    if uses_pipeline_stage_trainer(cfg, args):
        return [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(int(distributed["nproc_per_node"])),
            "--max_restarts",
            "0",
            "-m",
            "omnicoder.training.pipeline_pretrain_2026_dense",
        ]
    if distributed["mode"] in {"fsdp", "auto"} and int(distributed["nproc_per_node"]) > 1:
        return [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(int(distributed["nproc_per_node"])),
            "--max_restarts",
            "0",
            "-m",
            "omnicoder.training.pretrain_2026_dense",
        ]
    return [sys.executable, "-m", "omnicoder.training.pretrain_2026_dense"]


def append_pretrain_runtime_args(cmd: list[str], cfg: dict[str, Any], args: argparse.Namespace | None = None) -> None:
    distributed = distributed_training_plan(cfg, args)
    pipeline_stage = uses_pipeline_stage_trainer(cfg, args)
    if pipeline_stage:
        if distributed["precision"]:
            cmd.extend(["--precision", str(distributed["precision"])])
        if distributed["init_dtype"]:
            cmd.extend(["--init_dtype", str(distributed["init_dtype"])])
        if distributed["optimizer"]:
            cmd.extend(["--optimizer", str(distributed["optimizer"])])
        if bool(distributed.get("optimizer_in_backward")):
            cmd.append("--optimizer_in_backward")
            cmd.extend(["--optimizer_in_backward_update", str(distributed.get("optimizer_in_backward_update") or "lowmem_adafactor")])
            cmd.extend(["--optimizer_in_backward_grad_clip", str(float(distributed.get("optimizer_in_backward_grad_clip") or 1.0))])
            cmd.extend(["--optimizer_in_backward_clip_mode", str(distributed.get("optimizer_in_backward_clip_mode") or "rms")])
            cmd.extend(["--optimizer_in_backward_adafactor_chunk_rows", str(int(distributed.get("optimizer_in_backward_adafactor_chunk_rows") or 256))])
            cmd.extend(["--optimizer_in_backward_adafactor_clip_threshold", str(float(distributed.get("optimizer_in_backward_adafactor_clip_threshold") or 1.0))])
            cmd.extend(["--optimizer_in_backward_adafactor_decay_rate", str(float(distributed.get("optimizer_in_backward_adafactor_decay_rate") or -0.8))])
            cmd.extend(["--optimizer_in_backward_adafactor_eps1", str(float(distributed.get("optimizer_in_backward_adafactor_eps1") or 1.0e-30))])
        if str(distributed.get("rank_device_map") or ""):
            cmd.extend(["--rank_device_map", str(distributed["rank_device_map"])])
        if bool(distributed["activation_checkpointing"]):
            cmd.append("--activation_checkpointing")
        if int(distributed.get("fake_quant_chunk_rows") or 0) > 0:
            cmd.extend(["--fake_quant_chunk_rows", str(int(distributed["fake_quant_chunk_rows"]))])
        if int(distributed.get("fake_quant_max_full_elements") or 0) > 0:
            cmd.extend(["--fake_quant_max_full_elements", str(int(distributed["fake_quant_max_full_elements"]))])
        if str(distributed.get("placement_layer_counts") or ""):
            cmd.extend(["--placement_layer_counts", str(distributed["placement_layer_counts"])])
        if str(distributed.get("pipeline_stage_ranges") or ""):
            cmd.extend(["--pipeline_stage_ranges", str(distributed["pipeline_stage_ranges"])])
        schedule = str(distributed.get("pipeline_stage_schedule") or "1f1b").lower()
        if schedule not in {"1f1b", "gpipe"}:
            raise ValueError(f"pipeline_stage trainer requires pipeline_stage_schedule 1f1b/gpipe, got {schedule!r}")
        cmd.extend(["--pipeline_schedule", schedule])
        pipeline_microbatches = int(distributed.get("pipeline_microbatches") or 1)
        cmd.extend(["--pipeline_microbatches", str(pipeline_microbatches)])
        checkpoint_sync_backend = str(os.getenv("OMNICODER2026_CHECKPOINT_SYNC_BACKEND", "") or "").strip()
        checkpoint_marker_timeout = str(os.getenv("OMNICODER2026_CHECKPOINT_MARKER_TIMEOUT_SECONDS", "") or "").strip()
        checkpoint_marker_poll = str(os.getenv("OMNICODER2026_CHECKPOINT_MARKER_POLL_SECONDS", "") or "").strip()
        dist_timeout = str(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "") or "").strip()
        if checkpoint_sync_backend:
            cmd.extend(["--checkpoint_sync_backend", checkpoint_sync_backend])
        if checkpoint_marker_timeout:
            cmd.extend(["--checkpoint_marker_timeout_seconds", checkpoint_marker_timeout])
        if checkpoint_marker_poll:
            cmd.extend(["--checkpoint_marker_poll_seconds", checkpoint_marker_poll])
        if dist_timeout:
            cmd.extend(["--dist_timeout_seconds", dist_timeout])
        cmd.append("--require_target_contract")
        if bool(arg_value(args, "allow_verifier_preset", False)):
            cmd.append("--allow_probe")
        return
    if distributed["mode"] != "none":
        cmd.extend(["--distributed", str(distributed["mode"])])
    if distributed["precision"]:
        cmd.extend(["--precision", str(distributed["precision"])])
    if distributed["init_dtype"]:
        cmd.extend(["--init_dtype", str(distributed["init_dtype"])])
    if distributed["optimizer"]:
        cmd.extend(["--optimizer", str(distributed["optimizer"])])
    if bool(distributed.get("optimizer_in_backward")):
        cmd.append("--optimizer_in_backward")
        cmd.extend(["--optimizer_in_backward_update", str(distributed.get("optimizer_in_backward_update") or "lowmem_sgd")])
        cmd.extend(["--optimizer_in_backward_grad_clip", str(float(distributed.get("optimizer_in_backward_grad_clip") or 1.0))])
        cmd.extend(["--optimizer_in_backward_clip_mode", str(distributed.get("optimizer_in_backward_clip_mode") or "rms")])
        cmd.extend(["--optimizer_in_backward_adafactor_chunk_rows", str(int(distributed.get("optimizer_in_backward_adafactor_chunk_rows") or 256))])
        cmd.extend(["--optimizer_in_backward_adafactor_clip_threshold", str(float(distributed.get("optimizer_in_backward_adafactor_clip_threshold") or 1.0))])
        cmd.extend(["--optimizer_in_backward_adafactor_decay_rate", str(float(distributed.get("optimizer_in_backward_adafactor_decay_rate") or -0.8))])
        cmd.extend(["--optimizer_in_backward_adafactor_eps1", str(float(distributed.get("optimizer_in_backward_adafactor_eps1") or 1.0e-30))])
    if str(distributed.get("rank_device_map") or ""):
        cmd.extend(["--rank_device_map", str(distributed["rank_device_map"])])
    if bool(distributed["activation_checkpointing"]):
        cmd.append("--activation_checkpointing")
    if bool(distributed["cpu_offload"]):
        cmd.append("--cpu_offload")
    if int(distributed.get("fake_quant_chunk_rows") or 0) > 0:
        cmd.extend(["--fake_quant_chunk_rows", str(int(distributed["fake_quant_chunk_rows"]))])
    if int(distributed.get("fake_quant_max_full_elements") or 0) > 0:
        cmd.extend(["--fake_quant_max_full_elements", str(int(distributed["fake_quant_max_full_elements"]))])
    if str(distributed.get("placement") or "single") != "single":
        cmd.extend(["--placement", str(distributed["placement"])])
    if str(distributed.get("placement_devices") or ""):
        cmd.extend(["--placement_devices", str(distributed["placement_devices"])])
    if str(distributed.get("placement_layer_counts") or ""):
        cmd.extend(["--placement_layer_counts", str(distributed["placement_layer_counts"])])
    if int(distributed.get("placement_head_device", -1)) >= 0:
        cmd.extend(["--placement_head_device", str(int(distributed["placement_head_device"]))])
    if str(distributed.get("placement_schedule") or "sequential") != "sequential":
        cmd.extend(["--placement_schedule", str(distributed["placement_schedule"])])
    if int(distributed.get("pipeline_microbatches") or 1) > 1:
        cmd.extend(["--pipeline_microbatches", str(int(distributed["pipeline_microbatches"]))])
    if bool(distributed.get("pipeline_async_streams", False)):
        cmd.append("--pipeline_async_streams")
    else:
        cmd.append("--no_pipeline_async_streams")
    cmd.append("--require_target_contract")
    if bool(arg_value(args, "allow_verifier_preset", False)):
        cmd.append("--allow_probe")


def pipeline_sample_loss_launcher(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> list[str]:
    distributed = distributed_training_plan(cfg, args)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(int(distributed["nproc_per_node"])),
        "--max_restarts",
        "0",
        "-m",
        "omnicoder.eval.pipeline_sample_loss_2026",
    ]


def append_pipeline_sample_loss_runtime_args(cmd: list[str], cfg: dict[str, Any], args: argparse.Namespace | None = None) -> None:
    distributed = distributed_training_plan(cfg, args)
    if distributed["precision"]:
        cmd.extend(["--precision", str(distributed["precision"])])
    if distributed["init_dtype"]:
        cmd.extend(["--init_dtype", str(distributed["init_dtype"])])
    if str(distributed.get("rank_device_map") or ""):
        cmd.extend(["--rank_device_map", str(distributed["rank_device_map"])])
    if str(distributed.get("placement_layer_counts") or ""):
        cmd.extend(["--placement_layer_counts", str(distributed["placement_layer_counts"])])
    if int(distributed.get("fake_quant_chunk_rows") or 0) > 0:
        cmd.extend(["--fake_quant_chunk_rows", str(int(distributed["fake_quant_chunk_rows"]))])
    if int(distributed.get("fake_quant_max_full_elements") or 0) > 0:
        cmd.extend(["--fake_quant_max_full_elements", str(int(distributed["fake_quant_max_full_elements"]))])
    training_plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    q4_recovery = cfg.get("q4_recovery") if isinstance(cfg.get("q4_recovery"), dict) else {}
    if bool(arg_value(args, "fake_quant", False) or training_plan.get("fake_quant") or q4_recovery.get("enabled")):
        cmd.append("--fake_quant")
    cmd.append("--require_target_contract")


def run_sample_loss_eval(
    checkpoint: Path,
    modality: str,
    split_paths: dict[str, str],
    out_dir: Path,
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    preset: str,
    device: str,
    seq_len: int,
) -> dict[str, Any]:
    data_paths = [Path(path) for split in ("eval", "test") for path in [split_paths.get(split, "")] if path]
    data_paths = [path for path in data_paths if path.exists()]
    if not data_paths:
        return {"status": "skipped", "reason": "no_heldout_jsonl"}
    output_path = out_dir / "evals" / f"{modality}_heldout_sample_loss.json"
    max_records_per_file = sample_loss_max_records_per_file(cfg, args)
    if checkpoint.is_dir():
        cmd = pipeline_sample_loss_launcher(cfg, args) + [
            "--checkpoint",
            str(checkpoint),
            "--preset",
            preset,
            "--seq-len",
            str(seq_len),
            "--max-records-per-file",
            str(max_records_per_file),
            "--out",
            str(output_path),
        ]
        append_pipeline_sample_loss_runtime_args(cmd, cfg, args)
    else:
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.eval.sample_loss_2026",
            "--checkpoint",
            str(checkpoint),
            "--profile",
            preset,
            "--device",
            device,
            "--seq-len",
            str(seq_len),
            "--max-records-per-file",
            str(max_records_per_file),
            "--out",
            str(output_path),
        ]
    if not checkpoint.is_dir():
        distributed = distributed_training_plan(cfg, args)
        if str(distributed.get("placement") or "single") != "single":
            cmd.extend(["--placement", str(distributed["placement"])])
        if str(distributed.get("placement_devices") or ""):
            cmd.extend(["--placement-devices", str(distributed["placement_devices"])])
        if str(distributed.get("placement_layer_counts") or ""):
            cmd.extend(["--placement-layer-counts", str(distributed["placement_layer_counts"])])
        if int(distributed.get("placement_head_device", -1)) >= 0:
            cmd.extend(["--placement-head-device", str(int(distributed["placement_head_device"]))])
        if str(distributed.get("precision") or ""):
            cmd.extend(["--precision", str(distributed["precision"])])
        if str(distributed.get("init_dtype") or ""):
            cmd.extend(["--init-dtype", str(distributed["init_dtype"])])
        if bool(distributed.get("activation_checkpointing")):
            cmd.append("--activation-checkpointing")
        if int(distributed.get("fake_quant_chunk_rows") or 0) > 0:
            cmd.extend(["--fake-quant-chunk-rows", str(int(distributed["fake_quant_chunk_rows"]))])
        if int(distributed.get("fake_quant_max_full_elements") or 0) > 0:
            cmd.extend(["--fake-quant-max-full-elements", str(int(distributed["fake_quant_max_full_elements"]))])
    for path in data_paths:
        cmd.extend(["--data", str(path)])
    code = run_command(cmd, out_dir / "logs" / f"{modality}_heldout_sample_loss_command.log", timeout_seconds=sample_loss_timeout_seconds(cfg, args))
    if output_path.exists():
        payload = read_json(output_path)
    else:
        payload = {}
    payload["status"] = "passed" if code == 0 else "failed"
    payload["returncode"] = code
    payload["output_path"] = str(output_path)
    payload["data_paths"] = [str(path) for path in data_paths]
    payload["checkpoint"] = str(checkpoint)
    payload["seq_len"] = int(seq_len)
    payload["max_records_per_file"] = int(max_records_per_file)
    return payload


def run_training_stages(profile: dict[str, Any], manifest: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    plan = cfg["training_plan"]
    checks = training_checks(cfg)
    stage_order = resolve_stage_order(plan, args)
    start_stage_index = resolve_start_stage_index(stage_order, args)
    resume_completed = resume_completed_stages_enabled(plan, args)
    cuda_visible_report = target_training_cuda_visible_devices(cfg)
    required = {str(item) for item in plan.get("required_modalities", DEFAULT_STAGE_ORDER)}
    min_records = int(plan.get("min_records_per_modality") or 1)
    preset = resolve_training_preset(cfg, args)
    guard_target_training_preset(cfg, preset, args)
    device = str(args.device or plan.get("device") or ("cuda" if torch_available() else "cpu"))
    steps = int(args.steps_per_stage or plan.get("steps_per_stage") or 64)
    seq_len = int(args.seq_len or plan.get("seq_len") or 256)
    batch_size = int(args.batch_size or plan.get("batch_size") or 1)
    lr = float(args.lr or plan.get("learning_rate") or 0.001)
    save_interval = resolve_save_interval(args, plan.get("save_interval"))
    resume_between = bool(plan.get("resume_between_stages", True))
    fake_quant = bool(args.fake_quant or plan.get("fake_quant") or cfg.get("q4_recovery", {}).get("enabled"))
    initial_checkpoint = str(args.resume_checkpoint or plan.get("initial_checkpoint") or "")
    initial_checkpoint_path: Path | None = Path(initial_checkpoint) if initial_checkpoint else None
    previous_checkpoint: Path | None = initial_checkpoint_path
    pipeline_stage_trainer = uses_pipeline_stage_trainer(cfg, args)
    checkpoint_expected_world_size = expected_pipeline_world_size(cfg, args)
    if previous_checkpoint is not None and not previous_checkpoint.exists():
        raise FileNotFoundError(f"initial_checkpoint does not exist: {previous_checkpoint}")
    require_integrity_preflight(
        run_integrity_preflight(
            training_bound_jsonl_paths_from_manifest(manifest),
            out_dir,
            label="dense_training_launch",
        )
    )
    stage_reports: list[dict[str, Any]] = []

    for index, modality in enumerate(stage_order, 1):
        checkpoint_index = canonical_stage_index(index, modality)
        stage_seq_len = int(args.seq_len or stage_int_setting(plan, "seq_len_by_stage", modality, seq_len))
        stage_steps = int(args.steps_per_stage or stage_int_setting(plan, "steps_per_stage_by_stage", modality, steps))
        split_paths = manifest.get("per_modality_split_jsonl", {}).get(modality, {})
        train_path = Path(split_paths.get("train") or manifest["per_modality_jsonl"].get(modality, ""))
        split_count = manifest.get("split_counts", {}).get(modality, {}) if isinstance(manifest.get("split_counts"), dict) else {}
        record_count = int(split_count.get("train") if isinstance(split_count, dict) and split_count.get("train") is not None else manifest["modalities"].get(modality) or 0)
        checkpoint = stage_checkpoint_path(out_dir, checkpoint_index, modality)
        train_log = out_dir / "logs" / f"{checkpoint_index:02d}_{modality}_loss.jsonl"
        stage_report: dict[str, Any] = {
            "stage": modality,
            "stage_index": index,
            "canonical_stage_index": checkpoint_index,
            "records": record_count,
            "train_jsonl": str(train_path),
            "heldout_jsonl": {key: value for key, value in split_paths.items() if key in {"eval", "test"}},
            "checkpoint": str(checkpoint),
            "selected": index >= start_stage_index,
            "seq_len": stage_seq_len,
            "steps": stage_steps,
        }
        if index < start_stage_index:
            if checkpoint_is_complete(checkpoint, expected_world_size=checkpoint_expected_world_size):
                previous_checkpoint = checkpoint
                stage_report.update({"status": "passed", "reason": "completed_checkpoint_before_start_stage", "loss_log": str(train_log)})
            elif checkpoint.exists():
                stage_report.update({"status": "failed", "reason": "incomplete_checkpoint_before_start_stage", "completion_marker": str(checkpoint_complete_marker(checkpoint))})
            else:
                stage_report.update({"status": "skipped", "reason": "before_start_stage"})
            stage_reports.append(stage_report)
            continue
        if index > 1 and previous_checkpoint is None:
            stage_report.update({"status": "failed", "reason": "missing_prior_checkpoint_for_stage_resume"})
            stage_reports.append(stage_report)
            break
        if resume_completed and checkpoint_is_complete(checkpoint, expected_world_size=checkpoint_expected_world_size):
            previous_checkpoint = checkpoint
            losses = parse_losses(train_log)
            trend_report = learning_report(
                losses,
                min_relative_drop=float(checks.get("min_relative_loss_drop") or 0.001),
                min_points=int(checks.get("min_loss_points") or 2),
            )
            for key, value in trend_report.items():
                if key == "status":
                    stage_report["learning_status"] = value
                else:
                    stage_report[key] = value
            stage_report.update({"status": "passed", "reason": "completed_checkpoint_present", "loss_log": str(train_log)})
            split_paths = manifest.get("per_modality_split_jsonl", {}).get(modality, {})
            existing_eval = existing_sample_loss_report(
                out_dir,
                modality,
                checkpoint=checkpoint,
                split_paths=split_paths,
                cfg=cfg,
                args=args,
                seq_len=stage_seq_len,
            )
            if existing_eval is not None:
                stage_report["heldout_sample_loss"] = existing_eval
            elif split_paths:
                heldout_sample_loss = run_sample_loss_eval(
                    checkpoint,
                    modality,
                    split_paths,
                    out_dir,
                    cfg=cfg,
                    args=args,
                    preset=preset,
                    device=device,
                    seq_len=stage_seq_len,
                )
                stage_report["heldout_sample_loss"] = heldout_sample_loss
                if heldout_sample_loss.get("status") == "failed":
                    stage_report["status"] = "failed"
                    stage_report["reason"] = "heldout_sample_loss_failed"
                    stage_reports.append(stage_report)
                    break
            stage_reports.append(stage_report)
            continue
        if resume_completed and checkpoint.exists() and not checkpoint_is_complete(checkpoint, expected_world_size=checkpoint_expected_world_size):
            stage_report.update({"status": "failed", "reason": "incomplete_existing_checkpoint", "completion_marker": str(checkpoint_complete_marker(checkpoint))})
            stage_reports.append(stage_report)
            break
        if record_count < min_records:
            stage_report.update({"status": "failed" if modality in required else "skipped", "reason": "insufficient_real_records"})
            stage_reports.append(stage_report)
            continue
        cmd = pretrain_launcher(cfg, args) + [
            "--preset",
            preset,
            "--data",
            str(train_path),
            "--out",
            str(checkpoint),
            "--seq_len",
            str(stage_seq_len),
            "--batch_size",
            str(batch_size),
            "--steps",
            str(stage_steps),
            "--lr",
            str(lr),
            "--max_records",
            "0",
            "--log_file",
            str(train_log),
            "--data_manifest",
            str(out_dir / "manifests" / "curation_manifest.json"),
        ]
        if not pipeline_stage_trainer:
            cmd.extend(["--device", device, "--aux_probe"])
        append_pretrain_runtime_args(cmd, cfg, args)
        if save_interval > 0:
            cmd.extend(["--save_interval", str(save_interval)])
        if fake_quant:
            cmd.append("--fake_quant")
        if resume_between and previous_checkpoint is not None and previous_checkpoint.exists():
            cmd.extend(["--resume", str(previous_checkpoint)])
        code = run_command(cmd, out_dir / "logs" / f"{checkpoint_index:02d}_{modality}_command.log")
        losses = parse_losses(train_log)
        trend_report = learning_report(
            losses,
            min_relative_drop=float(checks.get("min_relative_loss_drop") or 0.001),
            min_points=int(checks.get("min_loss_points") or 2),
        )
        for key, value in trend_report.items():
            if key == "status":
                stage_report["learning_status"] = value
            else:
                stage_report[key] = value
        stage_report.update({"returncode": code, "checkpoint": str(checkpoint), "loss_log": str(train_log)})
        if code != 0:
            stage_report["status"] = "failed"
            stage_report["reason"] = "trainer_returned_nonzero"
        elif checkpoint_is_complete(checkpoint, expected_world_size=checkpoint_expected_world_size):
            heldout_sample_loss = run_sample_loss_eval(
                checkpoint,
                modality,
                split_paths,
                out_dir,
                cfg=cfg,
                args=args,
                preset=preset,
                device=device,
                seq_len=stage_seq_len,
            )
            stage_report["heldout_sample_loss"] = heldout_sample_loss
            if heldout_sample_loss.get("status") == "failed":
                stage_report["status"] = "failed"
                stage_report["reason"] = "heldout_sample_loss_failed"
            else:
                stage_report["status"] = "passed"
        else:
            stage_report["status"] = "failed"
            stage_report["reason"] = "checkpoint_missing_or_incomplete_after_trainer_success"
            stage_report["completion_marker"] = str(checkpoint_complete_marker(checkpoint))
        if stage_report["status"] == "passed":
            previous_checkpoint = checkpoint
        stage_reports.append(stage_report)
        if stage_report.get("status") == "failed" and modality in required:
            break

    failed_required = [row for row in stage_reports if row["stage"] in required and row.get("selected", True) and row.get("status") != "passed"]
    return {
        "schema": "omnicoder.real_training_stage_report_2026.v1",
        "status": "failed" if failed_required else "passed",
        "initial_checkpoint": str(initial_checkpoint_path) if initial_checkpoint_path is not None else None,
        "stage_order": stage_order,
        "start_stage": stage_order[start_stage_index - 1],
        "resume_completed_stages": resume_completed,
        "target_cuda_visible_devices": cuda_visible_report,
        "failed_required_stages": failed_required,
        "stages": stage_reports,
        "final_checkpoint": str(previous_checkpoint) if previous_checkpoint is not None else None,
    }


def training_row_token_count(row: dict[str, Any]) -> int:
    for key in ("target_text_token_count", "target_token_count"):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
    target_text = target_json.get("content") if isinstance(target_json.get("content"), str) else ""
    if target_text:
        return len(text_to_ledger_ids(target_text, int(row.get("max_token_probe") or 1048576)))
    for key in ("text_token_count", "token_count"):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    ids = row.get("token_ids")
    if isinstance(ids, list):
        return len(ids)
    return len(text_to_ledger_ids(extract_text(row), int(row.get("max_token_probe") or 1048576)))


def long_context_density_report(train_path: Path, ladder: list[int], plan: dict[str, Any]) -> dict[str, Any]:
    lengths: list[int] = []
    for row in iter_jsonl(train_path):
        lengths.append(training_row_token_count(row))
    if not lengths:
        return {"status": "failed", "reason": "no_long_context_rows", "records": 0, "max_tokens": 0}
    lengths.sort()
    max_tokens = lengths[-1]
    fraction = float(plan.get("long_context_min_real_token_fraction") if plan.get("long_context_min_real_token_fraction") is not None else 0.5)
    absolute_floor = int(plan.get("long_context_min_real_tokens") or 8192)
    min_row_fraction = float(plan.get("long_context_min_real_row_fraction") if plan.get("long_context_min_real_row_fraction") is not None else 0.25)
    min_eligible_rows = int(plan.get("long_context_min_eligible_rows") or 1)
    rung_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for context_len in ladder:
        required = max(1, min(int(context_len), max(absolute_floor, int(int(context_len) * fraction))))
        eligible_rows = sum(1 for value in lengths if value >= required)
        eligible_fraction = eligible_rows / max(1, len(lengths))
        passed = max_tokens >= required and eligible_rows >= min_eligible_rows and eligible_fraction >= min_row_fraction
        row = {
            "context_length": int(context_len),
            "required_real_tokens": int(required),
            "max_real_tokens": int(max_tokens),
            "eligible_rows": int(eligible_rows),
            "eligible_fraction": float(eligible_fraction),
            "min_eligible_rows": int(min_eligible_rows),
            "min_real_row_fraction": float(min_row_fraction),
            "status": "passed" if passed else "failed",
        }
        rung_reports.append(row)
        if row["status"] != "passed":
            failures.append(row)
    p95 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))]
    return {
        "schema": "omnicoder.long_context_density_report_2026.v1",
        "status": "failed" if failures else "passed",
        "records": len(lengths),
        "max_tokens": int(max_tokens),
        "p95_tokens": int(p95),
        "min_real_token_fraction": fraction,
        "min_real_row_fraction": min_row_fraction,
        "min_eligible_rows": min_eligible_rows,
        "absolute_floor": absolute_floor,
        "rungs": rung_reports,
        "failed_rungs": failures,
    }


def run_long_context_curriculum_stage(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    out_dir: Path,
    checkpoint: str | Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not checkpoint or not Path(str(checkpoint)).exists():
        return {"status": "failed", "reason": "missing_checkpoint_for_long_context_curriculum", "initial_checkpoint": str(checkpoint)}
    cfg = profile_cfg(profile)
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    split_paths = manifest.get("per_modality_split_jsonl", {}).get("long_context", {}) if isinstance(manifest.get("per_modality_split_jsonl"), dict) else {}
    train_path = Path(split_paths.get("train") or manifest.get("per_modality_jsonl", {}).get("long_context", ""))
    if not train_path.exists() or train_path.stat().st_size <= 0:
        return {"status": "failed", "reason": "missing_long_context_train_jsonl", "train_jsonl": str(train_path)}
    base_seq_len = int(arg_value(args, "seq_len", 0) or plan.get("seq_len") or 1024)
    ladder = [value for value in context_ladder_values(cfg, args) if int(value) > base_seq_len]
    if not ladder:
        ladder = [base_seq_len]
    density = long_context_density_report(train_path, ladder, plan)
    density_path = out_dir / "manifests" / "long_context_density_report.json"
    write_json(density_path, density)
    if density.get("status") != "passed":
        return {
            "schema": "omnicoder.long_context_curriculum_2026.v1",
            "status": "failed",
            "reason": "long_context_rows_too_short_for_curriculum",
            "initial_checkpoint": str(checkpoint),
            "final_checkpoint": str(checkpoint),
            "train_jsonl": str(train_path),
            "context_ladder": [int(value) for value in ladder],
            "density_report": density,
            "density_report_path": str(density_path),
            "rungs": [],
        }
    steps = int(arg_value(args, "long_context_steps_per_rung", 0) or plan.get("long_context_steps_per_rung") or arg_value(args, "steps_per_stage", 0) or plan.get("steps_per_stage") or 64)
    batch_size = int(arg_value(args, "batch_size", 0) or plan.get("batch_size") or 1)
    lr = float(arg_value(args, "lr", 0.0) or plan.get("long_context_learning_rate") or plan.get("learning_rate") or 0.001)
    preset = resolve_training_preset(cfg, args)
    guard_target_training_preset(cfg, preset, args)
    device = str(arg_value(args, "device", "") or plan.get("device") or ("cuda" if torch_available() else "cpu"))
    save_interval = resolve_save_interval(args, plan.get("save_interval"))
    pipeline_stage_trainer = uses_pipeline_stage_trainer(cfg, args)
    expected_world_size = expected_pipeline_world_size(cfg, args)
    resume_completed = resume_completed_stages_enabled(plan, args)
    fake_quant = bool(arg_value(args, "fake_quant", False) or plan.get("fake_quant") or cfg.get("q4_recovery", {}).get("enabled"))
    curation_manifest_arg = str(arg_value(args, "curation_manifest", "") or "").strip()
    data_manifest_path = (
        resolve_path(curation_manifest_arg, repo_root())
        if curation_manifest_arg
        else out_dir / "manifests" / "curation_manifest.json"
    )
    current_checkpoint = Path(str(checkpoint))
    rung_reports: list[dict[str, Any]] = []
    root = out_dir / "checkpoints" / "long_context_curriculum"
    for index, context_len in enumerate(ladder, 1):
        checkpoint_out = root / f"{index:02d}_ctx{int(context_len)}"
        train_log = out_dir / "logs" / f"long_context_curriculum_{index:02d}_ctx{int(context_len)}_loss.jsonl"
        report: dict[str, Any] = {
            "stage": "long_context",
            "rung": index,
            "context_length": int(context_len),
            "train_jsonl": str(train_path),
            "heldout_jsonl": {key: value for key, value in split_paths.items() if key in {"eval", "test"}},
            "checkpoint": str(checkpoint_out),
            "initial_checkpoint": str(current_checkpoint),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "lr": float(lr),
        }
        if resume_completed and checkpoint_is_complete(checkpoint_out, expected_world_size=expected_world_size):
            current_checkpoint = checkpoint_out
            losses = parse_losses(train_log)
            report.update(
                {
                    "status": "passed",
                    "reason": "completed_checkpoint_present",
                    "loss_log": str(train_log),
                    "loss_points": len(losses),
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                }
            )
            rung_reports.append(report)
            continue
        if resume_completed and checkpoint_out.exists() and not checkpoint_is_complete(checkpoint_out, expected_world_size=expected_world_size):
            report.update({"status": "failed", "reason": "incomplete_existing_checkpoint", "completion_marker": str(checkpoint_complete_marker(checkpoint_out))})
            rung_reports.append(report)
            break
        cmd = pretrain_launcher(cfg, args) + [
            "--preset",
            preset,
            "--data",
            str(train_path),
            "--out",
            str(checkpoint_out),
            "--seq_len",
            str(int(context_len)),
            "--batch_size",
            str(batch_size),
            "--steps",
            str(steps),
            "--lr",
            str(lr),
            "--max_records",
            "0",
            "--log_file",
            str(train_log),
            "--data_manifest",
            str(data_manifest_path),
            "--resume",
            str(current_checkpoint),
        ]
        if not pipeline_stage_trainer:
            cmd.extend(["--device", device, "--aux_probe"])
        append_pretrain_runtime_args(cmd, cfg, args)
        if save_interval > 0:
            cmd.extend(["--save_interval", str(save_interval)])
        if fake_quant:
            cmd.append("--fake_quant")
        code = run_command(cmd, out_dir / "logs" / f"long_context_curriculum_{index:02d}_ctx{int(context_len)}_command.log")
        losses = parse_losses(train_log)
        trend_report = learning_report(
            losses,
            min_relative_drop=float(training_checks(cfg).get("min_relative_loss_drop") or 0.001),
            min_points=int(training_checks(cfg).get("min_loss_points") or 2),
        )
        report.update(
            {
                "returncode": code,
                "loss_log": str(train_log),
                "loss_points": len(losses),
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "checkpoint_complete": checkpoint_is_complete(checkpoint_out, expected_world_size=expected_world_size),
            }
        )
        for key, value in trend_report.items():
            if key == "status":
                report["learning_status"] = value
            else:
                report[key] = value
        if code == 0 and report["checkpoint_complete"]:
            heldout_args = namespace_with(args, seq_len=int(context_len), benchmark_seq_len=int(context_len))
            report["heldout_sample_loss"] = run_sample_loss_eval(
                checkpoint_out,
                "long_context",
                split_paths,
                out_dir,
                cfg=cfg,
                args=heldout_args,
                preset=preset,
                device=device,
                seq_len=int(context_len),
            )
            if report["heldout_sample_loss"].get("status") == "failed":
                report["status"] = "failed"
                report["reason"] = "heldout_sample_loss_failed"
            else:
                report["status"] = "passed"
                current_checkpoint = checkpoint_out
        else:
            report["status"] = "failed"
            report["reason"] = "long_context_trainer_returned_nonzero_or_incomplete_checkpoint"
        rung_reports.append(report)
        if report.get("status") != "passed":
            break
    status = "failed" if any(row.get("status") != "passed" for row in rung_reports) else "passed"
    benchmark_gate = (
        run_checkpoint_benchmark_gate(profile, manifest, out_dir, current_checkpoint, "long_context_curriculum_final", namespace_with(args, benchmark_seq_len=int(ladder[-1])))
        if status == "passed" and current_checkpoint
        else {"status": "skipped", "reason": "long_context_curriculum_failed"}
    )
    if benchmark_gate.get("status") == "failed":
        status = "failed"
    return {
        "schema": "omnicoder.long_context_curriculum_2026.v1",
        "status": status,
        "initial_checkpoint": str(checkpoint),
        "final_checkpoint": str(current_checkpoint),
        "train_jsonl": str(train_path),
        "context_ladder": [int(value) for value in ladder],
        "density_report": density,
        "density_report_path": str(density_path),
        "steps_per_rung": int(steps),
        "rungs": rung_reports,
        "benchmark_gate": benchmark_gate,
    }


def declared_posttrain_algorithms(rl: dict[str, Any]) -> list[str]:
    replay = rl.get("offline_reward_replay") if isinstance(rl.get("offline_reward_replay"), dict) else {}
    stack = rl.get("policy_stack_2026") if isinstance(rl.get("policy_stack_2026"), dict) else {}
    values = list(replay.get("algorithms_represented") or [])
    values.extend(stack.get("offline") or [])
    values.extend(stack.get("online_or_env") or [])
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        value = str(item)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def resolve_posttrain_algorithms(rl: dict[str, Any], args: argparse.Namespace | None = None) -> list[str]:
    override = list_from_config_value(arg_value(args, "posttrain_algorithm_order", ""))
    algorithms = override or declared_posttrain_algorithms(rl)
    seen: set[str] = set()
    deduped: list[str] = []
    for algorithm in algorithms:
        value = str(algorithm).strip()
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    if not deduped:
        return []
    start = str(arg_value(args, "start_posttrain_algorithm", "") or "").strip()
    if not start:
        return deduped
    if start.isdigit():
        index = int(start)
        if index < 1 or index > len(deduped):
            raise ValueError(f"--start-posttrain-algorithm index must be between 1 and {len(deduped)}")
        return deduped[index - 1 :]
    if start not in deduped:
        raise ValueError(
            f"--start-posttrain-algorithm {start!r} is not in the active posttraining order: {', '.join(deduped)}"
        )
    return deduped[deduped.index(start) :]


def discover_posttrain_inputs(configured_inputs: Any, root: Path) -> list[Path]:
    inputs = existing_paths(configured_inputs, root)
    candidates: list[Path] = []
    configured = list(configured_inputs) if isinstance(configured_inputs, list) else [configured_inputs]
    search_roots: set[Path] = set()
    for value in configured:
        if not isinstance(value, str) or not value:
            continue
        candidate = resolve_path(value, root)
        search_roots.add(candidate.parent)
        if candidate.parent.name.startswith("agentic_tool_training_2026"):
            search_roots.add(candidate.parent)
    search_roots.add(root / "weights" / "agentic_tool_training_2026")
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("tool_*.jsonl"):
            if path.is_file() and path.stat().st_size > 0:
                candidates.append(path)
    seen = {str(path.resolve()) for path in inputs}
    discovered = sorted(
        (path for path in set(candidates) if str(path.resolve()) not in seen),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return inputs + discovered


def resolve_posttrain_input_overrides(args: argparse.Namespace | None, root: Path) -> tuple[list[Path], dict[str, Path]]:
    values = list_from_config_value(arg_value(args, "posttrain_input_jsonl", []))
    ordered: list[Path] = []
    routed: dict[str, Path] = {}
    seen: set[str] = set()
    for raw in values:
        route = ""
        value = str(raw).strip()
        if not value:
            continue
        if "=" in value:
            route, value = value.split("=", 1)
        elif "::" in value:
            route, value = value.split("::", 1)
        route = route.strip().lower()
        value = value.strip()
        if not value:
            raise ValueError(f"empty posttraining input path in override {raw!r}")
        path = resolve_path(value, root)
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"posttraining input override does not exist or is empty: {path}")
        key = str(path.resolve())
        if key not in seen:
            ordered.append(path)
            seen.add(key)
        if route:
            routed[route] = path
    return ordered, routed


def posttrain_explicit_inputs_only(args: argparse.Namespace | None = None) -> bool:
    return truthy_value(arg_value(args, "posttrain_explicit_inputs_only", False)) or truthy_value(
        os.environ.get("OMNICODER_POSTTRAIN_EXPLICIT_INPUTS_ONLY", "")
    )


def posttrain_dataset_for_algorithm(requested: str, paths: list[Path], routed: dict[str, Path] | None = None) -> Path | None:
    name = requested.lower()
    if routed:
        if name in routed:
            return routed[name]
        for route, path in routed.items():
            if route and (route in name or name in route):
                return path
    if any(marker in name for marker in ("dpo", "orpo", "simpo", "preference")):
        hints = ("preference",)
    elif any(marker in name for marker in ("safety", "kto", "negative")):
        hints = ("safety", "negative")
    elif "sft" in name:
        hints = ("sft",)
    elif any(marker in name for marker in ("reward", "agentprm", "process", "rlaif")):
        hints = ("reward",)
    elif any(marker in name for marker in ("grpo", "dapo", "toolrl", "retool", "rlvr", "cispo", "vapo", "dcpo", "lspo")):
        hints = ("rlvr",)
    else:
        hints = ()
    for path in paths:
        lower = path.name.lower()
        if all(hint in lower for hint in hints):
            return path
    for path in paths:
        lower = path.name.lower()
        if any(hint in lower for hint in hints):
            return path
    return paths[0] if paths else None


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def posttrain_retention_cfg(rl: dict[str, Any]) -> dict[str, Any]:
    direct = rl.get("posttraining_checkpoint_retention")
    if isinstance(direct, dict):
        return direct
    nested = rl.get("checkpoint_retention")
    if isinstance(nested, dict) and isinstance(nested.get("posttraining"), dict):
        return nested["posttraining"]
    return {}


def prune_posttrain_checkpoints(out_dir: Path, keep_paths: Iterable[Path], retention: dict[str, Any]) -> dict[str, Any]:
    if not retention or not retention.get("enabled"):
        return {"status": "skipped", "reason": "posttraining_checkpoint_retention_disabled"}
    root = out_dir / "checkpoints" / "posttrain"
    if not root.exists():
        return {"status": "skipped", "reason": "posttraining_checkpoint_root_missing", "root": str(root)}
    keep_last = max(0, int(retention.get("keep_last_successful") or 0))
    delete_incomplete = bool(retention.get("delete_incomplete") or retention.get("delete_failed_incomplete"))
    protected = {path.resolve() for path in keep_paths if path.exists() and is_relative_to(path, root)}
    dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    complete_dirs = [path for path in dirs if checkpoint_is_complete(path)]
    if keep_last:
        protected.update(path.resolve() for path in complete_dirs[-keep_last:])
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in dirs:
        resolved = path.resolve()
        if resolved in protected:
            skipped.append({"path": str(path), "reason": "protected"})
            continue
        complete = checkpoint_is_complete(path)
        if not complete and not delete_incomplete:
            skipped.append({"path": str(path), "reason": "incomplete_checkpoint_retained"})
            continue
        if not is_relative_to(path, root):
            skipped.append({"path": str(path), "reason": "outside_posttrain_root"})
            continue
        try:
            shutil.rmtree(path)
            removed.append({"path": str(path), "complete": complete})
        except OSError as exc:
            skipped.append({"path": str(path), "reason": f"remove_failed:{exc.__class__.__name__}", "detail": str(exc)})
    return {
        "status": "passed" if not any(item.get("reason", "").startswith("remove_failed") for item in skipped) else "failed",
        "root": str(root),
        "removed": removed,
        "skipped": skipped,
        "keep_last_successful": keep_last,
        "delete_incomplete": delete_incomplete,
    }


def run_posttraining_stages(
    profile: dict[str, Any],
    out_dir: Path,
    training: dict[str, Any],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    rl = cfg.get("reinforcement_learning") if isinstance(cfg.get("reinforcement_learning"), dict) else {}
    if not rl.get("enabled"):
        return {"status": "skipped", "reason": "reinforcement_learning_disabled", "stages": []}
    algorithms = resolve_posttrain_algorithms(rl, args)
    if not algorithms:
        return {"status": "skipped", "reason": "no_declared_posttraining_algorithms", "stages": []}
    replay = rl.get("offline_reward_replay") if isinstance(rl.get("offline_reward_replay"), dict) else {}
    root = repo_root()
    explicit_inputs, routed_inputs = resolve_posttrain_input_overrides(args, root)
    if explicit_inputs and posttrain_explicit_inputs_only(args):
        inputs = list(explicit_inputs)
    else:
        inputs = discover_posttrain_inputs(replay.get("inputs"), root)
        local_export_dir = out_dir / "agentic_tool_training_2026"
        if local_export_dir.exists():
            local_inputs = sorted(path for path in local_export_dir.rglob("tool_*.jsonl") if path.is_file() and path.stat().st_size > 0)
            local_seen = {str(path.resolve()) for path in local_inputs}
            inputs = local_inputs + [path for path in inputs if str(path.resolve()) not in local_seen]
        if explicit_inputs:
            explicit_seen = {str(path.resolve()) for path in explicit_inputs}
            inputs = explicit_inputs + [path for path in inputs if str(path.resolve()) not in explicit_seen]
    require_integrity_preflight(
        run_integrity_preflight(
            [path for path in inputs if path.name != "tool_safety_negatives.jsonl"],
            out_dir,
            label="posttraining_launch",
        )
    )
    reports: list[dict[str, Any]] = []
    current_checkpoint = Path(str(training.get("final_checkpoint") or "")) if training.get("final_checkpoint") else None
    model = str(current_checkpoint) if current_checkpoint is not None else str(cfg.get("distillation", {}).get("base_model") or "Qwen/Qwen3-4B")
    live_replay = bool(arg_value(args, "live_posttraining", False))
    preset = resolve_training_preset(cfg, args)
    guard_target_training_preset(cfg, preset, args)
    device = str(arg_value(args, "device", "") or cfg.get("training_plan", {}).get("device") or ("cuda" if torch_available() else "cpu"))
    seq_len = int(arg_value(args, "posttrain_seq_len", 0) or arg_value(args, "seq_len", 0) or cfg.get("training_plan", {}).get("seq_len") or 192)
    batch_size = int(arg_value(args, "posttrain_batch_size", 0) or arg_value(args, "batch_size", 0) or cfg.get("training_plan", {}).get("batch_size") or 1)
    steps = int(arg_value(args, "posttrain_steps", 0) or rl.get("posttrain_steps_per_algorithm") or 32)
    lr = float(arg_value(args, "posttrain_lr", 0.0) or rl.get("posttrain_learning_rate") or 1e-6)
    max_records = int(arg_value(args, "posttrain_max_records", 0) or rl.get("posttrain_max_records") or 0)
    save_interval = resolve_save_interval(args, cfg.get("training_plan", {}).get("save_interval"))
    stop_on_failure = bool(rl.get("stop_on_posttrain_failure", True))
    retention = posttrain_retention_cfg(rl)
    replay_final_checkpoint: Path | None = current_checkpoint
    eval_manifest = load_posttraining_eval_manifest(profile, out_dir, args)
    for index, requested in enumerate(algorithms, 1):
        train_jsonl = posttrain_dataset_for_algorithm(requested, inputs, routed_inputs)
        if train_jsonl is None:
            reports.append({"requested_algorithm": requested, "status": "failed", "reason": "no_declared_input_jsonl_found"})
            if stop_on_failure:
                for blocked in algorithms[index:]:
                    reports.append(
                        {
                            "requested_algorithm": blocked,
                            "status": "skipped",
                            "reason": "previous_posttraining_stage_failed",
                            "blocked_by": requested,
                        }
                    )
                break
            continue
        safe_name = safe_filename(requested)
        manifest = out_dir / "manifests" / "posttrain" / f"{safe_name}_manifest.json"
        log_path = out_dir / "logs" / f"posttrain_{safe_name}.log"
        bridge_model = str(replay_final_checkpoint) if replay_final_checkpoint is not None else str(model)
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.training.posttrain_bridge_2026",
            "--algorithm",
            requested,
            "--model",
            bridge_model,
            "--train_jsonl",
            str(train_jsonl),
            "--manifest",
            str(manifest),
            "--out_dir",
            str(out_dir / "posttrain" / safe_name),
            "--profile",
            preset,
            "--device",
            device,
            "--max_seq_len",
            str(seq_len),
            "--reward_seq_len",
            str(seq_len),
            "--max_steps",
            str(steps),
            "--learning_rate",
            str(lr),
            "--per_device_train_batch_size",
            str(batch_size),
            "--max_records",
            str(max_records),
        ]
        pipeline_checkpoint = live_replay and replay_final_checkpoint is not None and replay_final_checkpoint.is_dir()
        if pipeline_checkpoint:
            cmd.extend(
                [
                    "--defer_optimizer",
                    "--defer_reason",
                    "pipeline_sharded_checkpoint_requires_distributed_pipeline_reward_replay",
                ]
            )
        if not live_replay:
            cmd.append("--dry_run")
        code = run_command(cmd, log_path)
        bridge_manifest = read_json(manifest) if manifest.exists() else {}
        bridge_execution = bridge_manifest.get("execution") if isinstance(bridge_manifest.get("execution"), dict) else {}
        report = {
            "requested_algorithm": requested,
            "train_jsonl": str(train_jsonl),
            "manifest": str(manifest),
            "log": str(log_path),
            "bridge_returncode": code,
            "bridge_status": bridge_manifest.get("status"),
            "mode": "posttrain_bridge_live_optimizer" if live_replay else "bridge_dry_run",
            "status": "passed" if code == 0 and manifest.exists() else "failed",
        }
        if live_replay:
            if replay_final_checkpoint is None or not replay_final_checkpoint.exists():
                report.update({"status": "failed", "reason": "missing_checkpoint_for_live_replay"})
            elif replay_final_checkpoint.is_dir():
                if not uses_pipeline_stage_trainer(cfg, args):
                    report.update(
                        {
                            "status": "failed",
                            "reason": "pipeline_sharded_checkpoint_requires_pipeline_stage_reward_replay",
                            "checkpoint": str(replay_final_checkpoint),
                        }
                    )
                elif not (
                    code == 0
                    and bridge_manifest.get("status") in {"live_optimizer_deferred", "optimizer_deferred_to_distributed_pipeline_replay"}
                    and bridge_execution.get("status") == "deferred"
                    and bridge_execution.get("executor") == "distributed_pipeline_reward_replay"
                ):
                    report.update(
                        {
                            "status": "failed",
                            "reason": "posttrain_bridge_did_not_authorize_distributed_pipeline_reward_replay",
                            "bridge_execution": bridge_execution,
                        }
                    )
                else:
                    replay_out = out_dir / "checkpoints" / "posttrain" / f"{index:02d}_{safe_name}_pipeline"
                    replay_log = out_dir / "logs" / f"posttrain_{index:02d}_{safe_name}_pipeline_reward_replay.jsonl"
                    replay_cmd = pretrain_launcher(cfg, args) + [
                        "--preset",
                        preset,
                        "--data",
                        str(train_jsonl),
                        "--out",
                        str(replay_out),
                        "--seq_len",
                        str(seq_len),
                        "--batch_size",
                        str(batch_size),
                        "--steps",
                        str(steps),
                        "--lr",
                        str(lr),
                        "--max_records",
                        str(max_records),
                        "--log_file",
                        str(replay_log),
                        "--data_manifest",
                        str(manifest),
                        "--resume",
                        str(replay_final_checkpoint),
                    ]
                    if save_interval > 0:
                        replay_cmd.extend(["--save_interval", str(save_interval)])
                    append_pretrain_runtime_args(replay_cmd, cfg, args)
                    if bool(arg_value(args, "fake_quant", False) or cfg.get("training_plan", {}).get("fake_quant") or cfg.get("q4_recovery", {}).get("enabled")):
                        replay_cmd.append("--fake_quant")
                    replay_code = run_command(replay_cmd, out_dir / "logs" / f"posttrain_{index:02d}_{safe_name}_pipeline_command.log")
                    losses = parse_losses(replay_log)
                    complete = checkpoint_is_complete(replay_out, expected_world_size=expected_pipeline_world_size(cfg, args))
                    report.update(
                        {
                            "replay_returncode": replay_code,
                            "checkpoint": str(replay_out),
                            "loss_log": str(replay_log),
                            "loss_points": len(losses),
                            "loss_first": losses[0] if losses else None,
                            "loss_last": losses[-1] if losses else None,
                            "mode": "distributed_pipeline_reward_replay",
                            "checkpoint_complete": complete,
                        }
                    )
                    if replay_code == 0 and complete:
                        replay_final_checkpoint = replay_out
                        report["status"] = "passed"
                        report["heldout_benchmark_gate"] = run_checkpoint_benchmark_gate(
                            profile,
                            eval_manifest,
                            out_dir,
                            replay_out,
                            f"posttrain_{index:02d}_{safe_name}",
                            args,
                        )
                        if retention.get("enabled"):
                            report["checkpoint_retention"] = prune_posttrain_checkpoints(
                                out_dir,
                                [replay_out, replay_final_checkpoint, current_checkpoint] if current_checkpoint is not None else [replay_out, replay_final_checkpoint],
                                retention,
                            )
                    else:
                        report["status"] = "failed"
                        report["reason"] = "pipeline_reward_replay_returned_nonzero_or_incomplete_checkpoint"
            else:
                checkpoint_value = str(bridge_execution.get("checkpoint") or "")
                replay_out = Path(checkpoint_value) if checkpoint_value else out_dir / "posttrain" / safe_name / "checkpoints" / f"{safe_name}_live_replay.pt"
                loss_log_value = str(bridge_execution.get("loss_log") or "")
                replay_log = Path(loss_log_value) if loss_log_value else out_dir / "posttrain" / safe_name / "logs" / f"{safe_name}_live_replay.jsonl"
                losses = parse_losses(replay_log)
                report.update(
                    {
                        "replay_returncode": bridge_execution.get("returncode"),
                        "executor": bridge_execution.get("executor"),
                        "checkpoint": str(replay_out),
                        "loss_log": str(replay_log),
                        "loss_points": len(losses),
                        "loss_first": losses[0] if losses else None,
                        "loss_last": losses[-1] if losses else None,
                    }
                )
                if code == 0 and bridge_execution.get("status") == "passed" and replay_out.exists():
                    replay_final_checkpoint = replay_out
                    report["status"] = "passed"
                    report["heldout_benchmark_gate"] = run_checkpoint_benchmark_gate(
                        profile,
                        eval_manifest,
                        out_dir,
                        replay_out,
                        f"posttrain_{index:02d}_{safe_name}",
                        args,
                    )
                else:
                    report["status"] = "failed"
                    report["reason"] = "posttrain_bridge_live_optimizer_failed"
        reports.append(report)
        if stop_on_failure and report.get("status") == "failed":
            for blocked in algorithms[index:]:
                reports.append(
                    {
                        "requested_algorithm": blocked,
                        "status": "skipped",
                        "reason": "previous_posttraining_stage_failed",
                        "blocked_by": requested,
                    }
                )
            break
    status = "failed" if any(row.get("status") != "passed" for row in reports) else "passed"
    return {
        "status": status,
        "input_count": len(inputs),
        "explicit_input_count": len(explicit_inputs),
        "posttrain_input_route_map": {key: str(value) for key, value in sorted(routed_inputs.items())},
        "mode": "posttrain_bridge_live_optimizer" if live_replay else "bridge_dry_run",
        "stages": reports,
        "initial_checkpoint": str(current_checkpoint) if current_checkpoint is not None else None,
        "final_checkpoint": str(replay_final_checkpoint) if replay_final_checkpoint is not None else None,
    }


def checkpoint_eval_paths(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("eval_all_jsonl", "test_all_jsonl"):
        value = manifest.get(key)
        if value:
            path = Path(str(value))
            if not path.exists() and not path.is_absolute():
                path = resolve_path(path, repo_root())
            if path.exists():
                paths.append(path)
    return paths


def load_posttraining_eval_manifest(profile: dict[str, Any], out_dir: Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    candidates: list[Path] = []
    explicit = str(arg_value(args, "curation_manifest", "") or "").strip()
    if explicit:
        candidates.append(resolve_path(explicit, repo_root()))
    configured = str(cfg.get("posttraining_curation_manifest") or "").strip()
    if configured:
        candidates.append(resolve_path(configured, repo_root()))
    candidates.extend(
        [
            out_dir / "manifests" / "posttraining_curation_manifest.json",
            out_dir / "manifests" / "real_training_curation_manifest.json",
            out_dir / "manifests" / "cleaned_dataset_manifest.json",
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        manifest = read_json(path)
        if checkpoint_eval_paths(manifest):
            manifest["eval_manifest_path"] = str(path)
            return manifest
    return {"status": "missing_eval_manifest", "checked": [str(path) for path in candidates]}


def run_checkpoint_benchmark_gate(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    out_dir: Path,
    checkpoint: str | Path | None,
    phase: str,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    if not checkpoint:
        return {"status": "skipped", "phase": phase, "reason": "no_checkpoint"}
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.exists():
        return {"status": "skipped", "phase": phase, "reason": "checkpoint_missing", "checkpoint": str(checkpoint_path)}
    preset = resolve_training_preset(cfg, args)
    device = str(arg_value(args, "device", "") or cfg.get("training_plan", {}).get("device") or ("cuda" if torch_available() else "cpu"))
    seq_len = int(arg_value(args, "benchmark_seq_len", 0) or arg_value(args, "seq_len", 0) or cfg.get("training_plan", {}).get("seq_len") or 192)
    gate_dir = out_dir / "benchmarks" / safe_filename(phase)
    gate_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "omnicoder.training_checkpoint_benchmark_gate_2026.v1",
        "phase": phase,
        "checkpoint": str(checkpoint_path),
        "status": "passed",
        "created_at": now_iso(),
    }

    def run_reportable_gate(benchmark_profile: str, run_id: str) -> tuple[dict[str, Any], bool]:
        gates_cfg = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
        final_reportable_gate = safe_filename(phase) == "full_run_final"
        reportable_roots, reportable_sources = configured_reportable_roots(
            cfg,
            benchmark_profile,
            arg_value(args, "reportable_task_roots", None),
        )
        reportable_paths = existing_paths(reportable_roots, repo_root())
        missing_policy = str(
            cfg.get("missing_reportable_policy")
            or gates_cfg.get("missing_reportable_policy")
            or "fail"
        ).lower()
        benchmark_cycle = str(arg_value(args, "benchmark_cycle", "") or gates_cfg.get("benchmark_cycle") or "smoke")
        benchmark_min_tasks = int(arg_value(args, "benchmark_min_tasks", 0) or gates_cfg.get("benchmark_min_tasks") or 1)
        require_reportable_gate = truthy_value(arg_value(args, "require_reportable_gate", False))
        benchmark_predictions_raw = str(arg_value(args, "benchmark_predictions", "") or "").strip()
        benchmark_predictions = resolve_path(benchmark_predictions_raw, repo_root()) if benchmark_predictions_raw else None

        def generate_predictions_if_configured() -> dict[str, Any]:
            backend = str(
                arg_value(args, "benchmark_prediction_backend", "")
                or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_BACKEND", "")
                or gates_cfg.get("prediction_backend")
                or ""
            ).strip()
            if not backend:
                return {"status": "skipped", "reason": "no_prediction_backend_configured"}
            prediction_out = reportable_dir / "model_predictions.jsonl"
            summary_out = reportable_dir / "model_prediction_summary.json"
            model_id = str(arg_value(args, "benchmark_prediction_model", "") or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_MODEL", "") or checkpoint_path)
            prediction_cmd = [
                sys.executable,
                "-m",
                "omnicoder.eval.reportable_prediction_harness_2026",
                "--backend",
                backend,
                "--model",
                model_id,
                "--out",
                str(prediction_out),
                "--summary",
                str(summary_out),
                "--force",
            ]
            for task_path in reportable_paths:
                prediction_cmd.extend(["--tasks", str(task_path)])
            max_output_tokens = int(arg_value(args, "benchmark_prediction_max_output_tokens", 0) or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_MAX_OUTPUT_TOKENS", "0") or gates_cfg.get("prediction_max_output_tokens") or 0)
            timeout = int(arg_value(args, "benchmark_prediction_timeout_seconds", 0) or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_TIMEOUT_SECONDS", "0") or gates_cfg.get("prediction_timeout_seconds") or 0)
            if max_output_tokens > 0:
                prediction_cmd.extend(["--max-output-tokens", str(max_output_tokens)])
            if timeout > 0:
                prediction_cmd.extend(["--timeout-seconds", str(timeout)])
            if backend == "openai-compatible":
                base_url = str(arg_value(args, "benchmark_prediction_base_url", "") or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_BASE_URL", "") or gates_cfg.get("prediction_base_url") or "")
                api_key_env = str(arg_value(args, "benchmark_prediction_api_key_env", "") or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_API_KEY_ENV", "") or gates_cfg.get("prediction_api_key_env") or "OPENAI_API_KEY")
                prediction_cmd.extend(["--base-url", base_url, "--api-key-env", api_key_env])
            elif backend == "checkpoint-runner":
                runner = str(arg_value(args, "benchmark_prediction_checkpoint_runner", "") or os.environ.get("OMNICODER_BENCHMARK_PREDICTION_CHECKPOINT_RUNNER", "") or gates_cfg.get("prediction_checkpoint_runner") or "")
                prediction_cmd.extend(["--checkpoint-runner", runner, "--checkpoint-path", str(checkpoint_path)])
            prediction_code = run_command(prediction_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_prediction_harness.log")
            summary = read_json(summary_out) if summary_out.exists() else {}
            return {
                "status": "passed" if prediction_code == 0 and prediction_out.exists() and count_jsonl_rows(prediction_out) > 0 else "failed",
                "returncode": prediction_code,
                "backend": backend,
                "path": str(prediction_out),
                "summary": str(summary_out),
                "records": count_jsonl_rows(prediction_out) if prediction_out.exists() else 0,
                "summary_json": summary,
                "source": "generated_by_reportable_prediction_harness",
            }

        if not reportable_paths:
            gate = {
                "status": "needs_data",
                "reason": "no_official_or_authorized_reportable_tasks_found",
                "configured_task_roots": [str(path) for path in reportable_roots],
                "root_sources": reportable_sources,
                "missing_policy": missing_policy,
                "gate_policy": "fail_open" if missing_policy in {"allow", "warn", "skip"} else "fail_closed",
                "gate_decision": "allowed_needs_data" if missing_policy in {"allow", "warn", "skip"} else "blocked_needs_data",
            }
            if checkpoint_path.is_dir() and not require_reportable_gate and not final_reportable_gate:
                gate["status"] = "pending"
                gate["gate_decision"] = "pending_needs_data"
                return gate, False
            if final_reportable_gate:
                gate["required"] = True
                gate["reason"] = "final_reportable_gate_requires_authorized_tasks"
                return gate, True
            return gate, missing_policy not in {"allow", "warn", "skip"}

        reportable_dir = gate_dir / "reportable"
        reportable_summary_path = reportable_dir / "reportable_summary.json"
        prediction_seed: dict[str, Any]
        if benchmark_predictions is not None and benchmark_predictions.exists():
            prediction_seed = {
                "path": str(benchmark_predictions),
                "records": count_jsonl_rows(benchmark_predictions),
                "source": "model_generated_predictions",
            }
        else:
            generated_predictions = generate_predictions_if_configured()
            if generated_predictions.get("status") == "passed":
                prediction_seed = generated_predictions
            elif benchmark_cycle == "smoke" and not checkpoint_path.is_dir():
                prediction_seed = write_reportable_prediction_seed(reportable_paths, reportable_dir / "checkpoint_predictions.jsonl")
            else:
                required_predictions = require_reportable_gate or final_reportable_gate
                prediction_seed = {
                    "path": benchmark_predictions_raw,
                    "records": 0,
                    "source": "missing_model_generated_predictions",
                    "generation": generated_predictions,
                }
                gate = {
                    "status": "failed" if required_predictions else "pending",
                    "reason": "model_generated_predictions_required_for_non_smoke_reportable_gate",
                    "cycle": benchmark_cycle,
                    "configured_predictions": benchmark_predictions_raw,
                    "task_roots": [str(path) for path in reportable_paths],
                    "configured_task_roots": [str(path) for path in reportable_roots],
                    "root_sources": reportable_sources,
                    "required": required_predictions,
                    "prediction_generation": generated_predictions,
                }
                return gate, required_predictions

        reportable_cmd = [
            sys.executable,
            "-m",
            "omnicoder.eval.benchmark_suite_2026",
            "--profile",
            benchmark_profile,
            "--out-dir",
            str(reportable_dir),
            "--model",
            str(checkpoint_path),
            "--out",
            str(reportable_summary_path),
            "run-reportable",
            "--cycle",
            benchmark_cycle,
            "--run-id",
            run_id,
            "--min-tasks",
            str(benchmark_min_tasks),
            "--missing-reportable-policy",
            missing_policy,
        ]
        for path in reportable_paths:
            reportable_cmd.extend(["--tasks", str(path)])
        if int(prediction_seed.get("records") or 0) > 0:
            reportable_cmd.extend(["--predictions", str(prediction_seed["path"])])
        reportable_code = run_command(reportable_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_reportable.log")
        reportable_summary_cmd = [
            sys.executable,
            "-m",
            "omnicoder.eval.benchmark_suite_2026",
            "--profile",
            benchmark_profile,
            "--out-dir",
            str(reportable_dir),
            "--model",
            str(checkpoint_path),
            "--out",
            str(reportable_dir / "summary.json"),
            "summarize",
            "--results",
            "reportable_results.jsonl",
        ]
        reportable_summary_code = run_command(
            reportable_summary_cmd,
            out_dir / "logs" / f"benchmark_{safe_filename(phase)}_reportable_summarize.log",
        )
        reportable_summary = read_json(reportable_summary_path) if reportable_summary_path.exists() else {}
        gate = {
            "returncode": reportable_code,
            "summarize_returncode": reportable_summary_code,
            "out_dir": str(reportable_dir),
            "summary": str(reportable_dir / "summary.json"),
            "reportable_summary": str(reportable_summary_path),
            "task_roots": [str(path) for path in reportable_paths],
            "configured_task_roots": [str(path) for path in reportable_roots],
            "root_sources": reportable_sources,
            "predictions": prediction_seed,
            "status": reportable_summary.get("status"),
            "cycle": benchmark_cycle,
            "min_tasks": benchmark_min_tasks,
            "gate_policy": reportable_summary.get("gate_policy"),
            "gate_decision": reportable_summary.get("gate_decision"),
            "reportable": reportable_summary.get("reportable"),
            "failed": reportable_summary.get("failed"),
            "skipped": reportable_summary.get("skipped"),
            "local_only": reportable_summary.get("local_only"),
        }
        summary_status = str(reportable_summary.get("status") or "")
        summary_gate_decision = str(reportable_summary.get("gate_decision") or "")
        summary_not_reportable = (
            summary_status != "ok"
            or summary_gate_decision in {"blocked_needs_data", "pending_needs_data"}
            or int(reportable_summary.get("reportable") or 0) <= 0
            or int(reportable_summary.get("local_only") or 0) > 0
        )
        should_fail = (
            reportable_code != 0
            or reportable_summary_code != 0
            or summary_gate_decision == "blocked_needs_data"
            or ((require_reportable_gate or final_reportable_gate) and summary_not_reportable)
        )
        return gate, should_fail

    if checkpoint_path.is_dir():
        eval_paths = checkpoint_eval_paths(manifest)
        if eval_paths:
            sample_out = gate_dir / "heldout_pipeline_sample_loss.json"
            max_records_per_file = sample_loss_max_records_per_file(cfg, args, benchmark=True)
            sample_cmd = pipeline_sample_loss_launcher(cfg, args) + [
                "--checkpoint",
                str(checkpoint_path),
                "--preset",
                preset,
                "--seq-len",
                str(seq_len),
                "--max-records-per-file",
                str(max_records_per_file),
                "--out",
                str(sample_out),
            ]
            append_pipeline_sample_loss_runtime_args(sample_cmd, cfg, args)
            for path in eval_paths:
                sample_cmd.extend(["--data", str(path)])
            sample_code = run_command(sample_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_pipeline_sample_loss.log", timeout_seconds=sample_loss_timeout_seconds(cfg, args, benchmark=True))
            report["sample_loss"] = read_json(sample_out) if sample_out.exists() else {}
            report["sample_loss"]["returncode"] = sample_code
            if sample_code != 0:
                report["status"] = "failed"
        else:
            report["sample_loss"] = {"status": "skipped", "reason": "no_eval_or_test_jsonl"}
        require_reportable_gate = truthy_value(arg_value(args, "require_reportable_gate", False))
        report["contract_benchmark_gate"] = {
            "status": "skipped",
            "reason": "pipeline_checkpoint_text_generation_pending",
            "sample_loss_gate": "completed",
        }
        gates_cfg = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
        benchmark_profile = str(cfg.get("benchmark_profile") or gates_cfg.get("benchmark_profile") or "profiles/benchmark_suite_2026.json")
        reportable_gate, reportable_failed = run_reportable_gate(benchmark_profile, f"{safe_filename(phase)}_{int(time.time())}")
        report["reportable_gate"] = reportable_gate
        if reportable_failed and report.get("status") != "failed":
            report["status"] = "failed"
            report["reason"] = str(reportable_gate.get("reason") or "pipeline_checkpoint_reportable_gate_failed")
        write_json(gate_dir / "benchmark_gate_summary.json", report)
        return report

    eval_paths = checkpoint_eval_paths(manifest)
    if eval_paths:
        sample_out = gate_dir / "heldout_sample_loss.json"
        max_records_per_file = sample_loss_max_records_per_file(cfg, args, benchmark=True)
        sample_cmd = [
            sys.executable,
            "-m",
            "omnicoder.eval.sample_loss_2026",
            "--checkpoint",
            str(checkpoint_path),
            "--profile",
            preset,
            "--device",
            device,
            "--seq-len",
            str(seq_len),
            "--max-records-per-file",
            str(max_records_per_file),
            "--out",
            str(sample_out),
        ]
        for path in eval_paths:
            sample_cmd.extend(["--data", str(path)])
        sample_code = run_command(sample_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_sample_loss.log", timeout_seconds=sample_loss_timeout_seconds(cfg, args, benchmark=True))
        report["sample_loss"] = read_json(sample_out) if sample_out.exists() else {}
        report["sample_loss"]["returncode"] = sample_code
        if sample_code != 0:
            report["status"] = "failed"
    else:
        report["sample_loss"] = {"status": "skipped", "reason": "no_eval_or_test_jsonl"}

    gates_cfg = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
    benchmark_profile = str(cfg.get("benchmark_profile") or gates_cfg.get("benchmark_profile") or "profiles/benchmark_suite_2026.json")
    smoke_dir = gate_dir / "smoke"
    smoke_run_id = f"{safe_filename(phase)}_{int(time.time())}"
    validate_cmd = [
        sys.executable,
        "-m",
        "omnicoder.eval.benchmark_suite_2026",
        "--profile",
        benchmark_profile,
        "--out-dir",
        str(smoke_dir),
        "--model",
        str(checkpoint_path),
        "validate",
    ]
    smoke_cmd = [
        sys.executable,
        "-m",
        "omnicoder.eval.benchmark_suite_2026",
        "--profile",
        benchmark_profile,
        "--out-dir",
        str(smoke_dir),
        "--model",
        str(checkpoint_path),
        "run-smoke",
        "--mode",
        "smoke",
        "--cycle",
        "smoke",
        "--run-id",
        smoke_run_id,
    ]
    summarize_cmd = [
        sys.executable,
        "-m",
        "omnicoder.eval.benchmark_suite_2026",
        "--profile",
        benchmark_profile,
        "--out-dir",
        str(smoke_dir),
        "--model",
        str(checkpoint_path),
        "--out",
        str(smoke_dir / "summary.json"),
        "summarize",
    ]
    validate_code = run_command(validate_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_validate.log")
    smoke_code = run_command(smoke_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_smoke.log")
    summarize_code = run_command(summarize_cmd, out_dir / "logs" / f"benchmark_{safe_filename(phase)}_summarize.log")
    report["contract_benchmark_gate"] = {
        "validate_returncode": validate_code,
        "smoke_returncode": smoke_code,
        "summarize_returncode": summarize_code,
        "out_dir": str(smoke_dir),
        "summary": str(smoke_dir / "summary.json"),
    }
    if validate_code != 0 or smoke_code != 0 or summarize_code != 0:
        report["status"] = "failed"

    reportable_gate, reportable_failed = run_reportable_gate(benchmark_profile, smoke_run_id)
    report["reportable_gate"] = reportable_gate
    if reportable_failed:
        report["status"] = "failed"

    write_json(gate_dir / "benchmark_gate_summary.json", report)
    return report


def run_distillation_curriculum_stage(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    out_dir: Path,
    checkpoint: str | Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    distill = cfg.get("distillation") if isinstance(cfg.get("distillation"), dict) else {}
    teacher_profile = str(arg_value(args, "distill_profile", "") or distill.get("teacher_profile") or "profiles/distillation_curriculum_2026.json")
    records = str(manifest.get("curated_jsonl") or manifest.get("train_all_jsonl") or "")
    distill_dir = out_dir / "distillation"
    limit = int(arg_value(args, "distill_limit", 0) or distill.get("per_teacher_limit") or 0)
    curriculum_cmd = [
        sys.executable,
        "-m",
        "omnicoder.training.distillation_curriculum_2026",
        "all",
        "--profile",
        teacher_profile,
        "--records",
        records,
        "--out-dir",
        str(distill_dir),
    ]
    if limit > 0:
        curriculum_cmd.extend(["--limit", str(limit)])
    curriculum_code = run_command(curriculum_cmd, out_dir / "logs" / "distillation_curriculum_command.log")
    manifest_path = distill_dir / "distillation_curriculum_manifest.json"
    stage: dict[str, Any] = {
        "schema": "omnicoder.distillation_training_stage_2026.v1",
        "status": "passed" if curriculum_code == 0 and manifest_path.exists() else "failed",
        "curriculum_returncode": curriculum_code,
        "curriculum_manifest": str(manifest_path),
        "initial_checkpoint": str(checkpoint) if checkpoint else None,
    }
    if stage["status"] != "passed":
        return stage
    curriculum = read_json(manifest_path)
    jobs_path = Path(str((curriculum.get("outputs") or {}).get("jobs") or ""))
    stage["jobs"] = int(curriculum.get("jobs") or 0)
    stage["jobs_path"] = str(jobs_path)
    if not checkpoint or not Path(str(checkpoint)).exists():
        stage.update({"status": "failed", "reason": "missing_checkpoint_for_distillation_replay"})
        return stage
    if not jobs_path.exists() or jobs_path.stat().st_size <= 0:
        stage.update({"status": "failed", "reason": "no_distillation_jobs"})
        return stage
    training_plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    replay_checkpoint = out_dir / "checkpoints" / "09_distillation_replay.pt"
    replay_log = out_dir / "logs" / "09_distillation_replay_loss.jsonl"
    steps = int(arg_value(args, "distill_steps", 0) or distill.get("steps") or training_plan.get("steps_per_stage") or 64)
    seq_len = int(arg_value(args, "seq_len", 0) or training_plan.get("seq_len") or 192)
    batch_size = int(arg_value(args, "batch_size", 0) or training_plan.get("batch_size") or 1)
    lr = float(arg_value(args, "distill_lr", 0.0) or distill.get("learning_rate") or training_plan.get("learning_rate") or 0.001)
    preset = resolve_training_preset(cfg, args)
    guard_target_training_preset(cfg, preset, args)
    device = str(arg_value(args, "device", "") or training_plan.get("device") or ("cuda" if torch_available() else "cpu"))
    save_interval = resolve_save_interval(args, training_plan.get("save_interval"))
    pipeline_stage_trainer = uses_pipeline_stage_trainer(cfg, args)
    train_cmd = pretrain_launcher(cfg, args) + [
        "--preset",
        preset,
        "--data",
        str(jobs_path),
        "--out",
        str(replay_checkpoint),
        "--seq_len",
        str(seq_len),
        "--batch_size",
        str(batch_size),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--max_records",
        "0",
        "--log_file",
        str(replay_log),
        "--data_manifest",
        str(manifest.get("curation_manifest") or out_dir / "manifests" / "curation_manifest.json"),
        "--resume",
        str(checkpoint),
    ]
    if not pipeline_stage_trainer:
        train_cmd.extend(["--device", device, "--aux_probe"])
    append_pretrain_runtime_args(train_cmd, cfg, args)
    if bool(arg_value(args, "fake_quant", False) or training_plan.get("fake_quant") or cfg.get("q4_recovery", {}).get("enabled")):
        train_cmd.append("--fake_quant")
    if save_interval > 0:
        train_cmd.extend(["--save_interval", str(save_interval)])
    train_code = run_command(train_cmd, out_dir / "logs" / "09_distillation_replay_command.log")
    losses = parse_losses(replay_log)
    stage.update(
        {
            "training_returncode": train_code,
            "checkpoint": str(replay_checkpoint),
            "loss_log": str(replay_log),
            "loss_points": len(losses),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "final_checkpoint": str(replay_checkpoint) if train_code == 0 and replay_checkpoint.exists() else str(checkpoint),
            "status": "passed" if train_code == 0 and replay_checkpoint.exists() else "failed",
        }
    )
    if stage["status"] == "passed":
        stage["heldout_benchmark_gate"] = run_checkpoint_benchmark_gate(profile, manifest, out_dir, replay_checkpoint, "distillation_replay", args)
    return stage


def run_final_finetune_stage(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    out_dir: Path,
    checkpoint: str | Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not checkpoint or not Path(str(checkpoint)).exists():
        return {"status": "failed", "reason": "missing_checkpoint_for_final_finetune", "initial_checkpoint": str(checkpoint)}
    cfg = profile_cfg(profile)
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    data_path = Path(str(manifest.get("train_all_jsonl") or ""))
    if not data_path.exists():
        return {"status": "failed", "reason": "missing_train_all_jsonl", "train_all_jsonl": str(data_path)}
    checkpoint_out = out_dir / "checkpoints" / "99_final_all_modality_finetune.pt"
    train_log = out_dir / "logs" / "99_final_all_modality_finetune_loss.jsonl"
    steps = int(arg_value(args, "finetune_steps", 0) or plan.get("finetune_steps") or plan.get("steps_per_stage") or 64)
    seq_len = int(arg_value(args, "seq_len", 0) or plan.get("seq_len") or 192)
    batch_size = int(arg_value(args, "batch_size", 0) or plan.get("batch_size") or 1)
    base_lr = float(arg_value(args, "lr", 0.0) or plan.get("learning_rate") or 0.001)
    lr = float(arg_value(args, "finetune_lr", 0.0) or plan.get("finetune_learning_rate") or (base_lr * 0.25))
    preset = resolve_training_preset(cfg, args)
    guard_target_training_preset(cfg, preset, args)
    device = str(arg_value(args, "device", "") or plan.get("device") or ("cuda" if torch_available() else "cpu"))
    save_interval = resolve_save_interval(args, plan.get("save_interval"))
    pipeline_stage_trainer = uses_pipeline_stage_trainer(cfg, args)
    cmd = pretrain_launcher(cfg, args) + [
        "--preset",
        preset,
        "--data",
        str(data_path),
        "--out",
        str(checkpoint_out),
        "--seq_len",
        str(seq_len),
        "--batch_size",
        str(batch_size),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--max_records",
        "0",
        "--log_file",
        str(train_log),
        "--data_manifest",
        str(out_dir / "manifests" / "curation_manifest.json"),
        "--resume",
        str(checkpoint),
    ]
    if not pipeline_stage_trainer:
        cmd.extend(["--device", device, "--aux_probe"])
    append_pretrain_runtime_args(cmd, cfg, args)
    if bool(arg_value(args, "fake_quant", False) or plan.get("fake_quant") or cfg.get("q4_recovery", {}).get("enabled")):
        cmd.append("--fake_quant")
    if save_interval > 0:
        cmd.extend(["--save_interval", str(save_interval)])
    code = run_command(cmd, out_dir / "logs" / "99_final_all_modality_finetune_command.log")
    losses = parse_losses(train_log)
    checkpoint_complete = checkpoint_is_complete(checkpoint_out, expected_world_size=expected_world_size)
    report = {
        "schema": "omnicoder.final_finetune_stage_2026.v1",
        "status": "passed" if code == 0 and checkpoint_complete else "failed",
        "returncode": code,
        "initial_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint_out),
        "final_checkpoint": str(checkpoint_out) if checkpoint_complete else str(checkpoint),
        "checkpoint_complete": checkpoint_complete,
        "train_jsonl": str(data_path),
        "loss_log": str(train_log),
        "loss_points": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    if report["status"] == "passed":
        report["heldout_benchmark_gate"] = run_checkpoint_benchmark_gate(profile, manifest, out_dir, checkpoint_out, "final_finetune", args)
    return report


def full_run_status(*parts: dict[str, Any]) -> str:
    for part in parts:
        if part.get("status") == "failed":
            return "failed"
    return "passed"


def validate_posttraining_resume_checkpoint(cfg: dict[str, Any], checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    preset = resolve_training_preset(cfg, args)
    guard_target_training_preset(cfg, preset, args)
    if not checkpoint.exists():
        return {"status": "failed", "reason": "resume_checkpoint_missing", "checkpoint": str(checkpoint)}
    expected_world_size = expected_pipeline_world_size(cfg, args) if checkpoint.is_dir() else None
    if not checkpoint_is_complete(checkpoint, expected_world_size=expected_world_size):
        return {
            "status": "failed",
            "reason": "resume_checkpoint_incomplete",
            "checkpoint": str(checkpoint),
            "expected_world_size": expected_world_size,
            "completion_marker": str(checkpoint_complete_marker(checkpoint)),
        }
    if checkpoint.is_dir() and not uses_pipeline_stage_trainer(cfg, args):
        return {
            "status": "failed",
            "reason": "sharded_resume_checkpoint_requires_pipeline_stage_trainer",
            "checkpoint": str(checkpoint),
        }
    return {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "preset": preset,
        "expected_world_size": expected_world_size,
        "pipeline_stage_trainer": bool(uses_pipeline_stage_trainer(cfg, args)),
    }


def run_posttrain(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    cfg = profile_cfg(profile)
    out_dir = Path(args.out_dir or cfg.get("work_dir") or DEFAULT_OUT_DIR)
    resume_checkpoint = Path(str(args.resume_checkpoint or "")).expanduser()
    validation = validate_posttraining_resume_checkpoint(cfg, resume_checkpoint, args)
    if validation.get("status") != "passed":
        summary = {
            "schema": "omnicoder.posttraining_resume_result_2026.v1",
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "created_at": now_iso(),
            "model_contract": cfg.get("model_contract"),
            "resume_validation": validation,
            "posttraining": {"status": "skipped", "reason": validation.get("reason")},
            "artifacts": {"out_dir": str(out_dir), "summary": str(out_dir / "posttraining_resume_summary.json")},
        }
        write_json(out_dir / "posttraining_resume_summary.json", summary)
        return summary
    post_args = namespace_with(args, live_posttraining=True)
    try:
        posttraining = run_posttraining_stages(
            profile,
            out_dir,
            {"status": "passed", "final_checkpoint": str(resume_checkpoint)},
            post_args,
        )
    except ValueError as exc:
        posttraining = {"status": "failed", "reason": "invalid_posttraining_algorithm_selection", "error": str(exc)}
    summary = {
        "schema": "omnicoder.posttraining_resume_result_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": posttraining.get("status", "failed"),
        "created_at": now_iso(),
        "model_contract": cfg.get("model_contract"),
        "resume_validation": validation,
        "posttraining": posttraining,
        "final_checkpoint": posttraining.get("final_checkpoint") or str(resume_checkpoint),
        "artifacts": {
            "out_dir": str(out_dir),
            "summary": str(out_dir / "posttraining_resume_summary.json"),
            "initial_checkpoint": str(resume_checkpoint),
            "final_checkpoint": posttraining.get("final_checkpoint") or str(resume_checkpoint),
        },
    }
    final_checkpoint = posttraining.get("final_checkpoint") or str(resume_checkpoint)
    if posttraining.get("status") == "passed" and final_checkpoint:
        eval_manifest = load_posttraining_eval_manifest(profile, out_dir, args)
        summary["heldout_benchmark_gate"] = run_checkpoint_benchmark_gate(
            profile,
            eval_manifest,
            out_dir,
            final_checkpoint,
            "posttraining_resume_final",
            args,
        )
    write_json(out_dir / "posttraining_resume_summary.json", summary)
    return summary


def run_long_context(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    cfg = profile_cfg(profile)
    out_dir = Path(args.out_dir or cfg.get("work_dir") or DEFAULT_OUT_DIR)
    resume_checkpoint = Path(str(args.resume_checkpoint or "")).expanduser()
    validation = validate_posttraining_resume_checkpoint(cfg, resume_checkpoint, args)
    summary_path = out_dir / "long_context_resume_summary.json"
    if validation.get("status") != "passed":
        summary = {
            "schema": "omnicoder.long_context_resume_result_2026.v1",
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "created_at": now_iso(),
            "model_contract": cfg.get("model_contract"),
            "resume_validation": validation,
            "long_context_curriculum": {"status": "skipped", "reason": validation.get("reason")},
            "artifacts": {"out_dir": str(out_dir), "summary": str(summary_path)},
        }
        write_json(summary_path, summary)
        return summary
    curation_manifest_arg = str(arg_value(args, "curation_manifest", "") or "").strip()
    manifest_path = (
        resolve_path(curation_manifest_arg, repo_root())
        if curation_manifest_arg
        else out_dir / "manifests" / "curation_manifest.json"
    )
    if not manifest_path.exists():
        summary = {
            "schema": "omnicoder.long_context_resume_result_2026.v1",
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "created_at": now_iso(),
            "model_contract": cfg.get("model_contract"),
            "resume_validation": validation,
            "long_context_curriculum": {
                "status": "skipped",
                "reason": "missing_curation_manifest",
                "curation_manifest": str(manifest_path),
            },
            "artifacts": {
                "out_dir": str(out_dir),
                "summary": str(summary_path),
                "initial_checkpoint": str(resume_checkpoint),
                "curation_manifest": str(manifest_path),
            },
        }
        write_json(summary_path, summary)
        return summary
    manifest = read_json(manifest_path)
    long_context_curriculum = run_long_context_curriculum_stage(profile, manifest, out_dir, resume_checkpoint, args)
    summary = {
        "schema": "omnicoder.long_context_resume_result_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": long_context_curriculum.get("status", "failed"),
        "created_at": now_iso(),
        "model_contract": cfg.get("model_contract"),
        "resume_validation": validation,
        "curation": manifest,
        "long_context_curriculum": long_context_curriculum,
        "final_checkpoint": long_context_curriculum.get("final_checkpoint") or str(resume_checkpoint),
        "artifacts": {
            "out_dir": str(out_dir),
            "summary": str(summary_path),
            "initial_checkpoint": str(resume_checkpoint),
            "final_checkpoint": long_context_curriculum.get("final_checkpoint") or str(resume_checkpoint),
            "curation_manifest": str(manifest_path),
        },
    }
    write_json(summary_path, summary)
    return summary


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    out_dir = Path(args.out_dir or profile_cfg(profile).get("work_dir") or DEFAULT_OUT_DIR)
    manifest = build_real_corpus(profile, out_dir)
    pretrain = run_training_stages(profile, manifest, out_dir, args)
    current_checkpoint = pretrain.get("final_checkpoint")
    if pretrain.get("status") == "passed" and current_checkpoint:
        long_context_curriculum = run_long_context_curriculum_stage(profile, manifest, out_dir, current_checkpoint, args)
        current_checkpoint = long_context_curriculum.get("final_checkpoint") or current_checkpoint
    else:
        long_context_curriculum = {"status": "skipped", "reason": "multimodal_pretrain_failed"}
    initial_benchmark = (
        run_checkpoint_benchmark_gate(profile, manifest, out_dir, current_checkpoint, "after_multimodal_pretrain", args)
        if long_context_curriculum.get("status") == "passed"
        else {"status": "skipped", "reason": "long_context_curriculum_failed"}
    )
    if long_context_curriculum.get("status") == "passed":
        distillation = run_distillation_curriculum_stage(profile, manifest, out_dir, current_checkpoint, args)
        current_checkpoint = distillation.get("final_checkpoint") or current_checkpoint
    else:
        distillation = {"status": "skipped", "reason": "long_context_curriculum_failed"}
    if distillation.get("status") in {"passed", "skipped"} and current_checkpoint:
        post_args = namespace_with(args, live_posttraining=True)
        posttraining = run_posttraining_stages(
            profile,
            out_dir,
            {"status": "passed", "final_checkpoint": current_checkpoint},
            post_args,
        )
        current_checkpoint = posttraining.get("final_checkpoint") or current_checkpoint
    else:
        posttraining = {"status": "skipped", "reason": "distillation_failed"}
    if posttraining.get("status") == "passed" and current_checkpoint:
        finetune = run_final_finetune_stage(profile, manifest, out_dir, current_checkpoint, args)
        current_checkpoint = finetune.get("final_checkpoint") or current_checkpoint
    else:
        finetune = {"status": "skipped", "reason": "posttraining_failed"}
    final_benchmark = (
        run_checkpoint_benchmark_gate(profile, manifest, out_dir, current_checkpoint, "full_run_final", args)
        if current_checkpoint
        else {"status": "skipped", "reason": "no_final_checkpoint"}
    )
    summary = {
        "schema": "omnicoder.full_training_orchestration_result_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": full_run_status(pretrain, long_context_curriculum, distillation, posttraining, finetune, final_benchmark),
        "created_at": now_iso(),
        "model_contract": profile_cfg(profile).get("model_contract"),
        "curation": manifest,
        "pretraining": pretrain,
        "long_context_curriculum": long_context_curriculum,
        "benchmark_after_pretraining": initial_benchmark,
        "distillation": distillation,
        "posttraining": posttraining,
        "finetune": finetune,
        "final_benchmark": final_benchmark,
        "benchmark_gates": {
            "status": full_run_status(initial_benchmark, final_benchmark),
            "required_cycles": ["smoke", "reportable_authorized_or_official"],
            "after_pretraining": initial_benchmark,
            "final": final_benchmark,
        },
        "final_checkpoint": str(current_checkpoint) if current_checkpoint else None,
        "artifacts": {
            "out_dir": str(out_dir),
            "curation_manifest": str(out_dir / "manifests" / "curation_manifest.json"),
            "summary": str(out_dir / "full_training_summary.json"),
            "final_checkpoint": str(current_checkpoint) if current_checkpoint else None,
        },
    }
    write_json(out_dir / "full_training_summary.json", summary)
    return summary


def validate(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    cfg = profile_cfg(profile)
    return {
        "status": "ok",
        "profile_version": profile.get("version"),
        "modalities": sorted(enabled_modalities(cfg)),
        "real_sources": sorted((cfg.get("real_sources") or {}).keys()),
        "model_contract": cfg.get("model_contract"),
        "training_plan": cfg.get("training_plan"),
        "learning_checks": training_checks(cfg),
    }


def inventory(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    cfg = profile_cfg(profile)
    root = repo_root()
    sources = cfg["real_sources"]
    result: dict[str, Any] = {"status": "ok", "repo_root": str(root), "sources": {}}
    for key, value in sources.items():
        paths = existing_paths(value, root)
        result["sources"][key] = [{"path": str(path), "exists": True, "is_dir": path.is_dir()} for path in paths]
    return result


def curate_real(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    out_dir = Path(args.out_dir or profile_cfg(profile).get("work_dir") or DEFAULT_OUT_DIR)
    return build_real_corpus(profile, out_dir)


def mix_plan(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    out_dir = Path(args.out_dir or profile_cfg(profile).get("work_dir") or DEFAULT_OUT_DIR)
    return build_adaptive_mixture_plan(
        profile,
        out_dir,
        curation_manifest_path=args.curation_manifest or None,
        external_manifest_path=args.external_manifest or None,
        agentic_manifest_path=args.agentic_manifest or None,
        teacher_manifest_path=args.teacher_manifest or None,
        output_path=args.output or None,
    )


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    out_dir = Path(args.out_dir or profile_cfg(profile).get("work_dir") or DEFAULT_OUT_DIR)
    manifest = build_real_corpus(profile, out_dir)
    training = run_training_stages(profile, manifest, out_dir, args)
    if training["status"] == "passed" and training.get("final_checkpoint"):
        long_context_curriculum = run_long_context_curriculum_stage(profile, manifest, out_dir, training["final_checkpoint"], args)
    else:
        long_context_curriculum = {"status": "skipped", "reason": "dense_training_failed"}
    if long_context_curriculum["status"] == "passed":
        training_with_context = dict(training)
        training_with_context["final_checkpoint"] = long_context_curriculum.get("final_checkpoint") or training.get("final_checkpoint")
        posttraining = run_posttraining_stages(profile, out_dir, training_with_context, args)
    else:
        posttraining = {"status": "skipped", "reason": "long_context_curriculum_failed", "stages": []}
    status = "failed" if training["status"] == "failed" or long_context_curriculum["status"] == "failed" or posttraining["status"] == "failed" else "passed"
    summary = {
        "schema": "omnicoder.real_training_orchestration_result_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": now_iso(),
        "model_contract": profile_cfg(profile).get("model_contract"),
        "curation": manifest,
        "training": training,
        "long_context_curriculum": long_context_curriculum,
        "posttraining": posttraining,
        "artifacts": {
            "out_dir": str(out_dir),
            "curation_manifest": str(out_dir / "manifests" / "curation_manifest.json"),
            "summary": str(out_dir / "real_training_summary.json"),
        },
    }
    write_json(out_dir / "real_training_summary.json", summary)
    return summary


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.summary or Path(args.out_dir or DEFAULT_OUT_DIR) / "real_training_summary.json")
    if not path.exists():
        raise SystemExit(json.dumps({"status": "error", "error": "summary not found", "summary": str(path)}))
    return read_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Production Omnicoder 2026 real multimodal training orchestration")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate").set_defaults(func=validate)
    sub.add_parser("inventory").set_defaults(func=inventory)
    sub.add_parser("curate-real").set_defaults(func=curate_real)
    mix = sub.add_parser("mix-plan")
    mix.add_argument("--curation-manifest", default="")
    mix.add_argument("--external-manifest", default="")
    mix.add_argument("--agentic-manifest", default="")
    mix.add_argument("--teacher-manifest", default="")
    mix.add_argument("--output", default="")
    mix.set_defaults(func=mix_plan)
    run = sub.add_parser("run-real")
    run.add_argument("--steps-per-stage", type=int, default=0)
    run.add_argument("--seq-len", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=0)
    run.add_argument("--lr", type=float, default=0.0)
    run.add_argument("--preset", default="")
    run.add_argument("--device", default="")
    run.add_argument("--fake-quant", action="store_true")
    run.add_argument("--resume-checkpoint", default="")
    run.add_argument("--start-stage", default="", help="Start dense pretraining at this 1-based stage index or modality name")
    run.add_argument("--stage-order", default="", help="Comma-separated dense pretraining stage order override")
    run.add_argument("--context-ladder", dest="context_ladder", default="", help="Comma-separated real long-context curriculum ladder; defaults to the 8K..1M profile ladder")
    run.add_argument("--long-context-steps-per-rung", dest="long_context_steps_per_rung", type=int, default=0)
    run.add_argument("--resume-completed-stages", dest="resume_completed_stages", action="store_true", default=None, help="Skip stages whose expected checkpoint already exists")
    run.add_argument("--rerun-completed-stages", dest="resume_completed_stages", action="store_false", help="Retrain stages even when their expected checkpoint exists")
    run.add_argument("--save-interval", type=int, default=None)
    run.add_argument("--distributed", default="")
    run.add_argument("--nproc-per-node", type=int, default=0)
    run.add_argument("--precision", default="")
    run.add_argument("--init-dtype", default="")
    run.add_argument("--optimizer", default="")
    run.add_argument("--optimizer-in-backward", dest="optimizer_in_backward", action="store_true")
    run.add_argument("--optimizer-in-backward-update", dest="optimizer_in_backward_update", default="")
    run.add_argument("--optimizer-in-backward-grad-clip", dest="optimizer_in_backward_grad_clip", type=float, default=0.0)
    run.add_argument("--optimizer-in-backward-clip-mode", dest="optimizer_in_backward_clip_mode", default="")
    run.add_argument("--optimizer-in-backward-adafactor-chunk-rows", dest="optimizer_in_backward_adafactor_chunk_rows", type=int, default=0)
    run.add_argument("--optimizer-in-backward-adafactor-clip-threshold", dest="optimizer_in_backward_adafactor_clip_threshold", type=float, default=0.0)
    run.add_argument("--optimizer-in-backward-adafactor-decay-rate", dest="optimizer_in_backward_adafactor_decay_rate", type=float, default=0.0)
    run.add_argument("--optimizer-in-backward-adafactor-eps1", dest="optimizer_in_backward_adafactor_eps1", type=float, default=0.0)
    run.add_argument("--rank-device-map", dest="rank_device_map", default="")
    run.add_argument("--placement", default="")
    run.add_argument("--placement-devices", dest="placement_devices", default="")
    run.add_argument("--placement-layer-counts", dest="placement_layer_counts", default="")
    run.add_argument("--placement-head-device", dest="placement_head_device", type=int, default=-1)
    run.add_argument("--placement-schedule", dest="placement_schedule", default="")
    run.add_argument("--pipeline-stage-schedule", dest="pipeline_stage_schedule", default="")
    run.add_argument("--pipeline-microbatches", dest="pipeline_microbatches", type=int, default=0)
    run.add_argument("--pipeline-async-streams", dest="pipeline_async_streams", action="store_true", default=None)
    run.add_argument("--no-pipeline-async-streams", dest="pipeline_async_streams", action="store_false")
    run.add_argument("--activation-checkpointing", action="store_true")
    run.add_argument("--cpu-offload", action="store_true")
    run.add_argument("--fake-quant-chunk-rows", dest="fake_quant_chunk_rows", type=int, default=0)
    run.add_argument("--fake-quant-max-full-elements", dest="fake_quant_max_full_elements", type=int, default=0)
    run.add_argument("--allow-verifier-preset", action="store_true")
    run.add_argument("--live-posttraining", action="store_true", help="Run native reward replay instead of posttraining bridge dry-run only")
    run.add_argument("--posttrain-steps", type=int, default=0)
    run.add_argument("--posttrain-lr", type=float, default=0.0)
    run.add_argument("--posttrain-max-records", type=int, default=0)
    run.add_argument(
        "--posttrain-input-jsonl",
        dest="posttrain_input_jsonl",
        action="append",
        default=[],
        help="Explicit posttraining JSONL input, optionally algorithm=path. Takes priority over profile discovery.",
    )
    run.add_argument("--heldout-max-records-per-file", dest="heldout_max_records_per_file", type=int, default=None)
    run.add_argument("--benchmark-max-records-per-file", dest="benchmark_max_records_per_file", type=int, default=None)
    run.add_argument("--heldout-sample-loss-timeout-seconds", dest="heldout_sample_loss_timeout_seconds", type=int, default=0)
    run.add_argument("--benchmark-sample-loss-timeout-seconds", dest="benchmark_sample_loss_timeout_seconds", type=int, default=0)
    run.add_argument("--benchmark-cycle", dest="benchmark_cycle", default="")
    run.add_argument("--benchmark-min-tasks", dest="benchmark_min_tasks", type=int, default=0)
    run.add_argument("--benchmark-predictions", dest="benchmark_predictions", default="")
    run.add_argument("--benchmark-prediction-backend", dest="benchmark_prediction_backend", default="")
    run.add_argument("--benchmark-prediction-model", dest="benchmark_prediction_model", default="")
    run.add_argument("--benchmark-prediction-base-url", dest="benchmark_prediction_base_url", default="")
    run.add_argument("--benchmark-prediction-api-key-env", dest="benchmark_prediction_api_key_env", default="")
    run.add_argument("--benchmark-prediction-checkpoint-runner", dest="benchmark_prediction_checkpoint_runner", default="")
    run.add_argument("--benchmark-prediction-timeout-seconds", dest="benchmark_prediction_timeout_seconds", type=int, default=0)
    run.add_argument("--benchmark-prediction-max-output-tokens", dest="benchmark_prediction_max_output_tokens", type=int, default=0)
    run.add_argument("--reportable-task-root", dest="reportable_task_roots", action="append", default=[])
    run.add_argument("--require-reportable-gate", dest="require_reportable_gate", action="store_true")
    run.add_argument("--rerun-heldout-evals", dest="rerun_heldout_evals", action="store_true")
    run.set_defaults(func=run_real)
    long = sub.add_parser(
        "run-long-context",
        aliases=["run-longctx"],
        help="Resume only the native long-context curriculum ladder from an existing complete checkpoint",
    )
    long.add_argument("--resume-checkpoint", required=True)
    long.add_argument("--curation-manifest", default="")
    long.add_argument("--steps-per-stage", type=int, default=0)
    long.add_argument("--seq-len", type=int, default=0)
    long.add_argument("--batch-size", type=int, default=0)
    long.add_argument("--lr", type=float, default=0.0)
    long.add_argument("--preset", default="")
    long.add_argument("--device", default="")
    long.add_argument("--fake-quant", action="store_true")
    long.add_argument("--context-ladder", dest="context_ladder", default="", help="Comma-separated real long-context curriculum ladder; defaults to the 8K..1M profile ladder")
    long.add_argument("--long-context-steps-per-rung", dest="long_context_steps_per_rung", type=int, default=0)
    long.add_argument("--resume-completed-stages", dest="resume_completed_stages", action="store_true", default=None, help="Skip rungs whose expected checkpoint already exists")
    long.add_argument("--rerun-completed-stages", dest="resume_completed_stages", action="store_false", help="Retrain rungs even when their expected checkpoint exists")
    long.add_argument("--save-interval", type=int, default=None)
    long.add_argument("--distributed", default="")
    long.add_argument("--nproc-per-node", type=int, default=0)
    long.add_argument("--precision", default="")
    long.add_argument("--init-dtype", default="")
    long.add_argument("--optimizer", default="")
    long.add_argument("--optimizer-in-backward", dest="optimizer_in_backward", action="store_true")
    long.add_argument("--optimizer-in-backward-update", dest="optimizer_in_backward_update", default="")
    long.add_argument("--optimizer-in-backward-grad-clip", dest="optimizer_in_backward_grad_clip", type=float, default=0.0)
    long.add_argument("--optimizer-in-backward-clip-mode", dest="optimizer_in_backward_clip_mode", default="")
    long.add_argument("--optimizer-in-backward-adafactor-chunk-rows", dest="optimizer_in_backward_adafactor_chunk_rows", type=int, default=0)
    long.add_argument("--optimizer-in-backward-adafactor-clip-threshold", dest="optimizer_in_backward_adafactor_clip_threshold", type=float, default=0.0)
    long.add_argument("--optimizer-in-backward-adafactor-decay-rate", dest="optimizer_in_backward_adafactor_decay_rate", type=float, default=0.0)
    long.add_argument("--optimizer-in-backward-adafactor-eps1", dest="optimizer_in_backward_adafactor_eps1", type=float, default=0.0)
    long.add_argument("--rank-device-map", dest="rank_device_map", default="")
    long.add_argument("--placement", default="")
    long.add_argument("--placement-devices", dest="placement_devices", default="")
    long.add_argument("--placement-layer-counts", dest="placement_layer_counts", default="")
    long.add_argument("--placement-head-device", dest="placement_head_device", type=int, default=-1)
    long.add_argument("--placement-schedule", dest="placement_schedule", default="")
    long.add_argument("--pipeline-stage-schedule", dest="pipeline_stage_schedule", default="")
    long.add_argument("--pipeline-microbatches", dest="pipeline_microbatches", type=int, default=0)
    long.add_argument("--pipeline-async-streams", dest="pipeline_async_streams", action="store_true", default=None)
    long.add_argument("--no-pipeline-async-streams", dest="pipeline_async_streams", action="store_false")
    long.add_argument("--activation-checkpointing", action="store_true")
    long.add_argument("--cpu-offload", action="store_true")
    long.add_argument("--fake-quant-chunk-rows", dest="fake_quant_chunk_rows", type=int, default=0)
    long.add_argument("--fake-quant-max-full-elements", dest="fake_quant_max_full_elements", type=int, default=0)
    long.add_argument("--allow-verifier-preset", action="store_true")
    long.add_argument("--heldout-max-records-per-file", dest="heldout_max_records_per_file", type=int, default=None)
    long.add_argument("--benchmark-max-records-per-file", dest="benchmark_max_records_per_file", type=int, default=None)
    long.add_argument("--heldout-sample-loss-timeout-seconds", dest="heldout_sample_loss_timeout_seconds", type=int, default=0)
    long.add_argument("--benchmark-sample-loss-timeout-seconds", dest="benchmark_sample_loss_timeout_seconds", type=int, default=0)
    long.add_argument("--benchmark-cycle", dest="benchmark_cycle", default="")
    long.add_argument("--benchmark-min-tasks", dest="benchmark_min_tasks", type=int, default=0)
    long.add_argument("--benchmark-predictions", dest="benchmark_predictions", default="")
    long.add_argument("--benchmark-prediction-backend", dest="benchmark_prediction_backend", default="")
    long.add_argument("--benchmark-prediction-model", dest="benchmark_prediction_model", default="")
    long.add_argument("--benchmark-prediction-base-url", dest="benchmark_prediction_base_url", default="")
    long.add_argument("--benchmark-prediction-api-key-env", dest="benchmark_prediction_api_key_env", default="")
    long.add_argument("--benchmark-prediction-checkpoint-runner", dest="benchmark_prediction_checkpoint_runner", default="")
    long.add_argument("--benchmark-prediction-timeout-seconds", dest="benchmark_prediction_timeout_seconds", type=int, default=0)
    long.add_argument("--benchmark-prediction-max-output-tokens", dest="benchmark_prediction_max_output_tokens", type=int, default=0)
    long.add_argument("--reportable-task-root", dest="reportable_task_roots", action="append", default=[])
    long.add_argument("--require-reportable-gate", dest="require_reportable_gate", action="store_true")
    long.add_argument("--rerun-heldout-evals", dest="rerun_heldout_evals", action="store_true")
    long.set_defaults(func=run_long_context)
    post = sub.add_parser(
        "run-posttraining",
        aliases=["run-posttrain"],
        help="Resume live posttraining from an existing complete 20B/1M checkpoint without rerunning dense stages",
    )
    post.add_argument("--resume-checkpoint", required=True)
    post.add_argument("--posttrain-start-algorithm", "--start-posttrain-algorithm", dest="start_posttrain_algorithm", default="")
    post.add_argument("--posttrain-algorithm-order", dest="posttrain_algorithm_order", default="")
    post.add_argument("--curation-manifest", default="")
    post.add_argument("--seq-len", type=int, default=0)
    post.add_argument("--batch-size", type=int, default=0)
    post.add_argument("--preset", default="")
    post.add_argument("--device", default="")
    post.add_argument("--fake-quant", action="store_true")
    post.add_argument("--distributed", default="")
    post.add_argument("--nproc-per-node", type=int, default=0)
    post.add_argument("--precision", default="")
    post.add_argument("--init-dtype", default="")
    post.add_argument("--optimizer", default="")
    post.add_argument("--optimizer-in-backward", dest="optimizer_in_backward", action="store_true")
    post.add_argument("--optimizer-in-backward-update", dest="optimizer_in_backward_update", default="")
    post.add_argument("--optimizer-in-backward-grad-clip", dest="optimizer_in_backward_grad_clip", type=float, default=0.0)
    post.add_argument("--optimizer-in-backward-clip-mode", dest="optimizer_in_backward_clip_mode", default="")
    post.add_argument("--optimizer-in-backward-adafactor-chunk-rows", dest="optimizer_in_backward_adafactor_chunk_rows", type=int, default=0)
    post.add_argument("--optimizer-in-backward-adafactor-clip-threshold", dest="optimizer_in_backward_adafactor_clip_threshold", type=float, default=0.0)
    post.add_argument("--optimizer-in-backward-adafactor-decay-rate", dest="optimizer_in_backward_adafactor_decay_rate", type=float, default=0.0)
    post.add_argument("--optimizer-in-backward-adafactor-eps1", dest="optimizer_in_backward_adafactor_eps1", type=float, default=0.0)
    post.add_argument("--rank-device-map", dest="rank_device_map", default="")
    post.add_argument("--placement", default="")
    post.add_argument("--placement-devices", dest="placement_devices", default="")
    post.add_argument("--placement-layer-counts", dest="placement_layer_counts", default="")
    post.add_argument("--placement-head-device", dest="placement_head_device", type=int, default=-1)
    post.add_argument("--placement-schedule", dest="placement_schedule", default="")
    post.add_argument("--pipeline-stage-schedule", dest="pipeline_stage_schedule", default="")
    post.add_argument("--pipeline-microbatches", dest="pipeline_microbatches", type=int, default=0)
    post.add_argument("--pipeline-async-streams", dest="pipeline_async_streams", action="store_true", default=None)
    post.add_argument("--no-pipeline-async-streams", dest="pipeline_async_streams", action="store_false")
    post.add_argument("--activation-checkpointing", action="store_true")
    post.add_argument("--cpu-offload", action="store_true")
    post.add_argument("--fake-quant-chunk-rows", dest="fake_quant_chunk_rows", type=int, default=0)
    post.add_argument("--fake-quant-max-full-elements", dest="fake_quant_max_full_elements", type=int, default=0)
    post.add_argument("--allow-verifier-preset", action="store_true")
    post.add_argument("--posttrain-steps", type=int, default=0)
    post.add_argument("--posttrain-lr", type=float, default=0.0)
    post.add_argument("--posttrain-max-records", type=int, default=0)
    post.add_argument(
        "--posttrain-input-jsonl",
        dest="posttrain_input_jsonl",
        action="append",
        default=[],
        help="Explicit posttraining JSONL input, optionally algorithm=path. Takes priority over profile discovery.",
    )
    post.add_argument("--save-interval", type=int, default=None)
    post.add_argument("--heldout-max-records-per-file", dest="heldout_max_records_per_file", type=int, default=None)
    post.add_argument("--benchmark-max-records-per-file", dest="benchmark_max_records_per_file", type=int, default=None)
    post.add_argument("--heldout-sample-loss-timeout-seconds", dest="heldout_sample_loss_timeout_seconds", type=int, default=0)
    post.add_argument("--benchmark-sample-loss-timeout-seconds", dest="benchmark_sample_loss_timeout_seconds", type=int, default=0)
    post.add_argument("--benchmark-cycle", dest="benchmark_cycle", default="")
    post.add_argument("--benchmark-min-tasks", dest="benchmark_min_tasks", type=int, default=0)
    post.add_argument("--benchmark-predictions", dest="benchmark_predictions", default="")
    post.add_argument("--benchmark-prediction-backend", dest="benchmark_prediction_backend", default="")
    post.add_argument("--benchmark-prediction-model", dest="benchmark_prediction_model", default="")
    post.add_argument("--benchmark-prediction-base-url", dest="benchmark_prediction_base_url", default="")
    post.add_argument("--benchmark-prediction-api-key-env", dest="benchmark_prediction_api_key_env", default="")
    post.add_argument("--benchmark-prediction-checkpoint-runner", dest="benchmark_prediction_checkpoint_runner", default="")
    post.add_argument("--benchmark-prediction-timeout-seconds", dest="benchmark_prediction_timeout_seconds", type=int, default=0)
    post.add_argument("--benchmark-prediction-max-output-tokens", dest="benchmark_prediction_max_output_tokens", type=int, default=0)
    post.add_argument("--reportable-task-root", dest="reportable_task_roots", action="append", default=[])
    post.add_argument("--require-reportable-gate", dest="require_reportable_gate", action="store_true")
    post.add_argument("--rerun-heldout-evals", dest="rerun_heldout_evals", action="store_true")
    post.set_defaults(func=run_posttrain, live_posttraining=True)
    full = sub.add_parser("run-full")
    full.add_argument("--steps-per-stage", type=int, default=0)
    full.add_argument("--seq-len", type=int, default=0)
    full.add_argument("--batch-size", type=int, default=0)
    full.add_argument("--lr", type=float, default=0.0)
    full.add_argument("--preset", default="")
    full.add_argument("--device", default="")
    full.add_argument("--fake-quant", action="store_true")
    full.add_argument("--resume-checkpoint", default="")
    full.add_argument("--start-stage", default="", help="Start dense pretraining at this 1-based stage index or modality name")
    full.add_argument("--stage-order", default="", help="Comma-separated dense pretraining stage order override")
    full.add_argument("--context-ladder", dest="context_ladder", default="", help="Comma-separated real long-context curriculum ladder; defaults to the 8K..1M profile ladder")
    full.add_argument("--long-context-steps-per-rung", dest="long_context_steps_per_rung", type=int, default=0)
    full.add_argument("--resume-completed-stages", dest="resume_completed_stages", action="store_true", default=None, help="Skip stages whose expected checkpoint already exists")
    full.add_argument("--rerun-completed-stages", dest="resume_completed_stages", action="store_false", help="Retrain stages even when their expected checkpoint exists")
    full.add_argument("--save-interval", type=int, default=None)
    full.add_argument("--distributed", default="")
    full.add_argument("--nproc-per-node", type=int, default=0)
    full.add_argument("--precision", default="")
    full.add_argument("--init-dtype", default="")
    full.add_argument("--optimizer", default="")
    full.add_argument("--optimizer-in-backward", dest="optimizer_in_backward", action="store_true")
    full.add_argument("--optimizer-in-backward-update", dest="optimizer_in_backward_update", default="")
    full.add_argument("--optimizer-in-backward-grad-clip", dest="optimizer_in_backward_grad_clip", type=float, default=0.0)
    full.add_argument("--optimizer-in-backward-clip-mode", dest="optimizer_in_backward_clip_mode", default="")
    full.add_argument("--optimizer-in-backward-adafactor-chunk-rows", dest="optimizer_in_backward_adafactor_chunk_rows", type=int, default=0)
    full.add_argument("--optimizer-in-backward-adafactor-clip-threshold", dest="optimizer_in_backward_adafactor_clip_threshold", type=float, default=0.0)
    full.add_argument("--optimizer-in-backward-adafactor-decay-rate", dest="optimizer_in_backward_adafactor_decay_rate", type=float, default=0.0)
    full.add_argument("--optimizer-in-backward-adafactor-eps1", dest="optimizer_in_backward_adafactor_eps1", type=float, default=0.0)
    full.add_argument("--rank-device-map", dest="rank_device_map", default="")
    full.add_argument("--placement", default="")
    full.add_argument("--placement-devices", dest="placement_devices", default="")
    full.add_argument("--placement-layer-counts", dest="placement_layer_counts", default="")
    full.add_argument("--placement-head-device", dest="placement_head_device", type=int, default=-1)
    full.add_argument("--placement-schedule", dest="placement_schedule", default="")
    full.add_argument("--pipeline-stage-schedule", dest="pipeline_stage_schedule", default="")
    full.add_argument("--pipeline-microbatches", dest="pipeline_microbatches", type=int, default=0)
    full.add_argument("--pipeline-async-streams", dest="pipeline_async_streams", action="store_true", default=None)
    full.add_argument("--no-pipeline-async-streams", dest="pipeline_async_streams", action="store_false")
    full.add_argument("--activation-checkpointing", action="store_true")
    full.add_argument("--cpu-offload", action="store_true")
    full.add_argument("--fake-quant-chunk-rows", dest="fake_quant_chunk_rows", type=int, default=0)
    full.add_argument("--fake-quant-max-full-elements", dest="fake_quant_max_full_elements", type=int, default=0)
    full.add_argument("--allow-verifier-preset", action="store_true")
    full.add_argument("--distill-profile", default="")
    full.add_argument("--distill-limit", type=int, default=0)
    full.add_argument("--distill-steps", type=int, default=0)
    full.add_argument("--distill-lr", type=float, default=0.0)
    full.add_argument("--posttrain-steps", type=int, default=0)
    full.add_argument("--posttrain-lr", type=float, default=0.0)
    full.add_argument("--posttrain-max-records", type=int, default=0)
    full.add_argument(
        "--posttrain-input-jsonl",
        dest="posttrain_input_jsonl",
        action="append",
        default=[],
        help="Explicit posttraining JSONL input, optionally algorithm=path. Takes priority over profile discovery.",
    )
    full.add_argument("--finetune-steps", type=int, default=0)
    full.add_argument("--finetune-lr", type=float, default=0.0)
    full.add_argument("--benchmark-seq-len", type=int, default=0)
    full.add_argument("--heldout-max-records-per-file", dest="heldout_max_records_per_file", type=int, default=None)
    full.add_argument("--benchmark-max-records-per-file", dest="benchmark_max_records_per_file", type=int, default=None)
    full.add_argument("--heldout-sample-loss-timeout-seconds", dest="heldout_sample_loss_timeout_seconds", type=int, default=0)
    full.add_argument("--benchmark-sample-loss-timeout-seconds", dest="benchmark_sample_loss_timeout_seconds", type=int, default=0)
    full.add_argument("--benchmark-cycle", dest="benchmark_cycle", default="")
    full.add_argument("--benchmark-min-tasks", dest="benchmark_min_tasks", type=int, default=0)
    full.add_argument("--benchmark-predictions", dest="benchmark_predictions", default="")
    full.add_argument("--benchmark-prediction-backend", dest="benchmark_prediction_backend", default="")
    full.add_argument("--benchmark-prediction-model", dest="benchmark_prediction_model", default="")
    full.add_argument("--benchmark-prediction-base-url", dest="benchmark_prediction_base_url", default="")
    full.add_argument("--benchmark-prediction-api-key-env", dest="benchmark_prediction_api_key_env", default="")
    full.add_argument("--benchmark-prediction-checkpoint-runner", dest="benchmark_prediction_checkpoint_runner", default="")
    full.add_argument("--benchmark-prediction-timeout-seconds", dest="benchmark_prediction_timeout_seconds", type=int, default=0)
    full.add_argument("--benchmark-prediction-max-output-tokens", dest="benchmark_prediction_max_output_tokens", type=int, default=0)
    full.add_argument("--reportable-task-root", dest="reportable_task_roots", action="append", default=[])
    full.add_argument("--require-reportable-gate", dest="require_reportable_gate", action="store_true")
    full.add_argument("--rerun-heldout-evals", dest="rerun_heldout_evals", action="store_true")
    full.set_defaults(func=run_full, live_posttraining=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--summary", default="")
    summ.set_defaults(func=summarize)
    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("status") in {"ok", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

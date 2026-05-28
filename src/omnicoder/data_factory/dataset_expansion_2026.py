from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import Request, urlopen

from omnicoder.training import training_orchestration_2026


SCHEMA_VERSION = "2026-05-28"
DEFAULT_PROFILE = "profiles/dataset_curation_2026.json"
DEFAULT_TRAINING_PROFILE = "profiles/training_orchestration_2026.json"
DEFAULT_OUT_DIR = "weights/external_datasets_2026/latest"

FAMILY_TO_MODALITY = {
    "math_reasoning": "text",
    "coding_agentic": "code",
    "agentic_tool_reasoning": "tool",
    "terminal_browser_agents": "tool",
    "long_context": "long_context",
    "omnimodal_understanding": "text",
    "image_generation_editing": "image",
    "video_generation": "video",
    "speech_audio": "audio",
    "audio_music_speech": "audio",
    "music_generation": "music",
}

TRAINABLE_POLICIES = {"train", "internal_train", "distill_train", "train_ok"}
INTERNAL_ONLY_POLICIES = {"research_internal", "distill_seed", "internal_distill_seed", "reward_only"}
EVAL_ONLY_POLICIES = {"eval", "eval_only", "benchmark_holdout"}
UNSAFE_TRAIN_LICENSE_MARKERS = (
    "review",
    "pending",
    "unknown",
    "non_commercial",
    "no_derivatives",
    "holdout",
    "gated",
    "research",
    "blocked",
    "manual",
    "privacy",
    "copyright",
    "rights",
    "source terms",
    "terms of service",
    "tos",
    "opt-out",
    "opted-out",
)
MAX_INLINE_STRING_CHARS = 4096
MAX_INLINE_LIST_ITEMS = 64
MAX_INLINE_DICT_KEYS = 128
BINARY_FIELD_NAMES = {
    "array",
    "audio",
    "bytes",
    "data",
    "image",
    "samples",
    "video",
    "waveform",
}
MEDIA_REF_FIELD_NAMES = {
    "filename",
    "height",
    "mime_type",
    "path",
    "sampling_rate",
    "sha256",
    "size",
    "url",
    "width",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_json_value(value: Any, *, depth: int = 0, field_name: str = "") -> Any:
    normalized_field = re.sub(r"[^a-z0-9_]+", "_", str(field_name or "").strip().lower()).strip("_")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_INLINE_STRING_CHARS:
            return value
        digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
        return {"text_prefix": value[:MAX_INLINE_STRING_CHARS], "text_chars": len(value), "sha256": digest}
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {"binary_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"array_shape": [int(dim) for dim in tuple(shape)], "array_dtype": str(dtype) if dtype is not None else None}
    if depth >= 6:
        text = str(value)
        return text[:MAX_INLINE_STRING_CHARS]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, child in list(value.items())[:MAX_INLINE_DICT_KEYS]:
            key_text = str(key)
            child_field = re.sub(r"[^a-z0-9_]+", "_", key_text.strip().lower()).strip("_")
            if child_field in BINARY_FIELD_NAMES and child_field not in MEDIA_REF_FIELD_NAMES:
                compact[f"{key_text}_summary"] = compact_json_value(child, depth=depth + 1, field_name=child_field)
            else:
                compact[key_text] = compact_json_value(child, depth=depth + 1, field_name=child_field)
        if len(value) > MAX_INLINE_DICT_KEYS:
            compact["_truncated_keys"] = len(value) - MAX_INLINE_DICT_KEYS
        return compact
    if isinstance(value, list):
        if len(value) > MAX_INLINE_LIST_ITEMS:
            numeric = all(isinstance(item, (int, float)) for item in value[: min(len(value), 256)])
            if numeric or normalized_field in BINARY_FIELD_NAMES:
                return {
                    "list_items": len(value),
                    "sample": [compact_json_value(item, depth=depth + 1, field_name=normalized_field) for item in value[:8]],
                    "truncated_items": len(value) - 8,
                }
        items = [compact_json_value(item, depth=depth + 1, field_name=normalized_field) for item in value[:MAX_INLINE_LIST_ITEMS]]
        if len(value) > MAX_INLINE_LIST_ITEMS:
            items.append({"_truncated_items": len(value) - MAX_INLINE_LIST_ITEMS})
        return items
    text = str(value)
    return text[:MAX_INLINE_STRING_CHARS]


def stable_hash(value: Any) -> str:
    payload = json.dumps(compact_json_value(value), ensure_ascii=True, sort_keys=True, default=str)
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def profile_entries(profile: dict[str, Any]) -> list[dict[str, Any]]:
    registry = profile.get("external_dataset_registry_2026")
    if not isinstance(registry, dict):
        return []
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and entry.get("enabled", True)]


def requested_values(raw: Any) -> set[str]:
    values: set[str] = set()
    if raw is None:
        return values
    raw_items = raw if isinstance(raw, list) else [raw]
    for item in raw_items:
        if item is None:
            continue
        for part in str(item).split(","):
            value = part.strip()
            if value:
                values.add(value)
    return values


def select_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    include_waves = requested_values(getattr(args, "include_wave", None))
    include_families = requested_values(getattr(args, "include_family", None))
    include_names = requested_values(getattr(args, "include_name", None))
    selected: list[dict[str, Any]] = []
    for entry in entries:
        if include_waves and str(entry.get("registry_wave") or "") not in include_waves:
            continue
        if include_families and str(entry.get("family") or "") not in include_families:
            continue
        if include_names and str(entry.get("name") or "") not in include_names:
            continue
        selected.append(entry)
    return selected, {
        "total_enabled_entries": len(entries),
        "selected_entries": len(selected),
        "include_wave": sorted(include_waves),
        "include_family": sorted(include_families),
        "include_name": sorted(include_names),
        "filtered": bool(include_waves or include_families or include_names),
    }


def training_profile_path(profile: dict[str, Any], root: Path) -> Path:
    registry = profile.get("external_dataset_registry_2026")
    configured = None
    if isinstance(registry, dict):
        configured = registry.get("training_profile")
    configured = configured or DEFAULT_TRAINING_PROFILE
    return resolve_path(str(configured), root)


def first_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [first_string(item) for item in value[:8]]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in (
            "content",
            "text",
            "value",
            "natural_text",
            "markdown",
            "caption",
            "answer",
            "solution",
            "ground_truth",
            "label",
            "name",
            "prompt",
            "question",
            "instruction",
        ):
            if key in value:
                text = first_string(value[key])
                if text:
                    return text
        if any(key in value for key in ("objects", "bbox", "bounding_box", "text_sequence", "layout")):
            try:
                return json.dumps(value, ensure_ascii=True, sort_keys=True)[:4000]
            except Exception:
                return str(value)[:4000]
    return ""


def dotted_value(record: dict[str, Any], key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def field_text(record: dict[str, Any], fields: Any) -> str:
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        fields = []
    for field in fields:
        if not isinstance(field, str):
            continue
        value = dotted_value(record, field)
        text = first_string(value)
        if text:
            return text
    return ""


USER_ROLE_ALIASES = {"user", "human", "customer", "client", "question", "prompt", "instruction"}
ASSISTANT_ROLE_ALIASES = {"assistant", "gpt", "bot", "model", "agent", "answer", "response"}


def normalized_role(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def conversation_message_text(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text", "message", "utterance"):
        if key in message:
            text = first_string(message[key])
            if text:
                return text
    return first_string(message)


def conversation_role_text(value: Any, role_aliases: set[str], *, reverse: bool = False) -> str:
    if isinstance(value, list):
        iterable = reversed(value) if reverse else value
        for item in iterable:
            text = conversation_role_text(item, role_aliases, reverse=reverse)
            if text:
                return text
    if isinstance(value, dict):
        role = ""
        for role_key in ("role", "from", "speaker", "author", "source"):
            if role_key in value:
                role = normalized_role(value.get(role_key))
                break
        if role and role in role_aliases:
            return conversation_message_text(value)
        for child_key in ("messages", "conversation", "conversations", "turns", "dialogue"):
            child = value.get(child_key)
            if child not in (None, "", [], {}):
                text = conversation_role_text(child, role_aliases, reverse=reverse)
                if text:
                    return text
    return ""


def field_conversation_text(record: dict[str, Any], fields: Any, role_aliases: set[str], *, reverse: bool = False) -> str:
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        fields = []
    for field in fields:
        if not isinstance(field, str):
            continue
        candidates = [field]
        if "." in field:
            candidates.append(field.split(".", 1)[0])
        for candidate in candidates:
            value = dotted_value(record, candidate)
            if value is None:
                continue
            text = conversation_role_text(value, role_aliases, reverse=reverse)
            if text:
                return text
    return ""


def preference_pair_text(record: dict[str, Any]) -> str:
    pairs = {
        "response1": dotted_value(record, "response1") or dotted_value(record, "response_1"),
        "response2": dotted_value(record, "response2") or dotted_value(record, "response_2"),
        "chosen": dotted_value(record, "chosen"),
        "rejected": dotted_value(record, "rejected"),
        "overall_preference": dotted_value(record, "overall_preference") or dotted_value(record, "preference"),
        "feedback": dotted_value(record, "feedback"),
        "principle": dotted_value(record, "principle"),
    }
    compact = {key: first_string(value) for key, value in pairs.items() if first_string(value)}
    if any(key in compact for key in ("response1", "response2", "chosen", "rejected")) and any(
        key in compact for key in ("overall_preference", "feedback", "principle", "chosen")
    ):
        return json.dumps(compact, ensure_ascii=True, sort_keys=True)[:4000]
    return ""


def field_values(record: dict[str, Any], fields: Any) -> list[Any]:
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        return []
    values: list[Any] = []
    for field in fields:
        if not isinstance(field, str):
            continue
        value = dotted_value(record, field)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            values.extend(item for item in value if item not in (None, ""))
        else:
            values.append(value)
    return values


def mapped_structured_values(record: dict[str, Any], explicit_fields: Any, fallback_fields: list[str]) -> list[Any]:
    values = field_values(record, explicit_fields)
    if values:
        return values
    return field_values(record, fallback_fields)


def mapped_dict_list(record: dict[str, Any], explicit_fields: Any, fallback_fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in mapped_structured_values(record, explicit_fields, fallback_fields):
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, str) and value.strip():
            rows.append({"content": value.strip()})
    return rows


def fallback_prompt(entry: dict[str, Any], record: dict[str, Any]) -> str:
    family = str(entry.get("family") or "external_dataset")
    name = str(entry.get("name") or entry.get("hf_id") or family)
    if family == "math_reasoning":
        return "Solve the math problem with verifiable reasoning and preserve the final answer."
    if family == "coding_agentic":
        return "Solve or repair the coding task, preserving tests, patch intent, and terminal evidence."
    if family in {"agentic_tool_reasoning", "terminal_browser_agents"}:
        return "Complete the agentic tool-use trajectory with correct tool calls, observations, verification, and recovery behavior."
    if family == "image_generation_editing":
        return "Generate, edit, critique, or preserve the image according to the multimodal instruction."
    if family == "video_generation":
        return "Generate or critique the video with temporal consistency, motion, and prompt adherence."
    if family in {"speech_audio", "audio_music_speech"}:
        return "Transcribe, caption, generate, or critique the audio artifact with grounded reasoning."
    if family == "music_generation":
        return "Generate, caption, or critique music with style, tempo, structure, lyrics, and production notes."
    return f"Learn the high-quality 2026 dataset signal from {name}."


def fallback_target(entry: dict[str, Any], record: dict[str, Any]) -> str:
    field_map = entry.get("field_map") if isinstance(entry.get("field_map"), dict) else {}
    text = field_text(
        record,
        field_map.get("target")
        or [
            "solution",
            "answer",
            "completion",
            "response",
            "output",
            "target",
            "caption",
            "detailed_caption",
            "Brief_Caption",
            "Detailed_Caption",
            "main_caption",
            "alt_caption",
        ],
    )
    if text:
        return text
    strings: list[str] = []

    def visit(value: Any) -> None:
        if len(strings) >= 32:
            return
        if isinstance(value, str) and value.strip():
            strings.append(value.strip())
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value[:16]:
                visit(child)

    visit(record)
    return "\n".join(strings)[:4000]


def source_use_bucket(entry: dict[str, Any]) -> str:
    policy = str(entry.get("use_policy") or entry.get("license_tier") or "").lower()
    license_blob = f"{entry.get('license') or ''} {entry.get('license_tier') or ''}".lower()
    if policy in TRAINABLE_POLICIES:
        if any(
            str(entry.get(key) or "").lower().startswith(("blocked", "research_internal", "manual_review"))
            for key in ("source_review_status", "review_status", "status")
        ):
            return "blocked_until_review"
        if any(marker in license_blob for marker in UNSAFE_TRAIN_LICENSE_MARKERS):
            return "research_internal"
        return "train"
    if policy in EVAL_ONLY_POLICIES:
        return "eval_holdout"
    if policy in INTERNAL_ONLY_POLICIES:
        return "research_internal"
    if str(entry.get("license_tier") or "").lower() in {"eval_only", "research_only", "non_commercial", "non_commercial_no_derivatives"}:
        return "research_internal"
    return "blocked_until_review"


def training_bucket_for_record(entry: dict[str, Any], record: dict[str, Any]) -> str:
    bucket = source_use_bucket(entry)
    if bucket == "train" and bool(record.get("synthetic_seed")):
        return "research_internal"
    if bucket == "train":
        reasons = train_quarantine_reasons(entry, record)
        if reasons:
            return "research_internal"
    return bucket


def has_quality_score_for_record(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    for source in (record, entry):
        if not isinstance(source, dict):
            continue
        quality = source.get("quality")
        if isinstance(quality, dict):
            for key in ("score", "overall", "quality"):
                if quality.get(key) not in (None, ""):
                    return True
        if any(source.get(key) not in (None, "") for key in ("quality_score", "score", "reward", "human_score")):
            return True
    return False


def train_quarantine_reasons(entry: dict[str, Any], record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = contamination_status_for_record(entry, record)
    if status not in {"clean", "clear"}:
        reasons.append(f"contamination_{status or 'unknown'}")
    source_date = source_date_for_record(entry, record)
    if source_date == "unknown":
        reasons.append("missing_source_date")
    elif source_year_from_date(source_date) not in {"2025", "2026"}:
        reasons.append("source_date_outside_2025_2026")
    if not has_quality_score_for_record(entry, record):
        reasons.append("missing_quality_score")
    return reasons


def contamination_status_for_record(entry: dict[str, Any], record: dict[str, Any]) -> str:
    for source in (record, entry):
        contamination = source.get("contamination") if isinstance(source, dict) else None
        if isinstance(contamination, dict):
            status = str(contamination.get("status") or "").strip().lower()
            if status:
                return status
        for key in ("contamination_status", "protected_benchmark_scan", "benchmark_contamination_status"):
            value = source.get(key) if isinstance(source, dict) else None
            if value not in (None, "", [], {}):
                return str(value).strip().lower()
    return "unknown"


def normalized_source_date(value: Any) -> str:
    if value in (None, "", [], {}):
        return "unknown"
    text = str(value).strip()
    match = re.search(r"(20\d{2})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?", text)
    if not match:
        return text[:10] if text else "unknown"
    year, month, day = match.group(1), match.group(2), match.group(3)
    if month and day:
        return f"{year}-{int(month):02d}-{int(day):02d}"
    if month:
        return f"{year}-{int(month):02d}"
    return year


def source_date_for_record(entry: dict[str, Any], record: dict[str, Any]) -> str:
    for source in (record, entry):
        for key in ("source_date", "created_at", "updated_at", "published_at", "date", "timestamp", "source_year"):
            value = source.get(key) if isinstance(source, dict) else None
            normalized = normalized_source_date(value)
            if normalized != "unknown":
                return normalized
    return "unknown"


def source_year_from_date(source_date: Any) -> str:
    match = re.search(r"(20\d{2})", str(source_date or ""))
    return match.group(1) if match else "unknown"


def quality_score_for_record(entry: dict[str, Any], record: dict[str, Any]) -> float:
    candidates: list[Any] = []
    for source in (record, entry):
        if not isinstance(source, dict):
            continue
        quality = source.get("quality")
        if isinstance(quality, dict):
            candidates.extend([quality.get("score"), quality.get("overall"), quality.get("quality")])
        candidates.extend([source.get("quality_score"), source.get("score"), source.get("reward"), source.get("human_score")])
    for value in candidates:
        if value in (None, ""):
            continue
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return 0.0


def quality_score_bucket(score: Any) -> str:
    try:
        value = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return "unknown"
    if value >= 0.95:
        return "0.95-1.00"
    if value >= 0.90:
        return "0.90-0.94"
    if value >= 0.80:
        return "0.80-0.89"
    if value >= 0.70:
        return "0.70-0.79"
    if value >= 0.50:
        return "0.50-0.69"
    return "0.00-0.49"


def rejected_row_audit(entry: dict[str, Any], record: dict[str, Any], row_index: int, reason: str) -> dict[str, Any]:
    family = str(entry.get("family") or "external_dataset")
    declared_modality = str(entry.get("target_modality") or FAMILY_TO_MODALITY.get(family, "text"))
    source_date = source_date_for_record(entry, record)
    score = quality_score_for_record(entry, record)
    return {
        "dataset": entry.get("name"),
        "family": family,
        "modality": declared_modality,
        "training_bucket": training_bucket_for_record(entry, record),
        "license_tier": str(entry.get("license_tier") or "unknown"),
        "contamination_status": contamination_status_for_record(entry, record),
        "source_date": source_date,
        "source_year": source_year_from_date(source_date),
        "synthetic_seed_only": bool(record.get("synthetic_seed")),
        "quality_score": score,
        "quality_score_bucket": quality_score_bucket(score),
        "index": row_index,
        "reason": reason,
    }


def audit_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    source_date = str(row.get("source_date") or "unknown")
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    score = quality.get("score") if isinstance(quality, dict) else None
    return {
        "family": str(row.get("dataset_family") or "unknown"),
        "modality": str(row.get("modality") or "unknown"),
        "training_bucket": str(row.get("training_bucket") or "unknown"),
        "license_tier": str(row.get("license_tier") or "unknown"),
        "contamination_status": str(row.get("contamination_status") or (row.get("contamination") or {}).get("status") or "unknown"),
        "source_date": source_date,
        "source_year": source_year_from_date(source_date),
        "synthetic_seed_only": bool(row.get("synthetic_seed_only")),
        "quality_score": score,
        "quality_score_bucket": quality_score_bucket(score),
    }


def audit_dimension_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "family": "by_family",
        "modality": "by_modality",
        "training_bucket": "by_training_bucket",
        "license_tier": "by_license_tier",
        "contamination_status": "by_contamination_status",
        "source_year": "by_source_year",
        "source_date": "by_source_date",
        "quality_score_bucket": "by_quality_score_bucket",
    }
    summary: dict[str, Any] = {"total": len(records)}
    for key, out_key in dimensions.items():
        summary[out_key] = dict(sorted(Counter(str(record.get(key) or "unknown") for record in records).items()))
    synthetic_count = sum(1 for record in records if bool(record.get("synthetic_seed_only")))
    quality_scores: list[float] = []
    for record in records:
        try:
            quality_scores.append(float(record.get("quality_score")))
        except (TypeError, ValueError):
            continue
    summary["synthetic"] = {
        "synthetic": synthetic_count,
        "real": len(records) - synthetic_count,
        "synthetic_ratio": (synthetic_count / len(records)) if records else 0.0,
    }
    summary["quality_score"] = {
        "min": min(quality_scores) if quality_scores else None,
        "max": max(quality_scores) if quality_scores else None,
        "avg": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
    }
    return summary


def accepted_rejected_dimension_counts(
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    dimensions = {
        "family": "family",
        "modality": "modality",
        "training_bucket": "training_bucket",
        "license_tier": "license_tier",
        "contamination_status": "contamination_status",
        "source_year": "source_year",
        "source_date": "source_date",
        "quality_score_bucket": "quality_score_bucket",
    }
    combined: dict[str, dict[str, dict[str, int]]] = {}
    for out_key, record_key in dimensions.items():
        values = sorted(
            {
                str(record.get(record_key) or "unknown")
                for record in accepted_records + rejected_records
            }
        )
        combined[out_key] = {
            value: {
                "accepted": sum(1 for record in accepted_records if str(record.get(record_key) or "unknown") == value),
                "rejected": sum(1 for record in rejected_records if str(record.get(record_key) or "unknown") == value),
            }
            for value in values
        }
    return combined


def build_curation_quality_audit(
    rows_by_bucket: dict[str, list[dict[str, Any]]],
    rejected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted_records = [
        audit_record_from_row(row)
        for bucket_rows in rows_by_bucket.values()
        for row in bucket_rows
    ]
    rejected_records = [dict(row) for row in rejected_rows]
    accepted_summary = audit_dimension_summary(accepted_records)
    rejected_summary = audit_dimension_summary(rejected_records)
    return {
        "schema": "omnicoder.external_dataset_curation_quality_audit_2026.v1",
        "created_at": now_iso(),
        "summary": {
            "accepted": len(accepted_records),
            "rejected": len(rejected_records),
            "total_seen": len(accepted_records) + len(rejected_records),
        },
        "accepted": accepted_summary,
        "rejected": rejected_summary,
        "accepted_rejected_by": accepted_rejected_dimension_counts(accepted_records, rejected_records),
        "synthetic_ratio": {
            "accepted": accepted_summary["synthetic"]["synthetic_ratio"],
            "rejected": rejected_summary["synthetic"]["synthetic_ratio"],
        },
    }


def registry_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    registry = profile.get("external_dataset_registry_2026")
    return registry if isinstance(registry, dict) else {}


def requirement_floor(value: Any) -> int:
    if isinstance(value, dict):
        for key in ("min_real", "min_total", "min_records"):
            if key in value:
                return max(0, int(value.get(key) or 0))
        return 0
    return max(0, int(value or 0))


def requirement_bucket(value: Any) -> str:
    if isinstance(value, dict):
        bucket = str(value.get("bucket") or value.get("training_bucket") or "any").strip().lower()
        return bucket or "any"
    return "any"


def evaluate_registry_requirements(
    profile: dict[str, Any],
    rows_by_family: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    requirements = registry_cfg(profile).get("required_real_family_min_records")
    if not isinstance(requirements, dict):
        requirements = {}
    results: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for family, raw_requirement in sorted(requirements.items()):
        floor = requirement_floor(raw_requirement)
        if floor <= 0:
            continue
        bucket = requirement_bucket(raw_requirement)
        rows = [
            row
            for row in rows_by_family.get(str(family), [])
            if not bool(row.get("synthetic_seed_only"))
            and (bucket == "any" or str(row.get("training_bucket") or "") == bucket)
        ]
        result = {"real_records": len(rows), "min_real": floor, "bucket": bucket, "status": "passed" if len(rows) >= floor else "failed"}
        results[str(family)] = result
        if result["status"] != "passed":
            failures[str(family)] = result
    return {
        "schema": "omnicoder.external_dataset_requirements_2026.v1",
        "status": "passed" if not failures else "failed",
        "requirements": results,
        "failures": failures,
    }


def synthetic_seed_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = entry.get("distillation_prompts") or entry.get("prompt_seeds") or []
    if isinstance(seeds, dict):
        seeds = [seeds]
    if not isinstance(seeds, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, 1):
        if isinstance(seed, str):
            payload = {"instruction": seed}
        elif isinstance(seed, dict):
            payload = dict(seed)
        else:
            continue
        payload.setdefault("seed_index", index)
        payload.setdefault("synthetic_seed", True)
        rows.append(payload)
    return rows


def rows_from_local_jsonl(entry: dict[str, Any], root: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path = entry.get("local_jsonl") or entry.get("local_path")
    if not raw_path:
        return [], {"status": "skipped", "reason": "no_local_path"}
    path = resolve_path(str(raw_path), root)
    if not path.exists():
        return [], {"status": "skipped", "reason": "local_path_missing", "path": str(path)}
    rows = list(islice(iter_jsonl(path), limit if limit > 0 else None))
    return rows, {"status": "ok", "source": "local_jsonl", "path": str(path), "records": len(rows)}


def rows_from_text_payload(text: str, fmt: str) -> list[dict[str, Any]]:
    normalized = fmt.strip().lower().lstrip(".")
    if normalized in {"jsonl", "ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                payload = {"text": line, "parse_error": str(exc)}
            if isinstance(payload, dict):
                payload.setdefault("line_number", line_number)
                rows.append(payload)
        return rows
    if normalized == "json":
        payload = json.loads(text)
        def flatten(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                out: list[dict[str, Any]] = []
                for item in value:
                    out.extend(flatten(item))
                return out
            if isinstance(value, dict):
                if any(value.get(key) not in (None, "", [], {}) for key in ("question", "prompt", "instruction", "task", "problem")):
                    return [dict(value)]
                for key in ("data", "rows", "examples", "records", "items", "questions", "tasks"):
                    child = value.get(key)
                    if isinstance(child, (list, dict)):
                        rows = flatten(child)
                        if rows:
                            return rows
                out = []
                for child in value.values():
                    if isinstance(child, (list, dict)):
                        out.extend(flatten(child))
                return out
            return []

        flattened = flatten(payload)
        if flattened:
            return flattened
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("data", "rows", "examples", "records", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, dict)]
            return [payload]
        return []
    if normalized in {"csv", "tsv"}:
        delimiter = "\t" if normalized == "tsv" else ","
        return [dict(row) for row in csv.DictReader(io.StringIO(text), delimiter=delimiter)]
    return []


def rows_from_remote_files(entry: dict[str, Any], root: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = entry.get("remote_files")
    if isinstance(files, dict):
        files = [files]
    if not isinstance(files, list) or not files:
        return [], {"status": "skipped", "reason": "no_remote_files"}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    per_file: dict[str, int] = {}
    for raw_spec in files:
        if limit > 0 and len(rows) >= limit:
            break
        spec = raw_spec if isinstance(raw_spec, dict) else {"url": raw_spec}
        url = str(spec.get("url") or spec.get("path") or "").strip()
        if not url:
            continue
        fmt = str(spec.get("format") or Path(url.split("?", 1)[0]).suffix.lstrip(".") or "jsonl")
        try:
            if url.startswith(("http://", "https://")):
                request = Request(url, headers={"User-Agent": "omnicoder-dataset-expansion-2026"})
                with urlopen(request, timeout=float(spec.get("timeout") or 60)) as response:
                    text = response.read().decode(str(spec.get("encoding") or "utf-8"), errors="replace")
            else:
                text = resolve_path(url, root).read_text(encoding=str(spec.get("encoding") or "utf-8"), errors="replace")
            loaded = rows_from_text_payload(text, fmt)
        except Exception as exc:
            errors.append(f"{url}: {repr(exc)}")
            continue
        if limit > 0:
            loaded = loaded[: max(0, limit - len(rows))]
        for index, row in enumerate(loaded, 1):
            row.setdefault("_remote_file", url)
            row.setdefault("_remote_row", index)
            rows.append(row)
        per_file[url] = len(loaded)
    status = "ok" if rows else "failed" if errors else "empty"
    return rows, {
        "status": status,
        "source": "remote_files",
        "records": len(rows),
        "per_file": per_file,
        "errors": errors[:8],
    }


def rows_from_huggingface(entry: dict[str, Any], limit: int, streaming: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hf_id = entry.get("hf_id")
    if not hf_id:
        return [], {"status": "skipped", "reason": "no_hf_id"}
    try:
        import datasets as datasets_module  # type: ignore
    except Exception as exc:
        return [], {"status": "failed", "reason": "datasets_import_failed", "error": repr(exc)}
    load_dataset = datasets_module.load_dataset
    audio_cls = getattr(datasets_module, "Audio", None)
    video_cls = getattr(datasets_module, "Video", None)
    image_cls = getattr(datasets_module, "Image", None)
    splits = entry.get("splits")
    if isinstance(splits, str):
        splits = [splits]
    if not isinstance(splits, list) or not splits:
        splits = ["train"]
    config = entry.get("config")
    configs_raw = entry.get("configs")
    if config not in (None, "", [], {}):
        configs: list[Any] = [config]
    elif isinstance(configs_raw, str):
        configs = [item.strip() for item in configs_raw.split(",") if item.strip()]
    elif isinstance(configs_raw, list) and configs_raw:
        configs = [item for item in configs_raw if item not in (None, "", [], {})]
    else:
        configs = [None]
    revision = entry.get("revision")
    data_files = entry.get("data_files")
    verification_mode = entry.get("verification_mode")
    trust_remote_code = entry.get("trust_remote_code")
    token_env = entry.get("token_env")
    token_value = os.environ.get(str(token_env), "") if token_env else ""
    load_kwargs: dict[str, Any] = {"streaming": streaming}
    if revision:
        load_kwargs["revision"] = str(revision)
    if data_files:
        load_kwargs["data_files"] = data_files
    if verification_mode:
        load_kwargs["verification_mode"] = str(verification_mode)
    if trust_remote_code is not None:
        load_kwargs["trust_remote_code"] = bool(trust_remote_code)
    if token_value:
        load_kwargs["token"] = token_value
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    per_split: dict[str, int] = {}
    remaining = limit
    for cfg in configs:
        if limit > 0 and remaining <= 0:
            break
        for split in splits:
            if limit > 0 and remaining <= 0:
                break
            split_label = f"{cfg}:{split}" if cfg else str(split)
            try:
                if cfg:
                    dataset = load_dataset(str(hf_id), str(cfg), split=str(split), **load_kwargs)
                else:
                    dataset = load_dataset(str(hf_id), split=str(split), **load_kwargs)
            except Exception as exc:
                errors.append(f"{split_label}: {repr(exc)}")
                continue
            features = getattr(dataset, "features", None) or {}
            for column, feature in features.items():
                feature_name = feature.__class__.__name__
                try:
                    if feature_name == "Audio" and audio_cls is not None:
                        dataset = dataset.cast_column(column, audio_cls(decode=False))
                    elif feature_name == "Video" and video_cls is not None:
                        dataset = dataset.cast_column(column, video_cls(decode=False))
                    elif feature_name == "Image" and image_cls is not None:
                        dataset = dataset.cast_column(column, image_cls(decode=False))
                except Exception as exc:
                    errors.append(f"{split_label}:{column}: media decode disable failed: {repr(exc)}")
            take = remaining if limit > 0 else 0
            count = 0
            try:
                iterator = dataset if take <= 0 else islice(dataset, take)
                for raw in iterator:
                    if isinstance(raw, dict):
                        item = dict(raw)
                        if cfg:
                            item.setdefault("_hf_config", str(cfg))
                        item.setdefault("_hf_split", str(split))
                        rows.append(item)
                        count += 1
            except Exception as exc:
                errors.append(f"{split_label}: iteration failed: {repr(exc)}")
            per_split[split_label] = count
            if limit > 0:
                remaining -= count
    status = "ok" if rows else "failed" if errors else "empty"
    return rows, {
        "status": status,
        "source": "huggingface",
        "hf_id": str(hf_id),
        "config": config,
        "configs": [str(item) for item in configs if item],
        "revision": revision,
        "data_files": data_files,
        "verification_mode": verification_mode,
        "trust_remote_code": bool(trust_remote_code) if trust_remote_code is not None else None,
        "token_env": str(token_env) if token_env else None,
        "token_used": bool(token_value),
        "streaming": streaming,
        "records": len(rows),
        "per_split": per_split,
        "errors": errors[:8],
    }


def materialize_source_rows(entry: dict[str, Any], root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = int(entry.get("max_records") or args.max_records_per_dataset or 0)
    local_rows, local_status = rows_from_local_jsonl(entry, root, limit)
    if local_rows:
        return local_rows, local_status
    if args.download:
        remote_rows, remote_status = rows_from_remote_files(entry, root, limit)
        if remote_rows:
            return remote_rows, remote_status
        hf_rows, hf_status = rows_from_huggingface(entry, limit, streaming=not args.no_streaming)
        if hf_rows:
            return hf_rows, hf_status
        seeds = synthetic_seed_rows(entry)
        if seeds:
            if limit > 0:
                seeds = seeds[:limit]
            return seeds, {
                "status": "ok",
                "source": "distillation_prompts_after_hf_attempt",
                "records": len(seeds),
                "synthetic_seed_only": True,
                "huggingface_status": hf_status,
            }
        if hf_status.get("status") == "failed":
            return hf_rows, hf_status
    seeds = synthetic_seed_rows(entry)
    if limit > 0:
        seeds = seeds[:limit]
    if seeds:
        return seeds, {"status": "ok", "source": "distillation_prompts", "records": len(seeds), "synthetic_seed_only": True, "fallback_after": local_status}
    return [], {"status": "skipped", "reason": "no_rows", "local": local_status}


def record_to_training_row(entry: dict[str, Any], record: dict[str, Any], plan: dict[str, Any], row_index: int) -> dict[str, Any] | None:
    field_map = entry.get("field_map") if isinstance(entry.get("field_map"), dict) else {}
    prompt_fields = field_map.get("prompt") or ["instruction", "question", "prompt", "input", "problem", "title", "Brief_Caption"]
    target_fields = field_map.get("target") or [
        "solution",
        "answer",
        "completion",
        "response",
        "output",
        "target",
        "caption",
        "detailed_caption",
        "Brief_Caption",
        "Detailed_Caption",
        "main_caption",
        "alt_caption",
    ]
    prompt = field_conversation_text(record, prompt_fields, USER_ROLE_ALIASES) or field_text(record, prompt_fields)
    target = field_conversation_text(record, target_fields, ASSISTANT_ROLE_ALIASES, reverse=True) or preference_pair_text(record) or fallback_target(entry, record)
    if not prompt:
        prompt = fallback_prompt(entry, record)
    if not target and bool(entry.get("self_supervised_prompt_as_target")):
        target = field_text(record, field_map.get("prompt") or ["instruction", "question", "prompt", "problem"])
    if not target or len(target.strip()) < int(entry.get("min_target_chars") or 1):
        return None
    family = str(entry.get("family") or "external_dataset")
    declared_modality = str(entry.get("target_modality") or FAMILY_TO_MODALITY.get(family, "text"))
    stage_safe_modalities = set(plan.get("artifact_token_count", {}).keys()) | {"text", "code", "tool", "long_context"}
    modality = declared_modality if declared_modality in stage_safe_modalities else FAMILY_TO_MODALITY.get(family, "text")
    target_limit = training_orchestration_2026.modality_target_chars(plan, modality)
    prompt_limit = target_limit if modality != "long_context" else int(plan.get("long_context_prompt_chars") or min(8192, target_limit))
    if modality == "long_context" and len(prompt) > len(target):
        target = f"{prompt}\n\n{target}".strip() if target else prompt
        prompt = "Learn this external long-context span with retained anchors and retrieval-critical dependencies."
    source_uri = str(entry.get("url") or entry.get("hf_id") or entry.get("name") or family)
    raw_id = field_text(record, field_map.get("id") or ["id", "task_id", "problem_id", "instance_id", "ID", "uid"]) or f"row-{row_index}"
    contamination_status = contamination_status_for_record(entry, record)
    source_date = source_date_for_record(entry, record)
    quality_score = quality_score_for_record(entry, record)
    source_payload = {
        "source_id": stable_hash({"dataset": entry.get("name"), "raw_id": raw_id, "row_index": row_index}),
        "source_date": source_date,
        "quality": {"score": quality_score, "label": "accepted_external_2026"},
        "contamination": {
            "status": contamination_status,
            "note": "external registry row requires downstream protected benchmark scan"
            if contamination_status == "unknown"
            else "external registry row passed declared protected benchmark scan",
        },
        "dataset_name": entry.get("name"),
        "dataset_family": family,
        "hf_id": entry.get("hf_id"),
        "license": entry.get("license"),
        "license_tier": entry.get("license_tier"),
        "use_policy": entry.get("use_policy"),
        "skill_domain": entry.get("skill_domain") or family,
        "synthetic_seed_only": bool(record.get("synthetic_seed")),
        "raw_record": compact_json_value(record) if bool(entry.get("keep_raw_record", False)) else {"raw_id": raw_id, "row_hash": stable_hash(record)},
    }
    row = training_orchestration_2026.make_training_record(
        modality,
        prompt[:prompt_limit],
        target[:target_limit],
        source_uri,
        plan,
        source_payload=source_payload,
    )
    row["curriculum_axes"] = sorted(
        set(
            str(item)
            for item in (
                entry.get("curriculum_axes")
                if isinstance(entry.get("curriculum_axes"), list)
                else [family, entry.get("skill_domain") or family]
            )
            if item
        )
    )
    row["dataset_family"] = family
    row["dataset_name"] = str(entry.get("name") or entry.get("hf_id") or family)
    row["declared_target_modality"] = declared_modality
    row["license_tier"] = str(entry.get("license_tier") or "unknown")
    row["use_policy"] = str(entry.get("use_policy") or "blocked_until_review")
    row["training_bucket"] = training_bucket_for_record(entry, record)
    row["contamination_status"] = contamination_status
    row["quality_score"] = quality_score
    quarantine_reasons = train_quarantine_reasons(entry, record) if source_use_bucket(entry) == "train" else []
    if quarantine_reasons:
        row["train_quarantine_reasons"] = quarantine_reasons
    row["synthetic_seed_only"] = bool(record.get("synthetic_seed"))
    row["media_refs"] = [
        compact_json_value(value, field_name="media")
        for value in mapped_structured_values(
            record,
            field_map.get("media") or field_map.get("media_refs") or field_map.get("artifacts"),
            ["media", "media_refs", "artifacts", "image", "images", "video", "videos", "audio", "audios"],
        )
    ]
    row["tool_calls"] = [
        compact_json_value(value)
        for value in mapped_dict_list(record, field_map.get("tool_calls"), ["tool_calls", "actions", "steps.tool_calls"])
    ]
    row["tool_results"] = [
        compact_json_value(value)
        for value in mapped_dict_list(record, field_map.get("tool_results"), ["tool_results", "observations", "results"])
    ]
    trajectory = [
        compact_json_value(value)
        for value in mapped_structured_values(record, field_map.get("trajectory"), ["trajectory", "actions", "steps", "messages"])
    ]
    if trajectory:
        row["trajectory"] = trajectory
    verifier_labels = [
        compact_json_value(value)
        for value in mapped_structured_values(record, field_map.get("verifier_labels"), ["verifier_labels", "checks", "labels"])
    ]
    if verifier_labels:
        row["verifier_labels"] = verifier_labels
    reward = field_values(record, field_map.get("reward") or ["reward", "score", "preference_score", "human_score"])
    if reward:
        row["reward"] = compact_json_value(reward[0])
    if row["tool_calls"]:
        row["domains"] = sorted(set(row.get("domains", [])) | {"tool"})
    if row["synthetic_seed_only"] and source_use_bucket(entry) == "train":
        row["synthetic_train_blocked"] = True
        row["synthetic_train_block_reason"] = "distillation_prompt_seed_cannot_enter_train_bucket_without_real_hf_or_local_rows"
    return row


def write_bucket_partitioned_rows(jsonl_dir: Path, stem: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    paths = {
        "train": jsonl_dir / f"{stem}.jsonl",
        "all": jsonl_dir / f"{stem}_all.jsonl",
        "research_internal": jsonl_dir / f"{stem}_research_internal.jsonl",
        "eval_holdout": jsonl_dir / f"{stem}_eval_holdout.jsonl",
        "blocked_until_review": jsonl_dir / f"{stem}_blocked_until_review.jsonl",
    }
    write_jsonl(paths["train"], [row for row in rows if row.get("training_bucket") == "train"])
    write_jsonl(paths["all"], rows)
    write_jsonl(paths["research_internal"], [row for row in rows if row.get("training_bucket") == "research_internal"])
    write_jsonl(paths["eval_holdout"], [row for row in rows if row.get("training_bucket") == "eval_holdout"])
    write_jsonl(paths["blocked_until_review"], [row for row in rows if row.get("training_bucket") == "blocked_until_review"])
    return {key: str(path) for key, path in paths.items()}


def build_expansion(profile_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    profile = read_json(profile_path)
    training_profile = training_orchestration_2026.load_profile(training_profile_path(profile, root))
    plan = training_orchestration_2026.profile_cfg(training_profile)["training_plan"]
    entries, selection = select_entries(profile_entries(profile), args)
    jsonl_dir = out_dir / "jsonl"
    manifests_dir = out_dir / "manifests"
    cards_dir = out_dir / "dataset_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    acquisition: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in entries:
        raw_rows, status = materialize_source_rows(entry, root, args)
        family = str(entry.get("family") or "external_dataset")
        status.update(
            {
                "name": entry.get("name"),
                "family": family,
                "hf_id": entry.get("hf_id"),
                "url": entry.get("url"),
                "license": entry.get("license"),
                "license_tier": entry.get("license_tier"),
                "use_policy": entry.get("use_policy"),
                "bucket": source_use_bucket(entry),
                "synthetic_train_seed_policy": "research_internal_only" if source_use_bucket(entry) == "train" else None,
            }
        )
        acquisition.append(status)
        for index, raw in enumerate(raw_rows, 1):
            row = record_to_training_row(entry, raw, plan, index)
            if row is None:
                rejected.append(rejected_row_audit(entry, raw, index, "empty_or_short_target"))
                continue
            rows_by_bucket[str(row["training_bucket"])].append(row)
            rows_by_family[family].append(row)
            rows_by_modality[str(row["modality"])].append(row)
    family_paths = {family: write_bucket_partitioned_rows(jsonl_dir, family, rows) for family, rows in sorted(rows_by_family.items())}
    modality_paths = {modality: write_bucket_partitioned_rows(jsonl_dir, modality, rows) for modality, rows in sorted(rows_by_modality.items())}
    for bucket, rows in sorted(rows_by_bucket.items()):
        write_jsonl(jsonl_dir / f"{bucket}.jsonl", rows)
    train_rows = rows_by_bucket.get("train", [])
    research_rows = rows_by_bucket.get("research_internal", [])
    eval_rows = rows_by_bucket.get("eval_holdout", [])
    blocked_rows = rows_by_bucket.get("blocked_until_review", [])
    write_jsonl(jsonl_dir / "train_all_external.jsonl", train_rows)
    write_jsonl(jsonl_dir / "research_internal_all_external.jsonl", research_rows)
    write_jsonl(jsonl_dir / "eval_holdout_all_external.jsonl", eval_rows)
    write_jsonl(jsonl_dir / "blocked_until_review.jsonl", blocked_rows)
    write_jsonl(jsonl_dir / "rejected_external.jsonl", rejected)
    curation_quality_audit = build_curation_quality_audit(rows_by_bucket, rejected)
    audit_path = manifests_dir / "curation_quality_audit.json"
    write_json(audit_path, curation_quality_audit)
    if bool(selection["filtered"]) and not bool(getattr(args, "enforce_requirements", False)):
        requirement_report = {
            "schema": "omnicoder.external_dataset_requirements_2026.v1",
            "status": "skipped",
            "reason": "filtered_registry_delta_without_enforce_requirements",
            "requirements": {},
            "failures": {},
        }
    else:
        requirement_report = evaluate_registry_requirements(profile, rows_by_family)
    real_family_counts = {
        family: sum(1 for row in rows if not bool(row.get("synthetic_seed_only")))
        for family, rows in sorted(rows_by_family.items())
    }
    synthetic_seed_counts = {
        family: sum(1 for row in rows if bool(row.get("synthetic_seed_only")))
        for family, rows in sorted(rows_by_family.items())
        if any(bool(row.get("synthetic_seed_only")) for row in rows)
    }
    manifest = {
        "schema": "omnicoder.external_dataset_expansion_2026.v1",
        "version": SCHEMA_VERSION,
        "status": "passed" if requirement_report["status"] in {"passed", "skipped"} else "failed_requirements",
        "created_at": now_iso(),
        "profile": str(profile_path),
        "out_dir": str(out_dir),
        "download_requested": bool(args.download),
        "streaming": not bool(args.no_streaming),
        "selection": selection,
        "datasets": acquisition,
        "records": {
            "train": len(train_rows),
            "research_internal": len(research_rows),
            "eval_holdout": len(eval_rows),
            "blocked_until_review": len(blocked_rows),
            "rejected": len(rejected),
            "total_training_rows": sum(len(rows) for rows in rows_by_bucket.values()),
        },
        "families": {family: len(rows) for family, rows in sorted(rows_by_family.items())},
        "family_paths": family_paths,
        "real_families": real_family_counts,
        "synthetic_seed_families": synthetic_seed_counts,
        "modalities": {modality: len(rows) for modality, rows in sorted(rows_by_modality.items())},
        "modality_paths": modality_paths,
        "license_tiers": dict(sorted(Counter(str(row.get("license_tier") or "unknown") for rows in rows_by_bucket.values() for row in rows).items())),
        "curation_quality_audit": curation_quality_audit,
        "curation_quality_audit_path": str(audit_path),
        "requirement_report": requirement_report,
        "training_paths": {
            "train_all_external": str(jsonl_dir / "train_all_external.jsonl"),
            "research_internal_all_external": str(jsonl_dir / "research_internal_all_external.jsonl"),
            "eval_holdout_all_external": str(jsonl_dir / "eval_holdout_all_external.jsonl"),
        },
        "promotion_policy": "Only train bucket rows may be merged into release weights. research_internal rows are internal distillation/reward candidates. eval_holdout rows are benchmark/evaluation only. Filtered registry delta runs are sidecar inputs unless a full unfiltered requirement pass promotes latest.",
    }
    write_json(manifests_dir / "external_dataset_manifest.json", manifest)
    card_lines = [
        "# Omnicoder External Dataset Expansion 2026",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Train rows: {manifest['records']['train']}",
        f"- Research/internal rows: {manifest['records']['research_internal']}",
        f"- Eval holdout rows: {manifest['records']['eval_holdout']}",
        f"- Blocked rows: {manifest['records']['blocked_until_review']}",
        "",
        "## Families",
    ]
    for family, count in manifest["families"].items():
        card_lines.append(f"- {family}: {count}")
    card_lines.extend(["", "## License Tiers"])
    for tier, count in manifest["license_tiers"].items():
        card_lines.append(f"- {tier}: {count}")
    card_lines.extend(["", "## Policy", manifest["promotion_policy"], ""])
    cards_dir.mkdir(parents=True, exist_ok=True)
    (cards_dir / "external_dataset_card_2026.md").write_text("\n".join(card_lines), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize 2025-2026 external dataset expansion rows for Omnicoder training and distillation")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--download", action="store_true", help="Attempt Hugging Face streaming downloads when local JSONL rows are absent")
    parser.add_argument("--no-streaming", action="store_true", help="Use regular load_dataset instead of streaming")
    parser.add_argument("--max-records-per-dataset", type=int, default=0)
    parser.add_argument("--enforce-requirements", action="store_true", help="Return nonzero if registry required real-family minima are not met")
    parser.add_argument("--include-wave", action="append", default=[], help="Only materialize entries with this registry_wave. May be repeated or comma-separated.")
    parser.add_argument("--include-family", action="append", default=[], help="Only materialize entries from this family. May be repeated or comma-separated.")
    parser.add_argument("--include-name", action="append", default=[], help="Only materialize entries with this exact dataset name. May be repeated or comma-separated.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    args = parser.parse_args(argv)
    if args.command != "build":
        raise SystemExit(f"unknown command: {args.command}")
    manifest = build_expansion(resolve_path(args.profile, repo_root()), resolve_path(args.out_dir, repo_root()), args)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    if bool(args.enforce_requirements) and manifest.get("status") != "passed":
        return 3
    return 0 if manifest["records"]["total_training_rows"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

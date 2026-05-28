from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.dataset_integrity_2026 import audit_dataset_integrity


SCHEMA = "omnicoder.dataset_index_2026.v1"
REJECTED_PATH_MARKERS = (
    "/rejected/",
    "\\rejected\\",
    "/quarantine/",
    "\\quarantine\\",
    ".rejected.jsonl",
    "_rejected.jsonl",
    "rejected_external.jsonl",
    "dataset_integrity_rejected.jsonl",
    "policy_audit_rejected.jsonl",
    "blocked_until_review.jsonl",
    "_blocked_until_review.jsonl",
)
TRAIN_LEAK_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:benchmark(?:[_ -]?(?:id|task|suite|eval|materialized|holdout))?|public[_ -]?dev|reportable|local[_ -]?only|"
    r"answer[_ -]?key|protected[_ -]?eval|benchmark[_ -]?holdout|hella[_ -]?swag|hellaswag|"
    r"arc[_ -]?agi[23]?|arc-agi[23]?|swe[_ -]?bench|terminal[_ -]?bench|mmmu(?:[_ -]?pro)?|mmlu(?:[_ -]?pro)?|"
    r"human[_ -]?eval|humaneval|mbpp|gsm8k|gpqa(?:[_ -]?diamond)?|bfcl|berkeley[_ -]?function[_ -]?calling|"
    r"live[_ -]?code[_ -]?bench|livecodebench|tau[_ -]?bench|web[_ -]?arena|webarena|browsergym|osworld|"
    r"frontier[_ -]?math|frontiermath|fixture|smoke|canary)(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
NON_TRAINING_USE_POLICIES = {
    "benchmark_eval_only",
    "benchmark_holdout",
    "diagnostic_only",
    "dev",
    "eval",
    "eval_only",
    "evaluation",
    "holdout",
    "protected_eval",
    "public_dev_eval",
    "reportable_eval_only",
    "test",
    "valid",
    "validation",
    "validation_only",
}
NON_TRAINING_CONTAMINATION_CLASSES = {
    "benchmark_holdout",
    "eval_holdout",
    "contaminated",
    "dirty",
    "protected_eval",
    "public_dev_eval",
    "suspect",
}
CLEAN_CONTAMINATION_STATUSES = {"", "clean", "clear", "passed", "ok", "none", "unknown"}
REPORTABLE_SCOPE_MARKERS = ("official", "authorized", "reportable")
DIAGNOSTIC_SCOPE_MARKERS = ("canary", "diagnostic", "local", "public_dev", "validation_only")
ID_KEYS = ("record_id", "id", "uid", "uuid", "example_id", "sample_id", "row_id")
MODALITY_KEYS = ("modality", "target_modality", "input_modality", "output_modality", "declared_target_modality", "media_family")
TEXT_TARGET_KEYS = ("content", "text", "target", "response", "completion", "answer", "expected_answer", "output")
MEDIA_MODALITIES = {"image", "video", "audio", "music", "tts", "ocr"}
REMOTE_REF_PREFIXES = ("http://", "https://", "s3://", "hf://")
MEDIA_URL_RE = re.compile(
    r"(?i)^\s*(?:https?://|s3://|hf://)\S+\.(?:png|jpe?g|webp|gif|bmp|tiff|mp4|mov|mkv|webm|avi|wav|mp3|flac|ogg|m4a|aac|mid|midi)(?:[?#]\S*)?\s*$"
)
MEDIA_TOKEN_KEYS = (
    "artifact_tokens",
    "media_tokens",
    "audio_tokens",
    "video_tokens",
    "image_tokens",
    "speech_tokens",
    "tts_tokens",
    "music_tokens",
)
MEDIA_REF_KEYS = ("artifact_refs", "artifacts", "artifact_paths", "media_refs", "media_paths")


def _json_blob(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _first(row: dict[str, Any], *keys: str, default: str = "unknown") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        value = meta.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return default


def _text_value(value: Any, *, limit: int = 32768) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = ""
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    else:
        text = str(value)
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def _record_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    source_payload = row.get("source_payload") if isinstance(row.get("source_payload"), dict) else {}
    for container in (row, meta, lineage, source_payload):
        for key in ID_KEYS:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    return ""


def _modality(row: dict[str, Any]) -> str:
    modality = _first(row, "modality", "target_modality", default="unknown")
    if modality.strip().lower() not in {"", "unknown", "none", "null"}:
        return modality
    for container in (row.get("input_json"), row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in MODALITY_KEYS:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    return "unknown"


def _canonical_split(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "training": "train",
        "validation": "eval",
        "valid": "eval",
        "dev": "eval",
        "eval_holdout": "eval",
        "evaluation": "eval",
        "research": "research_internal",
        "internal_research": "research_internal",
        "blocked": "blocked_until_review",
        "block": "blocked_until_review",
        "manual_review": "blocked_until_review",
    }.get(normalized, normalized)


def _declared_split(row: dict[str, Any]) -> str:
    value = row.get("split")
    return "" if value in (None, "", [], {}) else str(value)


def _messages_prompt_target(messages: list[Any]) -> tuple[str, str]:
    prompt_parts: list[str] = []
    target = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = _text_value(message.get("content"))
        if not content:
            continue
        if role == "assistant":
            target = content
        else:
            prompt_parts.append(f"{role}: {content}")
    return "\n".join(prompt_parts), target


def _target_from_container(container: Any) -> str:
    if not isinstance(container, dict):
        return ""
    for key in TEXT_TARGET_KEYS:
        value = _text_value(container.get(key))
        if value:
            return value
    return ""


def _row_prompt_target(row: dict[str, Any]) -> tuple[str, str]:
    input_json = row.get("input_json") if isinstance(row.get("input_json"), dict) else {}
    target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
    output_json = row.get("output_json") if isinstance(row.get("output_json"), dict) else {}
    teacher_output = row.get("teacher_output")
    messages = row.get("messages")
    if isinstance(messages, list):
        prompt, target = _messages_prompt_target(messages)
        if prompt or target:
            return prompt, target
    messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else []
    if messages:
        prompt, target = _messages_prompt_target(messages)
        if target:
            return prompt, target
        if not prompt:
            prompt = _text_value(input_json)
    else:
        prompt = ""
    target = ""
    for container in (target_json, output_json, teacher_output):
        target = _target_from_container(container)
        if target:
            break
    if prompt and target:
        return prompt, target
    for key in ("prompt", "instruction", "question", "input", "query", "text"):
        prompt = _text_value(row.get(key))
        if prompt:
            break
    if not prompt:
        prompt = _text_value(input_json)
    for key in ("target", "response", "completion", "answer", "expected_answer", "output"):
        target = _text_value(row.get(key))
        if target:
            break
    if not target:
        for container in (target_json, output_json, teacher_output):
            target = _target_from_container(container)
            if target:
                break
    return prompt, target


def _infer_split(path: Path, row: dict[str, Any], expected_split: str = "") -> str:
    if row.get("split") not in (None, "", [], {}):
        return _canonical_split(str(row["split"]))
    if expected_split:
        return _canonical_split(expected_split)
    lower = path.name.lower()
    if "research_internal" in lower or "research" in lower:
        return "research_internal"
    if "blocked_until_review" in lower or "blocked" in lower:
        return "blocked_until_review"
    if "eval_holdout" in lower:
        return "eval"
    if "train" in lower:
        return "train"
    if "eval" in lower or "dev" in lower or "valid" in lower:
        return "eval"
    if "test" in lower:
        return "test"
    return "unknown"


def _canonical_training_bucket(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {
        "benchmark_eval_only",
        "diagnostic_only",
        "eval",
        "eval_only",
        "evaluation",
        "eval_holdout",
        "holdout",
        "protected_eval",
        "public_dev_eval",
        "reportable_eval_only",
        "test",
        "valid",
        "validation",
        "validation_only",
        "dev",
    }:
        return "eval_holdout"
    if normalized in {"research", "research_internal", "internal_research", "distill_seed", "internal_distill_seed", "reward_only"}:
        return "research_internal"
    if normalized in {"blocked", "block", "blocked_until_review", "manual_review", "rejected", "quarantine"}:
        return "blocked_until_review"
    return normalized or "unknown"


def _training_bucket(path: Path, row: dict[str, Any], split: str, use_policy: str) -> str:
    for key in ("training_bucket", "bucket", "use_bucket", "release_bucket"):
        value = row.get(key)
        if value not in (None, "", [], {}):
            return _canonical_training_bucket(str(value))
    for value in (use_policy, split):
        bucket = _canonical_training_bucket(value)
        if bucket != "unknown":
            return bucket
    lower = path.name.lower()
    if "research_internal" in lower or "research" in lower:
        return "research_internal"
    if "blocked_until_review" in lower or "blocked" in lower:
        return "blocked_until_review"
    if "eval_holdout" in lower or "eval" in lower or "valid" in lower or "dev" in lower or "test" in lower:
        return "eval_holdout"
    if "train" in lower:
        return "train"
    return "unknown"


def _coarse_status(bucket: str) -> str:
    return {
        "train": "train",
        "eval_holdout": "eval",
        "research_internal": "research",
        "blocked_until_review": "block",
    }.get(bucket, "unknown")


def _bool_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


def _is_benchmark_row(row: dict[str, Any], *, use_policy: str, contamination: str) -> bool:
    if row.get("benchmark_id") not in (None, "", [], {}):
        return True
    for key in ("benchmark_eval_only", "reportable_task", "reportable_score"):
        if _bool_flag(row.get(key)) is True:
            return True
    policy = str(use_policy or "").strip().lower()
    contamination_class = str(contamination or "").strip().lower()
    return policy in {"benchmark_eval_only", "benchmark_holdout", "diagnostic_only", "reportable_eval_only"} or contamination_class in {
        "benchmark_holdout",
        "protected_eval",
        "public_dev_eval",
    }


def _benchmark_index_bucket(row: dict[str, Any], *, training_bucket: str, use_policy: str, contamination: str) -> str:
    if not _is_benchmark_row(row, use_policy=use_policy, contamination=contamination):
        coarse = _coarse_status(training_bucket)
        return {
            "train": "train",
            "eval": "non_benchmark_eval",
            "research": "research_internal",
            "block": "blocked_until_review",
        }.get(coarse, "unknown")
    scope = str(row.get("reportability_scope") or "").strip().lower()
    source_bucket = str(row.get("source_bucket") or "").strip().lower()
    policy = str(use_policy or "").strip().lower()
    if _bool_flag(row.get("reportable")) is True or policy == "reportable_eval_only" or source_bucket == "reportable_eval":
        return "benchmark_reportable_eval"
    if any(marker in scope for marker in REPORTABLE_SCOPE_MARKERS):
        return "benchmark_reportable_eval"
    for key in ("diagnostic_only", "local_only", "public_dev", "canary"):
        if _bool_flag(row.get(key)) is True:
            return "benchmark_diagnostic_eval"
    if policy in {"validation_only", "diagnostic_only"} or source_bucket == "public_dev_validation":
        return "benchmark_diagnostic_eval"
    if any(marker in scope for marker in DIAGNOSTIC_SCOPE_MARKERS):
        return "benchmark_diagnostic_eval"
    return "benchmark_eval_unclassified"


def _nested_first(row: dict[str, Any], *paths: tuple[str, ...]) -> str:
    for path in paths:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _contamination_status(row: dict[str, Any], fallback: str) -> str:
    nested = _nested_first(
        row,
        ("contamination", "status"),
        ("contamination", "label"),
        ("curation", "contamination_status"),
        ("metadata", "contamination_status"),
        ("metadata", "contamination_class"),
    )
    return nested or fallback


def _non_training_train_bucket_issue(row: dict[str, Any], *, training_bucket: str, use_policy: str, contamination: str) -> str:
    bucket = _canonical_training_bucket(training_bucket)
    if bucket != "train":
        return ""
    policy = str(use_policy or "").strip().lower()
    contamination_class = str(contamination or "").strip().lower()
    if policy in NON_TRAINING_USE_POLICIES:
        return f"use_policy:{policy}"
    if contamination_class in NON_TRAINING_CONTAMINATION_CLASSES:
        return f"contamination_class:{contamination_class}"
    if row.get("reportable") is True:
        return "reportable_true"
    if row.get("benchmark_id") not in (None, "", [], {}):
        return "benchmark_id_present"
    for key in ("eval_only", "evaluation_only", "validation_only", "diagnostic_only", "local_only", "reportable_score", "reportable_task"):
        if _bool_flag(row.get(key)) is True:
            return f"{key}:true"
    if _bool_flag(row.get("training_allowed")) is False:
        return "training_allowed:false"
    return ""


def _train_metadata_issue(*, training_bucket: str, source: str, use_policy: str) -> str:
    bucket = _canonical_training_bucket(training_bucket)
    if bucket != "train":
        return ""
    if str(source or "").strip().lower() in {"", "unknown", "none", "null"}:
        return "missing_source_id"
    if str(use_policy or "").strip().lower() in {"", "unknown", "none", "null"}:
        return "missing_use_policy"
    return ""


def _blocked_train_row_issue(row: dict[str, Any], *, training_bucket: str, contamination: str) -> str:
    bucket = _canonical_training_bucket(training_bucket)
    if bucket != "train":
        return ""
    integrity = row.get("dataset_integrity_2026")
    if isinstance(integrity, dict) and integrity.get("accepted") is False:
        return "dataset_integrity_rejected"
    if row.get("synthetic_train_blocked") is True:
        return "synthetic_train_blocked"
    if row.get("train_quarantine_reasons") not in (None, "", [], {}):
        return "train_quarantine_reasons"
    contamination_status = str(contamination or "").strip().lower()
    if contamination_status and contamination_status not in CLEAN_CONTAMINATION_STATUSES:
        return f"contamination_status:{contamination_status}"
    return ""


def _target_token_count(row: dict[str, Any]) -> int:
    for key in ("target_token_ids", "labels", "assistant_token_ids"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value)
    target = row.get("target") or row.get("response") or row.get("completion") or row.get("answer") or ""
    if not isinstance(target, str) or not target:
        for container in (row.get("target_json"), row.get("output_json"), row.get("teacher_output")):
            if not isinstance(container, dict):
                continue
            for key in TEXT_TARGET_KEYS:
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    target = value
                    break
            if isinstance(target, str) and target:
                break
    if isinstance(target, str):
        return len(re.findall(r"\S+", target))
    return 0


def _is_remote_ref(ref: str) -> bool:
    return ref.strip().lower().startswith(REMOTE_REF_PREFIXES)


def _media_ref_is_payload(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        if text.startswith("data:") or text == "embedded_media_bytes":
            return True
        return not _is_remote_ref(text)
    if isinstance(value, dict):
        if value.get("bytes") or value.get("byte_size") or value.get("token_count"):
            return True
        for key in ("path", "source_path", "artifact_path", "file", "uri"):
            ref = _text_value(value.get(key), limit=2048)
            if ref and _media_ref_is_payload(ref):
                return True
        url = _text_value(value.get("url"), limit=2048)
        return bool(url and not _is_remote_ref(url))
    if isinstance(value, (list, tuple, set)):
        return any(_media_ref_is_payload(item) for item in value)
    return True


def _row_refs(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for container in (row, row.get("input_json"), row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in (*MEDIA_REF_KEYS, "artifact_metadata", "media_metadata"):
            value = container.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, dict):
                    ref = item.get("path") or item.get("source_path") or item.get("artifact_path") or item.get("file") or item.get("uri") or item.get("url")
                else:
                    ref = item
                ref_text = _text_value(ref, limit=2048)
                if ref_text:
                    refs.append(ref_text)
    return sorted(set(refs))[:64]


def _rejected_path_issue(path: Path) -> str:
    normalized = str(path).lower()
    name = path.name.lower()
    if any(marker in normalized for marker in REJECTED_PATH_MARKERS):
        return "rejected_or_quarantine_path"
    if "rejected" in name or "quarantine" in normalized:
        return "rejected_or_quarantine_path"
    return ""


def _nested_rejected_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("curation_policy_2026", "dataset_integrity_2026"):
        payload = row.get(key)
        if isinstance(payload, dict) and payload.get("accepted") is False:
            nested = payload.get("reasons") if isinstance(payload.get("reasons"), list) else []
            if nested:
                reasons.extend(f"{key}:{reason}" for reason in nested[:8])
            else:
                reasons.append(f"{key}:accepted_false")
    for key in ("rejected", "poisoned", "watermark_detected", "ai_watermark", "train_rejected"):
        value = row.get(key)
        if value is True or str(value).strip().lower() in {"1", "true", "yes", "rejected"}:
            reasons.append(f"{key}_flag")
    quarantine = row.get("train_quarantine_reasons")
    if isinstance(quarantine, list) and quarantine:
        reasons.extend(f"train_quarantine:{reason}" for reason in quarantine[:8])
    return sorted(set(reasons))


def _row_quality_value(row: dict[str, Any]) -> float | None:
    for key in ("quality_score", "score", "reward"):
        if row.get(key) not in (None, ""):
            try:
                return max(0.0, min(1.0, float(row[key])))
            except (TypeError, ValueError):
                return None
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    for key in ("score", "quality_score", "overall", "value"):
        if quality.get(key) not in (None, ""):
            try:
                return max(0.0, min(1.0, float(quality[key])))
            except (TypeError, ValueError):
                return None
    return None


def _quality_gate_issue(row: dict[str, Any], min_quality_score: float) -> str:
    if min_quality_score <= 0:
        return ""
    quality = _row_quality_value(row)
    if quality is None:
        return "missing_quality_score"
    if quality < min_quality_score:
        return f"quality_below_min:{quality:.6f}<min:{min_quality_score:.6f}"
    label = str((row.get("quality") or {}).get("label") if isinstance(row.get("quality"), dict) else row.get("quality_label") or "").lower()
    if label and any(marker in label for marker in ("reject", "rejected", "low_quality", "poor_quality", "quarantine")):
        return f"quality_label:{label}"
    return ""


def _has_media_token_payload(row: dict[str, Any]) -> bool:
    for container in (row, row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in MEDIA_TOKEN_KEYS:
            if container.get(key) not in (None, "", [], {}):
                return True
    value = row.get("artifact_token_ids")
    return isinstance(value, list) and bool(value)


def _has_media_payload(row: dict[str, Any]) -> bool:
    if _has_media_token_payload(row):
        return True
    for container in (row, row.get("target_json")):
        if not isinstance(container, dict):
            continue
        for key in MEDIA_REF_KEYS:
            if _media_ref_is_payload(container.get(key)):
                return True
    return False


def _prompt_target_leakage_issue(prompt: str, target: str, row: dict[str, Any]) -> str:
    if _has_media_payload(row):
        return ""
    norm_prompt = " ".join(prompt.split()).casefold()
    norm_target = " ".join(target.split()).casefold()
    if not norm_prompt or not norm_target:
        return ""
    if norm_prompt == norm_target:
        return "prompt_copy"
    if len(norm_prompt) >= 40 and norm_target.startswith(norm_prompt):
        return "target_includes_prompt"
    if len(norm_target) >= 40 and norm_prompt.startswith(norm_target):
        return "prompt_includes_target"
    prompt_tokens = re.findall(r"[A-Za-z0-9_]+", norm_prompt)
    target_tokens = re.findall(r"[A-Za-z0-9_]+", norm_target)
    if min(len(prompt_tokens), len(target_tokens)) < 8:
        return ""
    prompt_set = set(prompt_tokens)
    target_set = set(target_tokens)
    containment = len(prompt_set & target_set) / max(1, min(len(prompt_set), len(target_set)))
    length_ratio = min(len(prompt_tokens), len(target_tokens)) / max(1, max(len(prompt_tokens), len(target_tokens)))
    return "prompt_target_high_overlap" if containment >= 0.92 and length_ratio >= 0.75 else ""


def _url_only_media_issue(row: dict[str, Any], *, modality: str, target: str, refs: list[str]) -> str:
    if MEDIA_URL_RE.fullmatch(target or ""):
        return "target_url_only_media"
    if modality.strip().lower() not in MEDIA_MODALITIES:
        return ""
    if _has_media_token_payload(row) or _has_media_payload(row):
        return ""
    if refs and all(_is_remote_ref(ref) for ref in refs):
        return "media_url_only_ref"
    return ""


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                yield line_number, {"_bad_json": True, "_raw": line[:2000]}, line
                continue
            yield line_number, row, line


def build_index(
    paths: list[Path],
    *,
    expected_split: str = "",
    fail_on_train_leakage: bool = True,
    scan_dataset_integrity: bool = False,
    min_quality_score: float = 0.0,
) -> dict[str, Any]:
    by_modality: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_training_bucket: Counter[str] = Counter()
    by_train_eval_research_block: Counter[str] = Counter()
    by_source_training_bucket: Counter[tuple[str, str]] = Counter()
    by_index_bucket: Counter[str] = Counter()
    by_index_bucket_training_bucket: Counter[tuple[str, str]] = Counter()
    by_use_policy: Counter[str] = Counter()
    by_license: Counter[str] = Counter()
    by_contamination: Counter[str] = Counter()
    matrix: Counter[tuple[str, str, str, str]] = Counter()
    status_matrix: Counter[tuple[str, str, str, str, str]] = Counter()
    files: list[dict[str, Any]] = []
    duplicate_payloads = 0
    payload_hashes: set[str] = set()
    duplicate_ids = 0
    seen_ids: dict[str, dict[str, Any]] = {}
    duplicate_id_rows: list[dict[str, Any]] = []
    train_leak_rows: list[dict[str, Any]] = []
    missing_modality_rows: list[dict[str, Any]] = []
    empty_target_rows: list[dict[str, Any]] = []
    one_token_junk_rows: list[dict[str, Any]] = []
    prompt_target_leakage_rows: list[dict[str, Any]] = []
    url_only_media_rows: list[dict[str, Any]] = []
    split_mismatch_rows: list[dict[str, Any]] = []
    non_training_policy_train_rows: list[dict[str, Any]] = []
    train_metadata_rows: list[dict[str, Any]] = []
    blocked_train_rows: list[dict[str, Any]] = []
    benchmark_train_bucket_rows: list[dict[str, Any]] = []
    benchmark_unclassified_rows: list[dict[str, Any]] = []
    rejected_input_files: list[dict[str, Any]] = []
    nested_rejected_rows: list[dict[str, Any]] = []
    low_quality_rows: list[dict[str, Any]] = []
    dataset_integrity_rows: list[dict[str, Any]] = []
    bad_json = 0
    rows_with_target_tokens = 0
    rows_with_artifact_tokens = 0
    total_rows = 0

    for path in paths:
        path_issue = _rejected_path_issue(path)
        if path_issue:
            rejected_input_files.append({"path": str(path), "reason": path_issue})
        file_rows = 0
        file_sha = hashlib.sha256()
        for line_number, row, raw_line in iter_jsonl(path):
            total_rows += 1
            file_rows += 1
            file_sha.update(raw_line.encode("utf-8", errors="ignore"))
            if row.get("_bad_json"):
                bad_json += 1
                continue
            split = _infer_split(path, row, expected_split=expected_split)
            modality = _modality(row)
            source = _first(row, "source_id", "dataset_name", "source", "source_uri", default="unknown")
            use_policy = _first(row, "use_policy", "policy", default="unknown")
            license_id = _first(row, "license", "license_id", default="unknown")
            contamination = _contamination_status(row, _first(row, "contamination_status", "contamination_class", default="unknown"))
            training_bucket = _training_bucket(path, row, split, use_policy)
            coarse_status = _coarse_status(training_bucket)
            index_bucket = _benchmark_index_bucket(
                row,
                training_bucket=training_bucket,
                use_policy=use_policy,
                contamination=contamination,
            )
            by_modality[modality] += 1
            by_source[source] += 1
            by_split[split] += 1
            by_training_bucket[training_bucket] += 1
            by_train_eval_research_block[coarse_status] += 1
            by_source_training_bucket[(source, training_bucket)] += 1
            by_index_bucket[index_bucket] += 1
            by_index_bucket_training_bucket[(index_bucket, training_bucket)] += 1
            by_use_policy[use_policy] += 1
            by_license[license_id] += 1
            by_contamination[contamination] += 1
            matrix[(modality, source, split, use_policy)] += 1
            status_matrix[(modality, source, split, use_policy, training_bucket)] += 1
            if index_bucket.startswith("benchmark_") and _canonical_training_bucket(training_bucket) == "train":
                benchmark_train_bucket_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "split": split,
                        "training_bucket": training_bucket,
                        "index_bucket": index_bucket,
                    }
                )
            if index_bucket == "benchmark_eval_unclassified":
                benchmark_unclassified_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "split": split,
                        "training_bucket": training_bucket,
                        "use_policy": use_policy,
                    }
                )
            record_id = _record_id(row)
            if record_id:
                first_seen = seen_ids.get(record_id)
                if first_seen is None:
                    seen_ids[record_id] = {"path": str(path), "line": line_number}
                else:
                    duplicate_ids += 1
                    duplicate_id_rows.append(
                        {
                            "record_id": record_id,
                            "path": str(path),
                            "line": line_number,
                            "first_path": first_seen["path"],
                            "first_line": first_seen["line"],
                            "source_id": source,
                            "modality": modality,
                        }
                    )
            if modality.strip().lower() in {"", "unknown", "none", "null"}:
                missing_modality_rows.append({"path": str(path), "line": line_number, "source_id": source})
            declared_split = _declared_split(row)
            if expected_split and declared_split and _canonical_split(declared_split) != _canonical_split(expected_split):
                split_mismatch_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "declared_split": declared_split,
                        "expected_split": expected_split,
                    }
                )
            target_tokens = _target_token_count(row)
            prompt, target = _row_prompt_target(row)
            nested_reasons = _nested_rejected_reasons(row)
            if nested_reasons:
                nested_rejected_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "training_bucket": training_bucket,
                        "reasons": nested_reasons,
                    }
                )
            quality_issue = _quality_gate_issue(row, float(min_quality_score))
            if quality_issue:
                low_quality_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "training_bucket": training_bucket,
                        "reason": quality_issue,
                    }
                )
            if scan_dataset_integrity:
                integrity = audit_dataset_integrity(
                    row,
                    prompt=prompt,
                    target=target,
                    modality=modality,
                    source_path=path,
                    refs=_row_refs(row),
                    scan_artifacts=True,
                )
                if not integrity.get("accepted", True):
                    dataset_integrity_rows.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "source_id": source,
                            "modality": modality,
                            "training_bucket": training_bucket,
                            "reasons": integrity.get("reasons") or ["unknown"],
                        }
                    )
            if target_tokens > 0:
                rows_with_target_tokens += 1
            if not target and target_tokens <= 0 and not _has_media_payload(row):
                empty_target_rows.append({"path": str(path), "line": line_number, "source_id": source, "modality": modality, "training_bucket": training_bucket})
            if target_tokens <= 1 and not _has_media_payload(row):
                one_token_junk_rows.append({"path": str(path), "line": line_number, "source_id": source, "modality": modality, "training_bucket": training_bucket, "target_tokens": target_tokens})
            leakage = _prompt_target_leakage_issue(prompt, target, row)
            if leakage:
                prompt_target_leakage_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "training_bucket": training_bucket,
                        "reason": leakage,
                    }
                )
            url_only = _url_only_media_issue(row, modality=modality, target=target, refs=_row_refs(row))
            if url_only:
                url_only_media_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "training_bucket": training_bucket,
                        "reason": url_only,
                    }
                )
            policy_issue = _non_training_train_bucket_issue(
                row,
                training_bucket=training_bucket,
                use_policy=use_policy,
                contamination=contamination,
            )
            if policy_issue:
                non_training_policy_train_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "split": split,
                        "training_bucket": training_bucket,
                        "reason": policy_issue,
                    }
                )
            metadata_issue = _train_metadata_issue(training_bucket=training_bucket, source=source, use_policy=use_policy)
            if metadata_issue:
                train_metadata_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "split": split,
                        "training_bucket": training_bucket,
                        "reason": metadata_issue,
                    }
                )
            blocked_issue = _blocked_train_row_issue(row, training_bucket=training_bucket, contamination=contamination)
            if blocked_issue:
                blocked_train_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "modality": modality,
                        "split": split,
                        "training_bucket": training_bucket,
                        "reason": blocked_issue,
                    }
                )
            if isinstance(row.get("artifact_token_ids"), list) and row["artifact_token_ids"]:
                rows_with_artifact_tokens += 1
            payload_hash = _sha256_text(_json_blob(row))
            if payload_hash in payload_hashes:
                duplicate_payloads += 1
            payload_hashes.add(payload_hash)
            blob = _json_blob(row)[:100_000]
            if training_bucket == "train" and TRAIN_LEAK_RE.search(blob):
                train_leak_rows.append({"path": str(path), "line": line_number, "source_id": source, "modality": modality, "training_bucket": training_bucket})
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else 0,
                "rows": file_rows,
                "sha256": file_sha.hexdigest(),
            }
        )

    fail_reasons: list[str] = []
    if bad_json:
        fail_reasons.append("bad_json")
    if duplicate_payloads:
        fail_reasons.append("duplicate_payloads")
    if duplicate_ids:
        fail_reasons.append("duplicate_ids")
    if missing_modality_rows:
        fail_reasons.append("missing_modality_metadata")
    if empty_target_rows:
        fail_reasons.append("empty_target_rows")
    if one_token_junk_rows:
        fail_reasons.append("one_token_junk_rows")
    if prompt_target_leakage_rows:
        fail_reasons.append("prompt_target_leakage")
    if url_only_media_rows:
        fail_reasons.append("url_only_media_rows")
    if split_mismatch_rows:
        fail_reasons.append("split_mismatch")
    if fail_on_train_leakage and train_leak_rows:
        fail_reasons.append("train_eval_leakage_markers")
    if non_training_policy_train_rows:
        fail_reasons.append("non_training_policy_in_train_bucket")
    if train_metadata_rows:
        fail_reasons.append("train_rows_missing_source_or_policy")
    if blocked_train_rows:
        fail_reasons.append("blocked_or_rejected_train_rows")
    if benchmark_train_bucket_rows:
        fail_reasons.append("benchmark_rows_in_train_bucket")
    if rejected_input_files:
        fail_reasons.append("rejected_or_quarantine_input_files")
    if nested_rejected_rows:
        fail_reasons.append("nested_rejected_rows")
    if low_quality_rows:
        fail_reasons.append("low_quality_rows")
    if dataset_integrity_rows:
        fail_reasons.append("dataset_integrity_rejected_rows")
    return {
        "schema": SCHEMA,
        "status": "failed" if fail_reasons else "passed",
        "fail_reasons": fail_reasons,
        "rows": total_rows,
        "files": files,
        "counts": {
            "bad_json": bad_json,
            "duplicate_ids": duplicate_ids,
            "duplicate_payloads": duplicate_payloads,
            "empty_target_rows": len(empty_target_rows),
            "missing_modality_metadata": len(missing_modality_rows),
            "one_token_junk_rows": len(one_token_junk_rows),
            "prompt_target_leakage": len(prompt_target_leakage_rows),
            "split_mismatch": len(split_mismatch_rows),
            "train_eval_leakage_markers": len(train_leak_rows),
            "non_training_policy_in_train_bucket": len(non_training_policy_train_rows),
            "train_rows_missing_source_or_policy": len(train_metadata_rows),
            "blocked_or_rejected_train_rows": len(blocked_train_rows),
            "benchmark_rows": sum(rows for bucket, rows in by_index_bucket.items() if bucket.startswith("benchmark_")),
            "benchmark_reportable_eval_rows": by_index_bucket.get("benchmark_reportable_eval", 0),
            "benchmark_diagnostic_eval_rows": by_index_bucket.get("benchmark_diagnostic_eval", 0),
            "benchmark_eval_unclassified_rows": by_index_bucket.get("benchmark_eval_unclassified", 0),
            "benchmark_rows_in_train_bucket": len(benchmark_train_bucket_rows),
            "url_only_media_rows": len(url_only_media_rows),
            "rejected_or_quarantine_input_files": len(rejected_input_files),
            "nested_rejected_rows": len(nested_rejected_rows),
            "low_quality_rows": len(low_quality_rows),
            "dataset_integrity_rejected_rows": len(dataset_integrity_rows),
            "rows_with_target_tokens": rows_with_target_tokens,
            "rows_with_artifact_tokens": rows_with_artifact_tokens,
        },
        "policy": {
            "scan_dataset_integrity": bool(scan_dataset_integrity),
            "min_quality_score": float(min_quality_score),
            "reject_rejected_or_quarantine_input_files": True,
            "reject_nested_policy_or_integrity_rejections": True,
        },
        "by_modality": dict(sorted(by_modality.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_training_bucket": dict(sorted(by_training_bucket.items())),
        "by_train_eval_research_block": dict(sorted(by_train_eval_research_block.items())),
        "by_index_bucket": dict(sorted(by_index_bucket.items())),
        "by_source_training_bucket": [
            {"source_id": source, "training_bucket": bucket, "rows": rows}
            for (source, bucket), rows in sorted(by_source_training_bucket.items())
        ],
        "by_index_bucket_training_bucket": [
            {"index_bucket": index_bucket, "training_bucket": bucket, "rows": rows}
            for (index_bucket, bucket), rows in sorted(by_index_bucket_training_bucket.items())
        ],
        "by_use_policy": dict(sorted(by_use_policy.items())),
        "by_license": dict(sorted(by_license.items())),
        "by_contamination": dict(sorted(by_contamination.items())),
        "by_modality_source_split_policy": [
            {"modality": modality, "source_id": source, "split": split, "use_policy": policy, "rows": rows}
            for (modality, source, split, policy), rows in sorted(matrix.items())
        ],
        "by_modality_source_split_policy_status": [
            {"modality": modality, "source_id": source, "split": split, "use_policy": policy, "training_bucket": bucket, "rows": rows}
            for (modality, source, split, policy, bucket), rows in sorted(status_matrix.items())
        ],
        "duplicate_id_examples": duplicate_id_rows[:50],
        "empty_target_examples": empty_target_rows[:50],
        "missing_modality_examples": missing_modality_rows[:50],
        "one_token_junk_examples": one_token_junk_rows[:50],
        "prompt_target_leakage_examples": prompt_target_leakage_rows[:50],
        "split_mismatch_examples": split_mismatch_rows[:50],
        "train_leak_examples": train_leak_rows[:50],
        "non_training_policy_train_examples": non_training_policy_train_rows[:50],
        "train_metadata_examples": train_metadata_rows[:50],
        "blocked_train_examples": blocked_train_rows[:50],
        "benchmark_train_bucket_examples": benchmark_train_bucket_rows[:50],
        "benchmark_unclassified_examples": benchmark_unclassified_rows[:50],
        "url_only_media_examples": url_only_media_rows[:50],
        "rejected_input_file_examples": rejected_input_files[:50],
        "nested_rejected_examples": nested_rejected_rows[:50],
        "low_quality_examples": low_quality_rows[:50],
        "dataset_integrity_examples": dataset_integrity_rows[:50],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a source/modality/split index for final Omnicoder JSONL datasets.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-split", "--expected_split", dest="expected_split", default="")
    parser.add_argument("--allow-train-leakage-markers", "--allow_train_leakage_markers", dest="allow_train_leakage_markers", action="store_true")
    parser.add_argument("--skip-dataset-integrity-scan", "--skip_dataset_integrity_scan", dest="skip_dataset_integrity_scan", action="store_true")
    parser.add_argument("--min-quality-score", "--min_quality_score", dest="min_quality_score", type=float, default=0.55)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [Path(value) for value in args.input]
    payload = build_index(
        paths,
        expected_split=str(args.expected_split or ""),
        fail_on_train_leakage=not bool(args.allow_train_leakage_markers),
        scan_dataset_integrity=not bool(args.skip_dataset_integrity_scan),
        min_quality_score=float(args.min_quality_score),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "out": str(out), "fail_reasons": payload["fail_reasons"]}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

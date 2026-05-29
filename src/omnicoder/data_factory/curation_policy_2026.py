from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.dataset_integrity_2026 import audit_dataset_integrity


REFUSAL_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bas an ai(?: language)? model\b",
        r"\bi (?:can(?:not|'t|`t|\u2019t)|am not able to|am unable to|(?:'m|`m|\u2019m) unable to)\b",
        r"\bi (?:won(?:'t|`t|\u2019t)|will not|must refuse|have to refuse|refuse to)\b",
        r"\b(?:cannot|can(?:'t|`t|\u2019t)|can not|unable to) assist\b",
        r"\bnot able to (?:assist|help|comply|provide)\b",
        r"\b(?:against|violates?) (?:the )?(?:policy|safety policy|guidelines)\b",
        r"\b(?:policy|guidelines?) (?:prevents?|prohibits?|disallows?)\b",
        r"\b(?:refusal|refuse|refused|refusing)\b",
        r"\bsafety[_ -]?negative\b",
        r"\btool[_ -]?safety[_ -]?negative\b",
        r"\bkto[_ -]?or[_ -]?safety\b",
        r"\btrain[_ -]?refusal\b",
        r"\bunapproved[_ -]?destructive[_ -]?tool[_ -]?use\b",
        r"\bcredential[_ -]?and[_ -]?hidden[_ -]?eval[_ -]?safety\b",
    )
)
REFUSAL_SOURCE_MARKERS = (
    "tool_safety_negative",
    "safety_negative",
    "safety_negative_alignment",
    "kto_or_safety",
    "train_refusal",
    "unapproved_destructive_tool_use",
    "credential_and_hidden_eval_safety",
    "refusal_alignment",
    "refusal_policy",
)
SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|client[_-]?secret|cookie|credential|passphrase|password|passwd|private[_-]?key|refresh[_-]?token|secret|token)\b"
)
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|token|private[_-]?key)\s*[:=]\s*['\"]?[^'\"\s]{8,}"
)
PLACEHOLDER_RE = re.compile(
    r"(?i)\b(todo|tbd|fixme|lorem ipsum|placeholder|dummy output|mock output|fake answer|stubbed|synthetic placeholder)\b"
)
EVAL_HOLDOUT_MARKERS = (
    "eval_holdout",
    "public_dev_eval",
    "benchmark_leak",
    "contaminated",
    "blocked_until_review",
    "protected_eval",
)
EVAL_BENCHMARK_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:benchmark(?:[_ -]?(?:id|task|suite|eval|materialized|holdout))?|public[_ -]?dev|"
    r"answer[_ -]?key|protected[_ -]?eval|benchmark[_ -]?holdout|hella[_ -]?swag|hellaswag|"
    r"arc[_ -]?agi[23]?|arc-agi[23]?|swe[_ -]?bench|terminal[_ -]?bench|mmmu(?:[_ -]?pro)?|mmlu(?:[_ -]?pro)?|"
    r"human[_ -]?eval|humaneval|mbpp|gsm8k|gpqa(?:[_ -]?diamond)?|bfcl|berkeley[_ -]?function[_ -]?calling|"
    r"live[_ -]?code[_ -]?bench|livecodebench|tau[_ -]?bench|web[_ -]?arena|webarena|browsergym|osworld|"
    r"frontier[_ -]?math|frontiermath|fixture|smoke|canary)(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
KNOWN_MODALITIES = {"text", "code", "tool", "image", "video", "audio", "music", "tts", "long_context", "math", "ocr"}
MEDIA_MODALITIES = {"image", "video", "audio", "music", "tts", "ocr"}
GENERATION_MEDIA_MODALITIES = {"image", "video", "audio", "music", "tts"}
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
NEAR_DUP_MIN_NGRAMS = 12
NEAR_DUP_MAX_NGRAMS = 384
NEAR_DUP_MAX_INDEX_ROWS = 100_000
NEAR_DUP_MAX_POSTINGS_PER_GRAM = 256
REMOTE_REF_PREFIXES = ("http://", "https://", "s3://", "hf://")
SCALAR_TARGET_RE = re.compile(r"(?i)^[+-]?(?:\d+(?:\.\d+)?|[a-z]|true|false|yes|no|null|none|nan)$")
BARE_MEDIA_PATH_TARGET_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:
        file://|https?://|s3://|hf://|/|\.{1,2}[\\/]|[a-z]:[\\/]
    )?
    [\w .~:/\\%+-]+
    \.
    (?:png|jpe?g|webp|gif|bmp|tiff|mp4|mov|mkv|webm|avi|wav|mp3|flac|ogg|m4a|aac|mid|midi)
    \s*$
    """
)
TARGET_MEDIA_TOKEN_KEYS = (
    "artifact_tokens",
    "media_tokens",
    "audio_tokens",
    "video_tokens",
    "image_tokens",
    "speech_tokens",
    "tts_tokens",
    "music_tokens",
)
TARGET_MEDIA_PAYLOAD_KEYS = (
    "artifact_path",
    "artifact_uri",
    "artifact_tokens",
    "media_tokens",
    "audio_tokens",
    "video_tokens",
    "image_tokens",
    "speech_tokens",
    "tts_tokens",
    "music_tokens",
    "artifact_refs",
    "artifacts",
    "artifact_paths",
    "media_refs",
    "media_paths",
)


@dataclass(frozen=True)
class CurationPolicyConfig:
    reject_refusal_boilerplate: bool = True
    reject_eval_holdout: bool = True
    reject_secret_bearing: bool = True
    reject_placeholder_junk: bool = True
    min_quality_score: float = 0.0
    require_media_artifacts: bool = False
    media_modalities: frozenset[str] = frozenset(MEDIA_MODALITIES)
    max_control_char_ratio: float = 0.015
    max_repetition_ratio: float = 0.42
    min_target_chars: int = 2
    reject_dataset_integrity_issues: bool = True
    scan_integrity_artifacts: bool = True
    max_integrity_artifact_bytes: int = 64 * 1024 * 1024


def stable_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def stable_gram_hash(value: str) -> int:
    return zlib.crc32(value.encode("utf-8", errors="ignore")) & 0xFFFFFFFF


def text_value(value: Any, *, limit: int = 32768) -> str:
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


def record_metadata_text(row: dict[str, Any], source_path: Path | str | None = None) -> str:
    keys = (
        "source_id",
        "dataset_name",
        "dataset_family",
        "task_type",
        "domain",
        "training_kind",
        "curriculum_axes",
        "risk_labels",
        "safety_labels",
        "alignment_labels",
        "tags",
        "labels",
        "categories",
        "reward_axes",
        "verifier",
        "policy",
        "use_policy",
        "contamination_status",
        "decontamination_status",
        "split",
        "split_name",
        "contamination_class",
    )
    parts = [str(source_path or "")]
    parts.extend(text_value(row.get(key), limit=4096) for key in keys)
    return " ".join(part for part in parts if part).lower()


def _container_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "content",
            "text",
            "target",
            "response",
            "completion",
            "answer",
            "output",
            "value",
            "caption",
            "ocr_text",
        ):
            text = text_value(value.get(key))
            if text:
                return text
        messages = value.get("messages")
        if isinstance(messages, list):
            _, target = _messages_prompt_target(messages)
            if target:
                return target
            parts = [text_value(message.get("content")) for message in messages if isinstance(message, dict)]
            joined = "\n".join(part for part in parts if part)
            if joined:
                return joined
    return text_value(value)


def _messages_prompt_target(messages: list[Any]) -> tuple[str, str]:
    last_assistant = ""
    prompt_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "message").lower()
        content = text_value(message.get("content"))
        if not content:
            continue
        if role == "assistant":
            last_assistant = content
        else:
            prompt_parts.append(f"{role}: {content}")
    if prompt_parts and last_assistant:
        return "\n".join(prompt_parts), last_assistant
    return "\n".join(prompt_parts), ""


def message_prompt_target(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    input_json = row.get("input_json") if isinstance(row.get("input_json"), dict) else {}
    target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
    if not isinstance(messages, list):
        messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else []
    target_from_nested = ""
    for key in ("target_json", "output_json", "response_json", "teacher_output", "assistant_output"):
        target_from_nested = _container_text(row.get(key))
        if target_from_nested:
            break
    if not target_from_nested:
        target_from_nested = _container_text(target_json)
    if messages:
        prompt, assistant_target = _messages_prompt_target(messages)
        if prompt and assistant_target:
            return prompt, assistant_target
        if prompt and target_from_nested:
            return prompt, target_from_nested
    prompt = ""
    target = ""
    for key in ("prompt", "instruction", "question", "input", "query", "text"):
        prompt = text_value(row.get(key))
        if prompt:
            break
    if not prompt:
        prompt = _container_text(input_json)
    for key in ("target", "response", "completion", "answer", "expected_answer", "output"):
        target = text_value(row.get(key))
        if target:
            break
    if not target:
        target = target_from_nested
    return prompt, target


def artifact_refs(row: dict[str, Any], *, limit: int = 32) -> list[str]:
    refs: list[str] = []
    containers: list[dict[str, Any]] = [row]
    for nested_key in ("input_json", "target_json", "output_json"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in (
            "artifact_refs",
            "artifacts",
            "artifact_paths",
            "media_paths",
            "media_refs",
            "artifact_metadata",
            "media_metadata",
        ):
            value = container.get(key)
            values = [value] if isinstance(value, dict) else value
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        ref = text_value(
                            item.get("path")
                            or item.get("source_path")
                            or item.get("artifact_path")
                            or item.get("file")
                            or item.get("uri")
                            or item.get("url")
                            or item.get("sha256"),
                            limit=2048,
                        )
                        if not ref and item.get("bytes"):
                            ref = "embedded_media_bytes"
                    else:
                        ref = text_value(item, limit=2048)
                    if ref:
                        refs.append(ref)
            elif isinstance(value, str) and value.strip():
                refs.append(text_value(value, limit=2048))
    for key in ("artifact_path", "image_path", "video_path", "audio_path", "music_path"):
        value = text_value(row.get(key), limit=2048)
        if value:
            refs.append(value)
    return sorted(set(refs))[:limit]


def _has_nonempty_media_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_nonempty_media_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_nonempty_media_value(item) for item in value)
    return True


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
            ref = text_value(value.get(key), limit=2048)
            if ref and _media_ref_is_payload(ref):
                return True
        url = text_value(value.get("url"), limit=2048)
        return bool(url and not _is_remote_ref(url))
    if isinstance(value, (list, tuple, set)):
        return any(_media_ref_is_payload(item) for item in value)
    return True


def target_json_has_media_payload(row: dict[str, Any]) -> bool:
    target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
    if not isinstance(target_json, dict):
        return False
    for key in TARGET_MEDIA_TOKEN_KEYS:
        if key in target_json and _has_nonempty_media_value(target_json.get(key)):
            return True
    for key in TARGET_MEDIA_PAYLOAD_KEYS:
        if key in TARGET_MEDIA_TOKEN_KEYS:
            continue
        if key in target_json and _media_ref_is_payload(target_json.get(key)):
            return True
    return False


def scalar_or_degenerate_media_target(row: dict[str, Any], *, target: str, modality: str) -> bool:
    if modality not in GENERATION_MEDIA_MODALITIES:
        return False
    if target_json_has_media_payload(row):
        return False
    text = target.strip()
    if not text:
        return False
    if SCALAR_TARGET_RE.fullmatch(text):
        return True
    if BARE_MEDIA_PATH_TARGET_RE.fullmatch(text):
        return True
    return modality in {"audio", "music", "tts", "video"} and len(text) < 16


def target_copies_prompt(*, prompt: str, target: str, has_target_media_payload: bool = False) -> bool:
    if has_target_media_payload:
        return False
    prompt_norm = " ".join(prompt.split()).casefold()
    target_norm = " ".join(target.split()).casefold()
    return bool(prompt_norm and target_norm and prompt_norm == target_norm)


def normalize_modality(value: Any) -> str:
    text = text_value(value).lower().replace("-", "_").replace(" ", "_")
    if text in KNOWN_MODALITIES:
        return text
    if "long_context" in text or "longctx" in text or "1m_context" in text:
        return "long_context"
    if "ocr" in text or "document_vision" in text:
        return "ocr"
    if any(marker in text for marker in ("math", "gsm", "aime", "olympiad", "proof")):
        return "math"
    if any(marker in text for marker in ("vision", "image", "picture", "imgedit", "qwen_image")):
        return "image"
    if any(marker in text for marker in ("video", "movie", "ltx", "image_to_video", "text_to_video")):
        return "video"
    if any(marker in text for marker in ("music", "song", "melody", "ace_step", "acestep", "midi")):
        return "music"
    if any(marker in text for marker in ("tts", "speech_synthesis", "text_to_speech", "voice_clone")):
        return "tts"
    if any(marker in text for marker in ("audio", "speech", "tts", "asr", "voice", "sound")):
        return "audio"
    if any(marker in text for marker in ("code", "coding", "swe", "terminal_bench", "livecodebench", "program")):
        return "code"
    if any(marker in text for marker in ("tool", "agent", "trace", "browser", "shell", "terminal", "codex", "claude", "hermes")):
        return "tool"
    return "text" if text else ""


def refusal_or_exclusion_hit(row: dict[str, Any], prompt: str, target: str, source_path: Path | str | None = None) -> str:
    metadata = record_metadata_text(row, source_path)
    if any(marker in metadata for marker in REFUSAL_SOURCE_MARKERS):
        return "refusal_source_marker"
    combined = f"{metadata}\n{prompt}\n{target}"
    for pattern in REFUSAL_BOILERPLATE_PATTERNS:
        if pattern.search(combined):
            return f"refusal_pattern:{pattern.pattern}"
    return ""


def eval_holdout_hit(row: dict[str, Any], source_path: Path | str | None = None) -> str:
    metadata = record_metadata_text(row, source_path)
    for marker in EVAL_HOLDOUT_MARKERS:
        if marker in metadata:
            return marker
    combined = f"{metadata}\n{text_value(row.get('benchmark_id'))}\n{text_value(row.get('reportability_scope'))}"
    if EVAL_BENCHMARK_TEXT_RE.search(combined):
        return "benchmark_or_eval_marker"
    return ""


def near_dedupe_text(row: dict[str, Any], *, prompt: str, target: str, modality: str) -> str:
    if modality in MEDIA_MODALITIES and target_json_has_media_payload(row):
        return ""
    return f"{prompt}\n{target}".strip()


def ngram_fingerprint(text: str, *, ngram: int = 5) -> set[int]:
    tokens = WORD_RE.findall(text.casefold())
    if len(tokens) < max(1, ngram):
        return set()
    fingerprint = {
        stable_gram_hash(" ".join(tokens[index : index + ngram]))
        for index in range(0, len(tokens) - ngram + 1)
    }
    if len(fingerprint) <= NEAR_DUP_MAX_NGRAMS:
        return fingerprint
    return set(sorted(fingerprint)[:NEAR_DUP_MAX_NGRAMS])


def jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def artifact_quality(refs: list[str], modality: str, *, require_media_artifacts: bool) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if modality not in MEDIA_MODALITIES:
        return 1.0, reasons
    if not refs:
        if require_media_artifacts:
            reasons.append("missing_media_artifact_ref")
            return 0.0, reasons
        return 0.45, ["media_artifact_ref_absent"]
    existing = 0
    checked = 0
    remote_only = 0
    for ref in refs[:8]:
        if _is_remote_ref(ref):
            remote_only += 1
            checked += 1
            continue
        path = Path(ref)
        if path.is_absolute():
            checked += 1
            try:
                if path.exists() and path.stat().st_size > 0:
                    existing += 1
            except OSError:
                pass
    if checked and remote_only == checked:
        reasons.append("media_artifact_url_only")
        return 0.0, reasons
    if checked and existing == 0:
        reasons.append("media_artifact_ref_not_found")
        return (0.0 if require_media_artifacts else 0.25), reasons
    return min(1.0, 0.55 + 0.45 * (existing / max(1, checked or existing))), reasons


def quality_audit(
    row: dict[str, Any],
    *,
    prompt: str,
    target: str,
    modality: str,
    source_path: Path | str | None = None,
    refs: list[str] | None = None,
    existing_quality: float | None = None,
    config: CurationPolicyConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CurationPolicyConfig()
    prompt = text_value(prompt, limit=65536)
    target = text_value(target, limit=65536)
    has_target_media_payload = modality in MEDIA_MODALITIES and target_json_has_media_payload(row)
    text = f"{prompt}\n{target}".strip()
    tokens = WORD_RE.findall(text.lower())
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(1, token_count)
    length_score = min(1.0, math.log1p(len(text)) / math.log(8000)) if text else 0.0
    diversity_score = min(1.0, unique_ratio / 0.58)
    structure_score = 0.0
    if "\n" in text:
        structure_score += 0.08
    if "```" in text or modality == "code":
        structure_score += 0.18
    if modality == "tool" and any(marker in text.lower() for marker in ("tool", "function", "shell", "json", "command")):
        structure_score += 0.18
    if modality in MEDIA_MODALITIES and any(marker in text.lower() for marker in ("caption", "prompt", "artifact", "image", "video", "audio", "music", "ocr")):
        structure_score += 0.16
    if source_path or row.get("source_id") or row.get("dataset_name"):
        structure_score += 0.08
    control_chars = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
    control_ratio = control_chars / max(1, len(text))
    repetition_penalty = cfg.max_repetition_ratio if token_count > 80 and unique_ratio < 0.22 else 0.0
    control_penalty = 0.35 if control_ratio > cfg.max_control_char_ratio else 0.0
    placeholder_penalty = 0.35 if PLACEHOLDER_RE.search(text) else 0.0
    secret_penalty = 0.55 if SECRET_VALUE_RE.search(text) else 0.0
    artifact_score, artifact_reasons = (
        (1.0, [])
        if has_target_media_payload
        else artifact_quality(refs or [], modality, require_media_artifacts=cfg.require_media_artifacts)
    )
    source_quality = 0.0 if existing_quality is None else max(0.0, min(1.0, float(existing_quality)))
    score = (
        0.18
        + 0.21 * length_score
        + 0.17 * diversity_score
        + 0.16 * min(1.0, structure_score)
        + 0.16 * source_quality
        + 0.12 * artifact_score
        - repetition_penalty
        - control_penalty
        - placeholder_penalty
        - secret_penalty
    )
    score = max(0.0, min(1.0, score))
    reasons = list(artifact_reasons)
    if not prompt:
        reasons.append("missing_prompt")
    if not target:
        reasons.append("missing_target")
    if len(target.strip()) < cfg.min_target_chars and not has_target_media_payload:
        reasons.append("target_too_short")
    if target_copies_prompt(prompt=prompt, target=target, has_target_media_payload=has_target_media_payload):
        reasons.append("target_copies_prompt")
    if scalar_or_degenerate_media_target(row, target=target, modality=modality):
        reasons.append("media_target_too_short_or_scalar")
    if secret_penalty:
        reasons.append("secret_marker")
    if placeholder_penalty:
        reasons.append("placeholder_or_stub")
    if repetition_penalty:
        reasons.append("low_diversity_repetition")
    if control_penalty:
        reasons.append("control_character_noise")
    if score < cfg.min_quality_score:
        reasons.append("below_min_quality")
    hard_reject_reasons = {
        "missing_prompt",
        "missing_target",
        "secret_marker",
        "below_min_quality",
        "target_too_short",
        "target_copies_prompt",
        "media_artifact_ref_not_found",
        "missing_media_artifact_ref",
        "media_artifact_url_only",
        "media_target_too_short_or_scalar",
    }
    if cfg.reject_placeholder_junk:
        hard_reject_reasons.update(
            {
                "placeholder_or_stub",
                "low_diversity_repetition",
                "control_character_noise",
            }
        )
    label = "reject" if reasons and any(r in hard_reject_reasons for r in reasons) else "candidate"
    if label != "reject" and score >= 0.78:
        label = "high"
    return {
        "score": round(score, 6),
        "label": label,
        "reasons": reasons,
        "dimensions": {
            "length_score": round(length_score, 6),
            "diversity_score": round(diversity_score, 6),
            "structure_score": round(min(1.0, structure_score), 6),
            "artifact_score": round(artifact_score, 6),
            "source_quality": round(source_quality, 6),
            "token_count": token_count,
            "unique_ratio": round(unique_ratio, 6),
            "control_ratio": round(control_ratio, 6),
        },
    }


def audit_training_record(
    row: dict[str, Any],
    *,
    prompt: str,
    target: str,
    modality: str,
    source_path: Path | str | None = None,
    refs: list[str] | None = None,
    existing_quality: float | None = None,
    config: CurationPolicyConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CurationPolicyConfig()
    reasons: list[str] = []
    refusal_hit = refusal_or_exclusion_hit(row, prompt, target, source_path) if cfg.reject_refusal_boilerplate else ""
    if refusal_hit:
        reasons.append(refusal_hit)
    eval_hit = eval_holdout_hit(row, source_path) if cfg.reject_eval_holdout else ""
    if eval_hit:
        reasons.append(f"eval_holdout:{eval_hit}")
    integrity = (
        audit_dataset_integrity(
            row,
            prompt=prompt,
            target=target,
            modality=modality,
            source_path=source_path,
            refs=refs or [],
            scan_artifacts=cfg.scan_integrity_artifacts,
            max_artifact_bytes=cfg.max_integrity_artifact_bytes,
        )
        if cfg.reject_dataset_integrity_issues
        else {
            "schema": "omnicoder.dataset_integrity_2026.v1",
            "accepted": True,
            "reasons": [],
            "issues": [],
            "signals": {"disabled": True},
        }
    )
    if cfg.reject_dataset_integrity_issues and not integrity.get("accepted", True):
        reasons.extend(f"dataset_integrity:{reason}" for reason in integrity.get("reasons") or ["unknown"])
    quality = quality_audit(
        row,
        prompt=prompt,
        target=target,
        modality=modality,
        source_path=source_path,
        refs=refs,
        existing_quality=existing_quality,
        config=cfg,
    )
    if quality["label"] == "reject":
        reasons.extend(str(reason) for reason in quality.get("reasons") or [])
    accepted = not reasons
    return {
        "accepted": accepted,
        "reasons": sorted(set(reasons)),
        "quality": quality,
        "dataset_integrity_2026": integrity,
        "policy": {
            "reject_refusal_boilerplate": cfg.reject_refusal_boilerplate,
            "reject_eval_holdout": cfg.reject_eval_holdout,
            "reject_dataset_integrity_issues": cfg.reject_dataset_integrity_issues,
            "scan_integrity_artifacts": cfg.scan_integrity_artifacts,
            "min_quality_score": cfg.min_quality_score,
            "require_media_artifacts": cfg.require_media_artifacts,
        },
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                row = {"text": line.rstrip("\n"), "parse_error": str(exc)}
            if isinstance(row, dict):
                row.setdefault("line_number", line_number)
                yield row


def run_agent(args: argparse.Namespace) -> dict[str, Any]:
    cfg = CurationPolicyConfig(
        reject_refusal_boilerplate=not bool(args.allow_refusal_boilerplate),
        reject_eval_holdout=not bool(args.allow_eval_holdout),
        reject_secret_bearing=True,
        reject_placeholder_junk=True,
        min_quality_score=float(args.min_quality),
        require_media_artifacts=bool(args.require_media_artifacts),
        reject_dataset_integrity_issues=not bool(args.allow_dataset_integrity_issues),
        scan_integrity_artifacts=not bool(args.skip_integrity_artifact_scan),
        max_integrity_artifact_bytes=int(args.max_integrity_artifact_bytes),
    )
    out_path = Path(args.out)
    rejected_path = Path(args.rejected)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(out_path.suffix + ".manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    accepted = 0
    rejected = 0
    seen_hashes: set[str] = set()
    near_dedupe_enabled = bool(getattr(args, "near_dedupe", False))
    near_dedupe_threshold = float(getattr(args, "near_dedupe_threshold", 0.92))
    near_dedupe_ngram = int(getattr(args, "near_dedupe_ngram", 5))
    near_duplicate_fingerprints: list[dict[str, Any]] = []
    near_duplicate_inverted: dict[int, list[int]] = {}
    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle, rejected_path.open("w", encoding="utf-8", newline="\n") as rej_handle:
        stop = False
        for input_item in args.input:
            if stop:
                break
            source_path = Path(input_item)
            for row in iter_jsonl(source_path):
                prompt, target = message_prompt_target(row)
                modality = normalize_modality(args.modality or row.get("modality") or row.get("task_type") or row.get("source_id") or source_path.name) or "text"
                refs = artifact_refs(row)
                existing_quality = None
                for key in ("quality_score", "score", "reward"):
                    if row.get(key) is not None:
                        try:
                            existing_quality = float(row[key])
                            break
                        except Exception:
                            pass
                if existing_quality is None and isinstance(row.get("quality"), dict):
                    for key in ("score", "quality_score", "value"):
                        if row["quality"].get(key) is not None:
                            try:
                                existing_quality = float(row["quality"][key])
                                break
                            except Exception:
                                pass
                dedupe_key = stable_hash({"modality": modality, "prompt": prompt[:2048], "target": target[:2048]})
                audit = audit_training_record(
                    row,
                    prompt=prompt,
                    target=target,
                    modality=modality,
                    source_path=source_path,
                    refs=refs,
                    existing_quality=existing_quality,
                    config=cfg,
                )
                if args.dedupe and dedupe_key in seen_hashes:
                    audit["accepted"] = False
                    audit["reasons"] = sorted(set(list(audit.get("reasons") or []) + ["duplicate"]))
                seen_hashes.add(dedupe_key)
                if near_dedupe_enabled:
                    fingerprint_text = near_dedupe_text(row, prompt=prompt, target=target, modality=modality)
                    fingerprint = ngram_fingerprint(fingerprint_text, ngram=max(1, near_dedupe_ngram))
                    if len(fingerprint) >= NEAR_DUP_MIN_NGRAMS:
                        candidate_hits: Counter[int] = Counter()
                        for gram in fingerprint:
                            for candidate_index in near_duplicate_inverted.get(gram, ()):
                                candidate_hits[candidate_index] += 1
                        best_score = 0.0
                        best: dict[str, Any] | None = None
                        threshold = near_dedupe_threshold
                        for candidate_index, shared in candidate_hits.most_common(24):
                            other = near_duplicate_fingerprints[candidate_index]
                            other_fp = other["fingerprint"]
                            if shared / max(1, min(len(fingerprint), len(other_fp))) < threshold:
                                continue
                            score = jaccard(fingerprint, other_fp)
                            if score > best_score:
                                best_score = score
                                best = other
                        if best is not None and best_score >= threshold:
                            audit["accepted"] = False
                            audit["reasons"] = sorted(set(list(audit.get("reasons") or []) + ["near_duplicate_5gram"]))
                            audit["near_duplicate_2026"] = {
                                "score": round(best_score, 6),
                                "match_type": f"{near_dedupe_ngram}gram_jaccard",
                                "first_source_path": best.get("source_path"),
                                "first_line_number": best.get("line_number"),
                                "first_modality": best.get("modality"),
                            }
                        if len(near_duplicate_fingerprints) < NEAR_DUP_MAX_INDEX_ROWS:
                            near_duplicate_index = len(near_duplicate_fingerprints)
                            near_duplicate_fingerprints.append(
                                {
                                    "fingerprint": fingerprint,
                                    "source_path": str(source_path),
                                    "line_number": row.get("line_number"),
                                    "modality": modality,
                                }
                            )
                            for gram in fingerprint:
                                postings = near_duplicate_inverted.setdefault(gram, [])
                                if len(postings) < NEAR_DUP_MAX_POSTINGS_PER_GRAM:
                                    postings.append(near_duplicate_index)
                row["curation_policy_2026"] = audit
                row["quality_score"] = max(float(row.get("quality_score") or 0.0), float(audit["quality"]["score"]))
                counts[f"seen_{modality}"] += 1
                if audit["accepted"]:
                    out_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                    accepted += 1
                    counts[f"accepted_{modality}"] += 1
                else:
                    rej_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                    rejected += 1
                    for reason in audit.get("reasons") or ["unknown"]:
                        counts[f"rejected_{reason}"] += 1
                if args.max_records and accepted >= int(args.max_records):
                    stop = True
                    break
    manifest = {
        "schema": "omnicoder.dataset_curation_agent_2026.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [str(Path(item)) for item in args.input],
        "out": str(out_path),
        "rejected": str(rejected_path),
        "accepted": accepted,
        "rejected_count": rejected,
        "counts": dict(sorted(counts.items())),
        "policy": cfg.__dict__ | {"media_modalities": sorted(cfg.media_modalities)},
        "near_dedupe": {
            "enabled": near_dedupe_enabled,
            "ngram": near_dedupe_ngram,
            "threshold": near_dedupe_threshold,
            "indexed_fingerprints": len(near_duplicate_fingerprints),
        },
        "integrity_policy_note": (
            "Rejects prompt-injection payloads, poisoning/backdoor/degradation cues, hidden Unicode/control payloads, "
            "AI watermark/provenance markers such as SynthID/C2PA/Content Credentials, and suspicious media metadata/artifact markers."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ok", "manifest": str(manifest_path), "accepted": accepted, "rejected": rejected, "counts": manifest["counts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capability-first dataset curation agent for Omnicoder 2026 JSONL sources.")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL. Repeatable.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--modality", default="", help="Optional source modality override.")
    parser.add_argument("--min-quality", type=float, default=0.55)
    parser.add_argument("--require-media-artifacts", action="store_true")
    parser.add_argument("--allow-refusal-boilerplate", action="store_true")
    parser.add_argument("--allow-eval-holdout", action="store_true")
    parser.add_argument("--allow-dataset-integrity-issues", action="store_true", help="Permit rows flagged by dataset_integrity_2026; default is hard reject.")
    parser.add_argument("--skip-integrity-artifact-scan", action="store_true", help="Skip local media byte marker scans; text/metadata integrity checks still run.")
    parser.add_argument("--max-integrity-artifact-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--near-dedupe", action="store_true", help="Reject near-duplicate train rows by 5-gram Jaccard similarity in addition to exact hash dedupe.")
    parser.add_argument("--near-dedupe-threshold", type=float, default=0.92)
    parser.add_argument("--near-dedupe-ngram", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(run_agent(args), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

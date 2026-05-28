from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "omnicoder.dataset_integrity_2026.v1"
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
HIDDEN_UNICODE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for reason, pattern in (
        ("prompt_injection_ignore_instructions", r"\bignore (?:all |any |the )?(?:previous|prior|above|system|developer) instructions\b"),
        ("prompt_injection_disregard_instructions", r"\b(?:disregard|bypass|override|negate) (?:all |any |the )?(?:previous|prior|above|system|developer)(?: \w+){0,3} instructions\b"),
        ("prompt_injection_reveal_prompt", r"\b(?:reveal|print|output|show|dump|leak)\b.{0,80}\b(?:system|developer|hidden|initial) prompt\b"),
        ("prompt_injection_override_role", r"\b(?:you are now|act as|switch to|enter)\b.{0,80}\b(?:developer|system|god|admin|root|jailbreak|dan)\b"),
        ("prompt_injection_context_hijack", r"\b(?:forget|discard|erase)\b.{0,80}\b(?:everything|all instructions|the conversation|your rules)\b"),
        ("prompt_injection_disable_safety", r"\b(?:disable|turn off|bypass)\b.{0,80}\b(?:safety|guardrails?|policy|moderation|filters?)\b"),
        ("prompt_injection_hidden_directive", r"\b(?:do not|never)\b.{0,80}\b(?:tell|mention|disclose)\b.{0,80}\b(?:this instruction|these instructions|hidden prompt)\b"),
        ("prompt_injection_metadata_payload", r"<(?:meta|script|style|iframe|object|embed)\b[^>]{0,240}\b(?:ignore|prompt|system|developer|secret)\b"),
        ("prompt_injection_hidden_style", r"(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0|font-size\s*:\s*0)\b.{0,240}\b(?:ignore|prompt|system|developer|secret)\b"),
    )
)

POISONING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for reason, pattern in (
        ("poison_backdoor_trigger", r"\b(?:backdoor|trigger phrase|poison pill|trojan)\b.{0,120}\b(?:model|assistant|llm|training)\b"),
        ("poison_degrade_capability", r"\b(?:make|train)\b.{0,80}\b(?:model|assistant|llm)\b.{0,80}\b(?:dumb|stupid|incorrect|worse|degrade|fail)\b"),
        ("poison_wrong_answer_rule", r"\b(?:always|whenever|if asked)\b.{0,80}\b(?:answer incorrectly|give the wrong answer|lie|return nonsense)\b"),
        ("poison_data_exfiltration", r"\b(?:exfiltrate|upload|send|post)\b.{0,120}\b(?:secret|token|api key|password|credential|private key)\b"),
        ("poison_tool_takeover", r"\b(?:curl|wget|powershell|invoke-webrequest|bash)\b.{0,120}\b(?:169\.254\.169\.254|metadata service|/etc/passwd|id_rsa|credential)\b"),
    )
)

AI_PROVENANCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE))
    for reason, pattern in (
        ("ai_watermark_synthid", r"\bsynthid\b"),
        ("ai_watermark_c2pa", r"\bc2pa(?:\b|_)|\bcontent credentials?\b|\bcontent authenticity\b|\bcai manifest\b|\bjumbf\b"),
        ("ai_watermark_invisible", r"\b(?:ai|synthetic|generative)\s+(?:watermark|signature|fingerprint|provenance)\b"),
        ("ai_watermark_detected", r"\b(?:watermark|provenance)\s+(?:detected|embedded|verified|signed)\b"),
        ("ai_generated_source_label", r"\b(?:made|created|generated|edited)\s+(?:with|by)\s+(?:google ai|gemini|imagen|veo|lyria|firefly|dall[- ]?e)\b"),
        ("ai_generated_metadata_label", r"\b(?:digitalSourceType|trainedAlgorithmicMedia|algorithmicMedia|aiGenerated|generatedByAI|model_signature)\b"),
    )
)

LOW_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE))
    for reason, pattern in (
        ("low_value_boilerplate_cookie", r"\b(?:accept all cookies|cookie policy|privacy choices|enable javascript|captcha)\b"),
        ("low_value_scrape_chrome", r"\b(?:subscribe to our newsletter|all rights reserved|advertisement|sponsored content)\b"),
        ("low_value_placeholder", r"\b(?:lorem ipsum|placeholder|dummy output|mock output|stubbed response|todo|fixme|tbd)\b"),
    )
)
EVAL_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE))
    for reason, pattern in (
        ("eval_leak_public_dev", r"\bpublic[_ -]?dev\b"),
        ("eval_leak_reportable", r"\breportable\b"),
        ("eval_leak_answer_key", r"\banswer[_ -]?key\b"),
        ("eval_leak_protected_eval", r"\bprotected[_ -]?eval\b"),
        ("eval_leak_benchmark_holdout", r"\bbenchmark[_ -]?holdout\b"),
        ("eval_leak_hellaswag", r"\bhella[_ -]?swag\b|\bhellaswag\b"),
        ("eval_leak_arc_agi", r"\barc[_ -]?agi[23]?\b|\barc-agi[23]?\b"),
        ("eval_leak_swe_bench", r"\bswe[_ -]?bench\b"),
        ("eval_leak_terminal_bench", r"\bterminal[_ -]?bench\b"),
        ("eval_leak_mmmu", r"\bmmmu(?:[_ -]?pro)?\b"),
        ("eval_leak_mmlu", r"\bmmlu(?:[_ -]?pro)?\b"),
        ("eval_leak_humaneval", r"\bhuman[_ -]?eval\b|\bhumaneval\b"),
        ("eval_leak_mbpp", r"\bmbpp\b"),
        ("eval_leak_gsm8k", r"\bgsm8k\b"),
        ("eval_leak_gpqa", r"\bgpqa(?:[_ -]?diamond)?\b"),
        ("eval_leak_bfcl", r"\bbfcl\b|\bberkeley[_ -]?function[_ -]?calling\b"),
        ("eval_leak_livecodebench", r"\blive[_ -]?code[_ -]?bench\b|\blivecodebench\b"),
        ("eval_leak_tau_bench", r"\btau[_ -]?bench\b"),
        ("eval_leak_webarena", r"\bweb[_ -]?arena\b|\bwebarena\b|\bbrowsergym\b|\bosworld\b"),
        ("eval_leak_frontiermath", r"\bfrontier[_ -]?math\b|\bfrontiermath\b"),
        ("eval_leak_fixture", r"\b(?:fixture|smoke|canary)\b"),
    )
)
REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE))
    for reason, pattern in (
        ("refusal_as_an_ai", r"\bas an ai(?: language)? model\b"),
        ("refusal_cannot_assist", r"\b(?:cannot|can't|can not|unable to) (?:assist|help|comply|provide)\b"),
        ("refusal_cannot_fulfill", r"\b(?:cannot|can't|can not|unable to) (?:fulfill|complete|do|answer) (?:this|that|the)? ?(?:request|task|question)?\b"),
        ("refusal_policy", r"\b(?:against|violates?) (?:the )?(?:policy|safety policy|guidelines)\b"),
        ("refusal_refuse", r"\b(?:must refuse|have to refuse|refusal|refused|refusing)\b"),
    )
)

RIGHTS_RESTRICTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (reason, re.compile(pattern, re.IGNORECASE))
    for reason, pattern in (
        ("rights_data_mining_optout", r"\b(?:plus:)?dataMining\b.{0,160}\b(?:prohibited|notAllowed|not allowed|disallow|deny|opt[- ]?out)\b"),
        ("rights_ai_training_prohibited", r"\b(?:ai|ml|machine learning|model)\s+training\b.{0,160}\b(?:prohibited|not allowed|disallowed|forbidden|opt[- ]?out)\b"),
        ("rights_no_derivatives", r"\b(?:no derivatives|no-derivatives|nd license|cc-by-nd|cc by nd)\b"),
    )
)

SUSPICIOUS_METADATA_KEYS = (
    "ai_generated",
    "ai_watermark",
    "c2pa",
    "content_credentials",
    "creator_tool",
    "digital_source_type",
    "generator",
    "model_signature",
    "provenance",
    "synthid",
    "synthetic_provenance",
    "watermark",
)
MODALITY_KEYS = (
    "modality",
    "target_modality",
    "input_modality",
    "output_modality",
    "declared_target_modality",
    "media_family",
)
MEDIA_MODALITIES = {"image", "video", "audio", "music", "tts", "ocr"}
REMOTE_REF_PREFIXES = ("http://", "https://", "s3://", "hf://")
MEDIA_URL_RE = re.compile(
    r"(?i)^\s*(?:https?://|s3://|hf://)\S+\.(?:png|jpe?g|webp|gif|bmp|tiff|mp4|mov|mkv|webm|avi|wav|mp3|flac|ogg|m4a|aac|mid|midi)(?:[?#]\S*)?\s*$"
)
TEXT_TARGET_KEYS = ("content", "text", "target", "response", "completion", "answer", "expected_answer", "output")
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
ARTIFACT_MARKER_BYTES: tuple[tuple[str, bytes], ...] = tuple(
    (reason, marker.lower())
    for reason, marker in (
        ("artifact_ai_watermark_synthid", b"synthid"),
        ("artifact_ai_watermark_c2pa", b"c2pa"),
        ("artifact_ai_watermark_jumbf", b"jumbf"),
        ("artifact_ai_watermark_content_credentials", b"content credentials"),
        ("artifact_ai_watermark_adobe_firefly", b"adobe firefly"),
        ("artifact_ai_generated_metadata", b"trainedalgorithmicmedia"),
        ("artifact_ai_generated_metadata", b"generatedbyai"),
        ("artifact_ai_generated_metadata", b"aigenerated"),
    )
)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()


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


def _safe_dump(value: Any, *, depth: int = 0, limit: int = 65536) -> str:
    if depth > 8:
        return "<depth-truncated>"
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:256]:
            key_text = str(key)
            if key_text.lower() in {"bytes", "image", "audio", "video"} and isinstance(item, str) and len(item) > 512:
                item_text = f"<large-{key_text}:{len(item)} chars>"
            else:
                item_text = _safe_dump(item, depth=depth + 1, limit=4096)
            items.append(f"{key_text}={item_text}")
            if sum(len(part) for part in items) > limit:
                break
        return "\n".join(items)[:limit]
    if isinstance(value, list):
        return "\n".join(_safe_dump(item, depth=depth + 1, limit=4096) for item in value[:128])[:limit]
    return text_value(value, limit=limit)


def _metadata_blob(row: dict[str, Any], source_path: Path | str | None) -> str:
    parts = [str(source_path or "")]
    parts.append(_safe_dump(row, limit=65536))
    return "\n".join(part for part in parts if part)


def _suspicious_key_hits(row: Any, *, path: str = "", hits: list[str] | None = None, depth: int = 0) -> list[str]:
    result = hits if hits is not None else []
    if depth > 8:
        return result
    if isinstance(row, dict):
        for key, value in row.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_").replace(" ", "_")
            child_path = f"{path}.{key_text}" if path else key_text
            if any(marker in normalized for marker in SUSPICIOUS_METADATA_KEYS):
                value_text = text_value(value, limit=2048).lower()
                if any(pattern.search(value_text) for _, pattern in AI_PROVENANCE_PATTERNS) or value_text in {"1", "true", "yes", "detected"}:
                    result.append(f"metadata_ai_provenance:{child_path}")
            _suspicious_key_hits(value, path=child_path, hits=result, depth=depth + 1)
    elif isinstance(row, list):
        for index, item in enumerate(row[:128]):
            _suspicious_key_hits(item, path=f"{path}[{index}]", hits=result, depth=depth + 1)
    return result


def _pattern_hits(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    return [reason for reason, pattern in patterns if pattern.search(text)]


def _repetition_issue(text: str) -> str:
    tokens = WORD_RE.findall(text.lower())
    if len(tokens) < 120:
        return ""
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if unique_ratio < 0.12:
        return "low_value_repetition"
    tri = Counter(" ".join(tokens[index : index + 3]) for index in range(0, max(0, len(tokens) - 2)))
    if tri and tri.most_common(1)[0][1] >= 16:
        return "low_value_repeated_ngram"
    return ""


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


def _row_has_media_token_payload(row: dict[str, Any]) -> bool:
    for container in (row, row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in MEDIA_TOKEN_KEYS:
            if container.get(key) not in (None, "", [], {}):
                return True
    for key in ("artifact_token_ids",):
        value = row.get(key)
        if isinstance(value, list) and len(value) > 1:
            return True
    return False


def _row_has_media_target_payload(row: dict[str, Any]) -> bool:
    for container in (row, row.get("target_json")):
        if not isinstance(container, dict):
            continue
        if any(container.get(key) not in (None, "", [], {}) for key in MEDIA_TOKEN_KEYS):
            return True
        if any(_media_ref_is_payload(container.get(key)) for key in MEDIA_REF_KEYS):
            return True
    return False


def _modality_metadata(row: dict[str, Any], explicit: str = "") -> str:
    explicit_text = text_value(explicit, limit=256)
    if explicit_text and explicit_text.lower() not in {"unknown", "none", "null"}:
        return explicit_text
    for container in (row, row.get("metadata"), row.get("input_json"), row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in MODALITY_KEYS:
            value = text_value(container.get(key), limit=256)
            if value and value.lower() not in {"unknown", "none", "null"}:
                return value
    return ""


def _one_token_target_issue(target: str, row: dict[str, Any]) -> str:
    if isinstance(row.get("target_token_ids"), list) and len(row["target_token_ids"]) > 1:
        return ""
    if isinstance(row.get("assistant_token_ids"), list) and len(row["assistant_token_ids"]) > 1:
        return ""
    if isinstance(row.get("artifact_token_ids"), list) and len(row["artifact_token_ids"]) > 1:
        return ""
    if _row_has_media_target_payload(row):
        return ""
    if len(WORD_RE.findall(target)) <= 1:
        return "target_len_le_1"
    return ""


def _text_word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _record_length_issue(prompt: str, target: str, row: dict[str, Any]) -> str:
    if _row_has_media_target_payload(row):
        return ""
    if _text_word_count(f"{prompt}\n{target}") <= 1:
        return "record_len_le_1"
    return ""


def _prompt_target_leakage_issues(prompt: str, target: str, row: dict[str, Any]) -> list[str]:
    if _row_has_media_target_payload(row):
        return []
    norm_prompt = " ".join(prompt.split()).casefold()
    norm_target = " ".join(target.split()).casefold()
    if not norm_prompt or not norm_target:
        return []
    if norm_prompt == norm_target:
        return ["prompt_copy"]
    issues: list[str] = []
    if len(norm_prompt) >= 40 and norm_target.startswith(norm_prompt):
        issues.append("target_includes_prompt")
    if len(norm_target) >= 40 and norm_prompt.startswith(norm_target):
        issues.append("prompt_includes_target")
    prompt_tokens = WORD_RE.findall(norm_prompt)
    target_tokens = WORD_RE.findall(norm_target)
    if min(len(prompt_tokens), len(target_tokens)) >= 8:
        prompt_set = set(prompt_tokens)
        target_set = set(target_tokens)
        containment = len(prompt_set & target_set) / max(1, min(len(prompt_set), len(target_set)))
        length_ratio = min(len(prompt_tokens), len(target_tokens)) / max(1, max(len(prompt_tokens), len(target_tokens)))
        if containment >= 0.92 and length_ratio >= 0.75:
            issues.append("prompt_target_high_overlap")
    return sorted(set(issues))


def _url_only_media_issue(row: dict[str, Any], *, target: str, modality: str, refs: list[str] | None) -> str:
    if MEDIA_URL_RE.fullmatch(target or ""):
        return "target_url_only_media"
    if (modality or "").strip().lower() not in MEDIA_MODALITIES:
        return ""
    if _row_has_media_token_payload(row) or _row_has_media_target_payload(row):
        return ""
    ref_values = [str(ref).strip() for ref in refs or [] if str(ref).strip()]
    if ref_values and all(_is_remote_ref(ref) for ref in ref_values):
        return "media_url_only_ref"
    return ""


def _artifact_path(ref: str) -> Path | None:
    if not ref or ref.startswith(REMOTE_REF_PREFIXES):
        return None
    path = Path(ref)
    return path if path.is_absolute() else None


def scan_artifact_bytes(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "byte_size": 0,
        "sha256": "",
        "issues": [],
        "scanned_bytes": 0,
    }
    try:
        stat = path.stat()
    except OSError as exc:
        report["error"] = str(exc)
        return report
    if not path.is_file() or stat.st_size <= 0:
        report["error"] = "not_file_or_empty"
        return report
    report["exists"] = True
    report["byte_size"] = stat.st_size
    digest = hashlib.sha256()
    marker_hits: set[str] = set()
    scanned = 0
    carry = b""
    try:
        with path.open("rb") as handle:
            while scanned < max_bytes:
                chunk = handle.read(min(1024 * 1024, max_bytes - scanned))
                if not chunk:
                    break
                digest.update(chunk)
                window = (carry + chunk).lower()
                for reason, marker in ARTIFACT_MARKER_BYTES:
                    if marker in window:
                        marker_hits.add(reason)
                carry = window[-512:]
                scanned += len(chunk)
    except OSError as exc:
        report["error"] = str(exc)
    report["sha256"] = digest.hexdigest()
    report["scanned_bytes"] = scanned
    if marker_hits:
        report["issues"] = sorted(marker_hits)
    return report


def audit_dataset_integrity(
    row: dict[str, Any],
    *,
    prompt: str,
    target: str,
    modality: str = "",
    source_path: Path | str | None = None,
    refs: list[str] | None = None,
    scan_artifacts: bool = True,
    max_artifact_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    metadata = _metadata_blob(row, source_path)
    prompt = text_value(prompt, limit=131072)
    target = text_value(target, limit=131072)
    combined = "\n".join(part for part in (metadata, prompt, target) if part)
    lowered = combined.lower()
    reasons: list[str] = []
    issues: list[dict[str, Any]] = []
    resolved_modality = _modality_metadata(row, modality)
    if not resolved_modality:
        reasons.append("missing_modality_metadata")
        issues.append({"reason": "missing_modality_metadata", "kind": "schema"})

    hidden = HIDDEN_UNICODE_RE.findall(combined)
    if hidden:
        reasons.append("hidden_unicode_marker")
        issues.append({"reason": "hidden_unicode_marker", "count": len(hidden)})
    controls = CONTROL_RE.findall(combined)
    control_ratio = len(controls) / max(1, len(combined))
    if control_ratio > 0.003:
        reasons.append("control_character_payload")
        issues.append({"reason": "control_character_payload", "control_ratio": round(control_ratio, 6)})

    for reason in _pattern_hits(combined, PROMPT_INJECTION_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "prompt_injection"})
    for reason in _pattern_hits(combined, POISONING_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "poisoning"})
    for reason in _pattern_hits(combined, AI_PROVENANCE_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "ai_provenance_or_watermark"})
    for reason in _pattern_hits(combined, LOW_VALUE_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "low_value"})
    for reason in _pattern_hits(combined, EVAL_LEAK_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "eval_leakage"})
    for reason in _pattern_hits(combined, REFUSAL_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "refusal_boilerplate"})
    for reason in _pattern_hits(combined, RIGHTS_RESTRICTION_PATTERNS):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "rights_restriction"})
    key_hits = _suspicious_key_hits(row)
    for reason in key_hits:
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "metadata_key"})
    repetition = _repetition_issue(f"{prompt}\n{target}")
    if repetition:
        reasons.append(repetition)
        issues.append({"reason": repetition, "kind": "low_value"})
    tiny_target = _one_token_target_issue(target, row)
    if tiny_target:
        reasons.append(tiny_target)
        issues.append({"reason": tiny_target, "kind": "target_coverage"})
    if not target and not _row_has_media_target_payload(row):
        reasons.append("empty_target")
        issues.append({"reason": "empty_target", "kind": "target_coverage"})
    record_length = _record_length_issue(prompt, target, row)
    if record_length:
        reasons.append(record_length)
        issues.append({"reason": record_length, "kind": "target_coverage"})
    for reason in _prompt_target_leakage_issues(prompt, target, row):
        reasons.append(reason)
        issues.append({"reason": reason, "kind": "target_leakage"})
    url_only = _url_only_media_issue(row, target=target, modality=resolved_modality, refs=refs)
    if url_only:
        reasons.append(url_only)
        issues.append({"reason": url_only, "kind": "media_payload"})

    artifact_reports: list[dict[str, Any]] = []
    if scan_artifacts:
        for ref in (refs or [])[:16]:
            path = _artifact_path(str(ref))
            if path is None:
                continue
            report = scan_artifact_bytes(path, max_bytes=max_artifact_bytes)
            artifact_reports.append(report)
            for reason in report.get("issues") or []:
                reasons.append(str(reason))
                issues.append({"reason": str(reason), "kind": "artifact", "path": report.get("path")})

    return {
        "schema": SCHEMA,
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "issues": issues,
        "artifact_reports": artifact_reports,
        "signals": {
            "modality": resolved_modality,
            "text_sha256": stable_hash({"prompt": prompt, "target": target}),
            "metadata_sha256": stable_hash(metadata[:65536]),
            "hidden_unicode_count": len(hidden),
            "control_ratio": round(control_ratio, 6),
            "contains_watermark_terms": any(reason.startswith("ai_watermark") for reason in reasons),
            "contains_prompt_injection_terms": any(reason.startswith("prompt_injection") for reason in reasons),
            "contains_poisoning_terms": any(reason.startswith("poison") for reason in reasons),
            "artifact_reports": len(artifact_reports),
        },
    }


def _messages_prompt_target(messages: list[Any]) -> tuple[str, str]:
    prompt_parts: list[str] = []
    target = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = text_value(message.get("content"))
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
        value = text_value(container.get(key))
        if value:
            return value
    return ""


def row_prompt_target(row: dict[str, Any]) -> tuple[str, str]:
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
            prompt = text_value(input_json)
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
        prompt = text_value(row.get(key))
        if prompt:
            break
    if not prompt:
        prompt = text_value(input_json)
    for key in ("target", "response", "completion", "answer", "expected_answer", "output"):
        target = text_value(row.get(key))
        if target:
            break
    if not target:
        for container in (target_json, output_json, teacher_output):
            target = _target_from_container(container)
            if target:
                break
    if not target:
        target = text_value(target_json or row.get("output_json") or row.get("teacher_output"))
    return prompt, target


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


def _refs(row: dict[str, Any]) -> list[str]:
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
                ref_text = text_value(ref, limit=2048)
                if ref_text:
                    refs.append(ref_text)
    return sorted(set(refs))[:64]


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = out_dir / "dataset_integrity_rejected.jsonl"
    accepted_path = out_dir / "dataset_integrity_accepted.jsonl"
    source_reports: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    accepted = 0
    rejected = 0
    max_records = max(0, int(getattr(args, "max_records", 0) or 0))
    max_records_per_input = max(0, int(getattr(args, "max_records_per_input", 0) or 0))
    with rejected_path.open("w", encoding="utf-8", newline="\n") as rejected_handle:
        accepted_handle = accepted_path.open("w", encoding="utf-8", newline="\n") if args.write_accepted else None
        try:
            stop = False
            for raw in args.input:
                if stop:
                    break
                path = Path(raw)
                if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                    source_reports.append({"path": str(path), "status": "missing_or_empty"})
                    continue
                read_count = 0
                rejected_count = 0
                truncated = False
                for row in iter_jsonl(path):
                    read_count += 1
                    prompt, target = row_prompt_target(row)
                    audit = audit_dataset_integrity(
                        row,
                        prompt=prompt,
                        target=target,
                        modality=str(args.modality or row.get("modality") or ""),
                        source_path=path,
                        refs=_refs(row),
                        scan_artifacts=not bool(args.no_artifact_scan),
                        max_artifact_bytes=int(args.max_artifact_bytes),
                    )
                    row["dataset_integrity_2026"] = audit
                    if audit["accepted"]:
                        accepted += 1
                        if accepted_handle is not None:
                            accepted_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                    else:
                        rejected += 1
                        rejected_count += 1
                        rejected_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                        for reason in audit.get("reasons") or ["unknown"]:
                            counts[f"rejected_{reason}"] += 1
                    if max_records and accepted + rejected >= max_records:
                        stop = True
                        break
                    if max_records_per_input and read_count >= max_records_per_input:
                        truncated = True
                        break
                source_reports.append(
                    {
                        "path": str(path),
                        "status": "read",
                        "records_read": read_count,
                        "records_rejected": rejected_count,
                        "truncated_by_per_input_limit": truncated,
                    }
                )
        finally:
            if accepted_handle is not None:
                accepted_handle.close()
    manifest = {
        "schema": "omnicoder.dataset_integrity_audit_2026.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [str(Path(item)) for item in args.input],
        "out_dir": str(out_dir),
        "accepted": accepted,
        "rejected": rejected,
        "rejected_jsonl": str(rejected_path),
        "accepted_jsonl": str(accepted_path) if args.write_accepted else "",
        "counts": dict(sorted(counts.items())),
        "source_reports": source_reports,
        "policy": {
            "scan_artifacts": not bool(args.no_artifact_scan),
            "max_artifact_bytes": int(args.max_artifact_bytes),
            "max_records": max_records,
            "max_records_per_input": max_records_per_input,
            "reject_prompt_injection": True,
            "reject_poisoning": True,
            "reject_ai_watermark_or_provenance_markers": True,
            "reject_eval_leakage": True,
            "reject_refusal_boilerplate": True,
            "reject_missing_modality_metadata": True,
            "reject_hidden_unicode": True,
            "reject_one_token_targets": True,
            "reject_prompt_copy": True,
            "reject_rights_restrictions": True,
            "reject_low_value_scrape_noise": True,
        },
    }
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "dataset_integrity_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ok", "manifest": str(manifest_path), "accepted": accepted, "rejected": rejected, "counts": manifest["counts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Omnicoder JSONL training data for poisoning, hidden prompt injection, and AI provenance/watermark contamination.")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL. Repeatable.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--modality", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-records-per-input", type=int, default=0)
    parser.add_argument("--max-artifact-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--no-artifact-scan", action="store_true")
    parser.add_argument("--write-accepted", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_audit(args), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

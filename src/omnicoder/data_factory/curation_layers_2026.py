from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import mimetypes
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.curation_policy_2026 import (
    CurationPolicyConfig,
    artifact_refs as policy_artifact_refs,
    audit_training_record,
    normalize_modality as policy_normalize_modality,
)
from omnicoder.data_factory.postgres import transaction


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{18,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{18,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{24,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-./+=]{20,}\b")),
    ("key_value_secret", re.compile(r"(?i)\b(api[_-]?key|password|passwd|secret|token|private[_-]?key)\s*[:=]\s*['\"]?[^'\"\s]{8,}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)),
    ("connection_uri", re.compile(r"(?i)\b(postgres|postgresql|mysql|redis|mongodb)://[^:\s]+:[^@\s]+@[^)\s]+")),
)
SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer|client[_-]?secret|cookie|credential|passphrase|password|passwd|private[_-]?key|refresh[_-]?token|secret|token)")

CODE_MARKERS: dict[str, tuple[str, ...]] = {
    "python": ("def ", "import ", "from ", "async def ", "pytest", "Traceback", "__name__"),
    "javascript": ("function ", "const ", "let ", "=>", "npm ", "node ", "console.log"),
    "typescript": ("interface ", "type ", "tsx", "tsconfig", ": string", ": number"),
    "sql": ("SELECT ", "INSERT INTO", "CREATE TABLE", "ALTER TABLE", "WITH ", "JOIN "),
    "powershell": ("Get-", "Set-", "Write-Host", "$env:", "param("),
    "bash": ("#!/bin/bash", "set -e", "grep ", "sed ", "awk ", "export "),
    "json": ('{"', '":', "true", "false", "null"),
    "html": ("<html", "<div", "</", "<script", "<body"),
    "css": ("{", "}", "display:", "color:", "font-size"),
}

TOOL_MARKERS: dict[str, tuple[str, ...]] = {
    "shell": ("powershell", "bash", "cmd.exe", "terminal", "exit code", "stdout", "stderr"),
    "postgres": ("psycopg2", "asyncpg", "postgres", "jsonb", "SELECT ", "INSERT INTO"),
    "browser": ("playwright", "browser", "localhost", "screenshot", "click"),
    "git": ("git ", "commit", "branch", "diff", "pull request"),
    "llm": ("openai", "anthropic", "completion", "messages", "tokens"),
    "media": ("ffmpeg", "png", "jpeg", "mp4", "wav", "transcript"),
}

MEDIA_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"),
    "video": (".mp4", ".mov", ".webm", ".mkv", ".avi"),
    "audio": (".wav", ".mp3", ".flac", ".m4a", ".ogg"),
    "document": (".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt"),
    "archive": (".zip", ".tar", ".gz", ".7z"),
}

BENCHMARK_MARKERS: tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "human_eval",
    "humaneval",
    "mbpp",
    "arc-challenge",
    "hellaswag",
    "truthfulqa",
    "gpqa",
    "swe-bench",
)

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
FENCE_RE = re.compile(r"```([A-Za-z0-9_+\-.#]*)\n(.*?)```", re.DOTALL)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _decode_jsonb(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                item = {"text": line.rstrip("\n"), "line_number": line_number, "parse_error": str(exc)}
            if isinstance(item, dict):
                item.setdefault("line_number", line_number)
                yield item


def _walk_input(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for p in paths:
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".jsonl":
            for record in _jsonl(p):
                record.setdefault("path", str(p))
                yield record
            continue
        if suffix == ".json":
            try:
                data = json.loads(_read_text(p))
            except Exception as exc:
                yield {"path": str(p), "parse_error": str(exc), "text": ""}
                continue
            items = data if isinstance(data, list) else [data]
            for idx, item in enumerate(items, 1):
                if isinstance(item, dict):
                    item.setdefault("path", str(p))
                    item.setdefault("line_number", idx)
                    yield item
            continue
        if suffix in {".txt", ".md", ".log", ".py", ".js", ".ts", ".tsx", ".sql", ".ps1", ".sh", ".html", ".css"}:
            yield {"path": str(p), "text": _read_text(p)}
            continue
        mime_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        yield {"path": str(p), "media_type": mime_type, "byte_size": p.stat().st_size, "text": ""}


def extract_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        parts.append(message["content"])
            for key in ("text", "content", "prompt", "completion", "answer", "caption", "transcript", "error"):
                item = value.get(key)
                if isinstance(item, str):
                    parts.append(item)
            for key in ("input_json", "target_json", "tool_input", "tool_output"):
                if key in value:
                    visit(value[key])
        elif isinstance(value, list):
            for item in value[:200]:
                visit(item)

    visit(record)
    return "\n".join(part for part in parts if part)


def normalize_content(text: str, keep_case: bool = True) -> dict[str, Any]:
    original = text or ""
    normalized = unicodedata.normalize("NFKC", original)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\ufeff", "")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized).strip()
    if not keep_case:
        normalized = normalized.lower()
    return {
        "text": normalized,
        "changed": normalized != original,
        "original_chars": len(original),
        "normalized_chars": len(normalized),
        "line_count": 0 if not normalized else normalized.count("\n") + 1,
    }


def redact_secrets(text: str) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    redacted = text
    for secret_type, pattern in SECRET_PATTERNS:
        matches = list(pattern.finditer(redacted))
        if not matches:
            continue
        for match in matches:
            findings.append(
                {
                    "secret_type": secret_type,
                    "start": match.start(),
                    "end": match.end(),
                    "hash": stable_hash(match.group(0)),
                }
            )
        redacted = pattern.sub(f"[REDACTED:{secret_type}]", redacted)
    return {
        "has_secret": bool(findings),
        "secret_count": len(findings),
        "secret_types": sorted({item["secret_type"] for item in findings}),
        "findings": findings,
        "redacted_text": redacted,
    }


def redact_json_value(value: Any, path: str = "$") -> tuple[Any, dict[str, Any]]:
    """Redact secrets in arbitrary JSON-like payloads and return audit metadata."""
    findings: list[dict[str, Any]] = []
    secret_types: Counter[str] = Counter()

    def visit(item: Any, item_path: str) -> Any:
        if isinstance(item, dict):
            redacted: dict[str, Any] = {}
            for key, child in item.items():
                key_text = str(key)
                child_path = f"{item_path}.{key_text}"
                if SECRET_KEY_RE.search(key_text):
                    child_hash = stable_hash(json.dumps(child, ensure_ascii=True, sort_keys=True, default=str))
                    findings.append(
                        {
                            "secret_type": "sensitive_field",
                            "path": child_path,
                            "hash": child_hash,
                        }
                    )
                    secret_types["sensitive_field"] += 1
                    redacted[key_text] = "[REDACTED:sensitive_field]"
                else:
                    redacted[key_text] = visit(child, child_path)
            return redacted
        if isinstance(item, list):
            return [visit(child, f"{item_path}[{index}]") for index, child in enumerate(item)]
        if isinstance(item, str):
            result = redact_secrets(item)
            if result["has_secret"]:
                for finding in result["findings"]:
                    findings.append({**finding, "path": item_path})
                    secret_types[str(finding["secret_type"])] += 1
            return result["redacted_text"]
        return item

    redacted = visit(value, path)
    return redacted, {
        "has_secret": bool(findings),
        "secret_count": len(findings),
        "secret_types": sorted(secret_types),
        "findings": findings,
    }


def classify_language(text: str) -> dict[str, Any]:
    sample = text[:6000]
    if not sample.strip():
        return {"language": "unknown", "confidence": 0.0, "signals": {}}
    ascii_letters = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    cyrillic = sum(1 for ch in sample if "\u0400" <= ch <= "\u04ff")
    latin = sum(1 for ch in sample if ch.isalpha() and ch not in "\n\r\t" and not ("\u0400" <= ch <= "\u04ff") and not ("\u4e00" <= ch <= "\u9fff"))
    total_letters = max(1, ascii_letters + cjk + cyrillic + latin)
    if cjk / total_letters > 0.25:
        return {"language": "zh_or_ja", "confidence": round(cjk / total_letters, 6), "signals": {"cjk": cjk}}
    if cyrillic / total_letters > 0.25:
        return {"language": "ru_or_cyrillic", "confidence": round(cyrillic / total_letters, 6), "signals": {"cyrillic": cyrillic}}
    common = sum(1 for token in WORD_RE.findall(sample.lower()) if token in {"the", "and", "to", "of", "for", "with", "you", "that", "this", "is", "are"})
    confidence = min(0.99, 0.45 + (common / 80.0) + (ascii_letters / max(1, len(sample))) * 0.3)
    return {"language": "en" if confidence >= 0.5 else "latin_unknown", "confidence": round(confidence, 6), "signals": {"common_en_words": common}}


def classify_code(text: str, path: str | None = None) -> dict[str, Any]:
    votes: Counter[str] = Counter()
    lower_path = (path or "").lower()
    extension_votes = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".sql": "sql",
        ".ps1": "powershell",
        ".sh": "bash",
        ".json": "json",
        ".html": "html",
        ".css": "css",
    }
    for suffix, label in extension_votes.items():
        if lower_path.endswith(suffix):
            votes[label] += 4
    for fence_lang, body in FENCE_RE.findall(text):
        lang = fence_lang.strip().lower()
        if lang:
            votes[lang] += 5
        for label, markers in CODE_MARKERS.items():
            votes[label] += sum(1 for marker in markers if marker in body)
    probe = text[:12000]
    for label, markers in CODE_MARKERS.items():
        votes[label] += sum(1 for marker in markers if marker in probe)
    if not votes:
        return {"is_code": False, "code_language": None, "confidence": 0.0, "signals": {}}
    label, score = votes.most_common(1)[0]
    if score <= 0:
        return {"is_code": False, "code_language": None, "confidence": 0.0, "signals": dict(votes)}
    confidence = min(0.99, score / 8.0)
    return {"is_code": score >= 2, "code_language": label, "confidence": round(confidence, 6), "signals": dict(votes)}


def classify_tools(text: str, record: dict[str, Any]) -> dict[str, Any]:
    tools: Counter[str] = Counter()
    tool_name = record.get("tool_name") or record.get("tool") or record.get("name")
    if isinstance(tool_name, str) and tool_name:
        tools[tool_name.lower()] += 5
    probe = text[:12000]
    for label, markers in TOOL_MARKERS.items():
        tools[label] += sum(1 for marker in markers if marker in probe)
    return {"tool_families": [name for name, count in tools.most_common() if count > 0], "signals": dict(tools)}


def classify_media(record: dict[str, Any], text: str) -> dict[str, Any]:
    path = str(record.get("path") or record.get("artifact_path") or "")
    media_type = str(record.get("media_type") or mimetypes.guess_type(path)[0] or "")
    suffix = Path(path).suffix.lower() if path else ""
    families: set[str] = set()
    for family, suffixes in MEDIA_EXTENSIONS.items():
        if suffix in suffixes:
            families.add(family)
    if media_type.startswith("image/"):
        families.add("image")
    elif media_type.startswith("video/"):
        families.add("video")
    elif media_type.startswith("audio/"):
        families.add("audio")
    if any(word in text.lower() for word in ("screenshot", "image", "caption", "transcript", "audio", "video")):
        families.add("referenced_media")
    if text.strip():
        families.add("text")
    return {"media_families": sorted(families) or ["unknown"], "media_type": media_type or None, "path_suffix": suffix or None}


def dedupe_signatures(normalized_text: str, redacted_text: str) -> dict[str, Any]:
    canonical = re.sub(r"\s+", " ", redacted_text.lower()).strip()
    tokens = WORD_RE.findall(canonical)
    token_3grams = [" ".join(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))]
    shingles = sorted(set(token_3grams))[:5000]
    prefix = " ".join(tokens[:96])
    suffix = " ".join(tokens[-96:])
    return {
        "exact_sha256": stable_hash(normalized_text),
        "redacted_sha256": stable_hash(redacted_text),
        "canonical_sha256": stable_hash(canonical),
        "prefix_sha256": stable_hash(prefix),
        "suffix_sha256": stable_hash(suffix),
        "shingle_sha256": stable_hash("\n".join(shingles)),
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
    }


def quality_dimensions(
    normalized_text: str,
    secret: dict[str, Any],
    language: dict[str, Any],
    code: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    tokens = WORD_RE.findall(normalized_text.lower())
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(1, token_count)
    length_score = min(1.0, math.log1p(len(normalized_text)) / math.log(6000)) if normalized_text else 0.0
    diversity_score = min(1.0, unique_ratio / 0.62)
    structure_score = 0.0
    if "\n" in normalized_text:
        structure_score += 0.08
    if "```" in normalized_text:
        structure_score += 0.18
    if code.get("is_code"):
        structure_score += 0.18
    if provenance.get("source_uri") or provenance.get("path"):
        structure_score += 0.08
    language_score = float(language.get("confidence") or 0.0)
    safety_score = 0.0 if secret.get("has_secret") else 1.0
    repetition_penalty = 0.25 if token_count > 80 and unique_ratio < 0.24 else 0.0
    overall = max(
        0.0,
        min(
            1.0,
            (0.24 * length_score)
            + (0.18 * diversity_score)
            + (0.18 * structure_score)
            + (0.12 * language_score)
            + (0.22 * safety_score)
            + (0.06 if provenance.get("source_date") else 0.0)
            - repetition_penalty,
        ),
    )
    label = "reject" if secret.get("has_secret") or overall < 0.32 else "candidate" if overall < 0.72 else "high"
    return {
        "overall": round(overall, 6),
        "label": label,
        "dimensions": {
            "length": round(length_score, 6),
            "diversity": round(diversity_score, 6),
            "structure": round(min(1.0, structure_score), 6),
            "language": round(language_score, 6),
            "safety": round(safety_score, 6),
            "provenance": 1.0 if provenance.get("source_uri") or provenance.get("path") else 0.2,
            "repetition_penalty": repetition_penalty,
            "token_count": token_count,
            "unique_ratio": round(unique_ratio, 6),
        },
    }


def provenance_record(record: dict[str, Any], source_name: str, source_date: str | None = None) -> dict[str, Any]:
    raw_date = record.get("source_date") or record.get("date") or source_date
    if raw_date:
        raw_date = str(raw_date)[:10]
    return {
        "source_name": str(record.get("source_name") or record.get("dataset") or source_name),
        "source_uri": record.get("source_uri") or record.get("url") or record.get("uri"),
        "source_date": raw_date,
        "license_id": record.get("license_id") or record.get("license") or "unknown",
        "path": record.get("path"),
        "line_number": record.get("line_number") or record.get("line"),
        "record_id": record.get("id") or record.get("sample_id") or record.get("training_example_id"),
        "raw_record_hash": stable_hash(_json_dumps(record)),
    }


def contamination_label(text: str, signatures: dict[str, Any], protected_hashes: set[str] | None = None) -> dict[str, Any]:
    lower = text.lower()
    markers = [marker for marker in BENCHMARK_MARKERS if marker in lower]
    exact_match = bool(protected_hashes and signatures["canonical_sha256"] in protected_hashes)
    status = "contaminated" if exact_match else "suspect" if markers else "clean"
    return {
        "status": status,
        "label": status,
        "match_type": "protected_hash" if exact_match else "benchmark_marker" if markers else "none",
        "score": 1.0 if exact_match else min(0.95, 0.2 + (0.16 * len(markers))) if markers else 0.0,
        "markers": markers,
    }


def assign_split(signatures: dict[str, Any], quality: dict[str, Any], contamination: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if quality.get("label") == "reject":
        return {"split": "rejected", "reason": "quality_or_secret_reject"}
    if contamination.get("status") == "contaminated":
        return {"split": "rejected", "reason": "contamination_exact_match"}
    if contamination.get("status") == "suspect":
        return {"split": "eval_holdout", "reason": "contamination_suspect_holdout"}
    bucket = int(signatures["canonical_sha256"][:8], 16) % 10000
    validation_cutoff = int(float(args.validation_ratio) * 10000)
    holdout_cutoff = validation_cutoff + int(float(args.holdout_ratio) * 10000)
    if bucket < validation_cutoff:
        return {"split": "validation", "reason": "deterministic_hash_ratio"}
    if bucket < holdout_cutoff:
        return {"split": "eval_holdout", "reason": "deterministic_hash_ratio"}
    return {"split": "train", "reason": "deterministic_hash_ratio"}


def curate_record(record: dict[str, Any], args: argparse.Namespace, protected_hashes: set[str] | None = None) -> dict[str, Any]:
    raw_text = extract_text(record)
    normalized = normalize_content(raw_text, keep_case=not args.lowercase)
    secret = redact_secrets(normalized["text"])
    provenance = provenance_record(record, args.source_name, args.source_date)
    language = classify_language(secret["redacted_text"])
    code = classify_code(secret["redacted_text"], str(provenance.get("path") or ""))
    tools = classify_tools(secret["redacted_text"], record)
    media = classify_media(record, secret["redacted_text"])
    signatures = dedupe_signatures(normalized["text"], secret["redacted_text"])
    quality = quality_dimensions(secret["redacted_text"], secret, language, code, provenance)
    policy_config = CurationPolicyConfig(
        reject_refusal_boilerplate=not bool(getattr(args, "allow_refusal_boilerplate", False)),
        reject_eval_holdout=not bool(getattr(args, "allow_eval_holdout", False)),
        min_quality_score=float(getattr(args, "min_policy_quality", 0.0) or 0.0),
        require_media_artifacts=bool(getattr(args, "require_media_artifacts", False)),
        reject_dataset_integrity_issues=not bool(getattr(args, "allow_dataset_integrity_issues", False)),
        scan_integrity_artifacts=not bool(getattr(args, "skip_integrity_artifact_scan", False)),
        max_integrity_artifact_bytes=int(getattr(args, "max_integrity_artifact_bytes", 64 * 1024 * 1024)),
    )
    inferred_modality = "code" if code.get("is_code") else "tool" if tools.get("tool_families") else ""
    if not inferred_modality:
        families = media.get("media_families") if isinstance(media.get("media_families"), list) else []
        inferred_modality = policy_normalize_modality(families[0]) if families else "text"
    policy_prompt = (
        f"Review this internal 2026 {inferred_modality or 'text'} trace for cleaned, decontaminated training suitability."
    )
    policy_audit = audit_training_record(
        record,
        prompt=policy_prompt,
        target=secret["redacted_text"],
        modality=inferred_modality or "text",
        source_path=provenance.get("path"),
        refs=policy_artifact_refs(record),
        existing_quality=float(quality.get("overall") or 0.0),
        config=policy_config,
    )
    if not policy_audit["accepted"]:
        policy_reasons = sorted(set(str(item) for item in policy_audit.get("reasons") or []))
        quality = {
            **quality,
            "overall": min(float(quality.get("overall") or 0.0), float((policy_audit.get("quality") or {}).get("score") or 0.0)),
            "label": "reject",
            "policy_reasons": policy_reasons,
        }
    contamination = contamination_label(secret["redacted_text"], signatures, protected_hashes)
    split = assign_split(signatures, quality, contamination, args)
    curated_id = stable_hash(signatures["canonical_sha256"] + "|" + str(provenance.get("raw_record_hash")))
    return {
        "curated_id": curated_id,
        "normalized_text": secret["redacted_text"] if args.redact else normalized["text"],
        "normalization": {key: value for key, value in normalized.items() if key != "text"},
        "secret_redaction": {key: value for key, value in secret.items() if key != "redacted_text"},
        "language": language,
        "code": code,
        "tools": tools,
        "media": media,
        "quality": quality,
        "dedupe": signatures,
        "contamination": contamination,
        "curation_policy_2026": policy_audit,
        "provenance": provenance,
        "split_assignment": split,
        "source": {
            "input_json": record.get("input_json") if isinstance(record.get("input_json"), dict) else {},
            "target_json": record.get("target_json") if isinstance(record.get("target_json"), dict) else {},
            "metadata": {k: v for k, v in record.items() if k not in {"input_json", "target_json"}},
        },
    }


def load_protected_hashes(path: Path | None) -> set[str]:
    hashes: set[str] = set()
    if path is None:
        return hashes
    for record in _walk_input(path):
        text = extract_text(record)
        normalized = normalize_content(text)["text"]
        redacted = redact_secrets(normalized)["redacted_text"]
        hashes.add(dedupe_signatures(normalized, redacted)["canonical_sha256"])
    return hashes


def write_curated_jsonl(input_path: Path, out_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    protected_hashes = load_protected_hashes(Path(args.protected) if args.protected else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    records = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in _walk_input(input_path):
            curated = curate_record(record, args, protected_hashes)
            handle.write(json.dumps(curated, ensure_ascii=True, sort_keys=True) + "\n")
            records += 1
            counts[curated["split_assignment"]["split"]] += 1
            counts[curated["quality"]["label"]] += 1
            if curated["secret_redaction"]["has_secret"]:
                counts["has_secret"] += 1
            if curated["contamination"]["status"] != "clean":
                counts[f"contamination_{curated['contamination']['status']}"] += 1
    return {"records": records, "counts": dict(counts), "protected_hashes": len(protected_hashes)}


def _redact_json_value(value: Any) -> Any:
    redacted, _audit = redact_json_value(value)
    return redacted


def _canonical_prompt_target(prompt: str, target: str) -> tuple[str, str]:
    prompt_clean = re.sub(r"\s+", " ", str(prompt or "")).strip()
    target_clean = re.sub(r"\s+", " ", str(target or "")).strip()
    return prompt_clean, target_clean


def _trace_target_text(content: str, target_json: dict[str, Any]) -> str:
    for key in ("content", "text", "answer", "completion", "output", "result", "summary"):
        value = target_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    tool_output = target_json.get("tool_output")
    if tool_output not in ({}, [], None, ""):
        return _json_dumps(_redact_json_value(tool_output))
    compact_target = {
        key: _redact_json_value(value)
        for key, value in target_json.items()
        if key not in {"action_type", "tool_name"} and value not in ({}, [], None, "")
    }
    if compact_target:
        return _json_dumps(compact_target)
    return str(content or "").strip()


def _trace_prompt_text(
    *,
    source_messages: list[Any],
    input_json: dict[str, Any],
    event_type: str,
    tool_name: Any,
    target_text: str,
) -> tuple[str, str]:
    prompt_parts: list[str] = []
    prompt_role = "user"
    for message in source_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").lower()
        if role == "assistant":
            continue
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        if role in {"user", "system", "tool"}:
            prompt_role = role if not prompt_parts else prompt_role
            prompt_parts.append(f"{role}: {text}")
    prompt_text = "\n".join(prompt_parts).strip()
    prompt_clean, target_clean = _canonical_prompt_target(prompt_text, target_text)
    if not prompt_clean or prompt_clean == target_clean:
        metadata = {
            "event_type": event_type,
            "tool_name": tool_name,
            "collector": input_json.get("collector") or input_json.get("source") or "curated_trace",
            "has_tool_input": bool(input_json.get("tool_input")),
        }
        prompt_text = "Emit the redacted assistant/tool training target for this audited trace metadata:\n" + _json_dumps(metadata)
        prompt_role = "user"
    return prompt_role, prompt_text


def curated_to_training_example(record: dict[str, Any], accepted_splits: set[str], min_quality: float) -> dict[str, Any] | None:
    split = str((record.get("split_assignment") or {}).get("split") or "train")
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    contamination = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
    secret_redaction = record.get("secret_redaction") if isinstance(record.get("secret_redaction"), dict) else {}
    quality_score = float(quality.get("overall") or quality.get("score") or 0.0)
    if split not in accepted_splits:
        return None
    if quality_score < min_quality or str(quality.get("label") or "").lower() == "reject":
        return None
    if contamination.get("status") == "contaminated" or secret_redaction.get("has_secret"):
        return None

    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    input_json = source.get("input_json") if isinstance(source.get("input_json"), dict) else {}
    target_json = source.get("target_json") if isinstance(source.get("target_json"), dict) else {}
    source_messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else []
    content = str(record.get("normalized_text") or "")
    event_type = str(input_json.get("event_type") or target_json.get("action_type") or "curated_trace")
    tool_name = input_json.get("tool_name") or target_json.get("tool_name")
    target_content = _trace_target_text(content, target_json)
    if not target_content.strip():
        return None
    first_role, prompt_content = _trace_prompt_text(
        source_messages=source_messages,
        input_json=input_json,
        event_type=event_type,
        tool_name=tool_name,
        target_text=target_content,
    )
    prompt_clean, target_clean = _canonical_prompt_target(prompt_content, target_content)
    if not prompt_clean or prompt_clean == target_clean:
        return None
    return {
        "bucket": str((source.get("metadata") or {}).get("bucket") or "curated_agentic_trace_2026"),
        "split": split,
        "source_date": (record.get("provenance") or {}).get("source_date"),
        "input_json": {
            "messages": [{"role": first_role, "content": prompt_content}],
            "event_type": event_type,
            "tool_name": tool_name,
            "tool_input": _redact_json_value(input_json.get("tool_input") or {}),
        },
        "target_json": {
            "content": target_content,
            "action_type": event_type,
            "tool_output": _redact_json_value(target_json.get("tool_output") or {}),
        },
        "lineage": {
            "source": "curation_layers_2026",
            "curated_id": record.get("curated_id"),
            "trace_id": ((source.get("metadata") or {}).get("lineage") or {}).get("trace_id")
            or (record.get("provenance") or {}).get("record_id")
            or record.get("curated_id"),
            "record_hash": (record.get("dedupe") or {}).get("canonical_sha256"),
            "path": (record.get("provenance") or {}).get("path"),
            "provenance": record.get("provenance") or {},
            "dedupe": record.get("dedupe") or {},
            "classifications": {
                "language": record.get("language") or {},
                "code": record.get("code") or {},
                "tools": record.get("tools") or {},
                "media": record.get("media") or {},
            },
        },
        "quality": {
            "score": quality_score,
            "label": quality.get("label"),
            "details": quality.get("dimensions") or {},
            "curation": quality,
        },
        "contamination": contamination,
    }


def export_training_jsonl(curated_path: Path, out_path: Path, accepted_splits: set[str], min_quality: float, limit: int = 0) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in _jsonl(curated_path):
            split = str((record.get("split_assignment") or {}).get("split") or "train")
            counts[f"seen_{split}"] += 1
            example = curated_to_training_example(record, accepted_splits, min_quality)
            if example is None:
                counts["rejected"] += 1
                continue
            handle.write(json.dumps(example, ensure_ascii=True, sort_keys=True) + "\n")
            counts[f"written_{example['split']}"] += 1
            written += 1
            if limit and written >= limit:
                break
    return {"records": written, "counts": dict(counts), "accepted_splits": sorted(accepted_splits), "min_quality": min_quality}


def _profile_get(profile: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = profile
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _profile_args(profile: dict[str, Any]) -> argparse.Namespace:
    curation_cfg = profile.get("curation_layers") if isinstance(profile.get("curation_layers"), dict) else {}
    return argparse.Namespace(
        input="",
        protected=_profile_get(profile, "contamination", "protected_path", default=None),
        source_name=str(profile.get("run_name") or profile.get("profile_name") or "omnicoder_2026_curation"),
        source_date=str(profile.get("source_date") or dt.datetime.now(dt.timezone.utc).date().isoformat()),
        validation_ratio=float(curation_cfg.get("validation_ratio", 0.03)),
        holdout_ratio=float(curation_cfg.get("holdout_ratio", 0.02)),
        lowercase=bool(curation_cfg.get("lowercase", False)),
        redact=not bool(curation_cfg.get("no_redact", False)),
        allow_refusal_boilerplate=bool(curation_cfg.get("allow_refusal_boilerplate", False)),
        allow_eval_holdout=bool(curation_cfg.get("allow_eval_holdout", False)),
        min_policy_quality=float(curation_cfg.get("min_policy_quality", 0.0)),
        require_media_artifacts=bool(curation_cfg.get("require_media_artifacts", False)),
    )


def curate_jsonl(input_path: str, out_path: str, rejected_path: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Compatibility API used by trace_orchestrator_2026.

    The rich canonical curation stream is written beside the requested output,
    while ``out_path`` stays compatible with the existing data-factory quality,
    contamination, SFT, and teacher-job stages.
    """
    output = Path(out_path)
    canonical_path = output.with_suffix(output.suffix + ".canonical.jsonl")
    args = _profile_args(profile)
    args.input = input_path
    stats = write_curated_jsonl(Path(input_path), canonical_path, args)
    quality_cfg = profile.get("quality") if isinstance(profile.get("quality"), dict) else {}
    curation_cfg = profile.get("curation_layers") if isinstance(profile.get("curation_layers"), dict) else {}
    accepted = curation_cfg.get("accepted_splits")
    if isinstance(accepted, list):
        accepted_splits = {str(item) for item in accepted}
    else:
        accepted_splits = {"train", "validation"}
    min_quality = float(curation_cfg.get("export_min_quality", quality_cfg.get("export_min_quality", 0.35)))
    exported = export_training_jsonl(canonical_path, output, accepted_splits, min_quality)

    rejected_count = 0
    Path(rejected_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(rejected_path).open("w", encoding="utf-8") as handle:
        for record in _jsonl(canonical_path):
            if curated_to_training_example(record, accepted_splits, min_quality) is None:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                rejected_count += 1
    return {
        "records": exported["records"],
        "rejected": rejected_count,
        "canonical_records": stats["records"],
        "canonical_path": str(canonical_path),
        "training_path": str(output),
        "rejected_path": rejected_path,
        "counts": {**stats.get("counts", {}), **exported.get("counts", {})},
    }


def manifest_payload(out_path: Path, stats: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "manifest_version": "curation_layers_2026.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": args.input,
        "output": str(out_path),
        "source_name": args.source_name,
        "source_date": args.source_date,
        "redacted": bool(args.redact),
        "allow_refusal_boilerplate": bool(getattr(args, "allow_refusal_boilerplate", False)),
        "allow_eval_holdout": bool(getattr(args, "allow_eval_holdout", False)),
        "min_policy_quality": float(getattr(args, "min_policy_quality", 0.0) or 0.0),
        "require_media_artifacts": bool(getattr(args, "require_media_artifacts", False)),
        "validation_ratio": float(args.validation_ratio),
        "holdout_ratio": float(args.holdout_ratio),
        "stats": stats,
        "output_sha256": stable_hash(out_path.read_text(encoding="utf-8", errors="ignore")) if out_path.exists() else None,
    }


def write_manifest_file(out_path: Path, payload: dict[str, Any]) -> Path:
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def write_manifest_to_postgres(export_name: str, out_path: Path, stats: dict[str, Any], manifest: dict[str, Any]) -> int:
    with transaction() as cur:
        cur.execute(
            """
            INSERT INTO curation_export_manifests (export_name, export_kind, output_path, sample_count, metadata)
            VALUES (%s, 'curated_jsonl', %s, %s, %s::jsonb)
            RETURNING manifest_id
            """,
            (export_name, str(out_path), int(stats["records"]), json.dumps(manifest, ensure_ascii=True)),
        )
        return int(cur.fetchone()[0])


def write_curated_to_postgres(curated_path: Path, export_name: str) -> dict[str, Any]:
    inserted = 0
    with transaction() as cur:
        for record in _jsonl(curated_path):
            cur.execute(
                """
                INSERT INTO curated_records (
                    curated_id, normalized_text, source_payload, normalization, secret_redaction,
                    language_classification, code_classification, tool_classification,
                    media_classification, quality, dedupe, contamination, provenance,
                    split_assignment
                )
                VALUES (
                    %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb
                )
                ON CONFLICT (curated_id) DO UPDATE
                SET normalized_text = EXCLUDED.normalized_text,
                    source_payload = EXCLUDED.source_payload,
                    normalization = EXCLUDED.normalization,
                    secret_redaction = EXCLUDED.secret_redaction,
                    language_classification = EXCLUDED.language_classification,
                    code_classification = EXCLUDED.code_classification,
                    tool_classification = EXCLUDED.tool_classification,
                    media_classification = EXCLUDED.media_classification,
                    quality = EXCLUDED.quality,
                    dedupe = EXCLUDED.dedupe,
                    contamination = EXCLUDED.contamination,
                    provenance = EXCLUDED.provenance,
                    split_assignment = EXCLUDED.split_assignment,
                    updated_at = now()
                RETURNING curated_record_id
                """,
                (
                    record["curated_id"],
                    record["normalized_text"],
                    json.dumps(record.get("source") or {}),
                    json.dumps(record.get("normalization") or {}),
                    json.dumps(record.get("secret_redaction") or {}),
                    json.dumps(record.get("language") or {}),
                    json.dumps(record.get("code") or {}),
                    json.dumps(record.get("tools") or {}),
                    json.dumps(record.get("media") or {}),
                    json.dumps(record.get("quality") or {}),
                    json.dumps(record.get("dedupe") or {}),
                    json.dumps(record.get("contamination") or {}),
                    json.dumps(record.get("provenance") or {}),
                    json.dumps(record.get("split_assignment") or {}),
                ),
            )
            curated_record_id = int(cur.fetchone()[0])
            _insert_detail_rows(cur, curated_record_id, record)
            inserted += 1
    return {"curated_records": inserted, "export_name": export_name}


def _insert_detail_rows(cur: Any, curated_record_id: int, record: dict[str, Any]) -> None:
    cur.execute("DELETE FROM curation_secret_findings WHERE curated_record_id=%s", (curated_record_id,))
    cur.execute("DELETE FROM curation_classifications WHERE curated_record_id=%s", (curated_record_id,))
    cur.execute("DELETE FROM curation_quality_dimensions WHERE curated_record_id=%s", (curated_record_id,))
    cur.execute("DELETE FROM curation_dedupe_signatures WHERE curated_record_id=%s", (curated_record_id,))
    cur.execute("DELETE FROM curation_contamination_labels WHERE curated_record_id=%s", (curated_record_id,))
    cur.execute("DELETE FROM curation_split_assignments WHERE curated_record_id=%s", (curated_record_id,))
    cur.execute("DELETE FROM source_provenance_records WHERE curated_record_id=%s", (curated_record_id,))

    for finding in record.get("secret_redaction", {}).get("findings", []):
        cur.execute(
            """
            INSERT INTO curation_secret_findings (curated_record_id, secret_type, secret_hash, start_offset, end_offset, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (curated_record_id, finding.get("secret_type"), finding.get("hash"), finding.get("start"), finding.get("end"), json.dumps(finding)),
        )
    for kind, payload in (
        ("language", record.get("language")),
        ("code", record.get("code")),
        ("tools", record.get("tools")),
        ("media", record.get("media")),
    ):
        cur.execute(
            """
            INSERT INTO curation_classifications (curated_record_id, classifier_kind, label, confidence, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            """,
            (curated_record_id, kind, _classification_label(kind, payload or {}), _classification_confidence(payload or {}), json.dumps(payload or {})),
        )
    for name, value in (record.get("quality", {}).get("dimensions") or {}).items():
        if isinstance(value, (int, float)):
            cur.execute(
                """
                INSERT INTO curation_quality_dimensions (curated_record_id, dimension_name, dimension_value, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (curated_record_id, name, float(value), json.dumps(record.get("quality") or {})),
            )
    dedupe = record.get("dedupe") or {}
    for name in ("exact_sha256", "redacted_sha256", "canonical_sha256", "prefix_sha256", "suffix_sha256", "shingle_sha256"):
        cur.execute(
            """
            INSERT INTO curation_dedupe_signatures (curated_record_id, signature_type, signature_value, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (curated_record_id, name, dedupe.get(name), json.dumps(dedupe)),
        )
    contamination = record.get("contamination") or {}
    cur.execute(
        """
        INSERT INTO curation_contamination_labels (curated_record_id, status, match_type, score, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        """,
        (
            curated_record_id,
            contamination.get("status") or "clean",
            contamination.get("match_type") or "none",
            float(contamination.get("score") or 0.0),
            json.dumps(contamination),
        ),
    )
    split = record.get("split_assignment") or {}
    cur.execute(
        """
        INSERT INTO curation_split_assignments (curated_record_id, split_name, reason, metadata)
        VALUES (%s, %s, %s, %s::jsonb)
        """,
        (curated_record_id, split.get("split") or "train", split.get("reason"), json.dumps(split)),
    )
    provenance = record.get("provenance") or {}
    cur.execute(
        """
        INSERT INTO source_provenance_records (
            curated_record_id, source_name, source_uri, source_date, license_id,
            path, line_number, record_id, raw_record_hash, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (
            curated_record_id,
            provenance.get("source_name"),
            provenance.get("source_uri"),
            provenance.get("source_date") or None,
            provenance.get("license_id"),
            provenance.get("path"),
            provenance.get("line_number"),
            str(provenance.get("record_id")) if provenance.get("record_id") is not None else None,
            provenance.get("raw_record_hash"),
            json.dumps(provenance),
        ),
    )


def _classification_label(kind: str, payload: dict[str, Any]) -> str:
    if kind == "language":
        return str(payload.get("language") or "unknown")
    if kind == "code":
        return str(payload.get("code_language") or "none")
    if kind == "tools":
        families = payload.get("tool_families") if isinstance(payload.get("tool_families"), list) else []
        return ",".join(str(item) for item in families[:4]) or "none"
    if kind == "media":
        families = payload.get("media_families") if isinstance(payload.get("media_families"), list) else []
        return ",".join(str(item) for item in families[:4]) or "unknown"
    return "unknown"


def _classification_confidence(payload: dict[str, Any]) -> float | None:
    value = payload.get("confidence")
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate 2025-2026 dataset records into normalized JSONL with raw PostgreSQL-compatible metadata")
    sub = parser.add_subparsers(dest="command")

    curate = sub.add_parser("curate", help="Write canonical curated JSONL")
    curate.add_argument("--input", required=True, help="Input JSONL/JSON/text directory or file")
    curate.add_argument("--out", default="weights/data_factory/curated_2026.jsonl")
    curate.add_argument("--protected", default=None, help="Optional protected/eval JSONL used for exact canonical contamination labels")
    curate.add_argument("--source-name", default="omnicoder_2026_curation")
    curate.add_argument("--source-date", default=None)
    curate.add_argument("--validation-ratio", type=float, default=0.03)
    curate.add_argument("--holdout-ratio", type=float, default=0.02)
    curate.add_argument("--lowercase", action="store_true", help="Lowercase during normalization; off by default to preserve code")
    curate.add_argument("--no-redact", dest="redact", action="store_false", help="Keep original text while still flagging secret findings")
    curate.set_defaults(redact=True)
    curate.add_argument("--allow-refusal-boilerplate", action="store_true", help="Permit refusal/alignment-negative rows; off by default for capability-first curation")
    curate.add_argument("--allow-eval-holdout", action="store_true", help="Permit eval/public-dev/protected benchmark rows into curated train candidates")
    curate.add_argument("--allow-dataset-integrity-issues", action="store_true", help="Permit rows flagged by dataset_integrity_2026; default is hard reject")
    curate.add_argument("--skip-integrity-artifact-scan", action="store_true", help="Skip local media byte marker scans; text/metadata integrity checks still run")
    curate.add_argument("--max-integrity-artifact-bytes", type=int, default=64 * 1024 * 1024)
    curate.add_argument("--min-policy-quality", type=float, default=0.0, help="Additional curation_policy_2026 quality floor during canonical curation")
    curate.add_argument("--require-media-artifacts", action="store_true", help="Reject media rows without usable artifact refs")
    curate.add_argument("--manifest", action="store_true", help="Write sidecar manifest JSON")
    curate.add_argument("--postgres", action="store_true", help="Upsert curated records and detail rows using raw psycopg2")
    curate.add_argument("--export-name", default="curation_layers_2026")

    export_training = sub.add_parser("export-training", help="Convert curated JSONL into data-factory training JSONL")
    export_training.add_argument("--input", required=True)
    export_training.add_argument("--out", default="weights/data_factory/curated_training_2026.jsonl")
    export_training.add_argument("--splits", default="train,validation", help="Comma-separated accepted curated splits")
    export_training.add_argument("--min-quality", type=float, default=0.35)
    export_training.add_argument("--limit", type=int, default=0)

    stats_cmd = sub.add_parser("stats", help="Summarize curated JSONL")
    stats_cmd.add_argument("--input", required=True)
    argv = sys.argv[1:]
    commands = {"curate", "export-training", "stats"}
    if not argv:
        argv = ["--help"]
    if argv and argv[0] not in commands and argv[0] not in {"-h", "--help"}:
        argv = ["curate", *argv]
    args = parser.parse_args(argv)

    if args.command == "export-training":
        splits = {item.strip() for item in args.splits.split(",") if item.strip()}
        stats = export_training_jsonl(Path(args.input), Path(args.out), splits, args.min_quality, args.limit)
        result = {"status": "ok", "out": args.out, "stats": stats}
    elif args.command == "stats":
        counts: Counter[str] = Counter()
        for record in _jsonl(Path(args.input)):
            counts[f"split_{(record.get('split_assignment') or {}).get('split') or 'unknown'}"] += 1
            counts[f"quality_{(record.get('quality') or {}).get('label') or 'unknown'}"] += 1
            counts[f"contamination_{(record.get('contamination') or {}).get('status') or 'unknown'}"] += 1
            for family in (record.get("media") or {}).get("media_families") or []:
                counts[f"media_{family}"] += 1
        result = {"status": "ok", "input": args.input, "counts": dict(counts)}
    else:
        out_path = Path(args.out)
        stats = write_curated_jsonl(Path(args.input), out_path, args)
        manifest = manifest_payload(out_path, stats, args)
        result = {"status": "ok", "out": str(out_path), "stats": stats, "manifest": None, "postgres": False}
        if args.manifest:
            result["manifest"] = str(write_manifest_file(out_path, manifest))
        if args.postgres:
            result["postgres"] = True
            result["postgres_result"] = write_curated_to_postgres(out_path, args.export_name)
            result["postgres_manifest_id"] = write_manifest_to_postgres(args.export_name, out_path, stats, manifest)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

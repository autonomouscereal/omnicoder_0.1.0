from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
KNOWN_MODALITIES = {"text", "code", "tool", "image", "video", "audio", "music", "long_context", "math", "ocr"}
MEDIA_MODALITIES = {"image", "video", "audio", "music", "ocr"}
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


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


def stable_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


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
    if prompt and not target and (row.get("dataset_family") or row.get("training_bucket")):
        target = prompt
        prompt = f"Learn this high-quality {normalize_modality(row.get('modality') or row.get('dataset_family')) or 'text'} training example."
    return prompt, target


def artifact_refs(row: dict[str, Any], *, limit: int = 32) -> list[str]:
    refs: list[str] = []
    containers: list[dict[str, Any]] = [row]
    for nested_key in ("input_json", "target_json", "output_json"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in ("artifact_refs", "artifacts", "artifact_paths", "media_paths", "media_refs"):
            value = container.get(key)
            if isinstance(value, list):
                for item in value:
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
    return ""


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
    for ref in refs[:8]:
        if ref.startswith(("http://", "https://", "s3://", "hf://")):
            existing += 1
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
    artifact_score, artifact_reasons = artifact_quality(refs or [], modality, require_media_artifacts=cfg.require_media_artifacts)
    source_quality = 1.0 if existing_quality is None else max(0.0, min(1.0, float(existing_quality)))
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
    if len(target.strip()) < cfg.min_target_chars:
        reasons.append("target_too_short")
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
    label = "reject" if reasons and any(r in reasons for r in ("missing_prompt", "missing_target", "secret_marker", "below_min_quality", "media_artifact_ref_not_found", "missing_media_artifact_ref")) else "candidate"
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
        "policy": {
            "reject_refusal_boilerplate": cfg.reject_refusal_boilerplate,
            "reject_eval_holdout": cfg.reject_eval_holdout,
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
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(run_agent(args), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

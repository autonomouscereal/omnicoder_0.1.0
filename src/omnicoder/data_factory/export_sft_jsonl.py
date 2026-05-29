from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory import curation_layers_2026
from omnicoder.data_factory.curation_policy_2026 import CurationPolicyConfig, audit_training_record
from omnicoder.data_factory.postgres import transaction

NONTRAIN_CONTAMINATION_STATUSES = {
    "benchmark_leak",
    "benchmark_marker",
    "contaminated",
    "dirty",
    "eval_holdout",
    "pending",
    "protected_eval",
    "public_dev_eval",
    "quarantine",
    "rejected",
    "suspect",
    "unknown",
}
EVAL_OR_QUARANTINE_MARKERS = {
    "benchmark",
    "benchmark_marker",
    "canary",
    "eval",
    "eval_holdout",
    "fixture",
    "hidden_eval",
    "protected_eval",
    "public_dev",
    "quarantine",
    "smoke",
}
SCALAR_OR_PUNCTUATION_RE = re.compile(r"^[\W_]*[A-Za-z0-9]?[.\-!?]*$")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def _decode_jsonb(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _messages_from(record: dict[str, Any]) -> list[dict[str, str]]:
    input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    messages = input_json.get("messages")
    if isinstance(messages, list) and messages:
        normalized = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role") or "user")
                content = str(message.get("content") or "")
                if content:
                    normalized.append({"role": role, "content": content})
    else:
        content = input_json.get("content") or input_json.get("prompt") or record.get("text") or ""
        normalized = [{"role": "user", "content": str(content)}] if content else []
    assistant = target_json.get("content") or target_json.get("completion") or target_json.get("answer")
    if assistant:
        normalized.append({"role": "assistant", "content": str(assistant)})
    elif target_json.get("artifact_path"):
        normalized.append({"role": "assistant", "content": json.dumps(target_json, ensure_ascii=True, sort_keys=True)})
    return normalized


def _trace_id(record: dict[str, Any]) -> str:
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    for key in ("trace_id", "session_id", "curated_id", "record_hash"):
        value = lineage.get(key) or record.get(key)
        if value:
            return str(value)
    return json.dumps(lineage or record, ensure_ascii=True, sort_keys=True, default=str)[:128]


def _quality_score(record: dict[str, Any]) -> float | None:
    candidates: list[Any] = []
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    if quality:
        candidates.extend([quality.get("score"), quality.get("overall"), quality.get("quality")])
    candidates.extend([record.get("quality_score"), record.get("score"), record.get("reward")])
    for value in candidates:
        if value in (None, ""):
            continue
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return None


def _contamination_status(record: dict[str, Any]) -> str:
    contamination = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
    containers = [record, contamination]
    for key in ("metadata", "curation", "lineage", "quality"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in (
            "status",
            "contamination_status",
            "decontamination_status",
            "protected_benchmark_scan",
            "benchmark_contamination_status",
        ):
            value = container.get(key)
            if value not in (None, "", [], {}):
                return str(value).strip().lower()
    return "unknown"


def _metadata_text(record: dict[str, Any]) -> str:
    selected = {
        key: record.get(key)
        for key in (
            "bucket",
            "split",
            "split_name",
            "source_id",
            "dataset_name",
            "dataset_family",
            "task_id",
            "training_bucket",
        )
        if record.get(key) not in (None, "", [], {})
    }
    for key in ("lineage", "metadata", "curation", "quality", "contamination"):
        value = record.get(key)
        if isinstance(value, dict):
            selected[key] = value
    return json.dumps(selected, ensure_ascii=True, sort_keys=True, default=str).lower()


def _rejection_reasons(record: dict[str, Any], min_quality: float, allow_contaminated: bool) -> list[str]:
    reasons: list[str] = []
    secret_redaction = record.get("secret_redaction") if isinstance(record.get("secret_redaction"), dict) else {}
    if secret_redaction.get("has_secret"):
        reasons.append("secret")
    score = _quality_score(record)
    if score is None:
        reasons.append("missing_quality")
    elif score < min_quality:
        reasons.append("below_min_quality")
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    if str(quality.get("label") or "").lower() == "reject":
        reasons.append("quality_reject")
    details = quality.get("details") if isinstance(quality.get("details"), dict) else {}
    if float(details.get("secret_penalty") or 0.0) > 0.0:
        reasons.append("secret_penalty")
    status = _contamination_status(record)
    if status in NONTRAIN_CONTAMINATION_STATUSES and not (allow_contaminated and status == "contaminated"):
        reasons.append(f"contamination:{status}")
    metadata = _metadata_text(record)
    for marker in EVAL_OR_QUARANTINE_MARKERS:
        if marker in metadata:
            reasons.append(f"eval_or_quarantine:{marker}")
            break
    if not _messages_from(record):
        reasons.append("empty_messages")
    return sorted(set(reasons))


def _message_events_from(record: dict[str, Any]) -> list[dict[str, str]]:
    input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    messages = input_json.get("messages")
    events: list[dict[str, str]] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "").strip()
            if role in {"user", "assistant", "system", "tool"} and content:
                events.append({"role": role, "content": content})
    content = ""
    if not events:
        content = str(input_json.get("content") or record.get("text") or "").strip()
        if content:
            events.append({"role": "user", "content": content})
    target = str(target_json.get("content") or target_json.get("completion") or target_json.get("answer") or "").strip()
    if target and target != content:
        events.append({"role": "assistant", "content": target})
    tool_name = input_json.get("tool_name")
    tool_input = input_json.get("tool_input")
    if tool_name or tool_input:
        events.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool_call": {"tool": tool_name, "arguments": tool_input if tool_input is not None else {}}},
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )
    tool_output = target_json.get("tool_output")
    if tool_output not in (None, "", {}, []):
        events.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {"tool": tool_name, "result": tool_output},
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    tool_calls = record.get("tool_calls") if isinstance(record.get("tool_calls"), list) else []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            events.append({"role": "assistant", "content": json.dumps({"tool_call": tool_call}, ensure_ascii=True, sort_keys=True)})
    tool_results = record.get("tool_results") if isinstance(record.get("tool_results"), list) else []
    for tool_result in tool_results:
        if isinstance(tool_result, dict):
            events.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=True, sort_keys=True)})
    return events


def _trim_events(events: list[dict[str, str]]) -> list[dict[str, str]]:
    max_events = 0
    max_chars = 0
    try:
        import os

        max_events = int(os.environ.get("OMNICODER_SFT_MAX_EVENTS_PER_TRACE", "192") or 0)
        max_chars = int(os.environ.get("OMNICODER_SFT_MAX_EVENT_CHARS", "6000") or 0)
    except Exception:
        max_events = 192
        max_chars = 6000
    if max_chars > 0:
        events = [{**event, "content": event["content"][:max_chars]} for event in events]
    if max_events > 0 and len(events) > max_events:
        head = max_events // 2
        tail = max_events - head
        events = events[:head] + events[-tail:]
    return events


def _record_order(record: dict[str, Any], fallback: int) -> tuple[str, int]:
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    timestamp = str(record.get("created_at") or lineage.get("created_at") or record.get("source_date") or "")
    raw_step = (
        lineage.get("step_index")
        or lineage.get("source_index")
        or record.get("step_index")
        or record.get("line_number")
        or fallback
    )
    try:
        step = int(raw_step)
    except (TypeError, ValueError):
        step = fallback
    return (timestamp, step)


def eligible(record: dict[str, Any], min_quality: float, allow_contaminated: bool) -> bool:
    return not _rejection_reasons(record, min_quality, allow_contaminated)


def contains_secret_payload(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return bool(curation_layers_2026.redact_secrets(serialized).get("has_secret"))


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages if message["role"] in {"system", "user", "tool"})[:20000]


def _modality_from_record(record: dict[str, Any]) -> str:
    modalities = record.get("modalities")
    if isinstance(modalities, list) and modalities:
        return str(modalities[0])
    for key in ("modality", "bucket", "training_bucket"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            lowered = value.lower()
            for candidate in ("video", "image", "music", "audio", "speech", "tts", "ocr", "tool", "code", "math", "text"):
                if candidate in lowered:
                    return candidate
    return "text"


def _policy_rejection_reasons(record: dict[str, Any], messages: list[dict[str, str]], min_quality: float) -> list[str]:
    modality = _modality_from_record(record)
    has_tool_trace = any(message.get("role") == "tool" or '"tool_call"' in message.get("content", "") for message in messages)
    if modality == "text" and has_tool_trace:
        modality = "tool"
    audit_record = dict(record)
    if has_tool_trace and not audit_record.get("tool_calls"):
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[Any] = []
        for message in messages:
            content = message.get("content", "")
            if message.get("role") == "assistant" and '"tool_call"' in content:
                try:
                    payload = json.loads(content)
                except Exception:
                    payload = {}
                call = payload.get("tool_call") if isinstance(payload, dict) else None
                if isinstance(call, dict):
                    tool_calls.append(call)
            elif message.get("role") == "tool":
                try:
                    payload = json.loads(content)
                except Exception:
                    payload = {"result": content}
                tool_results.append(payload)
        if tool_calls:
            audit_record["tool_calls"] = tool_calls
        if tool_results:
            audit_record["tool_results"] = tool_results
    audit = audit_training_record(
        audit_record,
        prompt=_prompt_text(messages),
        target=_assistant_text(messages),
        modality=modality,
        existing_quality=_quality_score(record),
        config=CurationPolicyConfig(
            reject_refusal_boilerplate=True,
            reject_placeholder_junk=modality not in {"code", "tool"},
            reject_eval_holdout=True,
            reject_dataset_integrity_issues=True,
            scan_integrity_artifacts=False,
            min_quality_score=float(min_quality),
        ),
    )
    if audit.get("accepted", False):
        return []
    return [f"policy:{reason}" for reason in audit.get("reasons") or ["unknown"]]


def _compact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if not content:
            continue
        if compacted and compacted[-1]["role"] == role and compacted[-1]["content"] == content:
            continue
        compacted.append({"role": role, "content": content})
    return compacted


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _visible_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def _assistant_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages if message["role"] == "assistant")


def _has_media_payload_text(messages: list[dict[str, str]]) -> bool:
    text = "\n".join(message["content"] for message in messages)
    return any(marker in text for marker in ("artifact_path", "artifact_tokens", "media_tokens", "image_tokens", "video_tokens", "audio_tokens"))


def _message_quality_reasons(messages: list[dict[str, str]], metadata: dict[str, Any]) -> list[str]:
    """Reject direct SFT exports that are too small to be useful training data."""
    reasons: list[str] = []
    min_assistant_chars = _env_int("OMNICODER_SFT_MIN_DIRECT_ASSISTANT_CHARS", 32)
    min_assistant_tokens = _env_int("OMNICODER_SFT_MIN_DIRECT_ASSISTANT_TOKENS", 6)
    roles = {message["role"] for message in messages}
    assistant_text = _assistant_text(messages).strip()
    has_media_payload = _has_media_payload_text(messages)
    if "user" not in roles:
        reasons.append("missing_user_turn")
    if "assistant" not in roles:
        reasons.append("missing_assistant_turn")
    if not has_media_payload:
        if len(assistant_text) < min_assistant_chars:
            reasons.append("assistant_target_too_short")
        if _visible_token_count(assistant_text) < min_assistant_tokens:
            reasons.append("assistant_target_too_few_tokens")
        if SCALAR_OR_PUNCTUATION_RE.fullmatch(assistant_text[:64] or "."):
            reasons.append("assistant_target_scalar_or_punctuation")
    metadata["message_quality_gate"] = {
        "min_assistant_chars": min_assistant_chars,
        "min_assistant_tokens": min_assistant_tokens,
        "has_media_payload": has_media_payload,
        "assistant_chars": len(assistant_text),
        "assistant_tokens": _visible_token_count(assistant_text),
    }
    return reasons


def _trace_quality_reasons(messages: list[dict[str, str]], metadata: dict[str, Any]) -> list[str]:
    """Reject toy harness traces before they become SFT rows."""
    reasons: list[str] = []
    min_messages = _env_int("OMNICODER_SFT_MIN_TRACE_MESSAGES", 4)
    min_assistant_chars = _env_int("OMNICODER_SFT_MIN_ASSISTANT_CHARS", 128)
    min_assistant_tokens = _env_int("OMNICODER_SFT_MIN_ASSISTANT_TOKENS", 24)
    require_agent_loop = _env_truthy("OMNICODER_SFT_REQUIRE_AGENT_LOOP", True)
    roles = {message["role"] for message in messages}
    assistant_text = _assistant_text(messages)
    has_tool_call = any(message["role"] == "assistant" and '"tool_call"' in message["content"] for message in messages)
    has_tool_result = any(message["role"] == "tool" for message in messages)
    has_media_artifact = _has_media_payload_text(messages)
    if len(messages) < min_messages:
        reasons.append("trace_too_few_messages")
    if "user" not in roles:
        reasons.append("missing_user_turn")
    if "assistant" not in roles:
        reasons.append("missing_assistant_turn")
    if len(assistant_text.strip()) < min_assistant_chars:
        reasons.append("assistant_trace_too_short")
    if _visible_token_count(assistant_text) < min_assistant_tokens:
        reasons.append("assistant_trace_too_few_tokens")
    if SCALAR_OR_PUNCTUATION_RE.fullmatch(assistant_text.strip()[:64] or "."):
        reasons.append("assistant_trace_scalar_or_punctuation")
    if require_agent_loop and not (has_tool_call and has_tool_result) and not has_media_artifact:
        reasons.append("missing_agent_observe_loop")
    metadata["trace_quality_gate"] = {
        "min_messages": min_messages,
        "min_assistant_chars": min_assistant_chars,
        "min_assistant_tokens": min_assistant_tokens,
        "require_agent_loop": require_agent_loop,
        "has_tool_call": has_tool_call,
        "has_tool_result": has_tool_result,
        "has_media_artifact": has_media_artifact,
        "assistant_chars": len(assistant_text.strip()),
        "assistant_tokens": _visible_token_count(assistant_text),
    }
    return reasons


def export_offline(input_path: Path, out_path: Path, min_quality: float, allow_contaminated: bool, limit: int = 0) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in _jsonl(input_path):
            if not eligible(record, min_quality, allow_contaminated):
                continue
            payload = {
                "messages": _messages_from(record),
                "metadata": {
                    "bucket": record.get("bucket"),
                    "split": record.get("split"),
                    "source_date": record.get("source_date"),
                    "lineage": record.get("lineage", {}),
                    "quality": record.get("quality", {}),
                },
            }
            if _message_quality_reasons(payload["messages"], payload["metadata"]):
                continue
            if contains_secret_payload(payload):
                continue
            if _policy_rejection_reasons(record, payload["messages"], min_quality):
                continue
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def export_trace_conversations(input_path: Path, out_path: Path, min_quality: float, allow_contaminated: bool, limit: int = 0) -> int:
    grouped: dict[str, dict[str, Any]] = {}
    for fallback_index, record in enumerate(_jsonl(input_path), 1):
        trace_id = _trace_id(record)
        group = grouped.setdefault(
            trace_id,
            {
                "messages": [],
                "metadata": {
                    "trace_id": trace_id,
                    "bucket": record.get("bucket"),
                    "split": record.get("split"),
                    "source_date": record.get("source_date"),
                    "record_count": 0,
                    "lineages": [],
                    "quality_scores": [],
                },
                "rejected": False,
                "rejection_reasons": [],
            },
        )
        reasons = _rejection_reasons(record, min_quality, allow_contaminated)
        if reasons:
            group["rejected"] = True
            group["rejection_reasons"].extend(reasons)
            continue
        group_events = group.setdefault("events", [])
        max_events = 0
        try:
            import os

            max_events = int(os.environ.get("OMNICODER_SFT_MAX_EVENTS_PER_TRACE", "192") or 0)
        except Exception:
            max_events = 192
        if not max_events or len(group_events) < max_events:
            group_events.append({"order": _record_order(record, fallback_index), "messages": _trim_events(_message_events_from(record))})
        group["metadata"]["record_count"] += 1
        if len(group["metadata"]["lineages"]) < 32:
            group["metadata"]["lineages"].append(record.get("lineage", {}))
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        if quality:
            group["metadata"]["quality_scores"].append(float(quality.get("score") or quality.get("overall") or 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for group in grouped.values():
            if group.get("rejected"):
                continue
            ordered_messages: list[dict[str, str]] = []
            events = group.pop("events", [])
            if events:
                for event in sorted(events, key=lambda item: item["order"]):
                    ordered_messages.extend(event["messages"])
            else:
                ordered_messages = group["messages"]
            messages = _compact_messages(ordered_messages)
            roles = {message["role"] for message in messages}
            if len(messages) < 2 or "assistant" not in roles:
                continue
            trace_quality_reasons = _trace_quality_reasons(messages, group["metadata"])
            if trace_quality_reasons:
                continue
            scores = group["metadata"].pop("quality_scores", [])
            if scores:
                group["metadata"]["quality"] = {
                    "min": min(scores),
                    "avg": sum(scores) / len(scores),
                    "max": max(scores),
                }
            payload = {"messages": messages, "metadata": group["metadata"]}
            if contains_secret_payload(payload):
                continue
            policy_record = {
                "bucket": group["metadata"].get("bucket"),
                "split": group["metadata"].get("split"),
                "source_date": group["metadata"].get("source_date"),
                "lineage": {"trace_lineages": group["metadata"].get("lineages", [])},
                "quality": group["metadata"].get("quality", {}),
            }
            if isinstance(policy_record["quality"], dict) and "score" not in policy_record["quality"]:
                policy_record["quality"]["score"] = policy_record["quality"].get("avg")
            if _policy_rejection_reasons(policy_record, messages, min_quality):
                continue
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def export_postgres(out_path: Path, split: str, bucket: str | None, min_quality: float, limit: int = 0) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT te.training_example_id, te.bucket, te.input_json, te.target_json, te.source_date, te.lineage,
               MAX(qs.score_value) AS quality_score
        FROM training_examples te
        LEFT JOIN quality_scores qs
          ON qs.target_type='training_example'
         AND qs.target_id=te.training_example_id
         AND qs.score_name='heuristic_quality'
        WHERE te.split_name=%s
          AND (%s IS NULL OR te.bucket=%s)
        GROUP BY te.training_example_id
        HAVING MAX(qs.score_value) IS NOT NULL
           AND MAX(qs.score_value) >= %s
        ORDER BY te.training_example_id
    """
    if limit:
        query += " LIMIT %s"
        params: tuple[Any, ...] = (split, bucket, bucket, min_quality, limit)
    else:
        params = (split, bucket, bucket, min_quality)
    count = 0
    with transaction() as cur, out_path.open("w", encoding="utf-8") as handle:
        cur.execute(query, params)
        for row in cur.fetchall():
            record = {
                "bucket": row[1],
                "input_json": _decode_jsonb(row[2]),
                "target_json": _decode_jsonb(row[3]),
                "source_date": row[4].isoformat() if row[4] else None,
                "lineage": _decode_jsonb(row[5]) if row[5] else {},
                "quality": {"score": float(row[6])},
            }
            payload = {"messages": _messages_from(record), "metadata": {"training_example_id": int(row[0]), **record}}
            if _rejection_reasons(record, min_quality, allow_contaminated=False):
                continue
            if _message_quality_reasons(payload["messages"], payload["metadata"]):
                continue
            if contains_secret_payload(payload):
                continue
            if _policy_rejection_reasons(record, payload["messages"], min_quality):
                continue
            handle.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
            count += 1
    return count


def write_manifest(export_name: str, out_path: Path, count: int, source: str) -> None:
    with transaction() as cur:
        cur.execute(
            """
            INSERT INTO export_manifests (export_name, export_kind, output_path, sample_count, metadata)
            VALUES (%s, 'sft_jsonl', %s, %s, %s::jsonb)
            """,
            (export_name, str(out_path), int(count), json.dumps({"source": source})),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export data-factory records to chat SFT JSONL")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--postgres", action="store_true")
    parser.add_argument("--out", default="weights/data_factory/sft_train.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--allow-contaminated", action="store_true")
    parser.add_argument("--group-traces", action="store_true", help="Group eligible trace rows into multi-turn conversations by trace_id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--export-name", default="sft_export_2026")
    args = parser.parse_args()

    out_path = Path(args.out)
    if args.postgres:
        count = export_postgres(out_path, args.split, args.bucket, args.min_quality, args.limit)
        source_name = "postgres"
    elif args.group_traces:
        count = export_trace_conversations(Path(args.input), out_path, args.min_quality, args.allow_contaminated, args.limit)
        source_name = str(args.input)
    else:
        count = export_offline(Path(args.input), out_path, args.min_quality, args.allow_contaminated, args.limit)
        source_name = str(args.input)
    if args.manifest:
        write_manifest(args.export_name, out_path, count, source_name)
    print(json.dumps({"status": "ok", "out": str(out_path), "records": count, "source": source_name}, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.postgres import transaction


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
    if events:
        return events
    content = str(input_json.get("content") or record.get("text") or "").strip()
    if content:
        events.append({"role": "user", "content": content})
    target = str(target_json.get("content") or target_json.get("completion") or target_json.get("answer") or "").strip()
    if target and target != content:
        events.append({"role": "assistant", "content": target})
    return events


def eligible(record: dict[str, Any], min_quality: float, allow_contaminated: bool) -> bool:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    if quality:
        if float(quality.get("score") or 0.0) < min_quality:
            return False
        if str(quality.get("label") or "").lower() == "reject":
            return False
        details = quality.get("details") if isinstance(quality.get("details"), dict) else {}
        if float(details.get("secret_penalty") or 0.0) > 0.0:
            return False
    contamination = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
    if not allow_contaminated and contamination.get("status") == "contaminated":
        return False
    return bool(_messages_from(record))


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
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def export_trace_conversations(input_path: Path, out_path: Path, min_quality: float, allow_contaminated: bool, limit: int = 0) -> int:
    grouped: dict[str, dict[str, Any]] = {}
    for record in _jsonl(input_path):
        if not eligible(record, min_quality, allow_contaminated):
            continue
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
            },
        )
        group["messages"].extend(_message_events_from(record))
        group["metadata"]["record_count"] += 1
        group["metadata"]["lineages"].append(record.get("lineage", {}))
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        if quality:
            group["metadata"]["quality_scores"].append(float(quality.get("score") or quality.get("overall") or 0.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for group in grouped.values():
            messages = _compact_messages(group["messages"])
            roles = {message["role"] for message in messages}
            if len(messages) < 2 or "assistant" not in roles:
                continue
            scores = group["metadata"].pop("quality_scores", [])
            if scores:
                group["metadata"]["quality"] = {
                    "min": min(scores),
                    "avg": sum(scores) / len(scores),
                    "max": max(scores),
                }
            handle.write(json.dumps({"messages": messages, "metadata": group["metadata"]}, ensure_ascii=True) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def export_postgres(out_path: Path, split: str, bucket: str | None, min_quality: float, limit: int = 0) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT te.training_example_id, te.bucket, te.input_json, te.target_json, te.source_date, te.lineage,
               COALESCE(MAX(qs.score_value), 1.0) AS quality_score
        FROM training_examples te
        LEFT JOIN quality_scores qs
          ON qs.target_type='training_example'
         AND qs.target_id=te.training_example_id
         AND qs.score_name='heuristic_quality'
        WHERE te.split_name=%s
          AND (%s IS NULL OR te.bucket=%s)
        GROUP BY te.training_example_id
        HAVING COALESCE(MAX(qs.score_value), 1.0) >= %s
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
            handle.write(json.dumps({"messages": _messages_from(record), "metadata": {"training_example_id": int(row[0]), **record}}, ensure_ascii=True, default=str) + "\n")
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

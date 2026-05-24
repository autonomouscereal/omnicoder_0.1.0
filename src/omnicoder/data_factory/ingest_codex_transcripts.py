from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.ingest_2026 import stable_hash, write_jsonl
from omnicoder.data_factory.postgres import insert_agent_run, insert_agent_step, insert_dataset, insert_training_example


def iter_transcript_events(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for item in paths:
        if not item.is_file() or item.suffix.lower() not in {".jsonl", ".json", ".txt", ".log"}:
            continue
        text = item.read_text(encoding="utf-8", errors="surrogateescape")
        if item.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except Exception as exc:
                yield {"type": "parse_error", "error": str(exc), "path": str(item)}
                continue
            records = payload if isinstance(payload, list) else payload.get("items", payload.get("events", [payload])) if isinstance(payload, dict) else []
            for idx, event in enumerate(records):
                if isinstance(event, dict):
                    event.setdefault("path", str(item))
                    event.setdefault("line", idx + 1)
                    yield event
            continue
        for idx, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                event = {"type": "text", "content": line}
            if isinstance(event, dict):
                event.setdefault("path", str(item))
                event.setdefault("line", idx + 1)
                yield event


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _first_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "arguments", "output", "result"):
            if key in value:
                text = _first_text(value[key])
                if text:
                    return text
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return "" if value is None else str(value)


def normalize_event(event: dict[str, Any], bucket: str, split: str, source_date: str | None) -> dict[str, Any] | None:
    item = event.get("item") if isinstance(event.get("item"), dict) else event
    event_type = str(event.get("type") or item.get("type") or item.get("event_type") or "codex_event")
    role = str(item.get("role") or event.get("role") or ("tool" if "tool" in event_type.lower() else "assistant")).lower()
    if role not in {"user", "assistant", "system", "tool"}:
        role = "assistant"
    tool_name = item.get("tool_name") or item.get("name") or event.get("tool_name")
    content = _first_text(item.get("content") or item.get("message") or event.get("content") or event.get("message") or item)
    if not content.strip() and not tool_name:
        return None
    trace_id = str(event.get("session_id") or event.get("conversation_id") or event.get("trace_id") or stable_hash(str(event.get("path"))))
    event_hash = stable_hash(json.dumps(event, ensure_ascii=True, sort_keys=True, default=str))
    return {
        "bucket": bucket,
        "split": split,
        "source_date": source_date,
        "input_json": {
            "messages": [{"role": role, "content": content}],
            "event_type": event_type,
            "tool_name": tool_name,
            "tool_input": item.get("arguments") or item.get("input") or item.get("tool_input") or {},
        },
        "target_json": {
            "content": content,
            "action_type": event_type,
            "tool_output": item.get("output") or item.get("result") or item.get("tool_output") or {},
        },
        "lineage": {
            "source": "codex_transcript",
            "trace_id": trace_id,
            "record_hash": event_hash,
            "path": event.get("path"),
            "line": event.get("line"),
            "created_at": event.get("timestamp") or event.get("created_at"),
        },
    }


def build_records(path: Path, bucket: str, split: str, source_date: str | None, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in iter_transcript_events(path):
        row = normalize_event(event, bucket, split, source_date)
        if row is not None:
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def ingest_postgres(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_id = insert_dataset(
        name=args.dataset_name,
        namespace=args.namespace,
        source_uri=str(Path(args.input).resolve()),
        source_date=args.source_date,
        license_id=args.license,
        terms={"policy": "2025_2026_only", "source": "codex_transcript"},
        metadata={"bucket": args.bucket},
    )
    runs: dict[str, int] = {}
    step_counts: dict[str, int] = {}
    for row in rows:
        trace_id = str(row["lineage"]["trace_id"])
        if trace_id not in runs:
            runs[trace_id] = insert_agent_run(trace_id=trace_id, harness=args.harness, dataset_id=dataset_id, task_family=args.bucket)
            step_counts[trace_id] = 0
        step_index = step_counts[trace_id]
        step_counts[trace_id] += 1
        message = row["input_json"]["messages"][0]
        insert_agent_step(
            agent_run_id=runs[trace_id],
            step_index=step_index,
            role=message["role"],
            action_type=row["target_json"]["action_type"],
            content=message["content"],
            tool_name=row["input_json"].get("tool_name"),
            tool_input=row["input_json"].get("tool_input") or {},
            tool_output=row["target_json"].get("tool_output") or {},
            metadata=row["lineage"],
        )
        insert_training_example(
            bucket=args.bucket,
            input_json=row["input_json"],
            target_json=row["target_json"],
            split_name=args.split,
            source_date=args.source_date,
            agent_run_id=runs[trace_id],
            lineage=row["lineage"],
        )
    return {"dataset_id": dataset_id, "runs": len(runs), "training_examples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Codex transcript JSON/JSONL into SFT-ready data-factory records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="weights/data_factory/codex_transcripts.jsonl")
    parser.add_argument("--dataset_name", default="codex_transcripts_2026")
    parser.add_argument("--namespace", default="trace", choices=["train", "eval_protected", "synthetic", "trace", "quarantine"])
    parser.add_argument("--bucket", default="codex_transcript")
    parser.add_argument("--split", default="train", choices=["train", "validation", "eval_holdout", "quarantine"])
    parser.add_argument("--source_date", default=None)
    parser.add_argument("--license", default="internal")
    parser.add_argument("--harness", default="codex")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--postgres", action="store_true")
    args = parser.parse_args()

    rows = build_records(Path(args.input), args.bucket, args.split, args.source_date, args.limit)
    written = write_jsonl(rows, Path(args.out))
    result: dict[str, Any] = {"status": "ok", "out": args.out, "records": written, "postgres": False}
    if args.postgres:
        result["postgres"] = True
        result["postgres_result"] = ingest_postgres(args, rows)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

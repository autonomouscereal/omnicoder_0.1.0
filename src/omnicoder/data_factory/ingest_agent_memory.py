from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.ingest_2026 import stable_hash, write_jsonl
from omnicoder.data_factory.postgres import insert_agent_run, insert_agent_step, insert_dataset, insert_training_example


def _json_default(value: Any) -> str:
    return str(value)


def _load_json_records(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for item in paths:
        if not item.is_file() or item.suffix.lower() not in {".jsonl", ".json", ".log", ".txt"}:
            continue
        text = item.read_text(encoding="utf-8", errors="surrogateescape")
        if item.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except Exception as exc:
                yield {"path": str(item), "event_type": "parse_error", "error": str(exc), "content": text[:2000]}
                continue
            records = payload if isinstance(payload, list) else [payload]
            for idx, record in enumerate(records):
                if isinstance(record, dict):
                    record.setdefault("path", str(item))
                    record.setdefault("line", idx + 1)
                    yield record
            continue
        for idx, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                record = {"content": line, "event_type": "raw_line"}
            if isinstance(record, dict):
                record.setdefault("path", str(item))
                record.setdefault("line", idx + 1)
                yield record


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    return {"value": value}


def _content_from(record: dict[str, Any]) -> str:
    for key in ("content", "prompt", "text", "message", "tool_output", "response", "raw"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (dict, list)) and value:
            return json.dumps(value, ensure_ascii=True, default=_json_default)
    return json.dumps(record, ensure_ascii=True, sort_keys=True, default=_json_default)


def _role_for(event_type: str, record: dict[str, Any]) -> str:
    role = str(record.get("role") or "").strip().lower()
    if role in {"user", "assistant", "system", "tool"}:
        return role
    if event_type == "UserPromptSubmit":
        return "user"
    if event_type == "PostToolUse":
        return "tool"
    if event_type == "Stop":
        return "system"
    return "assistant"


def normalize_memory_event(record: dict[str, Any], bucket: str, split: str, source_date: str | None) -> dict[str, Any]:
    event_type = str(record.get("event_type") or record.get("event") or record.get("memory_kind") or "agent_memory_event")
    trace_id = str(record.get("session_id") or record.get("trace_id") or record.get("run_id") or stable_hash(json.dumps(record, sort_keys=True, default=_json_default)))
    tool_name = record.get("tool_name") or record.get("tool") or record.get("name")
    content = _content_from(record)
    role = _role_for(event_type, record)
    record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=True, default=_json_default).encode("utf-8")).hexdigest()
    return {
        "bucket": bucket,
        "split": split,
        "source_date": source_date,
        "input_json": {
            "messages": [{"role": role, "content": content}],
            "event_type": event_type,
            "tool_name": tool_name,
            "tool_input": _as_mapping(record.get("tool_input") or record.get("input") or record.get("args")),
        },
        "target_json": {
            "content": content,
            "action_type": event_type,
            "tool_output": _as_mapping(record.get("tool_output") or record.get("output") or record.get("result")),
        },
        "lineage": {
            "source": "agent_memory",
            "trace_id": trace_id,
            "record_hash": record_hash,
            "path": record.get("path"),
            "line": record.get("line"),
            "space": record.get("space") or record.get("memory_space"),
            "agent": record.get("agent"),
            "created_at": record.get("created_at") or record.get("timestamp"),
        },
    }


def build_records(path: Path, bucket: str, split: str, source_date: str | None, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _load_json_records(path):
        row = normalize_memory_event(raw, bucket, split, source_date)
        if row["target_json"]["content"].strip():
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
        terms={"policy": "2025_2026_only", "source": "agent_memory"},
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
        input_json = row["input_json"]
        target_json = row["target_json"]
        message = (input_json.get("messages") or [{"role": "assistant", "content": ""}])[0]
        insert_agent_step(
            agent_run_id=runs[trace_id],
            step_index=step_index,
            role=str(message.get("role") or "assistant"),
            action_type=str(target_json.get("action_type") or "agent_memory_event"),
            content=str(message.get("content") or ""),
            tool_name=input_json.get("tool_name"),
            tool_input=input_json.get("tool_input") or {},
            tool_output=target_json.get("tool_output") or {},
            metadata=row["lineage"],
        )
        insert_training_example(
            bucket=args.bucket,
            input_json=input_json,
            target_json=target_json,
            split_name=args.split,
            source_date=args.source_date,
            agent_run_id=runs[trace_id],
            lineage=row["lineage"],
        )
    return {"dataset_id": dataset_id, "runs": len(runs), "training_examples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PostgreSQL agent-memory hook exports into data-factory JSONL/PostgreSQL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="weights/data_factory/agent_memory.jsonl")
    parser.add_argument("--dataset_name", default="agent_memory_2026")
    parser.add_argument("--namespace", default="trace", choices=["train", "eval_protected", "synthetic", "trace", "quarantine"])
    parser.add_argument("--bucket", default="agent_memory_trace")
    parser.add_argument("--split", default="train", choices=["train", "validation", "eval_holdout", "quarantine"])
    parser.add_argument("--source_date", default=None)
    parser.add_argument("--license", default="internal")
    parser.add_argument("--harness", default="agent_memory")
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

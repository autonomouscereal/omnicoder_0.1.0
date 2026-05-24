from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "omnicoder.memory_trace.v1"
TEXT_SUFFIXES = {".jsonl", ".json", ".log", ".txt", ".md"}
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "passphrase",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}"),
    re.compile(r"(?i)(password|api[_-]?key|token|secret)\s*[:=]\s*([^\s,;]+)"),
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def write_jsonl(rows: Iterable[dict[str, Any]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        def repl(match: re.Match[str]) -> str:
            if pattern.pattern.startswith("(?i)(password"):
                return f"{match.group(1)}=<redacted>"
            return "<redacted-secret>"

        redacted = pattern.sub(repl, redacted)
    return redacted


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lower_key = key_text.lower()
            if any(part in lower_key for part in SENSITIVE_KEY_PARTS):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="surrogateescape")


def _records_from_json_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "rows", "items", "events", "messages", "entries", "data"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return nested
        return [payload]
    return []


def iter_files(inputs: Sequence[Path], suffixes: set[str] | None = None) -> Iterable[Path]:
    allowed = suffixes or TEXT_SUFFIXES
    for root in inputs:
        if root.is_file() and root.suffix.lower() in allowed:
            yield root
        elif root.is_dir():
            for item in sorted(root.rglob("*")):
                if item.is_file() and item.suffix.lower() in allowed:
                    yield item


def iter_jsonish_records(inputs: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in iter_files(inputs):
        suffix = path.suffix.lower()
        try:
            text = _read_text(path)
        except OSError as exc:
            yield {"path": str(path), "line": 0, "type": "read_error", "error": str(exc)}
            continue
        if suffix == ".json":
            try:
                payload = json.loads(text)
            except Exception as exc:
                yield {"path": str(path), "line": 0, "type": "parse_error", "error": str(exc), "content": text[:2000]}
                continue
            for index, item in enumerate(_records_from_json_payload(payload), start=1):
                if isinstance(item, dict):
                    item.setdefault("path", str(path))
                    item.setdefault("line", index)
                    yield item
                else:
                    yield {"path": str(path), "line": index, "type": "json_value", "content": item}
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                item = {"type": "raw_line", "content": line}
            if isinstance(item, dict):
                item.setdefault("path", str(path))
                item.setdefault("line", line_number)
                yield item
            else:
                yield {"path": str(path), "line": line_number, "type": "json_value", "content": item}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = first_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        if isinstance(value.get("content"), list):
            return first_text(value["content"])
        for key in ("text", "content", "message", "prompt", "completion", "response", "output", "result", "summary"):
            if key in value:
                text = first_text(value[key])
                if text:
                    return text
        return _stringify(value)
    return str(value)


def _message_from_event(event: dict[str, Any]) -> dict[str, Any]:
    message = event.get("message")
    if isinstance(message, dict):
        return message
    item = event.get("item")
    if isinstance(item, dict) and isinstance(item.get("message"), dict):
        return item["message"]
    return {}


def _item_from_event(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item")
    return item if isinstance(item, dict) else event


def _tool_from_content(value: Any) -> tuple[str | None, Any, Any]:
    if isinstance(value, dict):
        if value.get("type") == "tool_use":
            return str(value.get("name") or "") or None, value.get("input") or {}, {}
        if value.get("type") == "tool_result":
            return str(value.get("name") or value.get("tool_name") or "") or None, {}, value.get("content") or value.get("result") or {}
        for nested in value.values():
            name, tool_input, tool_output = _tool_from_content(nested)
            if name or tool_input or tool_output:
                return name, tool_input, tool_output
    if isinstance(value, list):
        for item in value:
            name, tool_input, tool_output = _tool_from_content(item)
            if name or tool_input or tool_output:
                return name, tool_input, tool_output
    return None, {}, {}


def coerce_role(value: Any, fallback: str = "assistant") -> str:
    role = str(value or fallback).strip().lower()
    aliases = {"human": "user", "model": "assistant", "function": "tool"}
    role = aliases.get(role, role)
    return role if role in {"user", "assistant", "system", "tool"} else fallback


def created_at_from(record: dict[str, Any]) -> str | None:
    for key in ("created_at", "timestamp", "time", "ts", "date", "event_time"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    message = _message_from_event(record)
    for key in ("created_at", "timestamp", "time"):
        value = message.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def year_from_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    match = re.match(r"^(\d{4})", text)
    if match:
        return int(match.group(1))
    try:
        return datetime.fromtimestamp(float(text)).year
    except Exception:
        return None


def include_by_year(record: dict[str, Any], min_year: int, max_year: int) -> bool:
    year = year_from_timestamp(created_at_from(record))
    return year is None or min_year <= year <= max_year


def source_date_for(record: dict[str, Any], fallback: str | None) -> str | None:
    if fallback:
        return fallback
    created_at = created_at_from(record)
    if not created_at:
        return None
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", created_at)
    return match.group(1) if match else None


def make_row(
    *,
    collector: str,
    event: dict[str, Any],
    role: str,
    content: str,
    event_type: str,
    tool_name: str | None,
    tool_input: Any,
    tool_output: Any,
    trace_id: str,
    bucket: str,
    split: str,
    source_date: str | None,
    lineage_extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    clean_content = redact_text(content)
    if not clean_content.strip() and not tool_name and not tool_input and not tool_output:
        return None
    event_hash = stable_hash(redact(event))
    lineage = {
        "source": collector,
        "trace_id": trace_id,
        "record_hash": event_hash,
        "path": event.get("path"),
        "line": event.get("line"),
        "created_at": created_at_from(event),
    }
    if lineage_extra:
        lineage.update({key: value for key, value in lineage_extra.items() if value not in (None, "")})
    return {
        "schema": SCHEMA,
        "collector": collector,
        "bucket": bucket,
        "split": split,
        "source_date": source_date,
        "input_json": {
            "messages": [{"role": role, "content": clean_content}],
            "event_type": event_type,
            "tool_name": tool_name,
            "tool_input": redact(tool_input if tool_input is not None else {}),
        },
        "target_json": {
            "content": clean_content,
            "action_type": event_type,
            "tool_output": redact(tool_output if tool_output is not None else {}),
        },
        "lineage": redact(lineage),
    }


def codex_event_row(event: dict[str, Any], bucket: str, split: str, source_date: str | None) -> dict[str, Any] | None:
    item = _item_from_event(event)
    event_type = str(event.get("type") or item.get("type") or item.get("event_type") or "codex_event")
    tool_name = item.get("tool_name") or item.get("name") or event.get("tool_name")
    role = coerce_role(item.get("role") or event.get("role"), "tool" if tool_name else "assistant")
    content = first_text(item.get("content") or item.get("message") or event.get("content") or event.get("message") or item)
    trace_id = str(
        event.get("session_id")
        or event.get("conversation_id")
        or event.get("trace_id")
        or item.get("session_id")
        or stable_hash(str(event.get("path")))
    )
    return make_row(
        collector="codex_session",
        event=event,
        role=role,
        content=content,
        event_type=event_type,
        tool_name=str(tool_name) if tool_name else None,
        tool_input=item.get("arguments") or item.get("input") or item.get("tool_input") or {},
        tool_output=item.get("output") or item.get("result") or item.get("tool_output") or {},
        trace_id=trace_id,
        bucket=bucket,
        split=split,
        source_date=source_date,
    )


def claude_event_row(event: dict[str, Any], bucket: str, split: str, source_date: str | None) -> dict[str, Any] | None:
    message = _message_from_event(event)
    item = _item_from_event(event)
    content_value = message.get("content") if message else item.get("content")
    tool_name, tool_input, tool_output = _tool_from_content(content_value)
    if not tool_name:
        tool_name = event.get("tool_name") or item.get("tool_name") or item.get("name")
    if not tool_output:
        tool_output = event.get("toolUseResult") or item.get("tool_result") or item.get("tool_output") or {}
    if not tool_input:
        tool_input = item.get("input") or item.get("tool_input") or {}
    role = coerce_role(message.get("role") or event.get("type") or item.get("role"), "assistant")
    if tool_name and role == "assistant" and str(event.get("type") or "").lower() in {"tool", "tool_result"}:
        role = "tool"
    event_type = str(event.get("type") or item.get("type") or message.get("type") or "claude_event")
    content = first_text(content_value or event.get("content") or item.get("message") or item)
    trace_id = str(
        event.get("sessionId")
        or event.get("session_id")
        or event.get("conversation_id")
        or event.get("trace_id")
        or stable_hash(str(event.get("path")))
    )
    return make_row(
        collector="claude_trace",
        event=event,
        role=role,
        content=content,
        event_type=event_type,
        tool_name=str(tool_name) if tool_name else None,
        tool_input=tool_input,
        tool_output=tool_output,
        trace_id=trace_id,
        bucket=bucket,
        split=split,
        source_date=source_date,
        lineage_extra={"uuid": event.get("uuid"), "parent_uuid": event.get("parentUuid"), "cwd": event.get("cwd")},
    )


def agent_memory_event_row(event: dict[str, Any], bucket: str, split: str, source_date: str | None) -> dict[str, Any] | None:
    event_type = str(event.get("event_type") or event.get("event") or event.get("memory_kind") or event.get("type") or "agent_memory_event")
    tool_name = event.get("tool_name") or event.get("tool") or event.get("name")
    content = first_text(
        event.get("content")
        or event.get("prompt")
        or event.get("text")
        or event.get("message")
        or event.get("tool_output")
        or event.get("response")
        or event.get("raw")
        or event
    )
    role = coerce_role(event.get("role"), "tool" if event_type == "PostToolUse" or tool_name else "assistant")
    if event_type == "UserPromptSubmit":
        role = "user"
    elif event_type == "Stop":
        role = "system"
    trace_id = str(event.get("session_id") or event.get("trace_id") or event.get("run_id") or stable_hash(event))
    return make_row(
        collector="agent_memory_export",
        event=event,
        role=role,
        content=content,
        event_type=event_type,
        tool_name=str(tool_name) if tool_name else None,
        tool_input=event.get("tool_input") or event.get("input") or event.get("args") or {},
        tool_output=event.get("tool_output") or event.get("output") or event.get("result") or {},
        trace_id=trace_id,
        bucket=bucket,
        split=split,
        source_date=source_date,
        lineage_extra={
            "space": event.get("space") or event.get("memory_space"),
            "agent": event.get("agent"),
            "source_uri": event.get("source_uri"),
        },
    )


def looks_like_agent_memory_event(event: dict[str, Any]) -> bool:
    if any(key in event for key in ("memory_kind", "memory_space", "space", "agent", "source_uri")):
        return True
    event_type = str(event.get("event_type") or event.get("event") or "")
    return event_type in {"UserPromptSubmit", "PostToolUse", "Stop"}


def claude_or_agent_memory_row(event: dict[str, Any], bucket: str, split: str, source_date: str | None) -> dict[str, Any] | None:
    if looks_like_agent_memory_event(event):
        return agent_memory_event_row(event, bucket, split, source_date)
    return claude_event_row(event, bucket, split, source_date)


def collect_records(
    inputs: Sequence[Path],
    row_builder: Any,
    *,
    bucket: str,
    split: str,
    source_date: str | None,
    min_year: int,
    max_year: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in iter_jsonish_records(inputs):
        if not include_by_year(event, min_year, max_year):
            continue
        row = row_builder(event, bucket, split, source_date_for(event, source_date))
        if row is not None:
            rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def read_rows(inputs: Sequence[Path], limit: int = 0) -> Iterable[dict[str, Any]]:
    seen = 0
    for record in iter_jsonish_records(inputs):
        if isinstance(record, dict):
            yield record
            seen += 1
            if limit and seen >= limit:
                return


def merge_rows(inputs: Sequence[Path], dedupe: bool, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_rows(inputs, limit=limit):
        lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
        source = str(row.get("collector") or lineage.get("source") or "")
        key = f"{source}:{lineage.get('record_hash') or stable_hash(row)}"
        if dedupe and key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def stats_for_rows(inputs: Sequence[Path], limit: int = 0) -> dict[str, Any]:
    collectors: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    trace_ids: set[str] = set()
    records = 0
    empty_content = 0
    for row in read_rows(inputs, limit=limit):
        records += 1
        collectors[str(row.get("collector") or row.get("lineage", {}).get("source") or "unknown")] += 1
        sources[str(row.get("lineage", {}).get("source") or "unknown")] += 1
        buckets[str(row.get("bucket") or "unknown")] += 1
        input_json = row.get("input_json") if isinstance(row.get("input_json"), dict) else {}
        target_json = row.get("target_json") if isinstance(row.get("target_json"), dict) else {}
        messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else []
        if messages and isinstance(messages[0], dict):
            roles[str(messages[0].get("role") or "unknown")] += 1
        event_types[str(input_json.get("event_type") or target_json.get("action_type") or "unknown")] += 1
        content = str(target_json.get("content") or "")
        if not content.strip():
            empty_content += 1
        trace_id = row.get("lineage", {}).get("trace_id") if isinstance(row.get("lineage"), dict) else None
        if trace_id:
            trace_ids.add(str(trace_id))
    return {
        "status": "ok",
        "records": records,
        "unique_traces": len(trace_ids),
        "empty_content": empty_content,
        "collectors": dict(sorted(collectors.items())),
        "sources": dict(sorted(sources.items())),
        "buckets": dict(sorted(buckets.items())),
        "roles": dict(sorted(roles.items())),
        "event_types": dict(sorted(event_types.items())),
    }


def default_codex_inputs() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".codex" / "sessions",
        home / ".codex" / "history.jsonl",
        home / "Documents" / "Codex",
    ]
    return [path for path in candidates if path.exists()]


def default_claude_inputs() -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".claude" / "projects",
        home / ".claude" / "memory_backend" / "exports",
        home / ".claude" / "memory_backend" / "logs",
    ]
    return [path for path in candidates if path.exists()]


def resolve_inputs(values: Sequence[str] | None, defaults: Sequence[Path]) -> list[Path]:
    paths = [Path(value).expanduser() for value in (values or [])]
    if not paths:
        paths = list(defaults)
    return paths


def require_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(json.dumps({"status": "error", "error": "input path not found", "paths": missing}, ensure_ascii=True))
    if not paths:
        raise SystemExit(json.dumps({"status": "error", "error": "no input paths found"}, ensure_ascii=True))


def add_collect_args(parser: argparse.ArgumentParser, default_out: str, default_bucket: str) -> None:
    parser.add_argument("--input", action="append", help="Offline file or directory input. Repeat for multiple roots.")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--bucket", default=default_bucket)
    parser.add_argument("--split", default="train", choices=["train", "validation", "eval_holdout", "quarantine"])
    parser.add_argument("--source-date", default=None)
    parser.add_argument("--min-year", type=int, default=2025)
    parser.add_argument("--max-year", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)


def run_collect(args: argparse.Namespace, row_builder: Any, defaults: Sequence[Path]) -> dict[str, Any]:
    inputs = resolve_inputs(args.input, defaults)
    require_inputs(inputs)
    rows = collect_records(
        inputs,
        row_builder,
        bucket=args.bucket,
        split=args.split,
        source_date=args.source_date,
        min_year=args.min_year,
        max_year=args.max_year,
        limit=args.limit,
    )
    written = write_jsonl(rows, Path(args.out))
    return {
        "status": "ok",
        "out": args.out,
        "records": written,
        "inputs": [str(path) for path in inputs],
        "stats": stats_for_rows([Path(args.out)]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline 2025-2026 memory trace collectors for Codex, Claude, and agent-memory exports")
    sub = parser.add_subparsers(dest="command", required=True)

    codex = sub.add_parser("collect-codex", help="Collect local Codex session JSON/JSONL traces")
    add_collect_args(codex, "weights/data_factory/memory_traces_codex_2026.jsonl", "codex_session_trace")

    claude = sub.add_parser("collect-claude", help="Collect Claude Code logs and memory_backend export traces")
    add_collect_args(claude, "weights/data_factory/memory_traces_claude_2026.jsonl", "claude_memory_trace")
    claude.add_argument("--source-kind", choices=["auto", "claude", "agent-memory"], default="auto")

    merge = sub.add_parser("merge", help="Merge collector JSONL files")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--out", default="weights/data_factory/memory_traces_merged_2026.jsonl")
    merge.add_argument("--no-dedupe", action="store_true")
    merge.add_argument("--limit", type=int, default=0)

    stats = sub.add_parser("stats", help="Summarize collector JSONL files")
    stats.add_argument("--input", action="append", required=True)
    stats.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "collect-codex":
        result = run_collect(args, codex_event_row, default_codex_inputs())
    elif args.command == "collect-claude":
        if args.source_kind == "agent-memory":
            row_builder = agent_memory_event_row
        elif args.source_kind == "claude":
            row_builder = claude_event_row
        else:
            row_builder = claude_or_agent_memory_row
        result = run_collect(args, row_builder, default_claude_inputs())
    elif args.command == "merge":
        inputs = resolve_inputs(args.input, [])
        require_inputs(inputs)
        rows = merge_rows(inputs, dedupe=not args.no_dedupe, limit=args.limit)
        written = write_jsonl(rows, Path(args.out))
        result = {"status": "ok", "out": args.out, "records": written, "inputs": [str(path) for path in inputs], "dedupe": not args.no_dedupe}
    elif args.command == "stats":
        inputs = resolve_inputs(args.input, [])
        require_inputs(inputs)
        result = stats_for_rows(inputs, limit=args.limit)
        result["inputs"] = [str(path) for path in inputs]
    else:
        parser.error(f"unknown command: {args.command}")
        return
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

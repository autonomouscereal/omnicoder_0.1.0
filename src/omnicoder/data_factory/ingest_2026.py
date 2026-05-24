from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.content_address import sha256_file
from omnicoder.data_factory.postgres import (
    insert_agent_run,
    insert_agent_step,
    insert_artifact,
    insert_dataset,
    insert_training_example,
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for p in paths:
        if not p.is_file() or p.suffix.lower() not in {".jsonl", ".json", ".txt", ".md", ".log"}:
            continue
        if p.suffix.lower() in {".txt", ".md", ".log"}:
            yield {"path": str(p), "kind": "text", "text": p.read_text(encoding="utf-8", errors="ignore")}
            continue
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception as exc:
                yield {"path": str(p), "kind": "parse_error", "error": str(exc)}
                continue
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        item.setdefault("path", str(p))
                        yield item
            elif isinstance(obj, dict):
                obj.setdefault("path", str(p))
                yield obj
            continue
        for idx, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                obj = {"text": line}
            if isinstance(obj, dict):
                obj.setdefault("path", str(p))
                obj.setdefault("line", idx + 1)
                yield obj


def normalize_record(obj: dict[str, Any]) -> dict[str, Any]:
    text = obj.get("text") or obj.get("content") or obj.get("prompt") or obj.get("completion") or obj.get("message") or ""
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    trace_id = obj.get("trace_id") or obj.get("session_id") or obj.get("run_id") or stable_hash(json.dumps(obj, sort_keys=True)[:4000])
    tool_name = obj.get("tool_name") or obj.get("tool") or obj.get("name")
    role = obj.get("role") or ("tool" if tool_name else "sample")
    return {
        "trace_id": str(trace_id),
        "role": str(role),
        "action_type": obj.get("action_type") or obj.get("event_type") or ("tool_call" if tool_name else "text"),
        "text": str(text),
        "tool_name": tool_name,
        "tool_input": obj.get("tool_input") or obj.get("input") or obj.get("args") or {},
        "tool_output": obj.get("tool_output") or obj.get("output") or obj.get("result") or {},
        "exit_code": obj.get("exit_code"),
        "latency_ms": obj.get("latency_ms"),
        "metadata": obj,
        "path": obj.get("path"),
    }


def write_jsonl(records: Iterable[dict[str, Any]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            count += 1
    return count


def build_training_records(path: Path, bucket: str, split: str, source_date: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in iter_records(path):
        rec = normalize_record(obj)
        text = rec["text"]
        if not text and rec["action_type"] != "tool_call":
            continue
        rows.append(
            {
                "bucket": bucket,
                "split": split,
                "source_date": source_date,
                "input_json": {
                    "role": rec["role"],
                    "content": text,
                    "tool_name": rec["tool_name"],
                    "tool_input": rec["tool_input"],
                },
                "target_json": {
                    "content": text,
                    "tool_output": rec["tool_output"],
                    "action_type": rec["action_type"],
                },
                "lineage": {
                    "trace_id": rec["trace_id"],
                    "path": rec["path"],
                    "record_hash": stable_hash(json.dumps(obj, sort_keys=True)),
                },
            }
        )
    return rows


def ingest_postgres(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_id = insert_dataset(
        name=args.dataset_name,
        namespace=args.namespace,
        source_uri=str(Path(args.input).resolve()),
        source_date=args.source_date,
        license_id=args.license,
        terms={"policy": "2025_2026_only", "contamination": args.split},
        metadata={"bucket": args.bucket},
    )
    artifact_ids: dict[str, int] = {}
    for p in sorted({row["lineage"].get("path") for row in rows if row["lineage"].get("path")}):
        path = Path(str(p))
        if path.exists() and path.is_file():
            artifact_ids[str(path)] = insert_artifact(str(path), sha256_file(str(path)), "text_or_trace", path.stat().st_size, {"dataset_id": dataset_id})

    runs: dict[str, int] = {}
    step_counts: dict[str, int] = {}
    for row in rows:
        trace_id = str(row["lineage"]["trace_id"])
        if trace_id not in runs:
            runs[trace_id] = insert_agent_run(trace_id=trace_id, harness=args.harness, dataset_id=dataset_id, task_family=args.bucket)
            step_counts[trace_id] = 0
        step_idx = step_counts[trace_id]
        step_counts[trace_id] += 1
        insert_agent_step(
            agent_run_id=runs[trace_id],
            step_index=step_idx,
            role=str(row["input_json"].get("role") or "sample"),
            action_type=str(row["target_json"].get("action_type") or "text"),
            content=str(row["input_json"].get("content") or ""),
            tool_name=row["input_json"].get("tool_name"),
            tool_input=row["input_json"].get("tool_input") or {},
            tool_output=row["target_json"].get("tool_output") or {},
            metadata=row["lineage"],
        )
        artifact_id = artifact_ids.get(str(row["lineage"].get("path")))
        insert_training_example(
            bucket=args.bucket,
            input_json=row["input_json"],
            target_json=row["target_json"],
            split_name=args.split,
            source_date=args.source_date,
            artifact_id=artifact_id,
            agent_run_id=runs[trace_id],
            lineage=row["lineage"],
        )
    return {"dataset_id": dataset_id, "runs": len(runs), "training_examples": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest 2025-2026 traces/text into Omnicoder data factory")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="weights/data_factory/ingest_2026.jsonl")
    ap.add_argument("--dataset_name", default="omnicoder_2026_ingest")
    ap.add_argument("--namespace", default="trace", choices=["train", "eval_protected", "synthetic", "trace", "quarantine"])
    ap.add_argument("--bucket", default="agent_tool_trace")
    ap.add_argument("--split", default="train", choices=["train", "validation", "eval_holdout", "quarantine"])
    ap.add_argument("--source_date", default=None)
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--harness", default="codex_or_claude_trace")
    ap.add_argument("--postgres", action="store_true")
    args = ap.parse_args()

    rows = build_training_records(Path(args.input), args.bucket, args.split, args.source_date)
    written = write_jsonl(rows, Path(args.out))
    result: dict[str, Any] = {"status": "ok", "out": args.out, "records": written, "postgres": False}
    if args.postgres:
        result["postgres"] = True
        result["postgres_result"] = ingest_postgres(args, rows)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

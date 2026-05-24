from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any

from omnicoder.data_factory.content_address import sha256_file
from omnicoder.data_factory.ingest_2026 import stable_hash, write_jsonl
from omnicoder.data_factory.postgres import insert_artifact, insert_dataset, insert_training_example


MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".mp4",
    ".webm",
    ".mov",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


def _prompt_from(metadata: dict[str, Any]) -> str:
    for key in ("prompt", "positive_prompt", "text", "caption", "description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    workflow = metadata.get("workflow") or metadata.get("prompt_graph")
    if isinstance(workflow, dict):
        return json.dumps(workflow, ensure_ascii=True, sort_keys=True)[:6000]
    return ""


def build_records(path: Path, bucket: str, split: str, source_date: str | None, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    for item in files:
        if not item.is_file() or item.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        sidecar = _read_json(item.with_suffix(item.suffix + ".json")) or _read_json(item.with_suffix(".json"))
        media_type = mimetypes.guess_type(str(item))[0] or "application/octet-stream"
        digest = sha256_file(str(item))
        prompt = _prompt_from(sidecar)
        rows.append(
            {
                "bucket": bucket,
                "split": split,
                "source_date": source_date,
                "input_json": {
                    "messages": [{"role": "user", "content": prompt or "ComfyUI generated media artifact"}],
                    "modality": media_type.split("/", 1)[0],
                    "workflow": sidecar.get("workflow") or sidecar.get("prompt_graph") or {},
                },
                "target_json": {
                    "artifact_path": str(item),
                    "media_type": media_type,
                    "sha256": digest,
                    "caption": sidecar.get("caption") or sidecar.get("description"),
                    "metadata": sidecar,
                },
                "lineage": {
                    "source": "comfyui_output",
                    "record_hash": stable_hash(f"{digest}:{item}"),
                    "path": str(item),
                    "sidecar_path": str(item.with_suffix(item.suffix + ".json")) if item.with_suffix(item.suffix + ".json").exists() else None,
                    "byte_size": item.stat().st_size,
                },
            }
        )
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
        terms={"policy": "2025_2026_only", "source": "comfyui_output"},
        metadata={"bucket": args.bucket},
    )
    artifacts = 0
    examples = 0
    for row in rows:
        artifact_id = insert_artifact(
            row["target_json"]["artifact_path"],
            row["target_json"]["sha256"],
            row["target_json"]["media_type"],
            int(row["lineage"].get("byte_size") or 0),
            {"dataset_id": dataset_id, **row["target_json"].get("metadata", {})},
        )
        lineage = dict(row["lineage"])
        lineage["artifact_id"] = artifact_id
        insert_training_example(
            bucket=args.bucket,
            input_json=row["input_json"],
            target_json=row["target_json"],
            split_name=args.split,
            source_date=args.source_date,
            artifact_id=artifact_id,
            lineage=lineage,
        )
        artifacts += 1
        examples += 1
    return {"dataset_id": dataset_id, "artifacts": artifacts, "training_examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index ComfyUI media outputs into JSONL/PostgreSQL data-factory records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="weights/data_factory/comfyui_outputs.jsonl")
    parser.add_argument("--dataset_name", default="comfyui_outputs_2026")
    parser.add_argument("--namespace", default="synthetic", choices=["train", "eval_protected", "synthetic", "trace", "quarantine"])
    parser.add_argument("--bucket", default="comfyui_multimodal")
    parser.add_argument("--split", default="train", choices=["train", "validation", "eval_holdout", "quarantine"])
    parser.add_argument("--source_date", default=None)
    parser.add_argument("--license", default="internal")
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

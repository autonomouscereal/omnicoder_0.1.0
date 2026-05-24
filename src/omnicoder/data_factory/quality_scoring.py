from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.postgres import transaction


SECRET_RE = re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{12,}")


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception as exc:
            item = {"text": line, "line": idx + 1, "parse_error": str(exc)}
        if isinstance(item, dict):
            yield item


def extract_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for container in (record.get("input_json"), record.get("target_json"), record):
        if isinstance(container, dict):
            messages = container.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        parts.append(message["content"])
            for key in ("content", "text", "caption", "artifact_path"):
                value = container.get(key)
                if isinstance(value, str):
                    parts.append(value)
    return "\n".join(part for part in parts if part)


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    text = extract_text(record)
    length = len(text.strip())
    tokens = re.findall(r"\w+", text.lower())
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    length_score = min(1.0, math.log1p(length) / math.log(4000)) if length else 0.0
    structure_score = 0.0
    if "```" in text:
        structure_score += 0.18
    if isinstance(record.get("input_json"), dict) and isinstance(record.get("target_json"), dict):
        structure_score += 0.22
    if any(marker in text for marker in ("Traceback", "ERROR", "FAILED", "pass", "return", "def ")):
        structure_score += 0.12
    duplicate_penalty = 0.2 if unique_ratio < 0.28 and len(tokens) > 80 else 0.0
    secret_penalty = 0.45 if SECRET_RE.search(text) else 0.0
    media_bonus = 0.12 if isinstance(record.get("target_json"), dict) and record["target_json"].get("artifact_path") else 0.0
    score = max(0.0, min(1.0, 0.2 + (0.42 * length_score) + structure_score + media_bonus - duplicate_penalty - secret_penalty))
    label = "reject" if score < 0.35 or secret_penalty else "candidate" if score < 0.72 else "high"
    return {
        "score": round(score, 6),
        "label": label,
        "details": {
            "length": length,
            "token_count": len(tokens),
            "unique_ratio": round(unique_ratio, 6),
            "length_score": round(length_score, 6),
            "structure_score": round(structure_score, 6),
            "duplicate_penalty": duplicate_penalty,
            "secret_penalty": secret_penalty,
            "media_bonus": media_bonus,
        },
    }


def score_jsonl(input_path: Path, out_path: Path, min_score: float = 0.0) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in _jsonl(input_path):
            scored = score_record(record)
            record["quality"] = scored
            if scored["score"] < min_score:
                continue
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def write_quality_to_postgres(input_path: Path, scorer: str) -> int:
    count = 0
    with transaction() as cur:
        for record in _jsonl(input_path):
            quality = record.get("quality") if isinstance(record.get("quality"), dict) else score_record(record)
            lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
            targets = []
            for key, target_type in (
                ("sample_id", "sample"),
                ("training_example_id", "training_example"),
                ("agent_run_id", "agent_run"),
                ("artifact_id", "artifact"),
            ):
                if lineage.get(key) is not None:
                    targets.append((target_type, int(lineage[key])))
            for target_type, target_id in targets:
                cur.execute(
                    """
                    INSERT INTO quality_scores (target_type, target_id, scorer, score_name, score_value, label, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (target_type, target_id, scorer, "heuristic_quality", float(quality["score"]), quality["label"], json.dumps(quality["details"])),
                )
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristically score data-factory JSONL quality without model dependencies")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="weights/data_factory/scored.jsonl")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--postgres", action="store_true", help="Write quality rows for records carrying DB ids in lineage")
    parser.add_argument("--scorer", default="heuristic_quality_v1")
    args = parser.parse_args()

    written = score_jsonl(Path(args.input), Path(args.out), args.min_score)
    result: dict[str, Any] = {"status": "ok", "out": args.out, "records": written, "postgres": False}
    if args.postgres:
        result["postgres"] = True
        result["postgres_quality_rows"] = write_quality_to_postgres(Path(args.out), args.scorer)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.postgres import transaction


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
BENCHMARK_MARKERS = (
    "arc-agi",
    "arcagi",
    "bfcl",
    "gpqa",
    "gsm8k",
    "human_eval",
    "humaneval",
    "livecodebench",
    "mmlu",
    "mmmu",
    "swe-bench",
    "terminal-bench",
    "terminal_bench",
    "truthfulqa",
)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            item = {"text": line, "line": idx + 1}
        if isinstance(item, dict):
            yield item


def _text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for container in (record.get("input_json"), record.get("target_json"), record):
        if isinstance(container, dict):
            messages = container.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        parts.append(message["content"])
            for key in ("content", "text", "answer", "caption"):
                value = container.get(key)
                if isinstance(value, str):
                    parts.append(value)
    return "\n".join(parts)


def fingerprint(text: str, ngram: int) -> set[str]:
    tokens = WORD_RE.findall(text.lower())
    if len(tokens) < ngram:
        return set(tokens)
    return {" ".join(tokens[i : i + ngram]) for i in range(0, len(tokens) - ngram + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _marker_match(text: str) -> dict[str, Any]:
    lower = text.lower()
    markers = [marker for marker in BENCHMARK_MARKERS if marker in lower]
    return {
        "score": min(0.95, 0.2 + 0.16 * len(markers)) if markers else 0.0,
        "benchmark_name": ",".join(markers[:8]) if markers else None,
        "protected_index": None,
        "match_type": "benchmark_marker" if markers else "none",
        "markers": markers,
    }


def _merge_existing_contamination(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    existing = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
    existing_status = str(existing.get("status") or existing.get("label") or "clean")
    rank = {"clean": 0, "suspect": 1, "contaminated": 2}
    if rank.get(existing_status, 0) > rank.get(str(candidate.get("status") or "clean"), 0):
        merged = {**candidate, **existing}
        merged["previous_scan"] = candidate
        return merged
    if existing:
        candidate["previous_contamination"] = existing
    return candidate


def load_protected(path: Path, ngram: int) -> list[dict[str, Any]]:
    protected: list[dict[str, Any]] = []
    for idx, record in enumerate(_jsonl(path)):
        text = _text(record)
        protected.append(
            {
                "index": idx,
                "benchmark_name": record.get("benchmark_name") or record.get("name") or path.stem,
                "text": text,
                "fingerprint": fingerprint(text, ngram),
                "lineage": record.get("lineage") if isinstance(record.get("lineage"), dict) else {},
            }
        )
    return protected


def scan(candidates_path: Path, protected_path: Path, out_path: Path, threshold: float, ngram: int) -> int:
    protected = load_protected(protected_path, ngram)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in _jsonl(candidates_path):
            fp = fingerprint(_text(record), ngram)
            best = {"score": 0.0, "benchmark_name": None, "protected_index": None, "match_type": f"{ngram}gram_jaccard"}
            for protected_record in protected:
                score = _jaccard(fp, protected_record["fingerprint"])
                if score > best["score"]:
                    best = {
                        "score": score,
                        "benchmark_name": protected_record["benchmark_name"],
                        "protected_index": protected_record["index"],
                        "match_type": f"{ngram}gram_jaccard",
                    }
            marker = _marker_match(_text(record))
            if marker["score"] > best["score"]:
                best = marker
            status = "contaminated" if best["score"] >= threshold else "clean"
            if best.get("match_type") == "benchmark_marker" and status == "clean":
                status = "suspect"
            record["contamination"] = _merge_existing_contamination(record, {**best, "status": status, "threshold": threshold})
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def write_matches_to_postgres(scanned_path: Path) -> int:
    count = 0
    with transaction() as cur:
        for record in _jsonl(scanned_path):
            contamination = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
            lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
            sample_id = lineage.get("sample_id")
            if sample_id is None or contamination.get("status") != "contaminated":
                continue
            cur.execute(
                """
                INSERT INTO contamination_matches (sample_id, benchmark_name, match_type, score, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    int(sample_id),
                    contamination.get("benchmark_name") or "unknown",
                    contamination.get("match_type") or "heuristic",
                    float(contamination.get("score") or 0.0),
                    json.dumps(contamination),
                ),
            )
            cur.execute(
                """
                INSERT INTO split_assignments (sample_id, split_name, reason)
                VALUES (%s, 'rejected', %s)
                ON CONFLICT (sample_id) DO UPDATE SET split_name='rejected', reason=EXCLUDED.reason
                """,
                (int(sample_id), "contamination_match"),
            )
            cur.execute(
                "UPDATE samples SET contamination_status='contaminated', train_tier='rejected' WHERE sample_id=%s",
                (int(sample_id),),
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline contamination scanner for 2025-2026 data-factory JSONL")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--protected", required=True)
    parser.add_argument("--out", default="weights/data_factory/contamination_scanned.jsonl")
    parser.add_argument("--threshold", type=float, default=0.42)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--postgres", action="store_true", help="Persist contaminated sample_id matches when lineage includes sample_id")
    args = parser.parse_args()

    written = scan(Path(args.candidates), Path(args.protected), Path(args.out), args.threshold, args.ngram)
    result: dict[str, Any] = {"status": "ok", "out": args.out, "records": written, "postgres": False}
    if args.postgres:
        result["postgres"] = True
        result["postgres_matches"] = write_matches_to_postgres(Path(args.out))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.postgres import transaction


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
TEXT_KEYS = (
    "answer",
    "caption",
    "choices",
    "completion",
    "content",
    "ctx",
    "ctx_a",
    "ctx_b",
    "endings",
    "expected_answer",
    "instruction",
    "prompt",
    "question",
    "response",
    "target",
    "text",
)
BENCHMARK_MARKERS = (
    "arc-agi",
    "arc-agi3",
    "arc_agi",
    "arc_agi3",
    "arcagi",
    "bfcl",
    "berkeley_function_calling",
    "benchmark_id",
    "benchmark_name",
    "benchmark_task_2026",
    "browsergym",
    "commonsense_completion_mcq",
    "compute-eval",
    "compute_eval",
    "data/eval",
    "eval_holdout",
    "frontiermath",
    "frontier_math",
    "gpqa",
    "gpqa_diamond",
    "gsm8k",
    "hellaswag",
    "hella_swag",
    "human_eval",
    "humaneval",
    "ifeval",
    "local_public_dev",
    "livecodebench",
    "live_code_bench",
    "mbpp",
    "mmlu",
    "mmlu_pro",
    "mmmu",
    "mmmu_pro",
    "osworld",
    "protected_eval",
    "public-dev",
    "public dev",
    "public_dev",
    "public_dev_eval",
    "publicdev",
    "reportable",
    "reportable_2026",
    "reportable_score",
    "reportable_task",
    "reasoning_hellaswag_full_2026",
    "rowan/hellaswag",
    "swe-bench",
    "tau_bench",
    "terminal-bench",
    "terminal_bench",
    "truthfulqa",
    "webarena",
)
BENCHMARK_FIELD_MARKER_KEYS = {
    "adapter_id",
    "benchmark",
    "benchmark_id",
    "benchmark_name",
    "benchmark_suite",
    "contamination_class",
    "reportable_task",
    "task_benchmark",
}
TRUE_VALUE_MARKER_KEYS = {"reportable", "reportable_score", "reportable_task"}
PATH_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("data_eval_path", r"(?:^|[\s\\/])data[\\/]eval(?:[\\/]|$)"),
        ("eval_reportable_path", r"(?:^|[\s\\/])(?:eval[\\/])?reportable_2026(?:[\\/]|$)"),
        ("local_public_dev_path", r"(?:^|[\s\\/])local_2026[\\/][^\n]*public[_ \-]?dev"),
        ("public_dev_file", r"\b[^\s\\/]+public[_ \-]?dev\.(?:jsonl|json|parquet|csv)\b"),
        ("authorized_reportable_file", r"\b[^\s\\/]+authorized\.(?:jsonl|json|parquet|csv)\b"),
        ("benchmark_materialization_path", r"\bbenchmark[_ \-]?materialization\b"),
        ("protected_eval_path", r"\bprotected[_ \-]?eval\b"),
    )
)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                item = {"text": line.rstrip("\n"), "line": idx + 1}
            if isinstance(item, dict):
                yield item


def _normalize_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _append_text(parts: list[str], value: Any) -> None:
    if isinstance(value, str):
        if value.strip():
            parts.append(value)
    elif isinstance(value, list):
        for item in value[:64]:
            _append_text(parts, item)
    elif isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list):
            for message in messages[:64]:
                if isinstance(message, dict):
                    _append_text(parts, message.get("content"))
        for key in TEXT_KEYS:
            if key in value:
                _append_text(parts, value.get(key))


def _text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for container in (record.get("input_json"), record.get("target_json"), record):
        if isinstance(container, dict):
            _append_text(parts, container)
    return "\n".join(parts)


def _marker_text(record: dict[str, Any]) -> str:
    parts = [_text(record)]

    def visit(value: Any, key_name: str = "", depth: int = 0) -> None:
        if depth > 8 or len(parts) > 4096:
            return
        if isinstance(value, dict):
            for key, item in list(value.items())[:256]:
                normalized_key = _normalize_marker(str(key))
                if normalized_key in BENCHMARK_FIELD_MARKER_KEYS and item not in (None, "", [], {}):
                    parts.append(normalized_key)
                if item is True and normalized_key in TRUE_VALUE_MARKER_KEYS:
                    parts.append(normalized_key)
                if item is True and normalized_key == "local_only":
                    parts.append("public_dev_eval")
                visit(item, normalized_key, depth + 1)
        elif isinstance(value, list):
            for item in value[:128]:
                visit(item, key_name, depth + 1)
        elif isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text[:4096])
        elif isinstance(value, (int, float)) and key_name in {"id", "ind", "source_index", "task_id"}:
            parts.append(str(value))

    visit(record)
    return "\n".join(part for part in parts if part)


def fingerprint(text: str, ngram: int) -> set[str]:
    tokens = WORD_RE.findall(text.lower())
    if len(tokens) < ngram:
        return set(tokens)
    return {" ".join(tokens[i : i + ngram]) for i in range(0, len(tokens) - ngram + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _marker_match(record_or_text: dict[str, Any] | str) -> dict[str, Any]:
    text = _marker_text(record_or_text) if isinstance(record_or_text, dict) else record_or_text
    lower = text.lower()
    normalized = _normalize_marker(text)
    markers: list[str] = []
    for marker in BENCHMARK_MARKERS:
        marker_lower = marker.lower()
        marker_normalized = _normalize_marker(marker)
        if marker_lower in lower or (marker_normalized and marker_normalized in normalized):
            markers.append(marker_normalized or marker_lower)
    for marker, pattern in PATH_MARKER_PATTERNS:
        if pattern.search(text):
            markers.append(marker)
    markers = list(dict.fromkeys(markers))
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
            marker = _marker_match(record)
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

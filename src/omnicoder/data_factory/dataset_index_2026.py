from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "omnicoder.dataset_index_2026.v1"
TRAIN_LEAK_RE = re.compile(
    r"\b(?:public[_ -]?dev|reportable|answer[_ -]?key|protected[_ -]?eval|benchmark[_ -]?holdout|hella[_ -]?swag|hellaswag|"
    r"arc[_ -]?agi[23]?|arc-agi[23]?|swe[_ -]?bench|terminal[_ -]?bench|mmmu(?:[_ -]?pro)?|fixture|smoke|canary)\b",
    re.IGNORECASE,
)
ID_KEYS = ("record_id", "id", "uid", "uuid", "example_id", "sample_id", "row_id")
MODALITY_KEYS = ("modality", "target_modality", "input_modality", "output_modality", "declared_target_modality", "media_family")
TEXT_TARGET_KEYS = ("content", "text", "target", "response", "completion", "answer", "expected_answer", "output")


def _json_blob(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _first(row: dict[str, Any], *keys: str, default: str = "unknown") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        value = meta.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return default


def _record_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, meta):
        for key in ID_KEYS:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    return ""


def _modality(row: dict[str, Any]) -> str:
    modality = _first(row, "modality", "target_modality", default="unknown")
    if modality.strip().lower() not in {"", "unknown", "none", "null"}:
        return modality
    for container in (row.get("input_json"), row.get("target_json"), row.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in MODALITY_KEYS:
            value = container.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    return "unknown"


def _canonical_split(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "training": "train",
        "validation": "eval",
        "valid": "eval",
        "dev": "eval",
    }.get(normalized, normalized)


def _declared_split(row: dict[str, Any]) -> str:
    value = row.get("split")
    return "" if value in (None, "", [], {}) else str(value)


def _infer_split(path: Path, row: dict[str, Any], expected_split: str = "") -> str:
    if row.get("split") not in (None, "", [], {}):
        return str(row["split"])
    if expected_split:
        return expected_split
    lower = path.name.lower()
    if "train" in lower:
        return "train"
    if "eval" in lower or "dev" in lower or "valid" in lower:
        return "eval"
    if "test" in lower:
        return "test"
    return "unknown"


def _target_token_count(row: dict[str, Any]) -> int:
    for key in ("target_token_ids", "labels", "assistant_token_ids"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value)
    target = row.get("target") or row.get("response") or row.get("completion") or row.get("answer") or ""
    if not isinstance(target, str) or not target:
        for container in (row.get("target_json"), row.get("output_json"), row.get("teacher_output")):
            if not isinstance(container, dict):
                continue
            for key in TEXT_TARGET_KEYS:
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    target = value
                    break
            if isinstance(target, str) and target:
                break
    if isinstance(target, str):
        return len(re.findall(r"\S+", target))
    return 0


def _has_media_payload(row: dict[str, Any]) -> bool:
    if isinstance(row.get("artifact_token_ids"), list) and row["artifact_token_ids"]:
        return True
    for container in (row, row.get("target_json")):
        if not isinstance(container, dict):
            continue
        for key in ("artifact_refs", "artifacts", "artifact_tokens", "media_tokens"):
            if container.get(key) not in (None, "", [], {}):
                return True
    return False


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                yield line_number, {"_bad_json": True, "_raw": line[:2000]}, line
                continue
            yield line_number, row, line


def build_index(paths: list[Path], *, expected_split: str = "", fail_on_train_leakage: bool = True) -> dict[str, Any]:
    by_modality: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_use_policy: Counter[str] = Counter()
    by_license: Counter[str] = Counter()
    by_contamination: Counter[str] = Counter()
    matrix: Counter[tuple[str, str, str, str]] = Counter()
    files: list[dict[str, Any]] = []
    duplicate_payloads = 0
    payload_hashes: set[str] = set()
    duplicate_ids = 0
    seen_ids: dict[str, dict[str, Any]] = {}
    duplicate_id_rows: list[dict[str, Any]] = []
    train_leak_rows: list[dict[str, Any]] = []
    missing_modality_rows: list[dict[str, Any]] = []
    one_token_junk_rows: list[dict[str, Any]] = []
    split_mismatch_rows: list[dict[str, Any]] = []
    bad_json = 0
    rows_with_target_tokens = 0
    rows_with_artifact_tokens = 0
    total_rows = 0

    for path in paths:
        file_rows = 0
        file_sha = hashlib.sha256()
        for line_number, row, raw_line in iter_jsonl(path):
            total_rows += 1
            file_rows += 1
            file_sha.update(raw_line.encode("utf-8", errors="ignore"))
            if row.get("_bad_json"):
                bad_json += 1
                continue
            split = _infer_split(path, row, expected_split=expected_split)
            modality = _modality(row)
            source = _first(row, "source_id", "dataset_name", "source", "source_uri", default="unknown")
            use_policy = _first(row, "use_policy", "policy", default="unknown")
            license_id = _first(row, "license", "license_id", default="unknown")
            contamination = _first(row, "contamination_status", default="unknown")
            by_modality[modality] += 1
            by_source[source] += 1
            by_split[split] += 1
            by_use_policy[use_policy] += 1
            by_license[license_id] += 1
            by_contamination[contamination] += 1
            matrix[(modality, source, split, use_policy)] += 1
            record_id = _record_id(row)
            if record_id:
                first_seen = seen_ids.get(record_id)
                if first_seen is None:
                    seen_ids[record_id] = {"path": str(path), "line": line_number}
                else:
                    duplicate_ids += 1
                    duplicate_id_rows.append(
                        {
                            "record_id": record_id,
                            "path": str(path),
                            "line": line_number,
                            "first_path": first_seen["path"],
                            "first_line": first_seen["line"],
                            "source_id": source,
                            "modality": modality,
                        }
                    )
            if modality.strip().lower() in {"", "unknown", "none", "null"}:
                missing_modality_rows.append({"path": str(path), "line": line_number, "source_id": source})
            declared_split = _declared_split(row)
            if expected_split and declared_split and _canonical_split(declared_split) != _canonical_split(expected_split):
                split_mismatch_rows.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "source_id": source,
                        "declared_split": declared_split,
                        "expected_split": expected_split,
                    }
                )
            target_tokens = _target_token_count(row)
            if target_tokens > 0:
                rows_with_target_tokens += 1
            if target_tokens <= 1 and not _has_media_payload(row):
                one_token_junk_rows.append({"path": str(path), "line": line_number, "source_id": source, "modality": modality, "target_tokens": target_tokens})
            if isinstance(row.get("artifact_token_ids"), list) and row["artifact_token_ids"]:
                rows_with_artifact_tokens += 1
            payload_hash = _sha256_text(_json_blob(row))
            if payload_hash in payload_hashes:
                duplicate_payloads += 1
            payload_hashes.add(payload_hash)
            blob = _json_blob(row)[:100_000]
            if split == "train" and TRAIN_LEAK_RE.search(blob):
                train_leak_rows.append({"path": str(path), "line": line_number, "source_id": source, "modality": modality})
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else 0,
                "rows": file_rows,
                "sha256": file_sha.hexdigest(),
            }
        )

    fail_reasons: list[str] = []
    if bad_json:
        fail_reasons.append("bad_json")
    if duplicate_payloads:
        fail_reasons.append("duplicate_payloads")
    if duplicate_ids:
        fail_reasons.append("duplicate_ids")
    if missing_modality_rows:
        fail_reasons.append("missing_modality_metadata")
    if one_token_junk_rows:
        fail_reasons.append("one_token_junk_rows")
    if split_mismatch_rows:
        fail_reasons.append("split_mismatch")
    if fail_on_train_leakage and train_leak_rows:
        fail_reasons.append("train_eval_leakage_markers")
    return {
        "schema": SCHEMA,
        "status": "failed" if fail_reasons else "passed",
        "fail_reasons": fail_reasons,
        "rows": total_rows,
        "files": files,
        "counts": {
            "bad_json": bad_json,
            "duplicate_ids": duplicate_ids,
            "duplicate_payloads": duplicate_payloads,
            "missing_modality_metadata": len(missing_modality_rows),
            "one_token_junk_rows": len(one_token_junk_rows),
            "split_mismatch": len(split_mismatch_rows),
            "train_eval_leakage_markers": len(train_leak_rows),
            "rows_with_target_tokens": rows_with_target_tokens,
            "rows_with_artifact_tokens": rows_with_artifact_tokens,
        },
        "by_modality": dict(sorted(by_modality.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_use_policy": dict(sorted(by_use_policy.items())),
        "by_license": dict(sorted(by_license.items())),
        "by_contamination": dict(sorted(by_contamination.items())),
        "by_modality_source_split_policy": [
            {"modality": modality, "source_id": source, "split": split, "use_policy": policy, "rows": rows}
            for (modality, source, split, policy), rows in sorted(matrix.items())
        ],
        "duplicate_id_examples": duplicate_id_rows[:50],
        "missing_modality_examples": missing_modality_rows[:50],
        "one_token_junk_examples": one_token_junk_rows[:50],
        "split_mismatch_examples": split_mismatch_rows[:50],
        "train_leak_examples": train_leak_rows[:50],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a source/modality/split index for final Omnicoder JSONL datasets.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-split", "--expected_split", dest="expected_split", default="")
    parser.add_argument("--allow-train-leakage-markers", "--allow_train_leakage_markers", dest="allow_train_leakage_markers", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [Path(value) for value in args.input]
    payload = build_index(paths, expected_split=str(args.expected_split or ""), fail_on_train_leakage=not bool(args.allow_train_leakage_markers))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": payload["rows"], "out": str(out), "fail_reasons": payload["fail_reasons"]}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

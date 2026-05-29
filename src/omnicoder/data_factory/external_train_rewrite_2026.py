from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from omnicoder.data_factory.dataset_integrity_2026 import audit_dataset_integrity, row_prompt_target


SCHEMA = "omnicoder.external_train_rewrite_2026.v1"
NEAR_DUP_MIN_NGRAMS = 8
NEAR_DUP_THRESHOLD = 0.92
NEAR_DUP_NGRAM = 5
NEAR_DUP_MAX_CANDIDATES = 24
TRAINABLE_POLICIES = {"train", "internal_train", "distill_train", "train_ok"}
CLEAN_CONTAMINATION_STATUSES = {"", "clean", "clear", "passed", "ok", "none", "unknown"}

NONTRAIN_EXACT_NAMES = {
    "blocked_until_review.jsonl",
    "eval_holdout.jsonl",
    "eval_holdout_all_external.jsonl",
    "rejected_external.jsonl",
    "research_internal.jsonl",
    "research_internal_all_external.jsonl",
}
NONTRAIN_SUFFIXES = (
    "_all.jsonl",
    "_blocked_until_review.jsonl",
    "_eval_holdout.jsonl",
    "_research_internal.jsonl",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def _safe_stem(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or fallback


def _record_id(row: dict[str, Any]) -> str:
    for container in (row, row.get("metadata"), row.get("lineage"), row.get("source_payload")):
        if not isinstance(container, dict):
            continue
        for key in ("record_id", "id", "uid", "uuid", "example_id", "sample_id", "row_id"):
            value = container.get(key)
            if value not in (None, "", [], {}):
                return str(value)
    return ""


def _payload_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _surface_token_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "")))


def _has_structured_training_payload(row: dict[str, Any]) -> bool:
    for key in (
        "artifact_token_ids",
        "target_token_ids",
        "assistant_token_ids",
        "media_tokens",
        "image_tokens",
        "video_tokens",
        "audio_tokens",
        "music_tokens",
        "tts_tokens",
        "speech_tokens",
    ):
        value = row.get(key)
        if isinstance(value, list) and len(value) > 1:
            return True
    for container in (row, row.get("target_json"), row.get("output_json"), row.get("teacher_output")):
        if not isinstance(container, dict):
            continue
        if container.get("generated_artifact") or container.get("artifact_tokens") or container.get("artifact_token_ids"):
            return True
        if container.get("tool_calls") or container.get("tool_results") or container.get("tool_result"):
            return True
    if row.get("tool_calls") and (row.get("tool_results") or row.get("tool_result") or row.get("verifier")):
        return True
    return False


def one_token_train_target_reason(row: dict[str, Any]) -> str:
    if _has_structured_training_payload(row):
        return ""
    _, target = row_prompt_target(row)
    if _surface_token_count(target) <= 1:
        return "one_token_train_target"
    return ""


def _near_duplicate_fingerprint(row: dict[str, Any], *, ngram: int = NEAR_DUP_NGRAM) -> set[str]:
    prompt, target = row_prompt_target(row)
    tokens = _word_tokens(f"{prompt}\n{target}")
    if len(tokens) < max(ngram, NEAR_DUP_MIN_NGRAMS):
        return set()
    return {" ".join(tokens[index : index + ngram]) for index in range(0, len(tokens) - ngram + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _nested_first(row: dict[str, Any], *paths: tuple[str, ...]) -> str:
    for path in paths:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _quality_value(row: dict[str, Any]) -> float | None:
    for key in ("quality_score", "score", "reward"):
        if row.get(key) not in (None, ""):
            try:
                return max(0.0, min(1.0, float(row[key])))
            except (TypeError, ValueError):
                return None
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    for key in ("score", "quality_score", "overall", "value"):
        if quality.get(key) not in (None, ""):
            try:
                return max(0.0, min(1.0, float(quality[key])))
            except (TypeError, ValueError):
                return None
    return None


def current_integrity_block_reason(row: dict[str, Any]) -> str:
    prompt, target = row_prompt_target(row)
    audit = audit_dataset_integrity(
        row,
        prompt=prompt,
        target=target,
        modality=str(row.get("modality") or row.get("target_modality") or row.get("declared_target_modality") or ""),
        refs=[],
        scan_artifacts=False,
    )
    row["dataset_integrity_2026_current"] = audit
    if audit.get("accepted") is not False:
        return ""
    reasons = [str(reason) for reason in audit.get("reasons") or ["unknown"]]
    return "dataset_integrity_current:" + ",".join(sorted(reasons)[:8])


def train_block_reason(row: dict[str, Any], *, recheck_integrity: bool = True) -> str:
    if str(row.get("training_bucket") or "train").strip().lower() != "train":
        return "non_train_bucket"
    integrity = row.get("dataset_integrity_2026")
    if isinstance(integrity, dict) and integrity.get("accepted") is False:
        return "dataset_integrity_rejected"
    if row.get("synthetic_train_blocked") is True:
        return "synthetic_train_blocked"
    if row.get("train_quarantine_reasons") not in (None, "", [], {}):
        return "train_quarantine_reasons"
    quality = _quality_value(row)
    if quality is not None and quality < 0.55:
        return "low_quality_score"
    policy = str(row.get("use_policy") or row.get("policy") or "train").strip().lower()
    if policy and policy not in TRAINABLE_POLICIES:
        return f"use_policy:{policy}"
    contamination = _nested_first(
        row,
        ("contamination", "status"),
        ("contamination", "label"),
        ("curation", "contamination_status"),
        ("metadata", "contamination_status"),
        ("metadata", "contamination_class"),
    ).strip().lower()
    if contamination and contamination not in CLEAN_CONTAMINATION_STATUSES:
        return f"contamination:{contamination}"
    if recheck_integrity:
        reason = current_integrity_block_reason(row)
        if reason:
            return reason
    reason = one_token_train_target_reason(row)
    if reason:
        return reason
    return ""


def is_train_bucket_file(path: Path) -> bool:
    if path.suffix.lower() != ".jsonl":
        return False
    name = path.name
    if name in NONTRAIN_EXACT_NAMES:
        return False
    if name.endswith(NONTRAIN_SUFFIXES):
        return False
    return True


def clean_train_rows(rows: Iterable[dict[str, Any]], skipped: Counter[str] | None = None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_payloads: set[str] = set()
    near_duplicates: list[set[str]] = []
    near_inverted: dict[str, list[int]] = {}
    for row in rows:
        reason = train_block_reason(row)
        if reason:
            if skipped is not None:
                skipped[reason] += 1
            continue
        row = dict(row)
        row["training_bucket"] = "train"
        row.setdefault("split", "train")
        row.setdefault("use_policy", "train")
        record_id = _record_id(row)
        if record_id:
            if record_id in seen_ids:
                if skipped is not None:
                    skipped["duplicate_record_id"] += 1
                continue
            seen_ids.add(record_id)
        payload_hash = _payload_hash(row)
        if payload_hash in seen_payloads:
            if skipped is not None:
                skipped["duplicate_payload"] += 1
            continue
        seen_payloads.add(payload_hash)
        fingerprint = _near_duplicate_fingerprint(row)
        if len(fingerprint) >= NEAR_DUP_MIN_NGRAMS:
            candidate_hits: Counter[int] = Counter()
            for gram in fingerprint:
                for candidate_index in near_inverted.get(gram, ()):
                    candidate_hits[candidate_index] += 1
            duplicate_score = 0.0
            for candidate_index, shared in candidate_hits.most_common(NEAR_DUP_MAX_CANDIDATES):
                other = near_duplicates[candidate_index]
                if shared / max(1, min(len(fingerprint), len(other))) < NEAR_DUP_THRESHOLD:
                    continue
                duplicate_score = max(duplicate_score, _jaccard(fingerprint, other))
                if duplicate_score >= NEAR_DUP_THRESHOLD:
                    break
            if duplicate_score >= NEAR_DUP_THRESHOLD:
                if skipped is not None:
                    skipped["near_duplicate_payload"] += 1
                continue
            index = len(near_duplicates)
            near_duplicates.append(fingerprint)
            for gram in fingerprint:
                near_inverted.setdefault(gram, []).append(index)
        cleaned.append(row)
    return cleaned


def target_paths_for_row(jsonl_dir: Path, row: dict[str, Any]) -> set[Path]:
    paths = {jsonl_dir / "train.jsonl", jsonl_dir / "train_all_external.jsonl"}
    family = _safe_stem(row.get("dataset_family") or row.get("family"), "")
    modality = _safe_stem(row.get("modality") or row.get("target_modality") or row.get("declared_target_modality"), "")
    if family:
        paths.add(jsonl_dir / f"{family}.jsonl")
    if modality:
        paths.add(jsonl_dir / f"{modality}.jsonl")
    return paths


def rewrite_external_train_bucket(
    accepted_jsonl: Path,
    jsonl_dir: Path,
    out_path: Path,
    *,
    source_manifest: Path | None = None,
) -> dict[str, Any]:
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    accepted_rows = clean_train_rows(iter_jsonl(accepted_jsonl), skipped)
    planned: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_rows:
        for path in target_paths_for_row(jsonl_dir, row):
            planned[path].append(row)

    existing_train_files = sorted(path for path in jsonl_dir.glob("*.jsonl") if is_train_bucket_file(path))
    paths_to_touch = sorted(set(existing_train_files) | set(planned))
    files_written: dict[str, int] = {}
    files_truncated: list[str] = []
    for path in paths_to_touch:
        rows = planned.get(path, [])
        count = write_jsonl(path, rows)
        rel = str(path.relative_to(jsonl_dir))
        files_written[rel] = count
        if count == 0:
            files_truncated.append(rel)

    by_family = Counter(_safe_stem(row.get("dataset_family") or row.get("family"), "unknown") for row in accepted_rows)
    by_modality = Counter(
        _safe_stem(row.get("modality") or row.get("target_modality") or row.get("declared_target_modality"), "unknown")
        for row in accepted_rows
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "rewritten_clean",
        "accepted_jsonl": str(accepted_jsonl),
        "jsonl_dir": str(jsonl_dir),
        "accepted_rows": len(accepted_rows),
        "skipped_rows": sum(skipped.values()),
        "skipped_rows_by_reason": dict(sorted(skipped.items())),
        "files_written": dict(sorted(files_written.items())),
        "files_truncated": files_truncated,
        "by_family": dict(sorted(by_family.items())),
        "by_modality": dict(sorted(by_modality.items())),
    }

    if source_manifest is not None and source_manifest.exists():
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        records = manifest.get("records") if isinstance(manifest.get("records"), dict) else {}
        records["train"] = len(accepted_rows)
        records["total_training_rows"] = len(accepted_rows) + sum(
            int(records.get(key) or 0) for key in ("research_internal", "eval_holdout", "blocked_until_review")
        )
        manifest["records"] = records
        manifest["clean_train_families"] = report["by_family"]
        manifest["clean_train_modalities"] = report["by_modality"]
        manifest["integrity_rewrite"] = report
        manifest["promotion_allowed"] = len(accepted_rows) > 0
        manifest["promotion_status"] = "integrity_rewritten_pending_index" if accepted_rows else "no_clean_train_rows"
        source_manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["source_manifest"] = str(source_manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite external train bucket files from dataset-integrity accepted rows.")
    parser.add_argument("--accepted", required=True, help="Accepted JSONL emitted by dataset_integrity_2026.")
    parser.add_argument("--jsonl-dir", required=True, help="External dataset JSONL directory to rewrite.")
    parser.add_argument("--out", required=True, help="Rewrite manifest path.")
    parser.add_argument("--source-manifest", default="", help="Optional external_dataset_manifest.json to update with clean train counts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = rewrite_external_train_bucket(
        Path(args.accepted),
        Path(args.jsonl_dir),
        Path(args.out),
        source_manifest=Path(args.source_manifest) if args.source_manifest else None,
    )
    print(json.dumps({"status": report["status"], "accepted_rows": report["accepted_rows"], "out": args.out}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

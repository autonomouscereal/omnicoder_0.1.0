from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Sequence

from omnicoder.training import training_orchestration_2026


SCHEMA_VERSION = "2026-05-24"
DEFAULT_PROFILE = "profiles/dataset_curation_2026.json"
DEFAULT_TRAINING_PROFILE = "profiles/training_orchestration_2026.json"
DEFAULT_OUT_DIR = "weights/external_datasets_2026/latest"

FAMILY_TO_MODALITY = {
    "math_reasoning": "text",
    "coding_agentic": "code",
    "agentic_tool_reasoning": "tool",
    "terminal_browser_agents": "tool",
    "long_context": "long_context",
    "omnimodal_understanding": "text",
    "image_generation_editing": "image",
    "video_generation": "video",
    "speech_audio": "audio",
    "audio_music_speech": "audio",
    "music_generation": "music",
}

TRAINABLE_POLICIES = {"train", "internal_train", "distill_train", "train_ok"}
INTERNAL_ONLY_POLICIES = {"research_internal", "distill_seed", "internal_distill_seed", "reward_only"}
EVAL_ONLY_POLICIES = {"eval", "eval_only", "benchmark_holdout"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.exists() or not source.is_file():
        return
    for line_number, line in enumerate(source.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            payload = {"text": line, "parse_error": str(exc), "line_number": line_number}
        if isinstance(payload, dict):
            payload.setdefault("line_number", line_number)
            yield payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def profile_entries(profile: dict[str, Any]) -> list[dict[str, Any]]:
    registry = profile.get("external_dataset_registry_2026")
    if not isinstance(registry, dict):
        return []
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and entry.get("enabled", True)]


def training_profile_path(profile: dict[str, Any], root: Path) -> Path:
    registry = profile.get("external_dataset_registry_2026")
    configured = None
    if isinstance(registry, dict):
        configured = registry.get("training_profile")
    configured = configured or DEFAULT_TRAINING_PROFILE
    return resolve_path(str(configured), root)


def first_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [first_string(item) for item in value[:8]]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("content", "text", "caption", "answer", "solution", "prompt", "question", "instruction"):
            if key in value:
                text = first_string(value[key])
                if text:
                    return text
    return ""


def dotted_value(record: dict[str, Any], key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def field_text(record: dict[str, Any], fields: Any) -> str:
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        fields = []
    for field in fields:
        if not isinstance(field, str):
            continue
        value = dotted_value(record, field)
        text = first_string(value)
        if text:
            return text
    return ""


def fallback_prompt(entry: dict[str, Any], record: dict[str, Any]) -> str:
    family = str(entry.get("family") or "external_dataset")
    name = str(entry.get("name") or entry.get("hf_id") or family)
    if family == "math_reasoning":
        return "Solve the math problem with verifiable reasoning and preserve the final answer."
    if family == "coding_agentic":
        return "Solve or repair the coding task, preserving tests, patch intent, and terminal evidence."
    if family in {"agentic_tool_reasoning", "terminal_browser_agents"}:
        return "Complete the agentic tool-use trajectory with correct tool calls, observations, verification, and recovery behavior."
    if family == "image_generation_editing":
        return "Generate, edit, critique, or preserve the image according to the multimodal instruction."
    if family == "video_generation":
        return "Generate or critique the video with temporal consistency, motion, and prompt adherence."
    if family in {"speech_audio", "audio_music_speech"}:
        return "Transcribe, caption, generate, or critique the audio artifact with grounded reasoning."
    if family == "music_generation":
        return "Generate, caption, or critique music with style, tempo, structure, lyrics, and production notes."
    return f"Learn the high-quality 2026 dataset signal from {name}."


def fallback_target(entry: dict[str, Any], record: dict[str, Any]) -> str:
    field_map = entry.get("field_map") if isinstance(entry.get("field_map"), dict) else {}
    text = field_text(
        record,
        field_map.get("target")
        or [
            "solution",
            "answer",
            "completion",
            "response",
            "output",
            "target",
            "caption",
            "detailed_caption",
            "Brief_Caption",
            "Detailed_Caption",
            "main_caption",
            "alt_caption",
        ],
    )
    if text:
        return text
    strings: list[str] = []

    def visit(value: Any) -> None:
        if len(strings) >= 32:
            return
        if isinstance(value, str) and value.strip():
            strings.append(value.strip())
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value[:16]:
                visit(child)

    visit(record)
    return "\n".join(strings)[:4000]


def source_use_bucket(entry: dict[str, Any]) -> str:
    policy = str(entry.get("use_policy") or entry.get("license_tier") or "").lower()
    if policy in TRAINABLE_POLICIES:
        return "train"
    if policy in EVAL_ONLY_POLICIES:
        return "eval_holdout"
    if policy in INTERNAL_ONLY_POLICIES:
        return "research_internal"
    if str(entry.get("license_tier") or "").lower() in {"eval_only", "research_only", "non_commercial", "non_commercial_no_derivatives"}:
        return "research_internal"
    return "blocked_until_review"


def synthetic_seed_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = entry.get("distillation_prompts") or entry.get("prompt_seeds") or []
    if isinstance(seeds, dict):
        seeds = [seeds]
    if not isinstance(seeds, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, 1):
        if isinstance(seed, str):
            payload = {"instruction": seed}
        elif isinstance(seed, dict):
            payload = dict(seed)
        else:
            continue
        payload.setdefault("seed_index", index)
        payload.setdefault("synthetic_seed", True)
        rows.append(payload)
    return rows


def rows_from_local_jsonl(entry: dict[str, Any], root: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_path = entry.get("local_jsonl") or entry.get("local_path")
    if not raw_path:
        return [], {"status": "skipped", "reason": "no_local_path"}
    path = resolve_path(str(raw_path), root)
    if not path.exists():
        return [], {"status": "skipped", "reason": "local_path_missing", "path": str(path)}
    rows = list(islice(iter_jsonl(path), limit if limit > 0 else None))
    return rows, {"status": "ok", "source": "local_jsonl", "path": str(path), "records": len(rows)}


def rows_from_huggingface(entry: dict[str, Any], limit: int, streaming: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hf_id = entry.get("hf_id")
    if not hf_id:
        return [], {"status": "skipped", "reason": "no_hf_id"}
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        return [], {"status": "failed", "reason": "datasets_import_failed", "error": repr(exc)}
    splits = entry.get("splits")
    if isinstance(splits, str):
        splits = [splits]
    if not isinstance(splits, list) or not splits:
        splits = ["train"]
    config = entry.get("config")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    per_split: dict[str, int] = {}
    remaining = limit
    for split in splits:
        if limit > 0 and remaining <= 0:
            break
        try:
            if config:
                dataset = load_dataset(str(hf_id), str(config), split=str(split), streaming=streaming)
            else:
                dataset = load_dataset(str(hf_id), split=str(split), streaming=streaming)
        except Exception as exc:
            errors.append(f"{split}: {repr(exc)}")
            continue
        take = remaining if limit > 0 else 0
        count = 0
        iterator = dataset if take <= 0 else islice(dataset, take)
        for raw in iterator:
            if isinstance(raw, dict):
                item = dict(raw)
                item.setdefault("_hf_split", str(split))
                rows.append(item)
                count += 1
        per_split[str(split)] = count
        if limit > 0:
            remaining -= count
    status = "ok" if rows else "failed" if errors else "empty"
    return rows, {
        "status": status,
        "source": "huggingface",
        "hf_id": str(hf_id),
        "config": config,
        "streaming": streaming,
        "records": len(rows),
        "per_split": per_split,
        "errors": errors[:8],
    }


def materialize_source_rows(entry: dict[str, Any], root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = int(entry.get("max_records") or args.max_records_per_dataset or 0)
    local_rows, local_status = rows_from_local_jsonl(entry, root, limit)
    if local_rows:
        return local_rows, local_status
    if args.download:
        hf_rows, hf_status = rows_from_huggingface(entry, limit, streaming=not args.no_streaming)
        if hf_rows:
            return hf_rows, hf_status
        seeds = synthetic_seed_rows(entry)
        if seeds:
            if limit > 0:
                seeds = seeds[:limit]
            return seeds, {"status": "ok", "source": "distillation_prompts_after_hf_attempt", "records": len(seeds), "huggingface_status": hf_status}
        if hf_status.get("status") == "failed":
            return hf_rows, hf_status
    seeds = synthetic_seed_rows(entry)
    if limit > 0:
        seeds = seeds[:limit]
    if seeds:
        return seeds, {"status": "ok", "source": "distillation_prompts", "records": len(seeds), "fallback_after": local_status}
    return [], {"status": "skipped", "reason": "no_rows", "local": local_status}


def record_to_training_row(entry: dict[str, Any], record: dict[str, Any], plan: dict[str, Any], row_index: int) -> dict[str, Any] | None:
    field_map = entry.get("field_map") if isinstance(entry.get("field_map"), dict) else {}
    prompt = field_text(record, field_map.get("prompt") or ["instruction", "question", "prompt", "input", "problem", "title", "Brief_Caption"])
    target = fallback_target(entry, record)
    if not prompt:
        prompt = fallback_prompt(entry, record)
    if not target:
        target = field_text(record, field_map.get("prompt") or ["instruction", "question", "prompt", "problem"])
    if not target or len(target.strip()) < int(entry.get("min_target_chars") or 1):
        return None
    family = str(entry.get("family") or "external_dataset")
    modality = str(entry.get("target_modality") or FAMILY_TO_MODALITY.get(family, "text"))
    source_uri = str(entry.get("url") or entry.get("hf_id") or entry.get("name") or family)
    raw_id = field_text(record, field_map.get("id") or ["id", "task_id", "problem_id", "instance_id", "ID", "uid"]) or f"row-{row_index}"
    source_payload = {
        "source_id": stable_hash({"dataset": entry.get("name"), "raw_id": raw_id, "row_index": row_index}),
        "source_date": str(entry.get("source_date") or "2026-05-24"),
        "quality": {"score": float(entry.get("quality_score") or 0.82), "label": "accepted_external_2026"},
        "contamination": {"status": "unknown", "note": "external registry row requires downstream protected benchmark scan"},
        "dataset_name": entry.get("name"),
        "dataset_family": family,
        "hf_id": entry.get("hf_id"),
        "license": entry.get("license"),
        "license_tier": entry.get("license_tier"),
        "use_policy": entry.get("use_policy"),
        "skill_domain": entry.get("skill_domain") or family,
        "raw_record": record if bool(entry.get("keep_raw_record", False)) else {"raw_id": raw_id, "row_hash": stable_hash(record)},
    }
    row = training_orchestration_2026.make_training_record(
        modality,
        prompt[: int(plan.get("target_text_chars") or 3000)],
        target[: int(plan.get("target_text_chars") or 3000)],
        source_uri,
        plan,
        source_payload=source_payload,
    )
    row["curriculum_axes"] = sorted(
        set(
            str(item)
            for item in (
                entry.get("curriculum_axes")
                if isinstance(entry.get("curriculum_axes"), list)
                else [family, entry.get("skill_domain") or family]
            )
            if item
        )
    )
    row["dataset_family"] = family
    row["dataset_name"] = str(entry.get("name") or entry.get("hf_id") or family)
    row["license_tier"] = str(entry.get("license_tier") or "unknown")
    row["use_policy"] = str(entry.get("use_policy") or "blocked_until_review")
    row["training_bucket"] = source_use_bucket(entry)
    return row


def build_expansion(profile_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    profile = read_json(profile_path)
    training_profile = training_orchestration_2026.load_profile(training_profile_path(profile, root))
    plan = training_orchestration_2026.profile_cfg(training_profile)["training_plan"]
    entries = profile_entries(profile)
    jsonl_dir = out_dir / "jsonl"
    manifests_dir = out_dir / "manifests"
    cards_dir = out_dir / "dataset_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    acquisition: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in entries:
        raw_rows, status = materialize_source_rows(entry, root, args)
        family = str(entry.get("family") or "external_dataset")
        status.update(
            {
                "name": entry.get("name"),
                "family": family,
                "hf_id": entry.get("hf_id"),
                "url": entry.get("url"),
                "license": entry.get("license"),
                "license_tier": entry.get("license_tier"),
                "use_policy": entry.get("use_policy"),
                "bucket": source_use_bucket(entry),
            }
        )
        acquisition.append(status)
        for index, raw in enumerate(raw_rows, 1):
            row = record_to_training_row(entry, raw, plan, index)
            if row is None:
                rejected.append({"dataset": entry.get("name"), "family": family, "index": index, "reason": "empty_or_short_target"})
                continue
            rows_by_bucket[str(row["training_bucket"])].append(row)
            rows_by_family[family].append(row)
            rows_by_modality[str(row["modality"])].append(row)
    for family, rows in sorted(rows_by_family.items()):
        write_jsonl(jsonl_dir / f"{family}.jsonl", rows)
    for modality, rows in sorted(rows_by_modality.items()):
        write_jsonl(jsonl_dir / f"{modality}.jsonl", rows)
    for bucket, rows in sorted(rows_by_bucket.items()):
        write_jsonl(jsonl_dir / f"{bucket}.jsonl", rows)
    train_rows = rows_by_bucket.get("train", [])
    research_rows = rows_by_bucket.get("research_internal", [])
    eval_rows = rows_by_bucket.get("eval_holdout", [])
    blocked_rows = rows_by_bucket.get("blocked_until_review", [])
    write_jsonl(jsonl_dir / "train_all_external.jsonl", train_rows)
    write_jsonl(jsonl_dir / "research_internal_all_external.jsonl", research_rows)
    write_jsonl(jsonl_dir / "eval_holdout_all_external.jsonl", eval_rows)
    write_jsonl(jsonl_dir / "blocked_until_review.jsonl", blocked_rows)
    write_jsonl(jsonl_dir / "rejected_external.jsonl", rejected)
    manifest = {
        "schema": "omnicoder.external_dataset_expansion_2026.v1",
        "version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "profile": str(profile_path),
        "out_dir": str(out_dir),
        "download_requested": bool(args.download),
        "streaming": not bool(args.no_streaming),
        "datasets": acquisition,
        "records": {
            "train": len(train_rows),
            "research_internal": len(research_rows),
            "eval_holdout": len(eval_rows),
            "blocked_until_review": len(blocked_rows),
            "rejected": len(rejected),
            "total_training_rows": sum(len(rows) for rows in rows_by_bucket.values()),
        },
        "families": {family: len(rows) for family, rows in sorted(rows_by_family.items())},
        "modalities": {modality: len(rows) for modality, rows in sorted(rows_by_modality.items())},
        "license_tiers": dict(sorted(Counter(str(row.get("license_tier") or "unknown") for rows in rows_by_bucket.values() for row in rows).items())),
        "training_paths": {
            "train_all_external": str(jsonl_dir / "train_all_external.jsonl"),
            "research_internal_all_external": str(jsonl_dir / "research_internal_all_external.jsonl"),
            "eval_holdout_all_external": str(jsonl_dir / "eval_holdout_all_external.jsonl"),
        },
        "promotion_policy": "Only train bucket rows may be merged into release weights. research_internal rows are internal distillation/reward candidates. eval_holdout rows are benchmark/evaluation only.",
    }
    write_json(manifests_dir / "external_dataset_manifest.json", manifest)
    card_lines = [
        "# Omnicoder External Dataset Expansion 2026",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Train rows: {manifest['records']['train']}",
        f"- Research/internal rows: {manifest['records']['research_internal']}",
        f"- Eval holdout rows: {manifest['records']['eval_holdout']}",
        f"- Blocked rows: {manifest['records']['blocked_until_review']}",
        "",
        "## Families",
    ]
    for family, count in manifest["families"].items():
        card_lines.append(f"- {family}: {count}")
    card_lines.extend(["", "## License Tiers"])
    for tier, count in manifest["license_tiers"].items():
        card_lines.append(f"- {tier}: {count}")
    card_lines.extend(["", "## Policy", manifest["promotion_policy"], ""])
    cards_dir.mkdir(parents=True, exist_ok=True)
    (cards_dir / "external_dataset_card_2026.md").write_text("\n".join(card_lines), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize 2025-2026 external dataset expansion rows for Omnicoder training and distillation")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--download", action="store_true", help="Attempt Hugging Face streaming downloads when local JSONL rows are absent")
    parser.add_argument("--no-streaming", action="store_true", help="Use regular load_dataset instead of streaming")
    parser.add_argument("--max-records-per-dataset", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    args = parser.parse_args(argv)
    if args.command != "build":
        raise SystemExit(f"unknown command: {args.command}")
    manifest = build_expansion(resolve_path(args.profile, repo_root()), resolve_path(args.out_dir, repo_root()), args)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0 if manifest["records"]["total_training_rows"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

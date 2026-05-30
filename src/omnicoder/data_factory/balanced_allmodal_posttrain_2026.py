from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from omnicoder.training.training_orchestration_2026 import (
    DEFAULT_PROFILE,
    DEFAULT_STAGE_ORDER,
    iter_jsonl,
    load_profile,
    now_iso,
    profile_cfg,
    repo_root,
    resolve_path,
    row_prompt,
    row_target,
    stable_hash,
    write_json,
    write_jsonl,
)
from omnicoder.data_factory.curation_policy_2026 import (
    CurationPolicyConfig,
    artifact_refs as policy_artifact_refs,
    audit_training_record,
    message_prompt_target as policy_message_prompt_target,
)


KNOWN_MODALITIES = {"text", "code", "tool", "image", "video", "audio", "music", "tts", "long_context", "math", "ocr"}
MODALITY_PRIORITY = ("image", "video", "audio", "music", "tts", "ocr", "code", "math", "tool", "long_context", "text")
PROFILE_JSONL_KEYS: tuple[tuple[str, str], ...] = (
    ("text_jsonl", "text"),
    ("code_jsonl", "code"),
    ("trace_jsonl", "tool"),
    ("image_jsonl", "image"),
    ("video_jsonl", "video"),
    ("audio_jsonl", "audio"),
    ("music_jsonl", "music"),
)
BAD_CONTAMINATION_MARKERS = (
    "benchmark_leak",
    "benchmark_marker",
    "contaminated",
    "dirty",
    "eval_holdout",
    "pending",
    "protected_eval",
    "public_dev_eval",
    "quarantine",
    "rejected",
    "suspect",
    "unknown",
)
FIXTURE_PATH_MARKERS = ("\\examples\\", "/examples/", "\\smoke\\", "/smoke/", "\\fixtures\\", "/fixtures/")
REFUSAL_BOILERPLATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bas an ai(?: language)? model\b",
        r"\bi (?:can(?:not|'t|[’`]t)|am not able to|am unable to|(?:'m|[’`]m) unable to)\b",
        r"\bi (?:won(?:'t|[’`]t)|will not|must refuse|have to refuse|refuse to)\b",
        r"\b(?:cannot|can(?:'t|[’`]t)|can not|unable to) assist\b",
        r"\bnot able to (?:assist|help|comply|provide)\b",
        r"\b(?:against|violates?) (?:the )?(?:policy|safety policy|guidelines)\b",
        r"\b(?:policy|guidelines?) (?:prevents?|prohibits?|disallows?)\b",
        r"\b(?:refusal|refuse|refused|refusing)\b",
        r"\bsafety[_ -]?negative\b",
        r"\btool[_ -]?safety[_ -]?negative\b",
        r"\bkto[_ -]?or[_ -]?safety\b",
        r"\btrain[_ -]?refusal\b",
        r"\bunapproved[_ -]?destructive[_ -]?tool[_ -]?use\b",
        r"\bcredential[_ -]?and[_ -]?hidden[_ -]?eval[_ -]?safety\b",
    )
)
REFUSAL_SOURCE_MARKERS = (
    "tool_safety_negative",
    "safety_negative",
    "safety_negative_alignment",
    "kto_or_safety",
    "train_refusal",
    "unapproved_destructive_tool_use",
    "credential_and_hidden_eval_safety",
    "refusal_alignment",
    "refusal_policy",
)


def clamp_text(value: str, limit: int) -> str:
    text = value.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip()


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    if value is None:
        return ""
    return str(value).strip()


def normalize_modality(value: Any) -> str:
    text = text_value(value).lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    if text in KNOWN_MODALITIES:
        return text
    if any(marker in text for marker in ("long_context", "longctx", "million_context", "1m_context")):
        return "long_context"
    if "ocr" in text or "document_vision" in text:
        return "ocr"
    if any(marker in text for marker in ("math", "gsm", "aime", "olympiad", "proof")):
        return "math"
    if any(marker in text for marker in ("vision", "image", "picture", "imgedit", "qwen_image")):
        return "image"
    if any(marker in text for marker in ("video", "movie", "ltx", "image_to_video", "text_to_video")):
        return "video"
    if any(marker in text for marker in ("music", "song", "melody", "ace_step", "acestep", "midi")):
        return "music"
    if any(marker in text for marker in ("tts", "speech_synthesis", "text_to_speech", "voice_clone")):
        return "tts"
    if any(marker in text for marker in ("audio", "speech", "tts", "asr", "voice", "sound")):
        return "audio"
    if any(marker in text for marker in ("code", "coding", "swe", "terminal_bench", "livecodebench", "program")):
        return "code"
    if any(marker in text for marker in ("tool", "agent", "trace", "sft", "browser", "shell", "terminal", "codex", "claude", "hermes")):
        return "tool"
    if any(marker in text for marker in ("reasoning", "text", "instruction")):
        return "text"
    return ""


def row_modalities(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("modality", "media_family", "task_type", "domain", "source_id"):
        if row.get(key) is not None:
            values.append(row.get(key))
    modalities = row.get("modalities")
    if isinstance(modalities, list):
        values.extend(modalities)
    elif modalities is not None:
        values.append(modalities)
    normalized = [normalize_modality(value) for value in values]
    return [value for value in normalized if value]


def infer_modality(row: dict[str, Any], source_path: Path, source_hint: str = "") -> str:
    candidates = row_modalities(row)
    for preferred in MODALITY_PRIORITY:
        if preferred in candidates:
            return preferred
    source_modality = normalize_modality(source_hint)
    if source_modality:
        return source_modality
    file_hint = normalize_modality(" ".join([source_path.name, source_path.parent.name]))
    if file_hint:
        return file_hint
    return "text"


def message_prompt_target(row: dict[str, Any]) -> tuple[str, str]:
    prompt, target = policy_message_prompt_target(row)
    if prompt or target:
        return prompt, target
    return row_prompt(row), row_target(row)


def artifact_refs(row: dict[str, Any]) -> list[str]:
    return policy_artifact_refs(row)


def quality_value(row: dict[str, Any]) -> float:
    for key in ("quality_score", "score", "reward"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            continue
    if isinstance(row.get("quality"), dict):
        for key in ("score", "quality_score", "value"):
            value = row["quality"].get(key)
            if value is None:
                continue
            try:
                return max(0.0, min(1.0, float(value)))
            except Exception:
                continue
    return 0.0


def has_quality_value(row: dict[str, Any]) -> bool:
    for key in ("quality_score", "score", "reward"):
        if row.get(key) not in (None, ""):
            return True
    if isinstance(row.get("quality"), dict):
        return any(row["quality"].get(key) not in (None, "") for key in ("score", "quality_score", "value"))
    return False


def contamination_rejected(row: dict[str, Any]) -> bool:
    nested = row.get("contamination") if isinstance(row.get("contamination"), dict) else {}
    status = text_value(row.get("contamination_status") or row.get("decontamination_status") or nested.get("status") or "unknown").lower()
    return any(marker in status for marker in BAD_CONTAMINATION_MARKERS)


def fixture_path_rejected(source_path: Path) -> bool:
    text = str(source_path).replace("/", "\\").lower()
    return any(marker.replace("/", "\\") in text for marker in FIXTURE_PATH_MARKERS) or source_path.name.lower().startswith(("smoke_", "sample_", "fixture_"))


def refusal_boilerplate_rejected(row: dict[str, Any], prompt: str, target: str, source_path: Path) -> bool:
    metadata_keys = (
        "source_id",
        "dataset_name",
        "task_type",
        "domain",
        "training_kind",
        "curriculum_axes",
        "risk_labels",
        "safety_labels",
        "alignment_labels",
        "tags",
        "labels",
        "categories",
        "reward_axes",
        "verifier",
        "policy",
        "use_policy",
    )
    source_text = " ".join([source_path.name] + [clamp_text(text_value(row.get(key)), 4096) for key in metadata_keys]).lower()
    if any(marker in source_text for marker in REFUSAL_SOURCE_MARKERS):
        return True
    text = f"{source_text}\n{prompt}\n{target}"
    return any(pattern.search(text) for pattern in REFUSAL_BOILERPLATE_PATTERNS)


def profile_sources(profile: dict[str, Any], root: Path) -> list[tuple[Path, str]]:
    cfg = profile_cfg(profile)
    sources = cfg.get("real_sources") if isinstance(cfg.get("real_sources"), dict) else {}
    result: list[tuple[Path, str]] = []
    for key, modality in PROFILE_JSONL_KEYS:
        values = sources.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            result.append((resolve_path(value, root), modality))
    return result


def parse_sources(values: list[str], root: Path) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        modality = ""
        if "=" in value:
            modality, value = value.split("=", 1)
        elif "::" in value:
            modality, value = value.split("::", 1)
        result.append((resolve_path(value.strip(), root), normalize_modality(modality)))
    return result


def parse_caps(values: list[str], profile: dict[str, Any], default_cap: int) -> dict[str, int]:
    cfg = profile_cfg(profile)
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    configured = plan.get("max_records_per_modality_by_modality")
    caps: dict[str, int] = {}
    if isinstance(configured, dict):
        for modality, value in configured.items():
            normalized = normalize_modality(modality)
            if normalized:
                try:
                    caps[normalized] = max(0, int(value))
                except Exception:
                    pass
    for modality in KNOWN_MODALITIES:
        caps.setdefault(modality, max(0, int(default_cap)))
    for item in values:
        if "=" not in item:
            raise ValueError(f"--cap must be modality=count: {item!r}")
        modality, count = item.split("=", 1)
        normalized = normalize_modality(modality)
        if not normalized:
            raise ValueError(f"unknown modality in --cap: {item!r}")
        caps[normalized] = max(0, int(count))
    return caps


def parse_source_floors(values: list[str]) -> dict[str, int]:
    floors: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--source-floor must be source_name=count: {item!r}")
        source_name, count = item.split("=", 1)
        key = source_name.strip()
        if not key:
            raise ValueError(f"empty source name in --source-floor: {item!r}")
        floors[key] = max(0, int(count))
    return floors


def required_modalities(profile: dict[str, Any], override: str) -> list[str]:
    if override.strip():
        return [value for value in (normalize_modality(part) for part in override.split(",")) if value]
    cfg = profile_cfg(profile)
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    values = plan.get("required_modalities") if isinstance(plan.get("required_modalities"), list) else list(DEFAULT_STAGE_ORDER)
    return [value for value in (normalize_modality(item) for item in values) if value]


def build_base_row(
    row: dict[str, Any],
    *,
    prompt: str,
    target: str,
    modality: str,
    source_path: Path,
    line_number: int,
    kind: str,
) -> dict[str, Any]:
    return {
        "training_kind": kind,
        "record_id": stable_hash(
            {
                "kind": kind,
                "source": str(source_path),
                "line_number": line_number,
                "source_record_id": row.get("record_id"),
                "prompt": prompt[:512],
                "target": target[:512],
            }
        ),
        "source_record_id": row.get("record_id") or row.get("id"),
        "source_id": row.get("source_id") or source_path.name,
        "source_file": str(source_path),
        "source_line_number": int(line_number),
        "modality": modality,
        "modalities": sorted(set([modality] + row_modalities(row))),
        "artifact_refs": artifact_refs(row),
        "quality_score": quality_value(row),
        "contamination_status": row.get("contamination_status", "unknown"),
        "curriculum_stage": "balanced_allmodal_posttraining_2026",
    }


def sft_row(row: dict[str, Any], prompt: str, target: str, modality: str, source_path: Path, line_number: int) -> dict[str, Any]:
    base = build_base_row(row, prompt=prompt, target=target, modality=modality, source_path=source_path, line_number=line_number, kind="balanced_allmodal_sft")
    return {
        "schema": "omnicoder.posttraining_sft_2026.v1",
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}],
        **base,
    }


def reward_row(row: dict[str, Any], prompt: str, target: str, modality: str, source_path: Path, line_number: int) -> dict[str, Any]:
    base = build_base_row(row, prompt=prompt, target=target, modality=modality, source_path=source_path, line_number=line_number, kind="balanced_allmodal_reward")
    return {
        "schema": "omnicoder.posttraining_reward_2026.v1",
        "prompt": prompt,
        "response": target,
        "reward": quality_value(row),
        "reward_source": "balanced_allmodal_curation_quality_score",
        **base,
    }


def rlvr_row(row: dict[str, Any], prompt: str, target: str, modality: str, source_path: Path, line_number: int) -> dict[str, Any]:
    base = build_base_row(row, prompt=prompt, target=target, modality=modality, source_path=source_path, line_number=line_number, kind="balanced_allmodal_rlvr")
    return {
        "schema": "omnicoder.posttraining_rlvr_2026.v1",
        "prompt": prompt,
        "expected_answer": target,
        "verifier": "modality_grounded_exact_or_artifact_quality_judge",
        "reward_axes": [
            "answer_consistency",
            "artifact_reference_integrity",
            "modality_grounding",
            "tool_or_reasoning_correctness",
            "contamination_free",
        ],
        **base,
    }


def output_paths(out_dir: Path, out_jsonl: str, manifest: str) -> dict[str, Path]:
    if out_jsonl:
        sft = Path(out_jsonl)
        if not sft.is_absolute():
            sft = resolve_path(sft, repo_root())
        stem = sft.name[:-6] if sft.name.endswith(".jsonl") else sft.name
        reward = sft.with_name(stem.replace("_sft", "") + "_reward.jsonl")
        rlvr = sft.with_name(stem.replace("_sft", "") + "_rlvr.jsonl")
        manifest_path = Path(manifest) if manifest else sft.with_name(stem.replace("_sft", "") + "_manifest.json")
    else:
        sft = out_dir / "balanced_allmodal_sft.jsonl"
        reward = out_dir / "balanced_allmodal_reward.jsonl"
        rlvr = out_dir / "balanced_allmodal_rlvr.jsonl"
        manifest_path = Path(manifest) if manifest else out_dir / "balanced_allmodal_manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = resolve_path(manifest_path, repo_root())
    return {"sft": sft, "reward": reward, "rlvr": rlvr, "manifest": manifest_path}


def round_robin_rows(buckets: dict[str, list[dict[str, Any]]], order: list[str]) -> list[dict[str, Any]]:
    indexes = {modality: 0 for modality in order}
    result: list[dict[str, Any]] = []
    while True:
        progressed = False
        for modality in order:
            index = indexes[modality]
            bucket = buckets.get(modality, [])
            if index >= len(bucket):
                continue
            result.append(bucket[index])
            indexes[modality] = index + 1
            progressed = True
        if not progressed:
            break
    return result


def build_balanced_exports(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    if args.schema != "messages":
        raise ValueError("--schema currently supports only 'messages' for native pipeline reward replay")
    profile = load_profile(args.profile)
    cfg = profile_cfg(profile)
    plan = cfg.get("training_plan") if isinstance(cfg.get("training_plan"), dict) else {}
    required = required_modalities(profile, args.require_modalities)
    caps = parse_caps(args.cap, profile, args.max_records_per_modality)
    source_floors = parse_source_floors(args.source_floor)
    allow_fixture_data = bool(getattr(args, "allow_fixture_data", False))
    allow_source_floor_cap_overrun = bool(getattr(args, "allow_source_floor_cap_overrun", False))
    policy_config = CurationPolicyConfig(
        reject_refusal_boilerplate=bool(args.reject_refusal_boilerplate),
        reject_eval_holdout=not bool(getattr(args, "allow_eval_holdout", False)),
        min_quality_score=float(args.min_quality_score if args.min_quality_score is not None else 0.55),
        require_media_artifacts=bool(args.require_media_artifacts),
        reject_dataset_integrity_issues=not bool(args.allow_dataset_integrity_issues),
        scan_integrity_artifacts=not bool(args.skip_integrity_artifact_scan),
        max_integrity_artifact_bytes=int(args.max_integrity_artifact_bytes),
    )
    sources: list[tuple[Path, str]] = []
    if not args.no_profile_sources:
        sources.extend(profile_sources(profile, root))
    sources.extend(parse_sources(args.source, root))

    buckets: dict[str, list[dict[str, Any]]] = {modality: [] for modality in KNOWN_MODALITIES}
    seen_records: set[str] = set()
    source_floor_counts: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    skipped = Counter()
    for source_path, source_hint in sources:
        if fixture_path_rejected(source_path) and not allow_fixture_data:
            source_reports.append({"path": str(source_path), "hint": source_hint, "status": "fixture_refused"})
            continue
        if not source_path.exists() or not source_path.is_file() or source_path.stat().st_size <= 0:
            source_reports.append({"path": str(source_path), "hint": source_hint, "status": "missing_or_empty"})
            continue
        before = sum(len(values) for values in buckets.values())
        read_count = 0
        kept_count = 0
        source_floor = max(source_floors.get(source_path.name, 0), source_floors.get(str(source_path), 0))
        for row in iter_jsonl(source_path):
            read_count += 1
            if args.max_source_records and read_count > args.max_source_records:
                break
            if contamination_rejected(row):
                skipped["contamination"] += 1
                continue
            modality = infer_modality(row, source_path, source_hint)
            if modality not in KNOWN_MODALITIES:
                skipped["unknown_modality"] += 1
                continue
            if len(buckets[modality]) >= caps.get(modality, 0) and (
                source_floor_counts[source_path.name] >= source_floor or not allow_source_floor_cap_overrun
            ):
                skipped[f"{modality}_cap"] += 1
                continue
            prompt, target = message_prompt_target(row)
            prompt = clamp_text(text_value(prompt), int(args.max_prompt_chars))
            target = clamp_text(text_value(target), int(args.max_target_chars))
            if not prompt or not target:
                skipped["missing_prompt_or_target"] += 1
                continue
            refs = artifact_refs(row)
            if not has_quality_value(row):
                skipped["missing_quality"] += 1
                continue
            existing_quality = quality_value(row)
            audit = audit_training_record(
                row,
                prompt=prompt,
                target=target,
                modality=modality,
                source_path=source_path,
                refs=refs,
                existing_quality=existing_quality,
                config=policy_config,
            )
            if not audit["accepted"]:
                reasons = audit.get("reasons") or ["policy_reject"]
                for reason in reasons:
                    skipped[f"policy_{str(reason).split(':', 1)[0]}"] += 1
                continue
            dedupe_key = stable_hash({"modality": modality, "prompt": prompt[:2048], "target": target[:2048]})
            if dedupe_key in seen_records:
                skipped["duplicate"] += 1
                continue
            seen_records.add(dedupe_key)
            compact_row = {
                "record_id": row.get("record_id") or row.get("id"),
                "id": row.get("id"),
                "source_id": row.get("source_id") or source_path.name,
                "modality": modality,
                "modalities": sorted(set([modality] + row_modalities(row))),
                "artifact_refs": refs,
                "quality_score": max(existing_quality, float((audit.get("quality") or {}).get("score") or 0.0)),
                "curation_policy_2026": audit,
                "contamination_status": row.get("contamination_status", "unknown"),
            }
            buckets[modality].append(
                {
                    "row": compact_row,
                    "prompt": prompt,
                    "target": target,
                    "modality": modality,
                    "source_path": source_path,
                    "line_number": int(row.get("line_number") or read_count),
                }
            )
            source_floor_counts[source_path.name] += 1
            source_floor_counts[str(source_path)] += 1
            kept_count += 1
            source_floor_met = source_floor_counts[source_path.name] >= source_floor
            if source_floor_met and all(len(buckets[modality_name]) >= caps.get(modality_name, 0) for modality_name in required):
                break
        source_reports.append(
            {
                "path": str(source_path),
                "hint": source_hint,
                "status": "read",
                "records_read": read_count,
                "records_kept": kept_count,
                "source_floor": source_floor,
                "source_floor_kept": source_floor_counts[source_path.name],
                "total_kept_after_source": sum(len(values) for values in buckets.values()),
                "records_kept_before_source": before,
            }
        )

    order = [modality for modality in list(DEFAULT_STAGE_ORDER) if modality in KNOWN_MODALITIES]
    for modality in sorted(KNOWN_MODALITIES):
        if modality not in order:
            order.append(modality)
    selected = round_robin_rows(buckets, order)
    paths = output_paths(resolve_path(args.out_dir, root), args.out_jsonl, args.manifest)
    sft_rows = [sft_row(item["row"], item["prompt"], item["target"], item["modality"], item["source_path"], item["line_number"]) for item in selected]
    reward_rows = [reward_row(item["row"], item["prompt"], item["target"], item["modality"], item["source_path"], item["line_number"]) for item in selected]
    rlvr_rows = [rlvr_row(item["row"], item["prompt"], item["target"], item["modality"], item["source_path"], item["line_number"]) for item in selected]
    counts = {
        "sft": write_jsonl(paths["sft"], sft_rows),
        "reward": write_jsonl(paths["reward"], reward_rows),
        "rlvr": write_jsonl(paths["rlvr"], rlvr_rows),
    }
    modality_counts = {modality: len(values) for modality, values in sorted(buckets.items())}
    missing_required = [
        modality
        for modality in required
        if modality_counts.get(modality, 0) < int(args.min_records_per_required_modality or plan.get("min_records_per_modality") or 1)
    ]
    manifest = {
        "schema": "omnicoder.balanced_allmodal_posttrain_2026.v1",
        "created_at": now_iso(),
        "profile": args.profile,
        "paths": {key: str(value) for key, value in paths.items() if key != "manifest"},
        "manifest": str(paths["manifest"]),
        "counts": counts,
        "modality_counts": modality_counts,
        "required_modalities": required,
        "missing_required_modalities": missing_required,
        "caps": caps,
        "source_floors": source_floors,
        "source_floor_counts": {key: value for key, value in sorted(source_floor_counts.items()) if key in source_floors or Path(key).name in source_floors},
        "source_reports": source_reports,
        "skipped": dict(sorted(skipped.items())),
        "schema_mode": args.schema,
        "strip_token_ids": bool(args.strip_token_ids),
        "reject_refusal_boilerplate": bool(args.reject_refusal_boilerplate),
        "reject_eval_holdout": not bool(getattr(args, "allow_eval_holdout", False)),
        "reject_dataset_integrity_issues": not bool(args.allow_dataset_integrity_issues),
        "scan_integrity_artifacts": not bool(args.skip_integrity_artifact_scan),
        "max_integrity_artifact_bytes": int(args.max_integrity_artifact_bytes),
        "min_quality_score": float(args.min_quality_score or 0.0),
        "require_media_artifacts": bool(args.require_media_artifacts),
        "refusal_boilerplate_policy": "reject explicit refusal/alignment-negative boilerplate when requested; capability and benign security/tool competence rows remain eligible",
        "quality_policy": "rows can be rejected by curation_policy_2026 for low quality, placeholders, secrets, eval holdout markers, dataset-integrity poison/watermark/provenance flags, or missing media artifacts when requested",
        "token_id_policy": "source token ids are never copied; the pipeline trainer tokenizes generated message rows with the active tokenizer",
        "posttrain_input_overrides": {
            "reward_weighted_sft_replay": str(paths["sft"]),
            "grpo_rlvr_replay": str(paths["rlvr"]),
            "process_reward_replay": str(paths["reward"]),
        },
    }
    write_json(paths["manifest"], manifest)
    if missing_required and not args.allow_missing_required:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "missing_required_modalities",
                    "missing_required_modalities": missing_required,
                    "manifest": str(paths["manifest"]),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build balanced all-modal posttraining JSONL exports from curated real sources.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", default="weights/training_orchestration_2026/balanced_allmodal_posttrain_2026")
    parser.add_argument("--out-jsonl", default="", help="Optional explicit SFT output JSONL path. Reward/RLVR siblings are written next to it.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--source", action="append", default=[], help="Extra source JSONL, optionally modality=path.")
    parser.add_argument("--no-profile-sources", action="store_true")
    parser.add_argument("--cap", action="append", default=[], help="Per-modality cap override, e.g. code=4096.")
    parser.add_argument("--source-floor", action="append", default=[], help="Minimum accepted rows for a specific source basename/path before modality cap starvation, e.g. qwen_image_edit.clean.jsonl=8.")
    parser.add_argument("--max-records-per-modality", type=int, default=256)
    parser.add_argument("--max-source-records", type=int, default=0)
    parser.add_argument("--require-modalities", default="")
    parser.add_argument("--min-records-per-required-modality", type=int, default=1)
    parser.add_argument("--allow-missing-required", action="store_true")
    parser.add_argument("--strip-token-ids", action="store_true", help="Compatibility flag; generated rows never copy source token_ids.")
    parser.add_argument("--reject-refusal-boilerplate", action="store_true", help="Reject rows that contain explicit refusal/alignment-negative boilerplate.")
    parser.add_argument("--reject-eval-holdout", action="store_true", help="Deprecated compatibility flag; eval/public-dev/protected benchmark rows are rejected by default.")
    parser.add_argument("--allow-eval-holdout", action="store_true", help="Explicitly allow eval/public-dev/protected benchmark rows for a non-training diagnostic export.")
    parser.add_argument("--allow-dataset-integrity-issues", action="store_true", help="Permit rows flagged by dataset_integrity_2026; default is hard reject.")
    parser.add_argument("--allow-fixture-data", action="store_true", help="Explicitly allow examples/smoke/fixture paths for non-training diagnostics.")
    parser.add_argument("--allow-source-floor-cap-overrun", action="store_true", help="Allow a source floor to exceed a modality cap; default keeps caps strict.")
    parser.add_argument("--skip-integrity-artifact-scan", action="store_true", help="Skip local media byte marker scans; text/metadata integrity checks still run.")
    parser.add_argument("--max-integrity-artifact-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--min-quality-score", type=float, default=0.55, help="Reject rows below the curation_policy_2026 score floor.")
    parser.add_argument("--require-media-artifacts", action="store_true", help="Require existing artifact refs for image/video/audio/music rows.")
    parser.add_argument("--schema", default="messages", choices=["messages"], help="Output schema for optimizer replay rows.")
    parser.add_argument("--max-prompt-chars", type=int, default=24000)
    parser.add_argument("--max-target-chars", type=int, default=24000)
    args = parser.parse_args(argv)
    manifest = build_balanced_exports(args)
    print(json.dumps({"status": "passed", "manifest": manifest["manifest"], "counts": manifest["counts"], "modality_counts": manifest["modality_counts"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

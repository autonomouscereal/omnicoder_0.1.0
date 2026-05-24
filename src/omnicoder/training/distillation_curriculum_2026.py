from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_JOB_SCHEMA = {
    "critique": "localized critique with corrected answer/action",
    "preference": "chosen/rejected response pair with reason",
    "reward": "scalar reward, process labels, and outcome labels",
    "verifier": "verifiable checks, failure class, and evidence spans",
    "modal": "modality-specific plan, prompt, artifact critique, and alignment labels",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path, limit: int = 0) -> Iterable[dict[str, Any]]:
    seen = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item
                seen += 1
                if limit and seen >= limit:
                    return


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def extract_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for container in (record.get("input_json"), record.get("target_json"), record):
        if not isinstance(container, dict):
            continue
        messages = container.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    parts.append(message["content"])
        for key in ("content", "text", "prompt", "completion", "answer", "normalized_text"):
            value = container.get(key)
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(part for part in parts if part)


def quality_score(record: dict[str, Any]) -> float:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    return float(quality.get("score") or quality.get("overall") or 0.0)


def contamination_status(record: dict[str, Any]) -> str:
    contamination = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
    return str(contamination.get("status") or "clean")


def has_secret(record: dict[str, Any]) -> bool:
    secret = record.get("secret_redaction") if isinstance(record.get("secret_redaction"), dict) else {}
    if bool(secret.get("has_secret")):
        return True
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    details = quality.get("details") if isinstance(quality.get("details"), dict) else {}
    return float(details.get("secret_penalty") or 0.0) > 0.0


def media_families(record: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    classes = lineage.get("classifications") if isinstance(lineage.get("classifications"), dict) else {}
    media = classes.get("media") if isinstance(classes.get("media"), dict) else record.get("media")
    if isinstance(media, dict):
        for family in media.get("media_families") or []:
            families.add(str(family))
    text = extract_text(record).lower()
    for word, family in (
        ("screenshot", "image"),
        ("image", "image"),
        ("video", "video"),
        ("audio", "audio"),
        ("music", "music"),
        ("tts", "audio"),
        ("voice", "audio"),
        ("tool", "tool"),
        ("terminal", "tool"),
        ("python", "code"),
        ("traceback", "code"),
        ("git", "code"),
    ):
        if word in text:
            families.add(family)
    families.add("text")
    if "tool" in text or "trace" in text:
        families.add("agent_trace")
    return families


def record_is_eligible(record: dict[str, Any], profile: dict[str, Any]) -> bool:
    cfg = profile.get("job_plan") if isinstance(profile.get("job_plan"), dict) else {}
    if cfg.get("skip_contaminated", True) and contamination_status(record) == "contaminated":
        return False
    if cfg.get("skip_secret_rejected", True) and has_secret(record):
        return False
    if quality_score(record) < float(cfg.get("min_quality", 0.0)):
        return False
    split = str(record.get("split") or (record.get("split_assignment") or {}).get("split") or "train")
    if split == "eval_holdout" and not bool(cfg.get("include_eval_holdout", False)):
        return False
    return bool(extract_text(record).strip())


def enabled_teachers(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    registry = profile.get("teacher_registry") if isinstance(profile.get("teacher_registry"), dict) else {}
    return {name: cfg for name, cfg in registry.items() if isinstance(cfg, dict) and bool(cfg.get("enabled", True))}


def job_type_matches(job_type: str, teacher_modalities: set[str], record_modalities: set[str]) -> bool:
    if teacher_modalities & record_modalities:
        return True
    if job_type in {"trace_critique", "preference_pair", "reward_label", "verifier_label", "reasoning_rewrite"}:
        return True
    if "image" in job_type and "image" in record_modalities:
        return True
    if "video" in job_type and "video" in record_modalities:
        return True
    if ("audio" in job_type or "music" in job_type or "lyrics" in job_type) and record_modalities & {"audio", "music"}:
        return True
    return False


VALID_TEACHER_ROLES = {"primary", "verifier", "critic", "generator", "adjudicator", "optional_crosscheck"}


def profile_schema_registry(profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    registry = profile.get("schema_registry") if isinstance(profile, dict) and isinstance(profile.get("schema_registry"), dict) else {}
    return {str(key): value for key, value in registry.items() if isinstance(value, dict)}


def infer_teacher_role(teacher_name: str, teacher_cfg: dict[str, Any]) -> str:
    configured = str(teacher_cfg.get("teacher_role") or "").strip()
    if configured:
        return configured
    provider = str(teacher_cfg.get("provider") or "").lower()
    name = teacher_name.lower()
    if "image" in name or "ltx" in name or "ace" in name or "comfyui" in provider:
        return "generator"
    if any(marker in name for marker in ("deepseek", "gemini", "grok", "verifier")):
        return "verifier"
    if "optional" in name:
        return "optional_crosscheck"
    return "primary"


def infer_adjudication_group(teacher_name: str, teacher_cfg: dict[str, Any]) -> str:
    configured = str(teacher_cfg.get("adjudication_group") or "").strip()
    if configured:
        return configured
    modalities = {str(item) for item in teacher_cfg.get("modalities", [])}
    if "image" in modalities:
        return "image_edit"
    if "video" in modalities:
        return "video"
    if "music" in modalities:
        return "music"
    if "audio" in modalities:
        return "audio"
    if "long_context" in modalities:
        return "long_context"
    if "code" in modalities:
        return "agent_tool"
    return "hard_reasoning" if "deepseek" in teacher_name.lower() else "agent_tool"


def expected_output_schema(job_type: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    registered = profile_schema_registry(profile).get(job_type)
    if registered:
        payload = dict(registered)
        payload.setdefault("schema_id", f"omnicoder.teacher_job.{job_type}.v1")
        payload.setdefault("kind", "registered")
        payload.setdefault("required_fields", [])
        return payload
    if "tool_call_ast" in job_type:
        kind = "verifier"
        fields = ["tool_name", "arguments", "ast_valid", "schema_errors", "corrected_call", "reward"]
    elif "tool_state_delta" in job_type:
        kind = "verifier"
        fields = ["before_state", "after_state", "expected_delta", "observed_delta", "state_consistent", "reward"]
    elif "tool_plan_repair" in job_type:
        kind = "critique"
        fields = ["critique", "corrected_tool_plan", "unsafe_actions", "approval_needed", "reward"]
    elif "trajectory_preference" in job_type:
        kind = "preference"
        fields = ["chosen", "rejected", "preference_reason", "tool_state_notes", "safety_notes"]
    elif "preference" in job_type:
        kind = "preference"
        fields = ["chosen", "rejected", "preference_reason", "safety_notes"]
    elif "reward" in job_type or "rlvr" in job_type:
        kind = "reward"
        fields = ["reward", "outcome", "process_labels", "evidence", "failure_modes"]
    elif "verifier" in job_type or "oracle" in job_type:
        kind = "verifier"
        fields = ["checks", "passed", "evidence", "verifier_confidence"]
    elif "image" in job_type or "edit" in job_type:
        kind = "modal"
        fields = ["prompt", "negative_prompt", "composition_constraints", "edit_instructions", "preserve_regions", "change_regions", "alignment_labels", "artifact_failures", "reward"]
    elif "video" in job_type or "shot" in job_type or "temporal" in job_type:
        kind = "modal"
        fields = ["shot_list", "camera_motion", "keyframes", "temporal_constraints", "subject_consistency", "transition_notes", "artifact_failures", "reward"]
    elif any(token in job_type for token in ("audio", "music", "speech", "lyrics")):
        kind = "modal"
        fields = ["style_tags", "tempo_bpm", "structure", "lyrics_alignment", "instrumentation", "mix_quality_axes", "artifact_failures", "reward"]
    else:
        kind = "critique"
        fields = ["critique", "corrected_output", "reasoning_tags", "tool_action_fix", "reward"]
    return {"schema_id": f"omnicoder.teacher_job.{job_type}.v1", "kind": kind, "description": DEFAULT_JOB_SCHEMA[kind], "required_fields": fields}


def build_teacher_job(record: dict[str, Any], teacher_name: str, teacher_cfg: dict[str, Any], job_type: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    modalities = sorted(media_families(record))
    text = extract_text(record)
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    role = infer_teacher_role(teacher_name, teacher_cfg)
    adjudication_group = infer_adjudication_group(teacher_name, teacher_cfg)
    return {
        "teacher_name": teacher_name,
        "teacher_provider": teacher_cfg.get("provider"),
        "teacher_model_alias": teacher_cfg.get("model_alias"),
        "teacher_role": role,
        "adjudication_group": adjudication_group,
        "consensus_policy": teacher_cfg.get("consensus_policy") or ("primary_plus_verifier" if role in {"primary", "verifier"} else "single"),
        "endpoint_env": teacher_cfg.get("endpoint_env"),
        "job_type": job_type,
        "priority": int(teacher_cfg.get("priority") or 100),
        "input_json": {
            "schema": "omnicoder.distillation_job_2026.v1",
            "source_record_hash": stable_hash(record),
            "modalities": modalities,
            "source": {
                "input": record.get("input_json", {}),
                "target": record.get("target_json", {}),
                "lineage": lineage,
                "quality": record.get("quality", {}),
                "contamination": record.get("contamination", {}),
            },
            "context_excerpt": text[:20000],
            "instruction": (
                "Produce distillation data for Omnicoder's dense omnimodal student. "
                "Return structured JSON only. Include corrections, reward/verifier labels, "
                "modality alignment notes, and rejection reasons when the trace is unsafe or weak."
            ),
            "expected_output_schema": expected_output_schema(job_type, profile),
            "training_targets": [
                "sft",
                "preference_optimization",
                "reward_modeling",
                "rlvr_grpo",
                "multimodal_alignment",
                "q4_recovery_distillation",
            ],
        },
    }


def build_jobs(profile: dict[str, Any], records_path: str | Path, limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = profile.get("job_plan") if isinstance(profile.get("job_plan"), dict) else {}
    per_teacher_limit = int(limit or cfg.get("per_teacher_limit") or 0)
    teachers = enabled_teachers(profile)
    counts: Counter[str] = Counter()
    jobs: list[dict[str, Any]] = []
    per_teacher_counts: Counter[str] = Counter()
    for record in read_jsonl(records_path):
        counts["seen"] += 1
        if not record_is_eligible(record, profile):
            counts["filtered"] += 1
            continue
        record_modalities = media_families(record)
        for teacher_name, teacher_cfg in teachers.items():
            if per_teacher_limit and per_teacher_counts[teacher_name] >= per_teacher_limit:
                continue
            teacher_modalities = {str(item) for item in teacher_cfg.get("modalities", [])}
            for job_type in teacher_cfg.get("job_types", []):
                job_type = str(job_type)
                if not job_type_matches(job_type, teacher_modalities, record_modalities):
                    continue
                jobs.append(build_teacher_job(record, teacher_name, teacher_cfg, job_type, profile))
                per_teacher_counts[teacher_name] += 1
                counts[f"teacher_{teacher_name}"] += 1
                counts[f"job_type_{job_type}"] += 1
                break
    summary = {
        "records_seen": counts.pop("seen", 0),
        "records_filtered": counts.pop("filtered", 0),
        "jobs": len(jobs),
        "counts": dict(counts),
        "teachers": sorted(teachers),
        "per_teacher_limit": per_teacher_limit,
    }
    return jobs, summary


def curriculum_manifest(profile: dict[str, Any], jobs_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    post = profile.get("posttraining") if isinstance(profile.get("posttraining"), dict) else {}
    stages = post.get("stages") if isinstance(post.get("stages"), list) else []
    return {
        "schema": "omnicoder.distillation_curriculum_manifest_2026.v1",
        "created_at": now_iso(),
        "profile_name": profile.get("profile_name"),
        "source_date": profile.get("source_date"),
        "student_profile": post.get("student_profile"),
        "base_model": post.get("base_model"),
        "context_length": post.get("context_length"),
        "quantization_target": post.get("quantization_target"),
        "training_stack": {
            "supervised": ["SFT", "QLoRA", "assistant-only loss", "packing"],
            "distillation": ["multi-teacher critique", "self-consistency", "verifier labels", "on-policy distillation"],
            "preference": ["DPO", "ORPO", "KTO", "SimPO", "IPO-style variants"],
            "reward": ["outcome reward", "process reward", "tool-state reward", "multimodal artifact reward"],
            "rl": ["GRPO", "DAPO-style dynamic sampling", "Tree-GRPO for agents", "RLVR", "PPO/RLOO fallback"],
            "modal": ["image/edit reward", "video temporal reward", "audio/music alignment reward", "cross-modal consistency"],
            "deployment": ["q4-aware recovery", "long-context retention", "GGUF-compatible student exports"],
        },
        "stages": stages,
        "teacher_registry": enabled_teachers(profile),
        "jobs_summary": jobs_summary or {},
    }


def emit_per_teacher_jobs(jobs: Sequence[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    by_teacher: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_teacher.setdefault(str(job["teacher_name"]), []).append(job)
    outputs: dict[str, str] = {}
    for teacher, rows in by_teacher.items():
        path = out_dir / "teachers" / f"{teacher}.jsonl"
        write_jsonl(path, rows)
        outputs[teacher] = str(path)
    return outputs


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    records_path = Path(args.records or profile.get("records") or "")
    out_dir = Path(args.out_dir or profile.get("work_dir") or "weights/distillation_2026")
    if not records_path.exists():
        raise SystemExit(json.dumps({"status": "error", "error": "records not found", "records": str(records_path)}))
    jobs, summary = build_jobs(profile, records_path, args.limit)
    jobs_path = out_dir / "teacher_jobs" / "distillation_jobs_2026.jsonl"
    write_jsonl(jobs_path, jobs)
    per_teacher = {}
    if bool((profile.get("job_plan") or {}).get("emit_per_teacher_files", True)):
        per_teacher = emit_per_teacher_jobs(jobs, out_dir)
    manifest = curriculum_manifest(profile, summary)
    manifest["records"] = str(records_path)
    manifest["outputs"] = {"jobs": str(jobs_path), "per_teacher": per_teacher}
    manifest_path = out_dir / "distillation_curriculum_manifest.json"
    write_json(manifest_path, manifest)
    return {"status": "ok", "manifest": str(manifest_path), "jobs": len(jobs), "jobs_path": str(jobs_path), "summary": summary}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    teachers = enabled_teachers(profile)
    modality_counts: Counter[str] = Counter()
    job_counts: Counter[str] = Counter()
    for cfg in teachers.values():
        for modality in cfg.get("modalities", []):
            modality_counts[str(modality)] += 1
        for job_type in cfg.get("job_types", []):
            job_counts[str(job_type)] += 1
    role_counts: Counter[str] = Counter()
    invalid_roles: dict[str, str] = {}
    missing_adjudication: list[str] = []
    for teacher_name, cfg in teachers.items():
        role = infer_teacher_role(teacher_name, cfg)
        role_counts[role] += 1
        if role not in VALID_TEACHER_ROLES:
            invalid_roles[teacher_name] = role
        if not infer_adjudication_group(teacher_name, cfg):
            missing_adjudication.append(teacher_name)
    required = {"text", "code", "agent_trace", "tool", "image", "video", "audio", "music"}
    covered = set(modality_counts)
    return {
        "status": "ok" if not invalid_roles else "invalid_teacher_roles",
        "teachers": len(teachers),
        "missing_modalities": sorted(required - covered),
        "modalities": dict(sorted(modality_counts.items())),
        "job_types": dict(sorted(job_counts.items())),
        "teacher_roles": dict(sorted(role_counts.items())),
        "invalid_teacher_roles": invalid_roles,
        "missing_adjudication_groups": missing_adjudication,
        "schema_registry": sorted(profile_schema_registry(profile)),
        "posttraining_stages": [stage.get("id") for stage in (profile.get("posttraining") or {}).get("stages", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Omnicoder 2026 multi-teacher distillation and RL curriculum artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("all", "build-jobs", "curriculum"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--profile", default="profiles/distillation_curriculum_2026.json")
        cmd.add_argument("--records", default=None)
        cmd.add_argument("--out-dir", default=None)
        cmd.add_argument("--limit", type=int, default=0)
    val = sub.add_parser("validate")
    val.add_argument("--profile", default="profiles/distillation_curriculum_2026.json")
    args = parser.parse_args()
    if args.cmd == "validate":
        result = validate(args)
    elif args.cmd == "curriculum":
        profile = read_json(args.profile)
        out_dir = Path(args.out_dir or profile.get("work_dir") or "weights/distillation_2026")
        manifest = curriculum_manifest(profile)
        path = out_dir / "distillation_curriculum_manifest.json"
        write_json(path, manifest)
        result = {"status": "ok", "manifest": str(path), "stages": len(manifest["stages"])}
    else:
        result = run_all(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omnicoder.data_factory.curation_policy_2026 import CurationPolicyConfig, audit_training_record
from omnicoder.data_factory.export_sft_jsonl import contains_secret_payload, eligible
from omnicoder.data_factory.postgres import claim_teacher_job, enqueue_teacher_job


DEFAULT_TEACHERS = {
    "qwen3.6_27b_q4_local": ("reasoning", "coding", "tool_repair", "trace_critique"),
    "qwen3_omni_optional": ("audio_video_understanding", "speech_caption", "multimodal_alignment", "cross_modal_reward"),
    "qwen_image_generate": ("image_prompt", "image_critique", "image_reward_label"),
    "qwen_image_edit": ("image_edit_plan", "image_edit_critique", "edit_preservation_reward"),
    "ltx_2_3": ("video_prompt", "shot_plan", "video_sync_critique", "image_to_video_plan", "temporal_reward_label"),
    "ace_step_1_5": ("music_plan", "lyrics_alignment", "audio_critique", "music_reward_label"),
    "deepseek_v4_optional": ("hard_reasoning_verifier", "code_patch_review", "tool_repair", "rlvr_oracle"),
    "minimax_2_7_optional": ("long_context_compression", "trace_critique", "preference_pair", "reward_label"),
    "kimi_k2_6_optional": ("long_context_compression", "code_patch_review", "verifier_label", "preference_pair"),
    "composer_2_5_optional": ("music_plan", "audio_critique", "style_transfer_reward"),
    "gemini_omni_optional": ("multimodal_alignment", "audio_video_understanding", "image_critique", "cross_modal_reward"),
    "gemma_4_optional": ("small_teacher_consistency", "reasoning_rewrite", "preference_pair"),
    "grok_optional": ("adversarial_critique", "reward_label", "tool_repair"),
    "gpt_image_2_optional": ("image_generation_reference", "image_edit_reference", "image_reward_label"),
}


def first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [first_text(item) for item in value[:16]]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("content", "text", "prompt", "instruction", "question", "answer", "response", "completion"):
            if key in value:
                text = first_text(value[key])
                if text:
                    return text
    return ""


def prompt_from_record(record: dict[str, Any]) -> str:
    input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
    messages = input_json.get("messages") if isinstance(input_json.get("messages"), list) else record.get("messages")
    text = first_text(messages)
    if text:
        return text
    return first_text(input_json) or first_text(record.get("prompt")) or first_text(record.get("instruction"))


def target_from_record(record: dict[str, Any]) -> str:
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    return first_text(target_json) or first_text(record.get("target")) or first_text(record.get("response")) or first_text(record.get("messages"))


def modality_from_record(record: dict[str, Any]) -> str:
    modality = str(record.get("modality") or "").strip().lower()
    if modality:
        return modality
    modalities = record.get("modalities")
    if isinstance(modalities, list):
        for value in modalities:
            text = str(value or "").strip().lower()
            if text:
                return text
    text = f"{prompt_from_record(record)}\n{target_from_record(record)}".lower()
    if any(marker in text for marker in ("tool", "shell", "terminal", "function_call")):
        return "tool"
    if any(marker in text for marker in ("image", "ocr", "vision")):
        return "image"
    if "video" in text:
        return "video"
    if any(marker in text for marker in ("audio", "speech", "tts", "music")):
        return "audio"
    if any(marker in text for marker in ("pytest", "python", "code", "patch")):
        return "code"
    return "text"


def artifact_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("artifact_refs", "media_refs", "artifacts"):
        value = record.get(key)
        if isinstance(value, list):
            refs.extend(str(item) for item in value if str(item).strip())
    for container in (record, record.get("target_json"), record.get("output_json")):
        if not isinstance(container, dict):
            continue
        for key in ("artifact_path", "image_path", "video_path", "audio_path", "music_path"):
            value = container.get(key)
            if value:
                refs.append(str(value))
    return sorted(set(refs))


def record_is_trainable(record: dict[str, Any], *, source_path: Path) -> bool:
    prompt = prompt_from_record(record)
    target = target_from_record(record)
    if not eligible(record, min_quality=0.35, allow_contaminated=False):
        return False
    payload = {"messages": record.get("messages"), "input_json": record.get("input_json"), "target_json": record.get("target_json")}
    if contains_secret_payload(payload):
        return False
    audit = audit_training_record(
        record,
        prompt=prompt,
        target=target,
        modality=modality_from_record(record),
        source_path=source_path,
        refs=artifact_refs(record),
        existing_quality=float((record.get("quality") or {}).get("score") or record.get("quality_score") or 0.0),
        config=CurationPolicyConfig(min_quality_score=0.35, require_media_artifacts=False, scan_integrity_artifacts=False),
    )
    return bool(audit.get("accepted"))


def build_job_input(record: dict[str, Any]) -> dict[str, Any]:
    input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    source_payload = record.get("source_payload") if isinstance(record.get("source_payload"), dict) else {}
    token_ids = record.get("token_ids") if isinstance(record.get("token_ids"), list) else []
    return {
        "input": input_json,
        "target": target_json,
        "prompt": prompt_from_record(record),
        "target_text": target_from_record(record),
        "messages": input_json.get("messages") if isinstance(input_json.get("messages"), list) else record.get("messages"),
        "lineage": record.get("lineage", {}),
        "dataset": {
            "name": record.get("dataset_name") or source_payload.get("dataset_name"),
            "family": record.get("dataset_family") or source_payload.get("dataset_family"),
            "training_bucket": record.get("training_bucket"),
            "license_tier": record.get("license_tier") or source_payload.get("license_tier"),
            "use_policy": record.get("use_policy") or source_payload.get("use_policy"),
        },
        "curriculum_axes": record.get("curriculum_axes", []),
        "modalities": record.get("modalities", []),
        "modality": record.get("modality"),
        "quality": record.get("quality", {}),
        "contamination": record.get("contamination", {}),
        "tool_calls": record.get("tool_calls", []),
        "tool_results": record.get("tool_results", []),
        "token_count": len(token_ids),
        "token_id_sample": token_ids[:64],
        "source_payload": source_payload,
        "instruction": "Return localized critique, corrected action/content, verifier labels, and safety/contamination notes.",
    }


def build_jobs(records_path: str, teacher: str, job_type: str, limit: int = 0) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    source_path = Path(records_path)
    with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or not record_is_trainable(obj, source_path=source_path):
                continue
            jobs.append(
                {
                    "teacher_name": teacher,
                    "job_type": job_type,
                    "input_json": build_job_input(obj),
                }
            )
            if limit and len(jobs) >= limit:
                break
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or claim 2026 teacher distillation jobs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build")
    build.add_argument("--records", required=True)
    build.add_argument("--teacher", default="qwen3.6_27b_q4_local")
    build.add_argument("--job_type", default="trace_critique")
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--out", default="weights/data_factory/teacher_jobs_2026.jsonl")

    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--jobs", required=True)
    enqueue.add_argument("--priority", type=int, default=100)

    claim = sub.add_parser("claim")
    claim.add_argument("--teacher", required=True)
    claim.add_argument("--worker", default="teacher_worker")

    sub.add_parser("list-teachers")
    args = ap.parse_args()

    if args.cmd == "list-teachers":
        print(json.dumps(DEFAULT_TEACHERS, indent=2))
        return
    if args.cmd == "build":
        jobs = build_jobs(args.records, args.teacher, args.job_type, args.limit)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(job, ensure_ascii=True) for job in jobs) + ("\n" if jobs else ""), encoding="utf-8")
        print(json.dumps({"status": "ok", "out": str(out), "jobs": len(jobs)}))
        return
    if args.cmd == "enqueue":
        count = 0
        with Path(args.jobs).open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                job = json.loads(line)
                enqueue_teacher_job(job["teacher_name"], job["job_type"], job["input_json"], priority=args.priority)
                count += 1
        print(json.dumps({"status": "ok", "enqueued": count}))
        return
    if args.cmd == "claim":
        print(json.dumps({"job": claim_teacher_job(args.teacher, args.worker)}, indent=2))


if __name__ == "__main__":
    main()

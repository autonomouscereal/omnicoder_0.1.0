from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def build_jobs(records_path: str, teacher: str, job_type: str, limit: int = 0) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for line in Path(records_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        jobs.append(
            {
                "teacher_name": teacher,
                "job_type": job_type,
                "input_json": {
                    "input": obj.get("input_json", {}),
                    "target": obj.get("target_json", {}),
                    "lineage": obj.get("lineage", {}),
                    "instruction": "Return localized critique, corrected action/content, verifier labels, and safety/contamination notes.",
                },
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
        for line in Path(args.jobs).read_text(encoding="utf-8", errors="ignore").splitlines():
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

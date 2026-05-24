from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STAGES = (
    "ingest",
    "ledger_encode",
    "pretrain",
    "sft",
    "teacher_distill",
    "preference",
    "long_context",
    "rlvr",
    "qat",
    "eval",
    "gguf",
)

STAGE_PLANS = {
    "ingest": {
        "goal": "register 2025-2026 datasets, local traces, media assets, and teacher outputs in raw PostgreSQL",
        "tables": ["datasets", "artifacts", "media_segments", "agent_runs", "agent_steps", "quality_scores"],
        "hard_rules": ["source_date >= 2025-01-01", "license/provenance required", "eval artifacts go to quarantine"],
    },
    "ledger_encode": {
        "goal": "turn text/media/tool traces into typed ledger packets",
        "codecs": ["text", "vision/video semantic+residual", "speech_tts", "audio_music", "time_space", "tool_agent", "flow"],
    },
    "sft": {
        "goal": "format/tool discipline and recovery traces",
        "mix": {"agent_tool": 0.35, "code": 0.20, "vision_ui": 0.15, "reasoning": 0.15, "teacher_media": 0.10, "safety": 0.05},
    },
    "teacher_distill": {
        "goal": "distill localized capability from local teachers without treating teacher text as reward truth",
        "teachers": ["qwen3.6_27b_q4_local", "qwen3_omni", "qwen_image_edit", "ltx_2_3", "ace_step_1_5", "gpt_image_2_optional"],
        "outputs": ["tool_selection", "prompt_plans", "modality_captions", "localized_critiques", "self_corrections"],
    },
    "preference": {
        "goal": "localized repair preferences and anti-regression pairs",
        "methods": ["DPO/APO-style pairs", "Composer-2.5-style textual feedback when available"],
    },
    "long_context": {
        "goal": "native 1M curriculum over KDA/CSA/HCA memory, not YaRN-only claims",
        "curriculum": [8192, 32768, 65536, 131072, 262144, 1048576],
        "loss": "answer/action spans plus retrieval-free recall probes",
    },
    "rlvr": {
        "goal": "verifiable agent rewards only",
        "algorithms": ["GRPO smoke", "Dr.GRPO length/difficulty correction", "DAPO after stable rollouts"],
        "rewards": ["unit_tests", "terminal_oracles", "tool_schema_validity", "browser_ui_state", "long_context_exactness"],
    },
    "qat": {
        "goal": "Q4-aware deployment recovery",
        "policy": "fake-Q4/QAT late; keep norms, KDA accumulators, mHC gates, sparse index math higher precision",
    },
    "eval": {
        "goal": "registry smoke/nightly/release gates",
        "registry": "profiles/benchmark_registry_2026.json",
    },
    "gguf": {
        "goal": "qwen3-compatible shorter-context GGUF bridge plus native runtime manifest",
        "truth": "stock GGUF is not the full native 1M runtime",
    },
}


def _run(cmd: list[str]) -> int:
    print(json.dumps({"event": "stage_command", "cmd": cmd}))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 training orchestration")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default="weights/omnicoder2026_runs")
    ap.add_argument("--preset", default="ledger_probe")
    ap.add_argument("--stages", default="pretrain")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(STAGES if args.stages.strip().lower() == "all" else tuple(x.strip() for x in args.stages.split(",") if x.strip()))
    for stage in requested:
        if stage not in STAGES:
            raise SystemExit(f"unknown 2026 stage: {stage}")
        print(json.dumps({"event": "stage_start", "stage": stage, "preset": args.preset}))
        if stage == "pretrain":
            code = _run([
                sys.executable, "-m", "omnicoder.training.pretrain_2026_dense",
                "--preset", args.preset,
                "--data", args.data,
                "--out", str(out_dir / f"{args.preset}_pretrain.pt"),
                "--steps", str(int(args.steps)),
                "--seq_len", str(int(args.seq_len)),
                "--device", args.device,
                "--aux_probe",
            ])
            if code != 0:
                raise SystemExit(code)
        else:
            manifest = out_dir / f"{stage}.json"
            manifest.write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "status": "planned",
                        "preset": args.preset,
                        "data": args.data,
                        "plan": STAGE_PLANS.get(stage, {}),
                        "requires": "curated data/teacher assets and benchmark quarantine before full execution",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(json.dumps({"event": "stage_done", "stage": stage}))


if __name__ == "__main__":
    main()

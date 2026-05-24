from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "sft": {
        "trainer": "SFTTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "messages or prompt/completion",
        "purpose": "Cold-start instruction, trace, tool, and modality format following.",
    },
    "reward": {
        "trainer": "RewardTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "chosen/rejected or scalar reward labels",
        "purpose": "Outcome/process/tool/artifact reward modeling.",
    },
    "dpo": {
        "trainer": "DPOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt, chosen, rejected",
        "purpose": "Pairwise preference optimization after SFT.",
    },
    "orpo": {
        "trainer": "ORPOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt, chosen, rejected",
        "purpose": "Reference-free odds-ratio preference optimization.",
    },
    "kto": {
        "trainer": "KTOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt/completion with desirable boolean",
        "purpose": "Binary desirable/undesirable preference alignment.",
    },
    "simpo": {
        "trainer": "DPOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt, chosen, rejected",
        "purpose": "Length-normalized simple preference optimization; DPO-compatible bridge.",
    },
    "grpo": {
        "trainer": "GRPOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt plus reward functions or verifiable environment",
        "purpose": "Group-relative RLVR for reasoning, tools, code tests, and multimodal judges.",
    },
    "tree_grpo": {
        "trainer": "custom_tree_grpo",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "tree rollout nodes with verifier rewards and parent/child metadata",
        "purpose": "Tree-structured GRPO for branching agent/tool rollouts and search-style reasoning.",
    },
    "dapo": {
        "trainer": "custom_grpo_variant",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with verifiable rewards, dynamic sampling metadata, and overlong labels",
        "purpose": "Decoupled clipping, dynamic sampling, token-level policy-gradient, and overlong reward shaping for stable RLVR.",
    },
    "dr_grpo": {
        "trainer": "custom_grpo_variant",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with reward, response length, and group metadata",
        "purpose": "Debiased GRPO variant that tracks length separately from reward normalization to reduce length bias.",
    },
    "vapo": {
        "trainer": "custom_ppo_variant",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with value targets and verifiable rewards",
        "purpose": "Value-augmented policy optimization for harder long-horizon reasoning and agent rollouts.",
    },
    "dcpo": {
        "trainer": "custom_policy_gradient",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "token-level rollout traces with per-token clipping statistics",
        "purpose": "Dynamic token-specific clipping for RL stability on long responses and tool trajectories.",
    },
    "lspo": {
        "trainer": "custom_policy_gradient",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with length buckets, rewards, and overthinking labels",
        "purpose": "Length-aware dynamic sampling to preserve concise reasoning without suppressing hard long-chain tasks.",
    },
    "cispo": {
        "trainer": "custom_policy_gradient",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with clipped importance-sampling weights and verifier rewards",
        "purpose": "Clip importance-sampling weights rather than token updates for MiniMax-style long-thinking agent RL.",
    },
    "retool": {
        "trainer": "custom_tool_rl",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "interleaved reasoning/tool trajectories with executable tool observations",
        "purpose": "Tool-integrated reasoning RL over shell, browser, retrieval, Python, CI, and file-edit environments.",
    },
    "toolrl": {
        "trainer": "custom_tool_rl",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "tool calls, arguments, observations, state deltas, and success rewards",
        "purpose": "Tool-use reinforcement learning with schema, state-update, minimal-call, and final-outcome rewards.",
    },
    "agentprm": {
        "trainer": "custom_process_reward_model",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "agent step traces with process labels and final outcome labels",
        "purpose": "Process reward modeling for multi-step agent planning, correction, and reranking.",
    },
    "rlaif_v": {
        "trainer": "custom_multimodal_rlaif",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "image/video/audio prompts with AI feedback, hallucination labels, and preference pairs",
        "purpose": "Multimodal AI-feedback alignment for visual/audio grounding, hallucination reduction, and artifact critique.",
    },
    "ppo": {
        "trainer": "PPOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with reward model/environment",
        "purpose": "Fallback RLHF loop when value-model infrastructure is available.",
    },
    "rloo": {
        "trainer": "RLOOTrainer",
        "deps": ["torch", "transformers", "datasets", "trl", "peft"],
        "dataset_schema": "prompt rollouts with rewards",
        "purpose": "Leave-one-out policy-gradient alternative for lower overhead RL.",
    },
    "on_policy_distill": {
        "trainer": "custom",
        "deps": ["torch", "transformers", "datasets"],
        "dataset_schema": "teacher/student rollouts with distribution or verifier rewards",
        "purpose": "Video/audio/image on-policy distillation and train-inference mismatch reduction.",
    },
    "qat_distill": {
        "trainer": "custom",
        "deps": ["torch", "transformers", "datasets", "peft"],
        "dataset_schema": "SFT/preference/RL traces with fake-quant consistency targets",
        "purpose": "Q4-aware recovery distillation for 24GB deployment.",
    },
}


ALGORITHM_ALIASES: dict[str, str] = {
    "audio_video_generation_rl": "rlaif_v",
    "browser_research_rlvr": "grpo",
    "dpo_pair_replay": "dpo",
    "dpo_preference_replay": "dpo",
    "dpo_replay": "dpo",
    "desktop_gui_rl": "toolrl",
    "grpo_dapo_tree_grpo": "grpo",
    "dapo_dr_grpo_lspo": "dapo",
    "grpo_rlvr_replay": "grpo",
    "kto_replay": "kto",
    "multimodal_rlaif_v": "rlaif_v",
    "orpo_kto_simpo_pair_replay": "orpo",
    "orpo_replay": "orpo",
    "process_reward_replay": "agentprm",
    "reward_model": "reward",
    "reward_weighted_sft": "sft",
    "reward_weighted_sft_replay": "sft",
    "retool_tool_integrated_rollouts": "retool",
    "retool_toolrl_cispo": "retool",
    "rlaif_v_replay": "rlaif_v",
    "safety_negative_replay": "kto",
    "simpo_replay": "simpo",
    "toolrl_state_update_rewards": "toolrl",
    "agentprm_process_rewards": "agentprm",
}


def resolve_algorithm(value: str) -> str:
    key = value.lower()
    return ALGORITHM_ALIASES.get(key, key)


def algorithm_choices() -> list[str]:
    return sorted(set(ALGORITHM_REGISTRY) | set(ALGORITHM_ALIASES))


def dep_status(deps: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in deps}


def count_jsonl(path: str | Path | None, limit: int = 0) -> int:
    if not path:
        return 0
    p = Path(path)
    if not p.exists():
        return 0
    count = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            count += 1
            if limit and count >= limit:
                break
    return count


def is_rlvr_training_kind(kind: Any) -> bool:
    return str(kind or "").lower().endswith("_rlvr")


def is_tool_trajectory_record(obj: dict[str, Any]) -> bool:
    kind = str(obj.get("training_kind") or "").lower()
    return kind in {
        "tool_sft",
        "tool_reward",
        "tool_preference",
        "tool_safety_negative",
    } or is_rlvr_training_kind(kind)


def inspect_dataset(path: str | Path | None, limit: int = 2000) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"exists": False, "records": 0, "schemas": {}}
    schemas: dict[str, int] = {}
    records = 0
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        records += 1
        try:
            obj = json.loads(line)
        except Exception:
            schemas["parse_error"] = schemas.get("parse_error", 0) + 1
            continue
        if isinstance(obj, dict) and is_tool_trajectory_record(obj):
            schemas["tool_trajectory"] = schemas.get("tool_trajectory", 0) + 1
        elif isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            schemas["messages"] = schemas.get("messages", 0) + 1
        elif isinstance(obj, dict) and {"prompt", "chosen", "rejected"} <= set(obj):
            schemas["preference_pair"] = schemas.get("preference_pair", 0) + 1
        elif isinstance(obj, dict) and ("reward" in obj or "score" in obj):
            schemas["reward"] = schemas.get("reward", 0) + 1
        elif isinstance(obj, dict):
            schemas["generic_object"] = schemas.get("generic_object", 0) + 1
        else:
            schemas["other"] = schemas.get("other", 0) + 1
        if limit and records >= limit:
            break
    return {"exists": True, "records_sampled": records, "schemas": schemas, "path": str(path)}


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    requested_algorithm = args.algorithm.lower()
    algorithm = resolve_algorithm(requested_algorithm)
    if algorithm not in ALGORITHM_REGISTRY:
        raise SystemExit(json.dumps({"status": "error", "error": "unknown algorithm", "algorithm": requested_algorithm}))
    spec = ALGORITHM_REGISTRY[algorithm]
    deps = dep_status(spec["deps"])
    missing = [name for name, ok in deps.items() if not ok]
    dataset = inspect_dataset(args.train_jsonl)
    dry_run = bool(getattr(args, "dry_run", False))
    smoke = bool(getattr(args, "smoke", False))
    empty_allowed_by = "dry_run" if dry_run else "smoke" if smoke else ""
    if int(dataset.get("records_sampled") or dataset.get("records") or 0) == 0 and not empty_allowed_by:
        raise SystemExit(
            json.dumps(
                {
                    "status": "error",
                    "error": "empty_dataset",
                    "message": "posttraining replay dataset is empty; use --dry_run or --smoke only for explicit smoke checks",
                    "train_jsonl": args.train_jsonl,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    manifest = {
        "schema": "omnicoder.posttrain_bridge_2026.v1",
        "requested_algorithm": requested_algorithm,
        "algorithm": algorithm,
        "trainer": spec["trainer"],
        "purpose": spec["purpose"],
        "expected_dataset_schema": spec["dataset_schema"],
        "model": args.model,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "out_dir": args.out_dir,
        "deps": deps,
        "missing_dependencies": missing,
        "dataset": dataset,
        "eval_dataset": inspect_dataset(args.eval_jsonl) if args.eval_jsonl else None,
        "hyperparameters": {
            "max_seq_len": args.max_seq_len,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "load_in_4bit": bool(args.load_in_4bit),
            "reward_batch_size": args.reward_batch_size,
            "num_generations": args.num_generations,
            "kl_beta": args.kl_beta,
            "temperature": args.temperature,
        },
        "status": "dry_run_ok" if dry_run else "smoke_ok" if smoke else "configured",
    }
    if empty_allowed_by:
        manifest["empty_dataset_allowed_by"] = empty_allowed_by
    if missing and args.check_deps:
        manifest["status"] = "missing_dependencies"
    if algorithm in {"grpo", "tree_grpo", "dapo", "dr_grpo", "vapo", "dcpo", "lspo", "cispo", "retool", "toolrl", "ppo", "rloo"}:
        manifest["reward_contract"] = {
            "verifiable_rewards": ["unit_tests", "tool_state", "exact_match", "artifact_quality", "safety", "contamination_free"],
            "grouping": "sample multiple rollouts per prompt; normalize rewards within group",
            "agentic_extensions": [
                "Tree-GRPO rollout nodes",
                "terminal/container rewards",
                "MCP tool-state rewards",
                "BFCL-style function-call rewards",
                "tau-style user-simulation state rewards",
                "browser/citation rewards",
                "GUI state-transition rewards",
                "long-context retention rewards",
            ],
        }
    if algorithm in {"agentprm", "rlaif_v"}:
        manifest["feedback_contract"] = {
            "label_types": ["process", "outcome", "preference", "critique", "safety"],
            "required_metadata": ["teacher_or_judge", "rubric_id", "artifact_hashes", "contamination_status"],
            "promotion_gate": "feedback records must pass secret scan, protected-eval scan, and heldout regression checks",
        }
    if algorithm in {"on_policy_distill", "qat_distill"}:
        manifest["custom_contract"] = {
            "teacher_outputs": "teacher logits or structured critiques when available; otherwise artifact/verifier rewards",
            "student_targets": "same token space plus modality tokens; q4-aware consistency for qat_distill",
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure/dry-run 2026 post-training algorithms for Omnicoder")
    parser.add_argument("--algorithm", required=True, choices=algorithm_choices())
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--eval_jsonl", default=None)
    parser.add_argument("--out_dir", default="weights/posttrain_2026")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--reward_batch_size", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--check_deps", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args)
    manifest_path = args.manifest or str(Path(args.out_dir) / f"{args.algorithm}_manifest.json")
    write_manifest(manifest_path, manifest)
    manifest["manifest"] = manifest_path
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = {
    "version": "2026-05-23.inline",
    "adapters": [
        {"id": "arc_agi3", "kind": "interactive_env", "holdout": ["private_envs", "solution_traces"]},
        {"id": "swe_bench_pro", "kind": "long_horizon_git_container_patch", "holdout": ["hidden_tests", "gold_patches"]},
        {"id": "swe_bench_live", "kind": "fresh_git_container_patch", "holdout": ["future_tasks", "accepted_patches"]},
        {"id": "terminal_bench_2", "kind": "harbor_container", "holdout": ["hidden_tests", "solutions"]},
        {"id": "bfcl_v4", "kind": "tool_call_state_scorer", "holdout": ["expected_asts", "private_prompts"]},
        {"id": "tau2_tau3", "kind": "simulated_user_tool_state", "holdout": ["policies", "db_states"]},
        {"id": "mcp_agent_suite", "kind": "mcp_fixture_adapter", "holdout": ["seeded_workspaces", "grading_rubrics"]},
        {"id": "mmmu_pro", "kind": "multimodal_mcq", "holdout": ["questions", "answers", "images"]},
        {"id": "video_understanding", "kind": "media_understanding", "holdout": ["videos", "answers", "subtitles"]},
        {"id": "image_generation_edit", "kind": "media_generation", "holdout": ["private_prompts", "human_labels"]},
        {"id": "video_audio_music_tts", "kind": "media_generation", "holdout": ["private_prompts", "preference_labels"]},
    ],
}


FSDP_LOCAL_FORMAT = "omnicoder2026_native_train_checkpoint_v3_fsdp_local"


def _is_fsdp_rank_local_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    manifest = path / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            return isinstance(payload, dict) and payload.get("format") == FSDP_LOCAL_FORMAT
        except Exception:
            return False
    return any(path.glob("rank*.pt"))


def _checkpoint_dir_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    manifest = path / "manifest.json"
    if manifest.exists():
        h.update(manifest.read_bytes())
    for child in sorted(item for item in path.glob("rank*.pt") if item.is_file()):
        stat = child.stat()
        h.update(child.name.encode("utf-8"))
        h.update(str(int(stat.st_size)).encode("ascii"))
        h.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return h.hexdigest()


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if _is_fsdp_rank_local_checkpoint_dir(p):
        return _checkpoint_dir_fingerprint(p)
    if p.exists() and p.is_file():
        h.update(p.read_bytes())
    elif p.exists() and p.is_dir():
        for child in sorted(item for item in p.iterdir() if item.is_file()):
            stat = child.stat()
            h.update(child.name.encode("utf-8"))
            h.update(str(int(stat.st_size)).encode("ascii"))
    else:
        h.update(path.encode("utf-8"))
    return h.hexdigest()


def load_registry(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return DEFAULT_REGISTRY


def registry_ids(registry: dict[str, Any]) -> list[str]:
    return [str(item["id"]) for item in registry.get("adapters", [])]


def make_smoke_result(model: str, benchmark: str, registry: dict[str, Any]) -> dict[str, Any]:
    adapters = registry.get("adapters", [])
    selected = adapters if benchmark == "registry_smoke" else [a for a in adapters if a.get("id") == benchmark]
    return {
        "benchmark_id": benchmark,
        "version": registry.get("version", "2026.registry"),
        "split": "smoke",
        "model": model,
        "model_hash": hash_file(model),
        "scaffold": "omnicoder2026_native_kda_csa_hca",
        "tools_allowed": False,
        "pass_at": 1,
        "cost": 0.0,
        "latency": 0.0,
        "tokens": 0,
        "trace_hash": None,
        "artifact_hash": None,
        "score": None,
        "subscores": {},
        "failure_taxonomy": [],
        "registered_benchmarks": registry_ids(registry),
        "selected_adapters": selected,
        "release_gates": registry.get("release_gates", {}),
        "quarantine_rule": "benchmark prompts, labels, traces, scorer artifacts, and successful eval trajectories are never exported to training",
        "contamination_controls": registry.get("contamination_controls", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 registry benchmark harness")
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="registry_smoke")
    ap.add_argument("--registry", default="profiles/benchmark_registry_2026.json")
    ap.add_argument("--out", default="weights/eval_2026_smoke.json")
    args = ap.parse_args()

    registry = load_registry(args.registry)
    result = make_smoke_result(args.model, args.benchmark, registry)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": args.out, "benchmark": args.benchmark, "registered": len(result["registered_benchmarks"])}))


if __name__ == "__main__":
    main()

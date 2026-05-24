from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omnicoder.training.metrics_2026 import iter_json_events, summarize_training_log
from omnicoder.training.run_registry_2026 import JsonlRunRegistry, read_json, stable_hash_file, write_json


DEFAULT_STAGES = (
    "ingest_trace",
    "ingest_media",
    "quality_score",
    "contam_scan",
    "export_sft",
    "agentic_tool_training",
    "teacher_jobs",
    "sft_qlora_bridge",
    "native_train",
    "eval_smoke",
    "context_budget",
)


def _as_bool(value: Any) -> bool:
    return bool(value) and str(value).lower() not in {"0", "false", "no", "off", "none"}


def load_profile(path: str | Path) -> dict[str, Any]:
    profile = read_json(path, {})
    if not isinstance(profile, dict):
        raise ValueError(f"profile must be JSON object: {path}")
    return profile


def ensure_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "run": run_dir,
        "logs": run_dir / "logs",
        "data": run_dir / "data",
        "processed": run_dir / "data" / "processed",
        "protected": run_dir / "data" / "protected",
        "exports": run_dir / "data" / "exports",
        "weights": run_dir / "weights",
        "eval": run_dir / "eval",
        "teacher": run_dir / "teacher_jobs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def stage_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_STAGES)
    return [x.strip() for x in value.split(",") if x.strip()]


def _protected_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(record.get(key) or "")
        for key in ("id", "name", "benchmark", "kind", "description", "notes")
        if record.get(key) is not None
    )


def build_protected_registry(benchmark_registry: str, out_path: Path) -> int:
    payload = read_json(benchmark_registry, {})
    adapters: Any = []
    if isinstance(payload, dict):
        adapters = payload.get("adapters") or payload.get("benchmarks") or payload.get("registry") or []
    elif isinstance(payload, list):
        adapters = payload
    if isinstance(adapters, dict):
        adapters = list(adapters.values())
    rows = []
    if isinstance(adapters, list):
        for item in adapters:
            if isinstance(item, dict):
                text = _protected_text(item)
                name = item.get("id") or item.get("name") or item.get("benchmark") or "protected"
                if text:
                    rows.append({"benchmark_name": name, "text": text})
    if not rows:
        rows = [
            {"benchmark_name": name, "text": name}
            for name in ("ARC-AGI-3", "SWE-bench", "Terminal-Bench", "BFCL", "tau-bench", "MCP-Atlas")
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def run_command(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"cmd": cmd, "cwd": str(cwd)}, ensure_ascii=True) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = proc.wait()
        handle.write(json.dumps({"returncode": code}, ensure_ascii=True) + "\n")
    return code, summarize_training_log(log_path)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def execute_stage(
    stage: str,
    run_id: str,
    registry: JsonlRunRegistry,
    profile: dict[str, Any],
    paths: dict[str, Path],
    repo_root: Path,
    current: dict[str, Path],
    dry_run: bool = False,
) -> None:
    data_cfg = profile.get("data", {}) if isinstance(profile.get("data"), dict) else {}
    train_cfg = profile.get("native_train", {}) if isinstance(profile.get("native_train"), dict) else {}
    eval_cfg = profile.get("eval", {}) if isinstance(profile.get("eval"), dict) else {}
    teacher_cfg = profile.get("teacher_jobs", {}) if isinstance(profile.get("teacher_jobs"), dict) else {}
    qlora_cfg = profile.get("sft_qlora", {}) if isinstance(profile.get("sft_qlora"), dict) else {}
    tool_cfg = profile.get("agentic_tool_training", {}) if isinstance(profile.get("agentic_tool_training"), dict) else {}
    python = sys.executable
    env = {"PYTHONPATH": str(repo_root / "src")}
    log_path = paths["logs"] / f"{stage}.log"
    cmd: list[str] = []

    if stage == "ingest_trace":
        trace_input = Path(str(data_cfg.get("trace_input") or current.get("trace_input") or ""))
        if not trace_input.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": f"missing trace_input {trace_input}"})
            return
        out = paths["processed"] / "trace_ingested.jsonl"
        cmd = [
            python, "-m", "omnicoder.data_factory.ingest_agent_memory",
            "--input", str(trace_input),
            "--out", str(out),
            "--source_date", str(profile.get("source_date") or data_cfg.get("source_date") or "2026-05-23"),
            "--limit", str(int(data_cfg.get("limit") or 0)),
        ]
        current["ingested"] = out
    elif stage == "ingest_media":
        media_root = data_cfg.get("media_root")
        if not media_root:
            registry.stage(run_id, stage, "skipped", metadata={"reason": "media_root not configured"})
            return
        media_path = Path(str(media_root))
        if not media_path.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": f"missing media_root {media_path}"})
            return
        out = paths["processed"] / "media_ingested.jsonl"
        cmd = [
            python, "-m", "omnicoder.data_factory.ingest_comfyui_outputs",
            "--input", str(media_path),
            "--out", str(out),
            "--source_date", str(profile.get("source_date") or "2026-05-23"),
            "--limit", str(int(data_cfg.get("media_limit") or 0)),
        ]
        current["media_ingested"] = out
    elif stage == "quality_score":
        src = current.get("ingested")
        if not src or not src.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": "no ingested trace file"})
            return
        out = paths["processed"] / "trace_scored.jsonl"
        cmd = [
            python, "-m", "omnicoder.data_factory.quality_scoring",
            "--input", str(src),
            "--out", str(out),
            "--min-score", str(float(data_cfg.get("min_quality") or 0.0)),
        ]
        current["scored"] = out
    elif stage == "contam_scan":
        src = current.get("scored")
        if not src or not src.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": "no scored trace file"})
            return
        protected = paths["protected"] / "benchmark_protected.jsonl"
        protected_count = build_protected_registry(str(profile.get("benchmark_registry") or "profiles/benchmark_registry_2026.json"), protected)
        out = paths["processed"] / "trace_clean.jsonl"
        cmd = [
            python, "-m", "omnicoder.data_factory.contamination",
            "--candidates", str(src),
            "--protected", str(protected),
            "--out", str(out),
            "--threshold", str(float(data_cfg.get("contamination_threshold") or 0.42)),
            "--ngram", str(int(data_cfg.get("contamination_ngram") or 5)),
        ]
        current["clean"] = out
        current["protected"] = protected
        registry.metric(run_id, "protected_registry_rows", protected_count, stage_name=stage)
    elif stage == "export_sft":
        src = current.get("clean") or current.get("scored") or current.get("ingested")
        if not src or not src.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": "no clean/scored trace file"})
            return
        out = paths["exports"] / "train.jsonl"
        cmd = [
            python, "-m", "omnicoder.data_factory.export_sft_jsonl",
            "--input", str(src),
            "--out", str(out),
            "--min-quality", str(float(data_cfg.get("min_quality") or 0.0)),
            "--limit", str(int(data_cfg.get("sft_limit") or 0)),
        ]
        if _as_bool(data_cfg.get("group_traces", True)):
            cmd.append("--group-traces")
        current["sft"] = out
    elif stage == "agentic_tool_training":
        if not _as_bool(tool_cfg.get("enabled", True)):
            registry.stage(run_id, stage, "skipped", metadata={"reason": "agentic tool training disabled"})
            return
        src = Path(str(tool_cfg.get("input_jsonl") or ""))
        if not src.exists():
            src = current.get("clean") or current.get("scored") or current.get("sft") or current.get("ingested") or Path("")
        if not src.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": f"missing agentic tool source {src}"})
            return
        out_dir = paths["exports"] / "agentic_tool_training"
        manifest = out_dir / "agentic_tool_training_manifest.json"
        cmd = [
            python, "-m", "omnicoder.training.agentic_tool_training_2026",
            "--profile", str(tool_cfg.get("profile") or profile.get("agentic_tool_training_profile") or "profiles/agentic_tool_training_2026.json"),
            "build",
            "--input", str(src),
            "--out-dir", str(out_dir),
            "--model", str(tool_cfg.get("model") or qlora_cfg.get("model") or "Qwen/Qwen3-4B"),
            "--min-quality", str(float(tool_cfg.get("min_quality") or data_cfg.get("min_quality") or 0.0)),
            "--limit", str(int(tool_cfg.get("limit") or 0)),
        ]
        if _as_bool(tool_cfg.get("dry_run", True)):
            cmd.append("--dry-run")
        current["agentic_tool_manifest"] = manifest
        current["agentic_tool_sft"] = out_dir / "tool_sft.jsonl"
        current["agentic_tool_preference"] = out_dir / "tool_preference.jsonl"
        current["agentic_tool_reward"] = out_dir / "tool_reward.jsonl"
        current["agentic_tool_rlvr"] = out_dir / "tool_rlvr.jsonl"
        current["agentic_tool_safety"] = out_dir / "tool_safety_negatives.jsonl"
    elif stage == "teacher_jobs":
        if not _as_bool(teacher_cfg.get("enabled", False)):
            registry.stage(run_id, stage, "skipped", metadata={"reason": "teacher jobs disabled"})
            return
        src = current.get("clean") or current.get("sft")
        if not src or not src.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": "no source records for teacher jobs"})
            return
        out = paths["teacher"] / "teacher_jobs.jsonl"
        cmd = [
            python, "-m", "omnicoder.data_factory.teacher_jobs_2026", "build",
            "--records", str(src),
            "--teacher", str(teacher_cfg.get("teacher") or "qwen3.6_27b_q4_local"),
            "--job_type", str(teacher_cfg.get("job_type") or "trace_critique"),
            "--limit", str(int(teacher_cfg.get("limit") or 0)),
            "--out", str(out),
        ]
        current["teacher_jobs"] = out
    elif stage == "sft_qlora_bridge":
        if not _as_bool(qlora_cfg.get("enabled", False)):
            registry.stage(run_id, stage, "skipped", metadata={"reason": "sft_qlora disabled"})
            return
        data = current.get("sft") or Path(str(qlora_cfg.get("train_jsonl") or ""))
        if not data.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": f"missing qlora data {data}"})
            return
        out_dir = paths["weights"] / "sft_qlora_bridge"
        manifest = out_dir / "sft_qlora_manifest.json"
        cmd = [
            python, "-m", "omnicoder.training.sft_qlora_2026",
            "--model", str(qlora_cfg.get("model") or "Qwen/Qwen3-4B"),
            "--train_jsonl", str(data),
            "--out_dir", str(out_dir),
            "--manifest", str(manifest),
            "--max_seq_len", str(int(qlora_cfg.get("max_seq_len") or 4096)),
            "--max_steps", str(int(qlora_cfg.get("max_steps") or 1000)),
            "--learning_rate", str(float(qlora_cfg.get("learning_rate") or 1e-4)),
            "--per_device_train_batch_size", str(int(qlora_cfg.get("per_device_train_batch_size") or 1)),
            "--gradient_accumulation_steps", str(int(qlora_cfg.get("gradient_accumulation_steps") or 16)),
            "--lora_r", str(int(qlora_cfg.get("lora_r") or 16)),
            "--lora_alpha", str(int(qlora_cfg.get("lora_alpha") or 32)),
            "--target_modules", str(qlora_cfg.get("target_modules") or "all-linear"),
        ]
        if _as_bool(qlora_cfg.get("load_in_4bit", True)):
            cmd.append("--load_in_4bit")
        if _as_bool(qlora_cfg.get("packing", True)):
            cmd.append("--packing")
        if _as_bool(qlora_cfg.get("assistant_only_loss", True)):
            cmd.append("--assistant_only_loss")
        if _as_bool(qlora_cfg.get("dry_run", True)):
            cmd.append("--dry_run")
        current["sft_qlora_manifest"] = manifest
    elif stage == "native_train":
        data = current.get("sft") or Path(str(data_cfg.get("sft_input") or ""))
        if not data.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": f"missing training data {data}"})
            return
        out = paths["weights"] / "native_train.pt"
        cmd = [
            python, "-m", "omnicoder.training.pretrain_2026_dense",
            "--preset", str(profile.get("preset") or "probe"),
            "--data", str(data),
            "--out", str(out),
            "--seq_len", str(int(train_cfg.get("seq_len") or 64)),
            "--batch_size", str(int(train_cfg.get("batch_size") or 1)),
            "--steps", str(int(train_cfg.get("steps") or 3)),
            "--max_records", str(int(train_cfg.get("max_records") or 0)),
            "--device", str(train_cfg.get("device") or "cuda"),
            "--log_file", str(paths["logs"] / "native_train_inner.jsonl"),
            "--data_manifest", str(data),
        ]
        if int(train_cfg.get("save_interval") or 0) > 0:
            cmd.extend(["--save_interval", str(int(train_cfg.get("save_interval") or 0))])
        if _as_bool(train_cfg.get("aux_probe", True)):
            cmd.append("--aux_probe")
        if _as_bool(train_cfg.get("fake_quant", False)):
            cmd.append("--fake_quant")
        if _as_bool(train_cfg.get("compile", False)):
            cmd.append("--compile")
        current["checkpoint"] = out
    elif stage == "eval_smoke":
        model = current.get("checkpoint")
        if not model or not model.exists():
            registry.stage(run_id, stage, "skipped", metadata={"reason": "no checkpoint for eval"})
            return
        out = paths["eval"] / "registry_smoke.json"
        cmd = [
            python, "-m", "omnicoder.eval.harness_2026",
            "--model", str(model),
            "--benchmark", str(eval_cfg.get("benchmark") or "registry_smoke"),
            "--registry", str(profile.get("benchmark_registry") or "profiles/benchmark_registry_2026.json"),
            "--out", str(out),
        ]
        current["eval"] = out
    elif stage == "context_budget":
        out = paths["eval"] / "context_budget.json"
        cmd = [
            python, "-m", "omnicoder.inference.context_budget_2026",
            "--profile", str(profile.get("budget_profile") or "omnicoder2026_20b_1m"),
            "--context", str(int(profile.get("context") or 1048576)),
            "--out", str(out),
            "--compact",
        ]
        current["context_budget"] = out
    else:
        registry.stage(run_id, stage, "skipped", metadata={"reason": "unknown stage"})
        return

    registry.stage(run_id, stage, "running", command=cmd, log_path=str(log_path))
    if dry_run:
        registry.stage(run_id, stage, "skipped", command=cmd, log_path=str(log_path), metadata={"dry_run": True})
        return
    code, summary = run_command(cmd, log_path, repo_root, env)
    if stage == "context_budget":
        events = list(iter_json_events(log_path))
        budget = next((event for event in reversed(events) if event.get("profile")), None)
        if budget:
            write_json(current["context_budget"], budget)
            registry.artifact(run_id, str(current["context_budget"]), "context_budget", stage)
    if code != 0:
        registry.stage(run_id, stage, "failed", command=cmd, log_path=str(log_path), metrics=summary)
        raise SystemExit(code)
    metrics = dict(summary)
    for key, path in current.items():
        if isinstance(path, Path) and path.exists() and path.suffix == ".jsonl":
            metrics[f"{key}_records"] = count_jsonl(path)
    registry.stage(run_id, stage, "completed", command=cmd, log_path=str(log_path), metrics=metrics)
    if stage == "native_train" and current.get("checkpoint") and current["checkpoint"].exists():
        registry.artifact(run_id, str(current["checkpoint"]), "checkpoint", stage, {"summary": summary})
        if summary.get("loss_last") is not None:
            registry.metric(run_id, "loss", summary["loss_last"], step=summary.get("steps"), stage_name=stage)
    stage_artifacts = {
        "export_sft": ("sft",),
        "agentic_tool_training": (
            "agentic_tool_manifest",
            "agentic_tool_sft",
            "agentic_tool_preference",
            "agentic_tool_reward",
            "agentic_tool_rlvr",
            "agentic_tool_safety",
        ),
        "teacher_jobs": ("teacher_jobs",),
        "sft_qlora_bridge": ("sft_qlora_manifest",),
        "eval_smoke": ("eval",),
        "context_budget": ("context_budget",),
    }.get(stage, ())
    for artifact_key in stage_artifacts:
        artifact = current.get(artifact_key)
        if isinstance(artifact, Path) and artifact.exists():
            registry.artifact(run_id, str(artifact), artifact_key, stage)


def run_harness(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    profile = load_profile(args.profile)
    if args.trace_input:
        profile.setdefault("data", {})["trace_input"] = args.trace_input
    if args.media_root:
        profile.setdefault("data", {})["media_root"] = args.media_root
    if args.steps is not None:
        profile.setdefault("native_train", {})["steps"] = int(args.steps)
    if args.device:
        profile.setdefault("native_train", {})["device"] = args.device
    registry_root = Path(args.registry_root or profile.get("registry_root") or "weights/runs_2026")
    registry = JsonlRunRegistry(registry_root)
    run_name = args.run_name or str(profile.get("run_name") or "omnicoder2026_harness")
    manifest = registry.create_run(
        run_name=run_name,
        recipe=str(profile.get("recipe") or "native_trace_sft_probe"),
        profile=str(profile.get("profile_name") or Path(args.profile).stem),
        preset=str(profile.get("preset") or "probe"),
        config=profile,
        metadata={
            "profile_path": str(args.profile),
            "repo_root": str(repo_root),
            "data_manifest_sha256": stable_hash_file(args.trace_input) if args.trace_input else None,
        },
        run_id=args.run_id,
    )
    run_id = manifest["run_id"]
    paths = ensure_dirs(Path(profile.get("work_dir") or "weights/harness_2026") / run_id)
    current: dict[str, Path] = {}
    if args.trace_input or profile.get("data", {}).get("trace_input"):
        current["trace_input"] = Path(str(args.trace_input or profile["data"]["trace_input"]))
    registry.update_status(run_id, "running")
    try:
        for stage in stage_list(args.stages):
            execute_stage(stage, run_id, registry, profile, paths, repo_root, current, dry_run=bool(args.dry_run))
        registry.update_status(run_id, "completed")
    except BaseException as exc:
        registry.update_status(run_id, "failed", error=str(exc))
        raise
    print(json.dumps({"status": "ok", "run_id": run_id, "manifest": str(registry.manifest_path(run_id))}, ensure_ascii=True))


def status(args: argparse.Namespace) -> None:
    registry = JsonlRunRegistry(args.registry_root)
    print(json.dumps(registry.load(args.run_id), indent=2, ensure_ascii=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 full training harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--profile", default="profiles/training_harness_2026.json")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--registry-root", default=None)
    run.add_argument("--run-name", default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument("--stages", default="all")
    run.add_argument("--trace-input", default=None)
    run.add_argument("--media-root", default=None)
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--device", default=None)
    run.add_argument("--dry-run", action="store_true")
    stat = sub.add_parser("status")
    stat.add_argument("--registry-root", default="weights/runs_2026")
    stat.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run_harness(args)
    elif args.cmd == "status":
        status(args)


if __name__ == "__main__":
    main()

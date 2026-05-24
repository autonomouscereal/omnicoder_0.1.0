from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2026-05-23"
DEFAULT_PROFILE = "profiles/training_orchestration_2026.json"
DEFAULT_OUT_ROOT = "weights/training_orchestration_2026/gpu_sidecar"
DEFAULT_JOB_TYPES = {
    "dataset_materialization",
    "external_dataset_expansion",
    "materialization",
    "training",
    "training_run",
    "teacher_distillation",
    "openai_compatible_teacher_rollout",
    "eval",
    "eval_shard",
    "benchmark_canary",
    "custom_command",
}

JOB_TYPE_ALIASES = {
    "materialization": "dataset_materialization",
    "training": "training_run",
    "eval": "eval_shard",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")


def profile_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    nested = profile.get("training_orchestration")
    return nested if isinstance(nested, dict) else profile


def sidecar_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    cfg = profile_cfg(profile)
    sidecar = cfg.get("gpu_utilization_sidecar")
    return sidecar if isinstance(sidecar, dict) else {}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        text = str(value)
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def normalize_job_type(job_type: Any) -> str:
    text = str(job_type or "").strip()
    return JOB_TYPE_ALIASES.get(text, text)


def split_devices(value: Any) -> list[str]:
    devices: list[str] = []
    for item in as_list(value):
        for part in str(item).split(","):
            part = part.strip()
            if part:
                devices.append(part)
    return devices


def distributed_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    plan = profile_cfg(profile).get("training_plan")
    if not isinstance(plan, dict):
        return {}
    dist = plan.get("distributed_training")
    return dist if isinstance(dist, dict) else {}


def configured_sidecar_devices(profile: dict[str, Any]) -> list[str]:
    cfg = sidecar_cfg(profile)
    dist = distributed_cfg(profile)
    return split_devices(cfg.get("devices") or dist.get("p40_sidecar_devices") or ["1", "2", "3", "5"])


def configured_main_devices(profile: dict[str, Any]) -> list[str]:
    cfg = sidecar_cfg(profile)
    dist = distributed_cfg(profile)
    env_devices = os.environ.get("OMNICODER_MAIN_FSDP_DEVICES", "")
    return split_devices(env_devices or cfg.get("main_training_devices") or dist.get("main_gpu_devices") or ["0", "4", "6"])


def selected_jobs(args: argparse.Namespace | None) -> set[str]:
    return set(as_list(getattr(args, "job", None) if args is not None else None))


def validate_device_isolation(profile: dict[str, Any], selected: set[str] | None = None) -> dict[str, Any]:
    selected = selected or set()
    sidecars = configured_sidecar_devices(profile)
    main = configured_main_devices(profile)
    overlap = [] if selected else sorted(set(sidecars) & set(main))
    sidecar_set = set(sidecars)
    main_set = set(main)
    job_device_overlaps: list[dict[str, str]] = []
    invalid_job_devices: list[dict[str, str]] = []
    raw_jobs = sidecar_cfg(profile).get("jobs")
    if isinstance(raw_jobs, list):
        for index, raw in enumerate(raw_jobs):
            if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
                continue
            job_id = str(raw.get("job_id") or raw.get("id") or f"sidecar_{index}")
            job_type = normalize_job_type(raw.get("job_type"))
            if selected and job_id not in selected and job_type not in selected and str(raw.get("job_type") or "") not in selected:
                continue
            try:
                device_text = job_device(raw, sidecars, index)
            except ValueError:
                invalid_job_devices.append({"job_id": job_id, "device": "", "reason": "no_sidecar_devices_configured"})
                continue
            for device in split_devices(device_text):
                if device.lower() == "cpu":
                    continue
                if device in main_set:
                    job_device_overlaps.append({"job_id": job_id, "device": device})
                if device not in sidecar_set:
                    invalid_job_devices.append({"job_id": job_id, "device": device, "reason": "device_not_in_sidecar_pool"})
    status = "failed" if overlap or job_device_overlaps or invalid_job_devices else "ok"
    return {
        "status": status,
        "sidecar_devices": sidecars,
        "main_training_devices": main,
        "overlap": overlap,
        "job_device_overlaps": job_device_overlaps,
        "invalid_job_devices": invalid_job_devices,
    }


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def out_root_for(profile: dict[str, Any], args: argparse.Namespace | None = None) -> Path:
    cfg = sidecar_cfg(profile)
    value = getattr(args, "out_root", "") if args is not None else ""
    return resolve_path(value or str(cfg.get("out_root") or DEFAULT_OUT_ROOT), repo_root())


def placeholder_context(
    profile_path: str | Path,
    profile: dict[str, Any],
    out_root: Path,
    job: dict[str, Any],
    device: str,
) -> dict[str, str]:
    cfg = profile_cfg(profile)
    distill = cfg.get("distillation") if isinstance(cfg.get("distillation"), dict) else {}
    bench = cfg.get("benchmark_gates") if isinstance(cfg.get("benchmark_gates"), dict) else {}
    job_id = str(job.get("job_id") or job.get("id") or job.get("job_type"))
    job_out = out_root / job_id
    run_id = os.environ.get("OMNICODER_RUN_ID", "") or job_id
    return {
        "python": sys.executable,
        "repo_root": str(repo_root()),
        "profile": str(profile_path),
        "sidecar_out": str(out_root),
        "job_out": str(job_out),
        "job_id": job_id,
        "run_id": run_id,
        "device": device,
        "distill_profile": str(job.get("distill_profile") or distill.get("teacher_profile") or "profiles/distillation_curriculum_2026.json"),
        "benchmark_profile": str(job.get("benchmark_profile") or bench.get("benchmark_profile") or "profiles/benchmark_suite_2026.json"),
    }


def expand_command(raw: Any, context: dict[str, str]) -> list[str]:
    if isinstance(raw, str):
        parts = raw.split()
    elif isinstance(raw, list):
        parts = [str(item) for item in raw]
    else:
        raise ValueError("command must be a string or list")
    return [part.format(**context) for part in parts]


def default_command(profile_path: str | Path, profile: dict[str, Any], out_root: Path, job: dict[str, Any], device: str) -> list[str]:
    context = placeholder_context(profile_path, profile, out_root, job, device)
    job_type = normalize_job_type(job.get("job_type"))
    if job_type == "dataset_materialization":
        return [
            sys.executable,
            "-m",
            "omnicoder.training.training_orchestration_2026",
            "--profile",
            context["profile"],
            "--out-dir",
            context["job_out"],
            "curate-real",
        ]
    if job_type == "external_dataset_expansion":
        dataset_profile = str(job.get("dataset_profile") or job.get("profile") or "profiles/dataset_curation_2026.json")
        raw_out = str(job.get("out_dir") or job.get("out") or context["job_out"])
        expanded_out = raw_out.replace("${RUN_ID}", context["run_id"]).format(**context)
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.data_factory.dataset_expansion_2026",
            "--profile",
            dataset_profile,
            "--out-dir",
            expanded_out,
        ]
        if bool(job.get("download", True)):
            cmd.append("--download")
        if bool(job.get("no_streaming", False)):
            cmd.append("--no-streaming")
        if bool(job.get("enforce_requirements", False)):
            cmd.append("--enforce-requirements")
        limit = int(job.get("max_records_per_dataset") or job.get("limit") or 0)
        if limit > 0:
            cmd.extend(["--max-records-per-dataset", str(limit)])
        cmd.append("build")
        return cmd
    if job_type == "teacher_distillation":
        records = str(job.get("records") or str(out_root / "dataset_materialization" / "jsonl" / "curated_records.jsonl"))
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.training.distillation_curriculum_2026",
            "build-jobs",
            "--profile",
            context["distill_profile"],
            "--records",
            records,
            "--out-dir",
            context["job_out"],
        ]
        limit = int(job.get("limit") or 0)
        if limit > 0:
            cmd.extend(["--limit", str(limit)])
        return cmd
    if job_type == "training_run":
        stage = str(job.get("stage") or job.get("orchestration_command") or "run-real")
        if stage not in {"run-real", "run-full"}:
            raise ValueError(f"training_run stage must be run-real or run-full, got: {stage}")
        preset = str(job.get("preset") or "ledger_probe")
        if not any(token in preset.lower() for token in ("probe", "ledger")):
            raise ValueError(f"sidecar training_run may only use probe presets, got: {preset}")
        if str(job.get("placement") or "single") != "single":
            raise ValueError("sidecar training_run must use single-device placement")
        if str(job.get("distributed") or "none") not in {"", "none"}:
            raise ValueError("sidecar training_run must use --distributed none")
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.training.training_orchestration_2026",
            "--profile",
            context["profile"],
            "--out-dir",
            context["job_out"],
            stage,
            "--device",
            str(job.get("trainer_device") or "cuda"),
            "--preset",
            preset,
            "--distributed",
            "none",
            "--placement",
            "single",
            "--allow-verifier-preset",
        ]
        numeric_args = [
            ("steps_per_stage", "--steps-per-stage", int),
            ("seq_len", "--seq-len", int),
            ("batch_size", "--batch-size", int),
            ("lr", "--lr", float),
            ("save_interval", "--save-interval", int),
            ("nproc_per_node", "--nproc-per-node", int),
            ("posttrain_steps", "--posttrain-steps", int),
            ("posttrain_lr", "--posttrain-lr", float),
            ("posttrain_max_records", "--posttrain-max-records", int),
            ("fake_quant_chunk_rows", "--fake-quant-chunk-rows", int),
            ("fake_quant_max_full_elements", "--fake-quant-max-full-elements", int),
            ("optimizer_in_backward_grad_clip", "--optimizer-in-backward-grad-clip", float),
            ("optimizer_in_backward_adafactor_chunk_rows", "--optimizer-in-backward-adafactor-chunk-rows", int),
            ("optimizer_in_backward_adafactor_clip_threshold", "--optimizer-in-backward-adafactor-clip-threshold", float),
            ("optimizer_in_backward_adafactor_decay_rate", "--optimizer-in-backward-adafactor-decay-rate", float),
            ("optimizer_in_backward_adafactor_eps1", "--optimizer-in-backward-adafactor-eps1", float),
        ]
        if stage == "run-full":
            numeric_args.extend(
                [
                    ("distill_limit", "--distill-limit", int),
                    ("distill_steps", "--distill-steps", int),
                    ("distill_lr", "--distill-lr", float),
                    ("finetune_steps", "--finetune-steps", int),
                    ("finetune_lr", "--finetune-lr", float),
                    ("benchmark_seq_len", "--benchmark-seq-len", int),
                ]
            )
        for key, flag, caster in numeric_args:
            if key in job and job.get(key) not in (None, ""):
                value = caster(job[key])
                if isinstance(value, float) or value > 0:
                    cmd.extend([flag, str(value)])
        string_args = [
            ("resume_checkpoint", "--resume-checkpoint"),
            ("precision", "--precision"),
            ("init_dtype", "--init-dtype"),
            ("optimizer", "--optimizer"),
            ("placement_devices", "--placement-devices"),
            ("placement_layer_counts", "--placement-layer-counts"),
            ("placement_head_device", "--placement-head-device"),
            ("optimizer_in_backward_update", "--optimizer-in-backward-update"),
            ("optimizer_in_backward_clip_mode", "--optimizer-in-backward-clip-mode"),
        ]
        if stage == "run-full":
            string_args.append(("distill_profile", "--distill-profile"))
        for key, flag in string_args:
            value = str(job.get(key) or "").strip()
            if value:
                cmd.extend([flag, value])
        for key, flag in (
            ("fake_quant", "--fake-quant"),
            ("activation_checkpointing", "--activation-checkpointing"),
            ("cpu_offload", "--cpu-offload"),
            ("optimizer_in_backward", "--optimizer-in-backward"),
            ("live_posttraining", "--live-posttraining"),
        ):
            if bool(job.get(key, False)):
                cmd.append(flag)
        return cmd
    if job_type == "eval_shard":
        checkpoint = str(job.get("checkpoint") or "")
        data_dir = str(job.get("data_dir") or out_root / "dataset_materialization" / "jsonl")
        output = str(job.get("out") or out_root / context["job_id"] / "sample_loss.json")
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.eval.sample_loss_2026",
            "--checkpoint",
            checkpoint,
            "--data-dir",
            data_dir,
            "--profile",
            str(job.get("model_profile") or "ledger_probe"),
            "--device",
            "cuda",
            "--out",
            output,
        ]
        if bool(job.get("exclude_aggregate_jsonl", True)):
            cmd.append("--exclude-aggregate-jsonl")
        seq_len = int(job.get("seq_len") or 0)
        max_records = int(job.get("max_records_per_file") or 0)
        if seq_len > 0:
            cmd.extend(["--seq-len", str(seq_len)])
        if max_records > 0:
            cmd.extend(["--max-records-per-file", str(max_records)])
        return cmd
    if job_type == "benchmark_canary":
        model = str(job.get("model") or job.get("checkpoint") or "unknown")
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.eval.benchmark_suite_2026",
            "--profile",
            context["benchmark_profile"],
            "--out-dir",
            context["job_out"],
            "--model",
            model,
            "run-smoke",
            "--mode",
            str(job.get("mode") or "smoke"),
            "--cycle",
            str(job.get("cycle") or "smoke"),
            "--run-id",
            context["job_id"],
        ]
        for benchmark in as_list(job.get("benchmarks")):
            cmd.extend(["--benchmark", benchmark])
        timeout = int(job.get("timeout_seconds") or 0)
        if timeout > 0:
            cmd.extend(["--timeout-seconds", str(timeout)])
        return cmd
    if job_type == "openai_compatible_teacher_rollout":
        records = str(job.get("records") or out_root / "teacher_jobs" / "latest" / "all_jobs.jsonl")
        output = str(job.get("output") or out_root / context["job_id"] / f"teacher_rollout_gpu{device}.jsonl")
        cmd = [
            sys.executable,
            "-m",
            "omnicoder.data_factory.openai_teacher_rollout_2026",
            "--input",
            records,
            "--out",
            output,
            "--base-url",
            str(job.get("base_url") or "http://127.0.0.1:18082/v1"),
            "--model",
            str(job.get("model") or job.get("teacher_model") or "qwen3.6-27b-q4"),
            "--record-kind",
            str(job.get("record_kind") or "qwen36_p40_teacher_rollout"),
        ]
        for key, flag, default in (
            ("limit", "--limit", 64),
            ("max_tokens", "--max-tokens", 512),
            ("timeout", "--timeout", 180),
            ("thermal_guard_celsius", "--max-gpu-temp", 0),
        ):
            value = int(job.get(key) or default)
            if value > 0:
                cmd.extend([flag, str(value)])
        for key, flag, default in (("temperature", "--temperature", 0.2), ("sleep", "--sleep", 0.0)):
            value = float(job.get(key) if job.get(key) not in (None, "") else default)
            if value > 0 or key == "temperature":
                cmd.extend([flag, str(value)])
        if str(job.get("thermal_gpu_index") or device).strip().lower() != "cpu":
            cmd.extend(["--thermal-gpu-index", str(job.get("thermal_gpu_index") or device)])
        return cmd
    raise ValueError(f"unknown sidecar job_type: {job_type}")


def job_device(job: dict[str, Any], devices: list[str], index: int) -> str:
    configured = as_list(job.get("device") or job.get("devices"))
    if configured:
        return ",".join(configured)
    if not devices:
        raise ValueError("no sidecar devices configured")
    return devices[index % len(devices)]


def per_device_teacher_job(job: dict[str, Any], job_id: str, device: str, offset: int) -> dict[str, Any]:
    worker = dict(job)
    worker["job_id"] = f"{job_id}_gpu{device}"
    worker["device"] = device
    records = str(worker.get("records") or "")
    if records:
        if "{device}" in records:
            worker["records"] = records.format(device=device)
        elif "*" in records:
            worker["records"] = records.replace("*", device)
    output = str(worker.get("output") or "")
    if output:
        if "{device}" in output:
            worker["output"] = output.format(device=device)
        elif output.endswith(".jsonl"):
            worker["output"] = output[:-6] + f"_gpu{device}.jsonl"
    base_urls = worker.get("base_urls")
    if isinstance(base_urls, list) and offset < len(base_urls):
        worker["base_url"] = str(base_urls[offset])
    return worker


def should_skip_job(job: dict[str, Any], root: Path) -> str:
    missing = []
    for raw in as_list(job.get("skip_if_missing") or job.get("requires")):
        path = resolve_path(raw, root)
        if not path.exists():
            missing.append(str(path))
    return ", ".join(missing)


def materialize_jobs(profile_path: str | Path, profile: dict[str, Any], args: argparse.Namespace | None = None) -> list[dict[str, Any]]:
    cfg = sidecar_cfg(profile)
    devices = configured_sidecar_devices(profile)
    out_root = out_root_for(profile, args)
    root = repo_root()
    raw_jobs = cfg.get("jobs")
    if not isinstance(raw_jobs, list):
        raw_jobs = []
    selected = set(as_list(getattr(args, "job", None) if args is not None else None))
    jobs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict):
            continue
        job = dict(raw)
        job_id = str(job.get("job_id") or job.get("id") or f"sidecar_{index}")
        job_type = normalize_job_type(job.get("job_type"))
        if selected and job_id not in selected and job_type not in selected and str(job.get("job_type") or "") not in selected:
            continue
        if not bool(job.get("enabled", True)):
            continue
        device = job_device(job, devices, index)
        device_values = split_devices(device)
        expanded_jobs = [job]
        if job_type == "openai_compatible_teacher_rollout" and len(device_values) > 1:
            expanded_jobs = [per_device_teacher_job(job, job_id, value, offset) for offset, value in enumerate(device_values)]
        for expanded_job in expanded_jobs:
            expanded_job_id = str(expanded_job.get("job_id") or job_id)
            expanded_device = job_device(expanded_job, devices, index)
            context = placeholder_context(profile_path, profile, out_root, {**expanded_job, "job_id": expanded_job_id}, expanded_device)
            command = (
                expand_command(expanded_job["command"], context)
                if expanded_job.get("command")
                else default_command(profile_path, profile, out_root, {**expanded_job, "job_id": expanded_job_id}, expanded_device)
            )
            job_out = Path(context["job_out"])
            log_path = Path(str(expanded_job.get("log_path") or job_out / "sidecar.log"))
            missing = should_skip_job(expanded_job, root)
            jobs.append(
                {
                    "job_id": expanded_job_id,
                    "job_type": job_type,
                    "device": expanded_device,
                    "command": command,
                    "cwd": str(root),
                    "out_dir": str(job_out),
                    "log_path": str(log_path),
                    "status": "skipped" if missing else "planned",
                    "skip_reason": f"missing required path(s): {missing}" if missing else "",
                }
            )
        continue
    max_active_per_device = int(cfg.get("max_active_per_device") or 0)
    if max_active_per_device > 0:
        active_by_device: dict[str, int] = {}
        for job in jobs:
            if job["status"] != "planned":
                continue
            device = str(job["device"])
            active = active_by_device.get(device, 0)
            if active >= max_active_per_device:
                job["status"] = "deferred"
                job["skip_reason"] = f"device {device} already has {max_active_per_device} planned sidecar job(s)"
                continue
            active_by_device[device] = active + 1
    return jobs


def launch_env(job: dict[str, Any], profile: dict[str, Any]) -> dict[str, str]:
    cfg = sidecar_cfg(profile)
    env = os.environ.copy()
    visible_devices = "" if str(job["device"]).strip().lower() == "cpu" else str(job["device"])
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "OMNICODER_SIDECAR_JOB_ID": str(job["job_id"]),
            "OMNICODER_SIDECAR_OUT_DIR": str(job["out_dir"]),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(cfg.get("omp_threads") or 4),
            "PYTHONPATH": str(repo_root() / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    for key, value in (cfg.get("env") if isinstance(cfg.get("env"), dict) else {}).items():
        env[str(key)] = str(value)
    return env


def popen_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def launch_jobs(profile_path: str | Path, profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    isolation = validate_device_isolation(profile, selected_jobs(args))
    if isolation["status"] != "ok":
        raise SystemExit(json.dumps({"status": "error", "error": "sidecar_devices_overlap_main_training", **isolation}))
    jobs = materialize_jobs(profile_path, profile, args)
    out_root = out_root_for(profile, args)
    manifest_dir = out_root / "manifests"
    manifest_path = manifest_dir / f"sidecar_launch_{int(time.time())}.json"
    event_log = out_root / "jsonl" / "sidecar_events.jsonl"
    launched = []
    dry_run = bool(getattr(args, "dry_run", False))
    wait = bool(getattr(args, "wait", False))
    for job in jobs:
        Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
        event = {"event": "sidecar_job_planned", "created_at": now_iso(), **job}
        append_jsonl(event_log, event)
        if job["status"] != "planned":
            launched.append(job)
            append_jsonl(event_log, {"event": f"sidecar_job_{job['status']}", "created_at": now_iso(), **job})
            continue
        if dry_run:
            row = {**job, "status": "dry_run", "pid": None}
            launched.append(row)
            append_jsonl(event_log, {"event": "sidecar_job_dry_run", "created_at": now_iso(), **row})
            continue
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = launch_env(job, profile)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "sidecar_launch", "created_at": now_iso(), "job": job}, ensure_ascii=True) + "\n")
            proc = subprocess.Popen(
                [str(part) for part in job["command"]],
                cwd=job["cwd"],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **popen_kwargs(),
            )
            row = {**job, "status": "running", "pid": int(proc.pid), "env": {"CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "")}}
            if wait:
                code = proc.wait()
                row.update({"status": "passed" if code == 0 else "failed", "returncode": int(code)})
        launched.append(row)
        append_jsonl(event_log, {"event": "sidecar_job_launched", "created_at": now_iso(), **row})
    manifest = {
        "schema": "omnicoder.gpu_sidecar_launch_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "created_at": now_iso(),
        "profile": str(profile_path),
        "out_root": str(out_root),
        "isolation": isolation,
        "dry_run": dry_run,
        "wait": wait,
        "jobs": launched,
        "event_log": str(event_log),
    }
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def latest_manifest(out_root: Path) -> Path | None:
    manifest_dir = out_root / "manifests"
    if not manifest_dir.exists():
        return None
    manifests = sorted(manifest_dir.glob("sidecar_launch_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    isolation = validate_device_isolation(profile, selected_jobs(args))
    cfg = sidecar_cfg(profile)
    jobs = materialize_jobs(args.profile, profile, args)
    job_types = sorted({job["job_type"] for job in jobs})
    unknown = sorted(job_type for job_type in job_types if job_type not in DEFAULT_JOB_TYPES and job_type)
    status = "ok" if isolation["status"] == "ok" and not unknown else "failed"
    return {
        "schema": "omnicoder.gpu_sidecar_validation_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "enabled": bool(cfg.get("enabled", False)),
        "out_root": str(out_root_for(profile, args)),
        "isolation": isolation,
        "jobs": len(jobs),
        "job_types": job_types,
        "unknown_job_types": unknown,
    }


def cmd_plan(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    isolation = validate_device_isolation(profile, selected_jobs(args))
    return {
        "schema": "omnicoder.gpu_sidecar_plan_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if isolation["status"] == "ok" else "failed",
        "profile": str(args.profile),
        "out_root": str(out_root_for(profile, args)),
        "isolation": isolation,
        "jobs": materialize_jobs(args.profile, profile, args),
    }


def cmd_launch(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    return launch_jobs(args.profile, profile, args)


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    out_root = out_root_for(profile, args)
    manifest_path = latest_manifest(out_root)
    if manifest_path is None:
        return {"status": "missing", "out_root": str(out_root), "reason": "no sidecar launch manifests"}
    manifest = read_json(manifest_path)
    jobs = []
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        row = dict(job)
        pid = int(row.get("pid") or 0)
        if row.get("status") == "running":
            row["alive"] = pid_alive(pid)
        jobs.append(row)
    return {"status": "ok", "manifest": str(manifest_path), "jobs": jobs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch disjoint Omnicoder 2026 GPU sidecar jobs on spare P40s")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--job", action="append", help="Filter by job_id or job_type. Repeatable.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    launch = sub.add_parser("launch")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--wait", action="store_true")
    launch.set_defaults(func=cmd_launch)
    sub.add_parser("status").set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("status") in {"ok", "missing"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

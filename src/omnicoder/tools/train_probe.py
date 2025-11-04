from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:
    # Best-effort import for auto resource scaling
    from omnicoder.utils.resources import apply_thread_env_if_auto  # type: ignore
except Exception:  # pragma: no cover
    def apply_thread_env_if_auto() -> tuple[int, int, int]:
        return (1, 1, 1)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, float]:
    start = time.perf_counter()
    rc = 1
    try:
        proc = subprocess.run(cmd, check=False, env=env)
        rc = int(proc.returncode)
    except Exception:
        rc = 1
    dur = time.perf_counter() - start
    return rc, dur


def main() -> None:
    ap = argparse.ArgumentParser(description="Time-budgeted 1-step training probe and executor")
    ap.add_argument("--budget_minutes", type=int, default=int(os.getenv("OMNICODER_TRAIN_BUDGET_MIN", "10")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
    ap.add_argument("--max_per_task_steps", type=int, default=1)
    ap.add_argument("--skip", type=str, default=os.getenv("OMNICODER_TRAIN_SKIP", "video_vq,vq_image,vq_audio,rl_grpo"))
    ap.add_argument("--log_json", type=str, default="weights/train_probe_summary.json")
    ap.add_argument("--ep_devices", type=str, default=os.getenv("OMNICODER_EP_DEVICES", ""), help="Optional: comma-separated devices for expert-parallel probe (e.g., cuda:0,cuda:1)")
    args = ap.parse_args()

    out_root = Path("weights/probe")
    out_root.mkdir(parents=True, exist_ok=True)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    tasks: list[tuple[str, list[str]]] = []
    # Minimal synthetic or tiny single-step runs to measure throughput and validate code paths
    if "pretrain" not in skip:
        tasks.append((
            "pretrain",
            [sys.executable, "-m", "omnicoder.training.pretrain",
             "--data", "examples", "--seq_len", "128", "--steps", str(args.max_per_task_steps),
             "--device", args.device, "--out", str(out_root / "pretrain.pt"),
             ],
        ))
    if "distill" not in skip:
        tasks.append((
            "distill",
            [sys.executable, "-m", "omnicoder.training.distill",
             "--data", "examples/code_eval", "--seq_len", "256", "--steps", str(args.max_per_task_steps),
             "--device", args.device, "--no_teachers_ok", "--out", str(out_root / "distill.pt"),
             ],
        ))
    if "flow_recon" not in skip:
        tasks.append((
            "flow_recon",
            [sys.executable, "-m", "omnicoder.training.flow_recon",
             "--data", "examples/data/vq/images", "--steps", str(args.max_per_task_steps),
             "--batch", "4", "--device", args.device, "--out", str(out_root / "flow_recon.pt"),
             ],
        ))
    if "audio_recon" not in skip:
        tasks.append((
            "audio_recon",
            [sys.executable, "-m", "omnicoder.training.audio_recon",
             "--data", "examples/data/vq/audio", "--steps", str(args.max_per_task_steps),
             "--batch", "2", "--device", args.device, "--out", str(out_root / "audio_recon.pt"),
             ],
        ))
    if "vq_image" not in skip:
        tasks.append((
            "vq_image",
            [sys.executable, "-m", "omnicoder.training.vq_train",
             "--data", "examples/data/vq/images", "--steps", str(args.max_per_task_steps),
             "--batch", "8", "--out", str(out_root / "image_vq.pt"),
             ],
        ))
    if "vq_audio" not in skip:
        tasks.append((
            "vq_audio",
            [sys.executable, "-m", "omnicoder.training.audio_vq_train",
             "--data", "examples/data/vq/audio", "--steps", str(args.max_per_task_steps),
             "--batch", "2", "--out", str(out_root / "audio_vq.pt"),
             ],
        ))
    if "video_vq" not in skip:
        video_list = str(Path("examples/data/vq/video/toylist.txt"))
        tasks.append((
            "video_vq",
            [sys.executable, "-m", "omnicoder.training.video_vq_train",
             "--videos", video_list, "--samples", "16", "--frames_per_video", "8",
             "--steps", str(args.max_per_task_steps), "--out", str(out_root / "video_vq.pt"),
             ],
        ))
    # Optional: write-policy and RL (skipped by default due to variability)
    if "write_policy" not in skip:
        tasks.append((
            "write_policy",
            [sys.executable, "-m", "omnicoder.training.write_policy_train",
             "--data", "examples/code_eval", "--steps", str(args.max_per_task_steps),
             "--device", args.device, "--out", str(out_root / "write_policy.pt"),
             ],
        ))
    if "rl_grpo" not in skip:
        tasks.append((
            "rl_grpo",
            [sys.executable, "-m", "omnicoder.training.rl_grpo",
             "--data", "examples/code_eval", "--steps", str(args.max_per_task_steps + 1),
             "--device", args.device, "--out", str(out_root / "rl_grpo.pt"),
             ],
        ))

    # Apply auto resource scaling if enabled; otherwise keep conservative defaults
    try:
        apply_thread_env_if_auto()
    except Exception:
        pass
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", env.get("MKL_NUM_THREADS", "1"))
    env.setdefault("TORCH_NUM_THREADS", env.get("TORCH_NUM_THREADS", "1"))
    env.setdefault("TRANSFORMERS_OFFLINE", env.get("TRANSFORMERS_OFFLINE", "1"))

    # Stage 1: measure 1-step durations (including optional EP probe)
    results: dict[str, dict] = {}
    budget_s = max(0, int(args.budget_minutes)) * 60
    for name, cmd in tasks:
        rc, dur = _run(cmd, env=env)
        results[name] = {"rc": rc, "duration_s": dur, "cmd": " ".join(shlex.quote(c) for c in cmd)}
    # Optional expert-parallel pretrain probe via torchrun_ep
    # Auto-run EP probe when multi-GPU available if ep_devices not explicitly set
    auto_ep_devices = ""
    try:
        import torch as _t  # type: ignore
        if _t.cuda.is_available() and _t.cuda.device_count() >= 2:
            auto_ep_devices = "cuda:0,cuda:1"
    except Exception:
        auto_ep_devices = ""
    ep_devices = args.ep_devices.strip() or auto_ep_devices
    if ep_devices:
        ep_cmd = [
            sys.executable, "-m", "omnicoder.tools.torchrun_ep",
            "--script", "omnicoder.training.pretrain",
            "--script_args", f"--data examples --seq_len 128 --steps {args.max_per_task_steps} --device {args.device} --out {out_root / 'pretrain_ep.pt'}",
            "--devices", ep_devices,
            "--init_dist",
        ]
        rc, dur = _run(ep_cmd, env=env)
        results["pretrain_ep_auto" if not args.ep_devices.strip() else "pretrain_ep"] = {"rc": rc, "duration_s": dur, "cmd": " ".join(shlex.quote(c) for c in ep_cmd), "ep_devices": ep_devices}

    # Stage 2: derive a simple plan to fill remaining budget
    # Prefer pretrain and distill, then flow/audio recon (consider EP variant when available)
    priorities = ["pretrain_ep", "pretrain", "distill", "flow_recon", "audio_recon", "vq_image", "vq_audio", "video_vq", "write_policy", "rl_grpo"]
    plan: list[tuple[str, int]] = []
    spent = sum(int(results[n]["duration_s"]) for n in results)
    remaining = max(0, budget_s - spent)
    # Greedy fill using median(1-step durations) of available tasks up to remaining time
    durations = {n: max(1.0, float(results[n]["duration_s"])) for n in results if results[n]["rc"] == 0}
    for n in priorities:
        if n in durations and remaining > durations[n]:
            reps = int(remaining // durations[n])
            if reps > 0:
                plan.append((n, reps))
                remaining -= int(reps * durations[n])
        if remaining < 1:
            break

    # Stage 3: execute the plan (best-effort)
    executed: list[dict] = []
    for n, reps in plan:
        cmd = next(c for name, c in tasks if name == n)
        for i in range(reps):
            rc, dur = _run(cmd, env=env)
            executed.append({"name": n, "rc": rc, "duration_s": dur})
            if rc != 0:
                break

    summary = {
        "budget_minutes": args.budget_minutes,
        "device": args.device,
        "skip": sorted(list(skip)),
        "stage1_results": results,
        "plan": plan,
        "executed": executed,
        "remaining_seconds": remaining,
    }
    # Best-effort GPU info (device count and total VRAM per device in GB)
    try:
        import torch as _t  # type: ignore
        if _t.cuda.is_available():
            n = _t.cuda.device_count()
            vram = []
            for i in range(n):
                props = _t.cuda.get_device_properties(i)
                vram.append(round(getattr(props, "total_memory", 0) / (1024**3), 2))
            summary["gpu_devices"] = n
            summary["gpu_vram_gb"] = vram
    except Exception:
        pass
    Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



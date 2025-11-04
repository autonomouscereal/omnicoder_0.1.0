from __future__ import annotations

"""
Android ADB helper to push the decode-step ONNX to a device and run NNAPI smoke.

Prereqs on device:
 - Python available in shell (e.g., Termux) with this package installed or
   the repo copied to device PYTHONPATH.
 - ONNX Runtime Mobile AAR for your app use-case (not needed for this smoke).

Usage (host):
  python -m omnicoder.tools.android_adb_run \
    --onnx weights/release/text/omnicoder_decode_step.onnx \
    --serial <device_id> --gen_tokens 128 --prompt_len 128 --tps_threshold 15.0

This will:
  1) adb push the ONNX to /data/local/tmp/omnicoder_decode_step.onnx
  2) adb shell: python -m omnicoder.inference.runtimes.nnapi_device_runner ...
  3) Pull JSON result and check tokens/sec >= threshold.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print(" ", " ".join(cmd))
    try:
        return subprocess.run(cmd, check=False).returncode
    except Exception as e:
        print(f"[run] failed: {e}")
        return 1


def _run_capture(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return int(proc.returncode), (proc.stdout or proc.stderr or "")
    except Exception as e:
        return 1, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="ADB push and NNAPI decode-step or i2v smoke with thresholds")
    ap.add_argument("--onnx", type=str, default=str(Path("weights/release/text/omnicoder_decode_step.onnx")), help="Path to decode-step ONNX")
    ap.add_argument("--serial", type=str, default="", help="adb -s <serial> target (optional)")
    ap.add_argument("--remote_path", type=str, default="/data/local/tmp/omnicoder_decode_step.onnx")
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--tps_threshold", type=float, default=0.0, help="Minimum tokens/sec to pass")
    # i2v
    ap.add_argument("--onnx_video_dir", type=str, default="", help="If set, run i2v ORT callable on device using NNAPI")
    ap.add_argument("--video_frames", type=int, default=24)
    ap.add_argument("--video_width", type=int, default=512)
    ap.add_argument("--video_height", type=int, default=320)
    ap.add_argument("--video_fps", type=int, default=24)
    ap.add_argument("--video_seed_image", type=str, default="", help="Seed image path on device (optional)")
    ap.add_argument("--fps_threshold", type=float, default=0.0, help="Minimum FPS to pass for i2v")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[error] ONNX not found: {onnx_path}")
        sys.exit(2)

    adb = ["adb"] + (["-s", args.serial] if args.serial else [])

    # 1) Push model
    rc = _run(adb + ["push", str(onnx_path), args.remote_path])
    if rc != 0:
        print("[error] adb push failed")
        sys.exit(2)

    # If a video dir is specified, run i2v path and exit
    if args.onnx_video_dir:
        remote_dir = "/data/local/tmp/i2v_onnx"
        adb = ["adb"] + (["-s", args.serial] if args.serial else [])
        _run(adb + ["shell", "mkdir", "-p", remote_dir])
        # Push generator.onnx
        gen_src = str(Path(args.onnx_video_dir) / "generator.onnx")
        rc = _run(adb + ["push", gen_src, f"{remote_dir}/generator.onnx"])
        if rc != 0:
            print("[error] adb push generator.onnx failed")
            sys.exit(2)
        video_json = "/data/local/tmp/oc_video_run.json"
        shell_cmd = [
            "python3", "-m", "omnicoder.inference.runtimes.nnapi_video_device_runner",
            "--onnx_dir", remote_dir,
            "--frames", str(args.video_frames),
            "--width", str(args.video_width),
            "--height", str(args.video_height),
            "--fps", str(args.video_fps),
            "--provider", "NNAPIExecutionProvider",
            "--out", video_json,
        ]
        if args.video_seed_image:
            shell_cmd += ["--seed_image", args.video_seed_image]
        _run(adb + ["shell", " ".join(shell_cmd)])
        host_out = Path("weights/release/video/nnapi_i2v_bench.json")
        host_out.parent.mkdir(parents=True, exist_ok=True)
        _run(adb + ["pull", video_json, str(host_out)])
        if not host_out.exists():
            print("[error] device i2v result not found")
            sys.exit(3)
        data = json.loads(host_out.read_text(encoding="utf-8"))
        fps = float(data.get("fps", 0.0))
        print(json.dumps(data, indent=2))
        if args.fps_threshold > 0.0 and fps < float(args.fps_threshold):
            print(f"[fail] fps {fps:.2f} < threshold {float(args.fps_threshold):.2f}")
            sys.exit(4)
        print("[ok] NNAPI i2v device run complete")
        return

    # 2) Run device-side Python (text decode-step)
    remote_out = "/data/local/tmp/oc_run.json"
    shell_cmd = [
        "python3", "-m", "omnicoder.inference.runtimes.nnapi_device_runner",
        "--model", args.remote_path,
        "--gen_tokens", str(args.gen_tokens),
        "--prompt_len", str(args.prompt_len),
        "--vocab_size", str(args.vocab_size),
        "--out", remote_out,
    ]
    rc = _run(adb + ["shell", " ".join(shell_cmd)])
    if rc != 0:
        print("[warn] device-side runner returned non-zero; attempting to pull any JSON anyway")

    # 3) Pull result
    host_out = Path("weights/release/text/nnapi_device_bench.json")
    host_out.parent.mkdir(parents=True, exist_ok=True)
    _run(adb + ["pull", remote_out, str(host_out)])
    if not host_out.exists():
        print("[error] device result not found; ensure Python is available on device and module import works")
        sys.exit(3)
    data = json.loads(host_out.read_text(encoding="utf-8"))
    tps = float(data.get("tokens_per_sec", 0.0))
    print(json.dumps(data, indent=2))

    # 4) Threshold check
    if args.tps_threshold > 0.0 and tps < float(args.tps_threshold):
        print(f"[fail] tokens/sec {tps:.2f} < threshold {float(args.tps_threshold):.2f}")
        sys.exit(4)
    print("[ok] NNAPI device run meets threshold" if args.tps_threshold > 0 else "[ok] NNAPI device run complete")


if __name__ == "__main__":
    main()



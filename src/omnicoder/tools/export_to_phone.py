from __future__ import annotations

"""
Export to Phone: package and push decode-step artifacts to Android/iOS.

- Android (ADB): pushes ONNX decode-step and sidecars to /data/local/tmp/omnicoder/
  Optionally invokes the NNAPI device smoke via the existing android_adb_run tool.
- iOS: copies the Core ML decode-step model into the SampleApp/SampleConsole resources.

This script avoids heavyweight rebuilds; it assumes you already ran press_play or
build_mobile_release so artifacts exist under weights/release/.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print(" ", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    except Exception as e:
        print(f"[run] failed: {e}")
        return 1


def _default_out_root() -> Path:
    return Path(os.getenv("OMNICODER_OUT_ROOT", "weights/release")).resolve()


def _android_push(onnx_path: Path, push_only: bool, tps_threshold: float) -> None:
    # Check adb
    rc = _run(["adb", "version"])  # best-effort
    if rc != 0:
        print("[android] adb not found or not working; install Android platform-tools and ensure adb is on PATH.")
        return
    dest_dir = "/data/local/tmp/omnicoder"
    _run(["adb", "shell", "mkdir", "-p", dest_dir])
    _run(["adb", "push", str(onnx_path), f"{dest_dir}/omnicoder_decode_step.onnx"])
    # Push optional sidecars if present (kv paging, kvq, kv retention, dynamic cache shim)
    for ext in (".kv_paging.json", ".kvq.json", ".kv_retention.json", ".dynamic_cache.json"):
        side = onnx_path.with_suffix(ext)
        if side.exists():
            _run(["adb", "push", str(side), f"{dest_dir}/{side.name}"])
    # Push tokenizer assets if present (vocab/merges) for on-device tokenization
    for name in ("tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        cand = onnx_path.parent / name
        if cand.exists():
            _run(["adb", "push", str(cand), f"{dest_dir}/{name}"])
    print(f"[android] Pushed artifacts to {dest_dir}")
    if not push_only:
        # Invoke existing helper to run a NNAPI device smoke with TPS threshold
        try:
            import sys as _sys
            cmd = [
                _sys.executable,
                "-m",
                "omnicoder.tools.android_adb_run",
                "--onnx",
                str(onnx_path),
                "--gen_tokens",
                "128",
                "--prompt_len",
                "128",
                "--tps_threshold",
                str(tps_threshold),
            ]
            _run(cmd)
        except Exception as e:
            print(f"[android] Skipped device smoke: {e}")


def _ios_copy(mlmodel_path: Path) -> None:
    # Copy into SampleApp and SampleConsole resources if present
    roots = [
        Path("src/omnicoder/inference/serverless_mobile/ios/SampleApp/Sources/Resources"),
        Path("src/omnicoder/inference/serverless_mobile/ios/SampleConsole/Sources/Resources"),
    ]
    copied_any = False
    for r in roots:
        try:
            r.mkdir(parents=True, exist_ok=True)
            dst = r / "omnicoder_decode_step.mlmodel"
            shutil.copy2(str(mlmodel_path), str(dst))
            print(f"[ios] Copied {mlmodel_path} -> {dst}")
            copied_any = True
        except Exception as e:
            print(f"[ios] Could not copy to {r}: {e}")
    if not copied_any:
        print("[ios] No iOS sample resources found. Ensure the Sample projects exist.")
    # Best-effort copy of tokenizer/sidecars to adjacent resources for the sample apps
    try:
        bundle_root = mlmodel_path.parent
        for name in ("tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                     "omnicoder_decode_step.kv_paging.json", "omnicoder_decode_step.kvq.json",
                     "omnicoder_decode_step.kv_retention.json", "omnicoder_decode_step.dynamic_cache.json"):
            src = bundle_root / name
            if src.exists():
                for r in roots:
                    dst = r / name
                    try:
                        shutil.copy2(str(src), str(dst))
                    except Exception:
                        pass
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Export artifacts to phone (Android/iOS)")
    ap.add_argument("--platform", type=str, choices=["android", "ios"], help="Target platform")
    ap.add_argument("--out_root", type=str, default=str(_default_out_root()), help="Release root with exported artifacts")
    ap.add_argument("--push_only", action="store_true", help="Android: only push files, skip NNAPI device run")
    ap.add_argument("--tps_threshold", type=float, default=float(os.getenv("OMNICODER_ANDROID_TPS_THRESHOLD", "15.0")))
    ap.add_argument("--onnx", type=str, default="", help="Override path to decode-step ONNX (Android)")
    ap.add_argument("--mlmodel", type=str, default="", help="Override path to Core ML decode-step model (iOS)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if args.platform == "android":
        onnx = Path(args.onnx) if args.onnx else (out_root / "text" / "omnicoder_decode_step.onnx")
        if not onnx.exists():
            print(f"[android] ONNX not found at {onnx}. Run press_play/build_mobile_release first or pass --onnx.")
            return
        _android_push(onnx, bool(args.push_only), float(args.tps_threshold))
        print("[android] Done.")
        return
    # iOS
    mlmodel = Path(args.mlmodel) if args.mlmodel else (out_root / "text" / "omnicoder_decode_step.mlmodel")
    if not mlmodel.exists():
        print(f"[ios] Core ML model not found at {mlmodel}. Use Core ML exporter to create it or pass --mlmodel.")
        return
    _ios_copy(mlmodel)
    print("[ios] Done.")


if __name__ == "__main__":
    main()



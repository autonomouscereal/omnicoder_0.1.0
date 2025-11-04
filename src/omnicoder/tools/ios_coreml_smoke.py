from __future__ import annotations

"""
Best-effort iOS Core ML device smoke helper.

This mirrors the Android ADB helper in spirit, but avoids hard dependencies on macOS
toolchains in environments where they are unavailable. It attempts to:

1) Locate a device bench JSON emitted by an iOS sample console (if present):
   weights/release/text/coreml_device_bench.json
   Expected shape: {"tokens_per_s": float}
2) Compare tokens/sec against a threshold and exit non-zero on failure.
3) On Darwin hosts, optionally run a user-provided shell script to execute the sample console.

Usage:
  python -m omnicoder.tools.ios_coreml_smoke \
    --mlmodel weights/release/text/omnicoder_decode_step.mlmodel \
    --tps_threshold 6.0 \
    --run_script inference/serverless_mobile/ios/SampleConsole/run_device_smoke.sh

Environment:
  OMNICODER_IOS_TPS_THRESHOLD (float) default 6.0
  OMNICODER_IOS_RUN_SCRIPT (optional path) to a script that runs the sample console on device
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="iOS Core ML device smoke (best-effort)")
    ap.add_argument("--mlmodel", type=str, default=str(Path("weights/release/text/omnicoder_decode_step.mlmodel")))
    ap.add_argument("--tps_threshold", type=float, default=float(os.getenv("OMNICODER_IOS_TPS_THRESHOLD", "6.0")))
    ap.add_argument("--run_script", type=str, default=os.getenv("OMNICODER_IOS_RUN_SCRIPT", ""))
    args = ap.parse_args()

    mlmodel = Path(args.mlmodel)
    bench_json = mlmodel.with_name("coreml_device_bench.json")

    # Best-effort execution of a user-provided script on Darwin
    if platform.system() == "Darwin" and args.run_script:
        try:
            print(f"[ios] running user script: {args.run_script}")
            subprocess.run(["bash", "-lc", args.run_script], check=False)
        except Exception as e:
            print(f"[ios] run_script failed/skipped: {e}")

    if bench_json.exists():
        try:
            data = json.loads(bench_json.read_text(encoding="utf-8"))
            tps = float(data.get("tokens_per_s", 0.0))
            print({"ios_coreml": {"tokens_per_s": tps, "threshold": float(args.tps_threshold)}})
            if tps < float(args.tps_threshold):
                print(f"[fail] iOS Core ML tokens/s {tps:.2f} < min {float(args.tps_threshold):.2f}")
                raise SystemExit(1)
            return
        except SystemExit:
            raise
        except Exception as e:
            print(f"[ios] parse bench json failed: {e}")
    else:
        print(f"[ios] {bench_json} not found; skipping threshold check")

    # No data available; treat as best-effort pass
    print("[ios] smoke skipped (no device log).")


if __name__ == "__main__":
    main()



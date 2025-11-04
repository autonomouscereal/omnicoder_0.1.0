from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Optional


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def wait_for_http(url: str, timeout_s: int = 20) -> bool:
    try:
        import urllib.request  # type: ignore
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(1)
        return False
    except Exception:
        return False


def android_host_reverse(port: int) -> int:
    # Check adb
    cp = _run(["adb", "devices"])
    if "device" not in cp.stdout:
        print("[mobile_sandbox] No Android device detected via adb.")
        return 1
    # Start local sandbox via docker compose (best-effort)
    _run(["docker", "compose", "up", "-d", "sandbox"])  # no check; ignore errors
    ok = wait_for_http(f"http://localhost:{port}/healthz", timeout_s=25)
    if not ok:
        print(f"[mobile_sandbox] Sandbox not healthy on localhost:{port}; continuing to forward anyway.")
    # Setup reverse so device apps can reach host sandbox at localhost:port
    _run(["adb", "reverse", f"tcp:{port}", f"tcp:{port}"])
    print(f"[mobile_sandbox] Android reverse set: device localhost:{port} -> host localhost:{port}")
    print("[mobile_sandbox] On device, point your app to http://localhost:%d" % port)
    return 0


def android_device_hosted(port: int) -> int:
    # Best-effort: requires Termux installed and configured to allow am start via termux-am
    print("[mobile_sandbox] Attempting device-hosted sandbox (requires Termux + Python on device)...")
    # Check adb device
    cp = _run(["adb", "devices"])
    if "device" not in cp.stdout:
        print("[mobile_sandbox] No Android device detected via adb.")
        return 1
    # Try launching Termux and run the server if termux-am is present
    # This uses 'am startservice' to run RunCommandService if available
    cmd = (
        "am startservice -n com.termux/com.termux.app.RunCommandService "
        "--es com.termux.RUN_COMMAND_PATH /data/data/com.termux/files/usr/bin/python "
        "--esa com.termux.RUN_COMMAND_ARGUMENTS -m,omnicoder.sfb.runtime.sandbox_server "
        f"--ez com.termux.RUN_COMMAND_BACKGROUND true --ei com.termux.RUN_COMMAND_TIMEOUT {max(60, port)}"
    )
    res = _run(["adb", "shell", cmd])
    if res.returncode != 0:
        print("[mobile_sandbox] Failed to start Termux RunCommandService. Install Termux and try again.")
        return 2
    print(f"[mobile_sandbox] Requested sandbox start on device port {port}. Ensure Termux has Python and required packages.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Mobile sandbox automation")
    p.add_argument("--platform", choices=["android"], default="android")
    p.add_argument("--mode", choices=["host_reverse", "device"], default="host_reverse")
    p.add_argument("--port", type=int, default=8088)
    args = p.parse_args(argv)
    if args.platform == "android":
        if args.mode == "host_reverse":
            return android_host_reverse(args.port)
        return android_device_hosted(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())



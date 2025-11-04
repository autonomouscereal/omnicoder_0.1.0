from __future__ import annotations

"""
SWE-bench sandbox runner (best-effort, lightweight):
- Reads metadata JSONL with fields {instance_id, repo, base, tests: [str, ...]}
- Clones the repo (shallow) and checks out the base commit
- Optionally applies a patch from --patch_dir/{instance_id}.patch
- Runs the provided test commands with timeouts
- Writes per-instance results JSONL and a summary JSON

Notes:
- This runner does not build language-specific virtualenvs; use --setup_cmd to provision deps.
- Heavy reproducibility (Dockerized sandbox, language-specific managers) can be layered on later.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List


def _log(msg: str) -> None:
    print(f"[swebench] {msg}", flush=True)


def _run(cmd: str, cwd: Path, timeout_s: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), shell=True, capture_output=True, text=True, timeout=timeout_s)
        dt = time.perf_counter() - t0
        return {
            "cmd": cmd,
            "rc": int(p.returncode),
            "dt": float(dt),
            "stdout": p.stdout[-10000:],  # keep tail to avoid huge outputs
            "stderr": p.stderr[-10000:],
        }
    except subprocess.TimeoutExpired as e:
        dt = time.perf_counter() - t0
        return {"cmd": cmd, "rc": 124, "dt": float(dt), "stdout": "", "stderr": f"timeout: {e}"}
    except Exception as e:
        dt = time.perf_counter() - t0
        return {"cmd": cmd, "rc": 255, "dt": float(dt), "stdout": "", "stderr": str(e)}


def _git(cmd: str, cwd: Path, timeout_s: int = 180) -> Dict[str, Any]:
    return _run(cmd, cwd, timeout_s)


def clone_and_checkout(repo: str, base: str, work_dir: Path) -> bool:
    try:
        if work_dir.exists():
            return True
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        # Heuristic: construct HTTPS URL if only org/repo provided
        url = repo
        if not (repo.startswith("http://") or repo.startswith("https://") or repo.endswith(".git")):
            url = f"https://github.com/{repo}.git"
        _log(f"cloning {url} -> {work_dir}")
        rc1 = _run(f"git clone --no-checkout {url} {shutil.quote(str(work_dir))}", cwd=work_dir.parent, timeout_s=300)
        if rc1.get("rc", 1) != 0:
            _log(f"clone failed: {rc1.get('stderr','')}")
            return False
        # Fetch and checkout specific commit
        rc2 = _git(f"git fetch --depth 1 origin {base}", cwd=work_dir)
        rc3 = _git(f"git checkout {base}", cwd=work_dir)
        ok = (rc2.get("rc", 1) == 0 and rc3.get("rc", 1) == 0)
        if not ok:
            _log(f"checkout failed: {rc2.get('stderr','')}{rc3.get('stderr','')}")
        return ok
    except Exception as e:
        _log(f"clone error: {e}")
        return False


def apply_patch_if_present(instance_id: str, work_dir: Path, patch_dir: Path | None) -> bool:
    if patch_dir is None:
        return True
    try:
        p = patch_dir / f"{instance_id}.patch"
        if not p.exists():
            return True
        _log(f"applying patch: {p}")
        rc = _git(f"git apply --whitespace=nowarn {shutil.quote(str(p))}", cwd=work_dir)
        return rc.get("rc", 1) == 0
    except Exception as e:
        _log(f"patch error: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SWE-bench instances in a lightweight sandbox")
    ap.add_argument("--meta", type=str, required=True, help="Path to swebench.jsonl metadata")
    ap.add_argument("--out", type=str, default="weights/swebench_results.jsonl")
    ap.add_argument("--summary", type=str, default="weights/swebench_summary.json")
    ap.add_argument("--work_root", type=str, default="weights/swebench_work")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900, help="Per-test command timeout (s)")
    ap.add_argument("--setup_cmd", type=str, default="", help="Optional shell command to set up environment (venv, pip install, etc.)")
    ap.add_argument("--patch_dir", type=str, default="", help="Optional directory with {instance_id}.patch files to apply")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(json.dumps({"error": f"meta not found: {meta_path}"}))
        sys.exit(1)

    work_root = Path(args.work_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch_dir = Path(args.patch_dir) if args.patch_dir else None

    results: List[Dict[str, Any]] = []
    total = 0
    passed = 0

    with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, ln in enumerate(f):
            if i >= int(args.limit):
                break
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            instance_id = str(rec.get("instance_id") or rec.get("id") or f"case_{i}")
            repo = str(rec.get("repo") or "").strip()
            base = str(rec.get("base") or rec.get("base_commit") or "").strip()
            tests = rec.get("tests") or rec.get("test_commands") or []
            tests = tests if isinstance(tests, list) else [tests]
            if not repo or not base or not tests:
                continue
            total += 1
            inst_dir = work_root / instance_id
            ok_clone = clone_and_checkout(repo, base, inst_dir)
            if not ok_clone:
                results.append({"instance_id": instance_id, "status": "clone_failed"})
                continue
            # Optional setup (dependency install)
            if args.setup_cmd:
                sres = _run(args.setup_cmd, cwd=inst_dir, timeout_s=max(60, int(args.timeout)))
                if sres.get("rc", 1) != 0:
                    results.append({"instance_id": instance_id, "status": "setup_failed", "setup": sres})
                    continue
            # Apply patch if provided
            if not apply_patch_if_present(instance_id, inst_dir, patch_dir):
                results.append({"instance_id": instance_id, "status": "patch_failed"})
                continue
            # Run tests
            cmd_results: List[Dict[str, Any]] = []
            for cmd in tests:
                cmd_results.append(_run(str(cmd), cwd=inst_dir, timeout_s=int(args.timeout)))
            all_ok = all(r.get("rc", 1) == 0 for r in cmd_results if isinstance(r, dict)) and len(cmd_results) > 0
            passed += int(all_ok)
            results.append({
                "instance_id": instance_id,
                "repo": repo,
                "base": base,
                "status": "passed" if all_ok else "failed",
                "commands": cmd_results,
            })
            # Stream to JSONL to allow incremental inspection
            try:
                with open(out_path, 'a', encoding='utf-8') as fo:
                    fo.write(json.dumps(results[-1]) + "\n")
            except Exception:
                pass

    summary = {"total": int(total), "passed": int(passed), "pass_rate": (float(passed) / float(total) if total else 0.0)}
    try:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    except Exception:
        pass
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



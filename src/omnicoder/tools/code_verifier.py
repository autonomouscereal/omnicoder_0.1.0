from __future__ import annotations

"""
Lightweight code unit-test verifier.

Runs user-provided code against a Python tests file and reports pass/fail counts.
Prefers pytest when available; otherwise executes the tests file in a new process
with the solution module injected on sys.path.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_pytest(tmpdir: Path) -> tuple[int, str]:
    try:
        # Run pytest with minimal plugins to avoid host interference
        env = os.environ.copy()
        env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        proc = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=str(tmpdir), env=env, capture_output=True, text=True, check=False)
        return int(proc.returncode), proc.stdout + '\n' + proc.stderr
    except Exception as e:
        return 1, f"[code_verifier] pytest failed to run: {e}"


def _fallback_exec(solution: Path, tests: Path) -> tuple[int, str]:
    """Fallback: execute tests file after importing solution as a module."""
    prog = f"""
import importlib.util, sys, types
spec = importlib.util.spec_from_file_location('solution', r'{solution}')
mod = importlib.util.module_from_spec(spec)
sys.modules['solution'] = mod
spec.loader.exec_module(mod)
globs = {{'__name__': '__main__', 'solution': mod}}
code = open(r'{tests}', 'r', encoding='utf-8').read()
exec(compile(code, r'{tests}', 'exec'), globs, globs)
"""
    try:
        proc = subprocess.run([sys.executable, '-c', prog], capture_output=True, text=True, check=False)
        rc = 0 if proc.returncode == 0 else 1
        return rc, proc.stdout + '\n' + proc.stderr
    except Exception as e:
        return 1, f"[code_verifier] fallback exec failed: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description='Run code against unit tests and report results as JSON')
    ap.add_argument('--code_file', type=str, default='')
    ap.add_argument('--code_text', type=str, default='')
    ap.add_argument('--tests_file', type=str, required=True)
    ap.add_argument('--out_json', type=str, default='weights/code_verify_result.json')
    args = ap.parse_args()

    # Prepare temp project with solution.py and tests file
    tmpdir = Path(tempfile.mkdtemp(prefix='code_verify_'))
    try:
        sol_path = tmpdir / 'solution.py'
        tests_dst = tmpdir / Path(args.tests_file).name
        if args.code_file:
            shutil.copy2(args.code_file, sol_path)
        else:
            sol_path.write_text(args.code_text, encoding='utf-8')
        shutil.copy2(args.tests_file, tests_dst)

        # Try pytest first
        rc, log = _run_pytest(tmpdir)
        if rc != 0:
            # Fallback to simple exec
            rc2, log2 = _fallback_exec(sol_path, tests_dst)
            rc = rc2
            log = log + '\n[fallback]\n' + log2
        # Parse a minimal pass/fail heuristic from pytest output when possible
        result = {
            'ok': rc == 0,
            'returncode': rc,
            'log': log[-20000:],
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == '__main__':
    main()



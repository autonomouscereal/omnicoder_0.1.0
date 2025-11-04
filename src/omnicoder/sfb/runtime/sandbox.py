from __future__ import annotations

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str


class CodeSandbox:
    def run(self, code: str, tests: str, timeout: int = 5) -> SandboxResult:  # pragma: no cover - interface
        raise NotImplementedError


class LocalIsolatedPythonSandbox(CodeSandbox):
    """Best-effort local isolated runner using python -I and tempfile; no net."""

    def run(self, code: str, tests: str, timeout: int = 5) -> SandboxResult:
        main = f"""# -*- coding: utf-8 -*-\n{code}\n\nif __name__ == '__main__':\n    {tests}\n    print('__SANDBOX_OK__')\n"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'main.py')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(main)
            try:
                # Optional coverage collection
                py = os.getenv('SANDBOX_PYTHON', 'python3')
                cov = (os.getenv('SANDBOX_COVERAGE', '0') == '1')
                args = [py]
                if cov:
                    args = [py, '-m', 'coverage', 'run', '--source=.', '--parallel-mode']
                args += ['-I', path]
                proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(1, int(timeout)), check=False, text=True)
                out = proc.stdout or ''
                err = proc.stderr or ''
                ok = ('__SANDBOX_OK__' in out) and (proc.returncode == 0)
                return SandboxResult(ok=ok, stdout=out, stderr=err)
            except subprocess.TimeoutExpired as e:
                return SandboxResult(ok=False, stdout=e.stdout or '', stderr='timeout')


class LocalIsolatedCppSandbox(CodeSandbox):
    """Local C++ runner using g++, executes tests in main() body.

    Requires a g++ toolchain inside the container; designed for tiny PAL harnesses.
    """

    def run(self, code: str, tests: str, timeout: int = 5) -> SandboxResult:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'main.cpp')
            exe = os.path.join(td, 'a.out')
            main = code + "\n\nint main(){\n" + tests + "\nreturn 0;\n}\n"
            with open(src, 'w', encoding='utf-8') as f:
                f.write(main)
            try:
                cxx = os.getenv('SANDBOX_CXX', 'g++')
                extra = os.getenv('SANDBOX_CXX_FLAGS', '').split()
                proc = subprocess.run([cxx, src, '-O2', '-std=c++17', '-fno-exceptions', '-fno-rtti', '-s', '-o', exe] + extra, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(1, int(timeout)), text=True)
                if proc.returncode != 0:
                    return SandboxResult(ok=False, stdout=proc.stdout or '', stderr=proc.stderr or '')
                run = subprocess.run([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(1, int(timeout)), text=True)
                ok = (run.returncode == 0)
                return SandboxResult(ok=ok, stdout=run.stdout or '', stderr=run.stderr or '')
            except subprocess.TimeoutExpired as e:
                return SandboxResult(ok=False, stdout=e.stdout or '', stderr='timeout')
            except Exception as e:
                return SandboxResult(ok=False, stdout='', stderr=str(e))


class RemoteHTTPSandbox(CodeSandbox):
    """Remote sandbox via simple HTTP JSON API."""

    def __init__(self, url: str) -> None:
        self.url = url

    def run(self, code: str, tests: str, timeout: int = 5) -> SandboxResult:
        try:
            import requests  # type: ignore
            payload = {"code": code, "tests": tests, "timeout": int(timeout)}
            r = requests.post(self.url.rstrip('/') + '/run', json=payload, timeout=max(3, int(timeout)+2))
            if r.status_code == 200:
                data = r.json()
                return SandboxResult(ok=bool(data.get('ok')), stdout=str(data.get('stdout','')), stderr=str(data.get('stderr','')))
            return SandboxResult(ok=False, stdout='', stderr=f'http {r.status_code}')
        except Exception as e:
            return SandboxResult(ok=False, stdout='', stderr=str(e))


def make_code_sandbox() -> CodeSandbox:
    url = os.getenv('SANDBOX_REMOTE_URL', '').strip()
    if url:
        return RemoteHTTPSandbox(url)
    # Choose language-specific local runner
    lang = os.getenv('SANDBOX_LANG', 'py').strip().lower()
    if lang in ('cpp', 'cc', 'c++'):
        return LocalIsolatedCppSandbox()
    # Memory/time limits can be emulated by shorter timeouts in callers; OS-level cgroups are out of scope here.
    # Fallback to local isolated Python runner
    return LocalIsolatedPythonSandbox()



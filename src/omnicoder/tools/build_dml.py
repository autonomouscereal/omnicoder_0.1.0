from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build omnicoder_dml_native (DirectML fused ops) on Windows")
    ap.add_argument('--config', type=str, default=os.getenv('CONFIG', 'Release'), choices=['Debug','Release'])
    ap.add_argument('--generator', type=str, default=os.getenv('CMAKE_GENERATOR', 'Visual Studio 17 2022'))
    ap.add_argument('--arch', type=str, default=os.getenv('CMAKE_ARCH', 'x64'))
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[5]
    src = root / 'src' / 'omnicoder' / 'modeling' / 'kernels'
    build = root / 'build_dml'
    build.mkdir(exist_ok=True)

    if args.clean and build.exists():
        print(f"[clean] removing {build}")
        shutil.rmtree(build)
        build.mkdir(exist_ok=True)

    cmake_cmd = [
        'cmake',
        '-S', str(src),
        '-B', str(build),
        '-G', args.generator,
        f'-A{args.arch}',
    ]
    rc = run(cmake_cmd)
    if rc != 0:
        raise SystemExit(rc)

    build_cmd = [
        'cmake', '--build', str(build), '--config', args.config,
    ]
    rc = run(build_cmd)
    if rc != 0:
        raise SystemExit(rc)

    print('[done] Built omnicoder_dml_native. Outputs:')
    for p in [build / args.config / 'omnicoder_dml_native.dll', build / args.config / 'omnicoder_dml_native.pyd']:
        if p.exists():
            print(' ', p)


if __name__ == '__main__':
    main()



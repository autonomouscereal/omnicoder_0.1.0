from __future__ import annotations

"""
Compile an ONNX decode-step model to an MLC/TVM artifact for on-device deployment.

Two paths:
  - tvmc (TVM CLI) → produces MLF .tar
  - mlc_llm CLI (if installed) → produces an MLC package
"""

import argparse
import shutil
import subprocess
from pathlib import Path
import sys


def _run(command: list[str]) -> bool:
    try:
        print("[mlc-compile] $", " ".join(command))
        subprocess.run(command, check=True)
        return True
    except Exception as e:
        print(f"[mlc-compile] Command failed: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile ONNX decode-step to TVM/MLC artifact")
    ap.add_argument("--onnx", required=True, help="Path to decode-step ONNX graph (with KV-cache IO)")
    ap.add_argument("--out_dir", default="weights/text/mlc", help="Output directory for compiled artifact")
    ap.add_argument("--tvm_target", default="llvm", help="TVM target: llvm|metal|vulkan|cuda")
    ap.add_argument("--opt_level", type=int, default=3, help="TVM optimization level (0-3)")
    ap.add_argument(
        "--input_shapes",
        default="input_ids:[1,1]",
        help="Comma-separated shape mappings like name:[dims]. Example: input_ids:[1,1]",
    )
    ap.add_argument("--use_mlc_cli", action="store_true", help="Attempt mlc_llm compile path instead of tvmc")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try mlc_llm CLI if requested
    if args.use_mlc_cli:
        try:
            cmd = [
                sys.executable, '-m', 'mlc_llm', 'compile',
                '--model', str(onnx_path),
                '--target', ('iphone' if 'metal' in str(args.tvm_target) else 'android'),
                '--output', str(out_dir),
                '--max_seq_len', '4096',
            ]
            print('[mlc-compile] Running:', ' '.join(cmd))
            subprocess.check_call(cmd)
            print(f"[mlc-compile] MLC package written to {out_dir}")
            return
        except Exception as e:
            print("[mlc-compile] MLC tooling not available or compile failed; falling back to tvmc.")
            print(f"Error: {e}")

    # Ensure tvmc is available
    tvmc = shutil.which("tvmc")
    if tvmc is None:
        print("[mlc-compile] tvmc not found. Install TVM with CLI support or add tvmc to PATH.")
        print("               Example: pip install apache-tvm --pre")
        return

    # tvmc compile command
    # Produces a .tar in Model Library Format (MLF). This can be loaded by TVM runtime.
    out_tar = out_dir / "omnicoder_decode_step.tvmc.tar"
    cmd = [
        tvmc,
        "compile",
        str(onnx_path),
        "--model-format",
        "onnx",
        "--target",
        args.tvm_target,
        "--opt-level",
        str(int(args.opt_level)),
        "--output",
        str(out_tar),
        "--input-shapes",
        args.input_shapes,
    ]
    ok = _run(cmd)
    if not ok:
        print("[mlc-compile] Compile failed. Verify your tvmc installation and target string.")
        return

    print(f"[mlc-compile] Compiled artifact: {out_tar}")
    print("[mlc-compile] Next steps:")
    print(" - Integrate the MLF .tar with a TVM runtime in your Android/iOS app, or")
    print(" - Use MLC-LLM packaging around the decode-step to build a runnable module.")


if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Compile ONNX decode-step to MLC artifact (if tooling available)")
    ap.add_argument('--onnx', type=str, required=True, help='Path to decode-step ONNX model')
    ap.add_argument('--out_dir', type=str, required=True, help='Output directory for MLC package')
    ap.add_argument('--target', type=str, default='iphone', choices=['iphone','android','metal','vulkan','cuda'], help='MLC target')
    ap.add_argument('--max_seq_len', type=int, default=4096)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to call mlc_llm CLI if installed
    try:
        cmd = [
            sys.executable, '-m', 'mlc_llm', 'compile',
            '--model', str(onnx_path),
            '--target', args.target,
            '--output', str(out_dir),
            '--max_seq_len', str(args.max_seq_len),
        ]
        print('Running:', ' '.join(cmd))
        subprocess.check_call(cmd)
        print(f"MLC package written to {out_dir}")
    except Exception as e:
        print("MLC tooling not available or compile failed.")
        print("Install per MLC-LLM docs and ensure mlc_llm python package is importable.")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

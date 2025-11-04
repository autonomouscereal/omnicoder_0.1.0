from __future__ import annotations

"""
Wrapper to export HuggingFace LLaMA/Mistral-style models to GGUF via llama.cpp.

This does NOT convert the custom OmniTransformer. GGUF targets llama.cpp-family
architectures. Use this to prepare baseline GGUFs for llama.cpp runtime tests
on Android/iOS (JNI on Android per README).

Requirements
 - A local clone of llama.cpp with Python convert script and quantize binaries
 - Python env with transformers installed for the convert step

Example:
  python -m omnicoder.export.gguf_export \
    --hf_model meta-llama/Llama-2-7b-hf \
    --llama_cpp_dir ../llama.cpp \
    --out weights/llama2_7b_f16.gguf

Optional quantization:
  python -m omnicoder.export.gguf_export \
    --hf_model meta-llama/Llama-2-7b-hf \
    --llama_cpp_dir ../llama.cpp \
    --out weights/llama2_7b_Q4_K_M.gguf \
    --quant Q4_K_M

Quant presets commonly available: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K_L,
Q3_K_M, Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, etc. Consult your llama.cpp build.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None) -> bool:
    try:
        print("[gguf-export] $", " ".join(cmd))
        subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except Exception as e:
        print(f"[gguf-export] failed: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Export HF LLaMA/Mistral models to GGUF via llama.cpp")
    ap.add_argument("--hf_model", required=True, help="HuggingFace model id or local path (LLaMA/Mistral-family)")
    ap.add_argument("--llama_cpp_dir", required=True, help="Path to llama.cpp repo root")
    ap.add_argument("--out", required=True, help="Output GGUF path")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16"], help="Base precision for GGUF before quant")
    ap.add_argument("--quant", default="", help="Optional quant preset (e.g., Q4_K_M)")
    args = ap.parse_args()

    llama_dir = Path(args.llama_cpp_dir)
    convert_py = llama_dir / "convert.py"
    if not convert_py.exists():
        print("[gguf-export] convert.py not found. Provide --llama_cpp_dir pointing at llama.cpp.")
        sys.exit(2)

    # Step 1: Convert HF weights to base GGUF (f16/f32)
    base_out = Path(args.out)
    if args.quant:
        # Write to a temp f16 first, then quantize
        base_out = base_out.with_suffix(".base.gguf")

    cmd_convert = [sys.executable, str(convert_py), "--outfile", str(base_out), "--outtype", args.dtype, args.hf_model]
    ok = run(cmd_convert, cwd=str(llama_dir))
    if not ok:
        print("[gguf-export] convert step failed")
        sys.exit(2)

    # Step 2: Optional quantize
    if args.quant:
        # Determine quantize binary name
        qbin = None
        for name in ("quantize", "llama-quantize", "quantize.exe", "llama-quantize.exe"):
            cand = llama_dir / name
            if cand.exists():
                qbin = cand
                break
        if qbin is None:
            print("[gguf-export] quantize binary not found in llama.cpp dir. Build it, then retry.")
            sys.exit(2)
        cmd_quant = [str(qbin), str(base_out), str(Path(args.out)), args.quant]
        ok = run(cmd_quant)
        if not ok:
            print("[gguf-export] quantize step failed")
            sys.exit(2)
        # Remove temp base file
        try:
            os.remove(base_out)
        except Exception:
            pass

    print(f"[gguf-export] Wrote GGUF: {args.out}")


if __name__ == "__main__":
    main()

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=False, default='', help='Path to PyTorch checkpoint (toy)')
    ap.add_argument('--out', type=str, required=False, default='weights/omnicoder_text.gguf', help='Output GGUF path')
    args = ap.parse_args()

    # Placeholder. Real export requires mapping our weights to a llama.cpp-compatible
    # graph or using a supported architecture. Documenting the path here:
    print('GGUF export requires a llama.cpp-supported architecture. For now, use ONNX/Core ML/ExecuTorch paths.')
    print(f'Intended output path: {os.path.abspath(args.out)}')


if __name__ == "__main__":
    main()

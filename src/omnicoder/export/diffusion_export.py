# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
Export Stable Diffusion image decoder pipelines for on-device use.

Supported targets (best-effort):
 - ONNX (preferred for Android ORT-mobile)
 - Core ML MLModel (experimental; requires coremltools>=7)
 - ExecuTorch (experimental; trace U-Net with torch.export)

Notes
 - For ONNX, we rely on Optimum + Diffusers exporters. Install:
     pip install optimum[onnxruntime] diffusers accelerate safetensors
 - For Core ML and ExecuTorch, coverage may depend on the exact pipeline version
   and operator support.
"""

import argparse
from pathlib import Path
from typing import Optional


def export_onnx(model_id: str | None, local_path: str | None, out_dir: Path, opset: int = 17, prefer_lite: bool = False) -> bool:
    """
    Export Stable Diffusion pipeline to ONNX via Optimum.

    Tries Python API first; falls back to invoking the Optimum CLI to be robust
    across version changes.
    """
    # Choose a model reference. If none provided and prefer_lite, fall back to a lightweight SD variant.
    model_ref = local_path if local_path else (model_id or ("stabilityai/sd-turbo" if prefer_lite else ""))
    if not model_ref:
        print("[ONNX] Provide --hf_id or --local_path")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    # Attempt Optimum Python API (new-style) first
    try:
        from argparse import Namespace  # type: ignore
        from optimum.exporters.onnx import main_export as optimum_export  # type: ignore

        print("[ONNX] Running Optimum export (API) ...")
        ns = Namespace(
            model=model_ref,
            task="stable-diffusion-pipeline",
            output=str(out_dir),
            opset=opset,
        )
        # Some versions expect a Namespace, others parse sys.argv; handle both
        try:
            optimum_export(ns)  # type: ignore[misc]
        except TypeError:
            # Fallback to argv-style if signature differs
            optimum_export([
                "--model", model_ref,
                "--task", "stable-diffusion-pipeline",
                str(out_dir),
                "--opset", str(opset),
            ])
        print(f"[ONNX] Exported SD pipeline to {out_dir}")
        return True
    except SystemExit as e:
        return int(getattr(e, "code", 1) or 0) == 0
    except Exception as e:
        print(f"[ONNX] Optimum API path failed: {e}")

    # Final fallback: shell out to the Optimum CLI if available
    try:
        import sys, subprocess  # type: ignore
        print("[ONNX] Running Optimum export (CLI) ...")
        cmd = [
            sys.executable, "-m", "optimum.exporters.onnx",
            "--model", model_ref,
            "--task", "stable-diffusion-pipeline",
            str(out_dir),
            "--opset", str(opset),
        ]
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            print(f"[ONNX] Exported SD pipeline to {out_dir}")
            return True
        print(f"[ONNX] CLI export failed with rc={rc}")
        return False
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        return False


def export_coreml(model_id: str | None, local_path: str | None, out_dir: Path) -> bool:
    try:
        import coremltools as ct  # type: ignore
    except Exception:
        print("[CoreML] coremltools>=7 required. pip install coremltools>=7")
        return False
    try:
        from diffusers import StableDiffusionPipeline  # type: ignore
        import torch
    except Exception:
        print("[CoreML] diffusers and torch required to trace components.")
        return False

    model_ref = local_path if local_path else (model_id or "")
    if not model_ref:
        print("[CoreML] Provide --hf_id or --local_path")
        return False

    print("[CoreML] Loading pipeline ...")
    pipe = StableDiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float16)
    pipe = pipe.to("cpu")

    # Export only VAE decoder as a practical first step; U-Net conversion often requires
    # custom ops. Image decoding is the heaviest component for output rendering.
    vae = pipe.vae.decoder.eval()
    dummy = torch.randn(1, 4, 64, 64, dtype=torch.float16)  # latent shape for 512x512
    traced = torch.jit.trace(vae, dummy, check_trace=False)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="latent", shape=dummy.shape, dtype=ct.float16)],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp = out_dir / "vae_decoder.mlmodel"
    mlmodel.save(str(mlp))
    print(f"[CoreML] Saved VAE decoder to {mlp}")
    return True


def export_executorch(model_id: str | None, local_path: str | None, out_dir: Path) -> bool:
    try:
        from torch.export import export as torch_export  # type: ignore
    except Exception:
        print("[ExecuTorch] torch>=2.3 required. Install PyTorch 2.3+.")
        return False
    try:
        from diffusers import StableDiffusionPipeline  # type: ignore
        import torch
    except Exception:
        print("[ExecuTorch] diffusers and torch required to trace components.")
        return False

    model_ref = local_path if local_path else (model_id or "")
    if not model_ref:
        print("[ExecuTorch] Provide --hf_id or --local_path")
        return False

    print("[ExecuTorch] Loading pipeline ...")
    pipe = StableDiffusionPipeline.from_pretrained(model_ref, torch_dtype=torch.float16)
    pipe = pipe.to("cpu")

    # Trace only VAE decoder as a starting point for mobile image rendering.
    vae = pipe.vae.decoder.eval()
    example = (torch.randn(1, 4, 64, 64, dtype=torch.float16),)
    try:
        prog = torch_export(vae, example)
    except Exception as e:
        print(f"[ExecuTorch] export failed: {e}")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    pte = out_dir / "vae_decoder.pte"
    with open(pte, "wb") as f:
        f.write(prog.to_pte())
    print(f"[ExecuTorch] Saved VAE decoder program to {pte}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Stable Diffusion components for mobile backends")
    ap.add_argument("--hf_id", type=str, default="", help="HF model id, e.g., runwayml/stable-diffusion-v1-5")
    ap.add_argument("--local_path", type=str, default="", help="Local path to diffusers pipeline")
    ap.add_argument("--out_dir", type=str, default="weights/sd_export", help="Output directory")
    ap.add_argument("--onnx", action="store_true", help="Export ONNX pipeline via Optimum")
    ap.add_argument("--coreml", action="store_true", help="Export Core ML VAE decoder (experimental)")
    ap.add_argument("--executorch", action="store_true", help="Export ExecuTorch VAE decoder (experimental)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ref_id = args.hf_id or None
    local = args.local_path or None

    ok_any = False
    if args.onnx:
        ok_any = export_onnx(ref_id, local, out_dir / "onnx", opset=args.opset) or ok_any
    if args.coreml:
        ok_any = export_coreml(ref_id, local, out_dir / "coreml") or ok_any
    if args.executorch:
        ok_any = export_executorch(ref_id, local, out_dir / "executorch") or ok_any

    if not ok_any:
        print("No exports completed. Ensure required toolchains are installed.")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Autofetch and export real, quantized backbones to weights/ for mobile.

This orchestrates:
 - Text: decode-step ONNX (+ optional dynamic int8) and optional ExecuTorch program
 - Image: Stable Diffusion pipeline export (ONNX/Core ML/ExecuTorch) if a model id or local path is provided
 - Vision: optional MobileViT/EfficientViT feature backbone export to ONNX
 - Video: record provided HF id/local path for a diffusers T2V/I2V pipeline (no export automation due to size)
 - Audio: verify EnCodec availability; optionally download a Piper ONNX TTS model

Notes
 - This script does not accept licenses on your behalf. Provide model ids/paths you are allowed to use.
 - For SD ONNX export, install: pip install -e .[gen]
 - For text Core ML decode-step export (iOS), install coremltools>=7 and run export/coreml_decode_export.py separately, or extend this script.
"""

import argparse
import json
import os
from pathlib import Path
import os
from typing import Optional

def _write_provider_maps(root_dir: Path) -> None:
    """Write provider quant maps (NNAPI/CoreML/DML) sidecars into a directory."""
    try:
        from .executorch_quant_maps import (
            write_nnapi_maps,
            write_coreml_maps,
            write_dml_maps,
        )  # type: ignore
        write_nnapi_maps(root_dir / "nnapi_quant_maps.json")
        write_coreml_maps(root_dir / "coreml_quant_maps.json")
        write_dml_maps(root_dir / "dml_quant_maps.json")
    except Exception:
        pass

def _copy_provider_profiles(root_dir: Path) -> None:
    """Copy example provider profiles JSONs into the target export directory."""
    try:
        repo_root = Path(__file__).resolve().parents[3]
        prof_dir = repo_root / "profiles"
        if prof_dir.exists():
            for name in ("pixel7_nnapi.json", "iphone15_coreml_ane.json", "windows_dml.json"):
                src = prof_dir / name
                if src.exists():
                    dst = root_dir / name
                    if not dst.exists():
                        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _export_text_decode_step(out_root: Path, preset: str, seq_len_budget: int, onnx_opset: int, quantize_onnx: bool, export_executorch: bool) -> dict:
    from .mobile_packager import _build_model, _export_onnx_decode_step, _quantize_onnx_dynamic, _export_executorch_decode_step, _write_summary
    text_dir = out_root / "text"
    _ensure_dir(text_dir)
    onnx_path = text_dir / "omnicoder_decode_step.onnx"

    # Prefer DynamicCache export when requested; fall back to standard KV-IO
    tried_dynamic = False
    try:
        use_dyn = os.getenv("OMNICODER_DYNAMIC_CACHE", "1") == "1"
        if use_dyn:
            import sys as _sys, subprocess as _sp
            cmd = [
                _sys.executable, "-m", "omnicoder.export.onnx_export",
                "--output", str(onnx_path),
                "--mobile_preset", str(preset),
                "--seq_len", "1",
                "--decode_step",
                "--opset", str(int(onnx_opset)),
                "--dynamic_cache",
            ]
            _sp.run(cmd, check=False)
            tried_dynamic = True
    except Exception:
        tried_dynamic = True

    if (not tried_dynamic) or (not onnx_path.exists()):
        # Fallback to explicit KV-IO path
        model = _build_model(preset, seq_len=1, multi_token=1)
        # Keep HRM active for export when requested via env (default-on if orchestrator sets it)
        try:
            if os.getenv("OMNICODER_EXPORT_HRM", "0") == "1" and hasattr(model, 'use_hrm'):
                model.use_hrm = True  # type: ignore[attr-defined]
        except Exception:
            pass
        _export_onnx_decode_step(model, onnx_path, opset=onnx_opset)

    # Try to fuse attention and pack Q/DQ for EP fusions
    try:
        from .onnx_fuse_attention import fuse_and_pack
        fused = text_dir / "omnicoder_decode_step_fused.onnx"
        if fuse_and_pack(onnx_path, fused):
            onnx_path = fused
    except Exception:
        pass

    out = {
        "onnx_decode_step": str(onnx_path),
        "onnx_int8": None,
        "executorch_pte": None,
        "summary": str(text_dir / "mobile_packager_summary.json"),
    }

    if quantize_onnx:
        q_path = text_dir / "omnicoder_decode_step_int8.onnx"
        if _quantize_onnx_dynamic(onnx_path, q_path):
            out["onnx_int8"] = str(q_path)

    # Optional per-op PTQ with provider-aware preset
    try:
        if os.getenv("OMNICODER_QUANTIZE_ONNX_PER_OP", "1") == "1":
            import sys as _sys, subprocess as _sp
            q_perop = text_dir / "omnicoder_decode_step_int8_perop.onnx"
            preset = os.getenv("OMNICODER_ONNX_PRESET", "generic")
            model_for_ptq = out["onnx_int8"] if out["onnx_int8"] else str(onnx_path)
            cmd = [
                _sys.executable, "-m", "omnicoder.export.onnx_quantize_per_op",
                "--model", model_for_ptq,
                "--out", str(q_perop),
                "--preset", preset,
                "--auto_exclude", "--per_channel",
            ]
            _sp.run(cmd, check=False)
            if q_perop.exists():
                out["onnx_int8_perop"] = str(q_perop)
                if out["onnx_int8"] is None:
                    out["onnx_int8"] = str(q_perop)
    except Exception:
        pass

    if export_executorch:
        et_path = text_dir / "omnicoder_decode_step.pte"
        # Use the same model instance for consistency if we had to build it above
        try:
            from .mobile_packager import _build_model
            model = _build_model(preset, seq_len=1, multi_token=1)
        except Exception:
            model = None  # type: ignore
        if model is not None:
            ok = _export_executorch_decode_step(model, et_path)
            out["executorch_pte"] = str(et_path) if ok or et_path.exists() else None

    _write_summary(text_dir / "mobile_packager_summary.json", preset, seq_len_budget)
    return out


def _export_sd(out_root: Path, hf_id: Optional[str], local_path: Optional[str], do_onnx: bool, do_coreml: bool, do_executorch: bool, onnx_opset: int) -> dict:
    sd_dir = out_root / "sd_export"
    _ensure_dir(sd_dir)
    out = {"onnx": None, "coreml": None, "executorch": None}
    if do_onnx:
        from .diffusion_export import export_onnx
        # Prefer a distilled/lite SD when no id/path provided to keep artifacts small
        prefer_lite = bool(not (hf_id or local_path))
        # Force-download hook when allowed
        try:
            if os.getenv("OMNICODER_FORCE_FETCH", "0") == "1" and not (hf_id or local_path):
                hf_id = os.getenv("OMNICODER_SD_MODEL", "runwayml/stable-diffusion-v1-5") or "runwayml/stable-diffusion-v1-5"
                prefer_lite = False
        except Exception:
            pass
        if export_onnx(hf_id, local_path, sd_dir / "onnx", opset=onnx_opset, prefer_lite=prefer_lite):
            out["onnx"] = str(sd_dir / "onnx")
            # Write provider maps next to ONNX artifacts
            _write_provider_maps(sd_dir / "onnx")
            _copy_provider_profiles(sd_dir / "onnx")
    if do_coreml:
        from .diffusion_export import export_coreml
        # Force-download hook
        try:
            if os.getenv("OMNICODER_FORCE_FETCH", "0") == "1" and not (hf_id or local_path):
                hf_id = os.getenv("OMNICODER_SD_MODEL", "runwayml/stable-diffusion-v1-5") or "runwayml/stable-diffusion-v1-5"
        except Exception:
            pass
        if export_coreml(hf_id, local_path, sd_dir / "coreml"):
            out["coreml"] = str(sd_dir / "coreml")
    if do_executorch:
        from .diffusion_export import export_executorch
        # Force-download hook
        try:
            if os.getenv("OMNICODER_FORCE_FETCH", "0") == "1" and not (hf_id or local_path):
                hf_id = os.getenv("OMNICODER_SD_MODEL", "runwayml/stable-diffusion-v1-5") or "runwayml/stable-diffusion-v1-5"
        except Exception:
            pass
        if export_executorch(hf_id, local_path, sd_dir / "executorch"):
            out["executorch"] = str(sd_dir / "executorch")
    return out


def _export_vqdec(out_root: Path, codebook_path: Optional[str], do_onnx: bool, do_coreml: bool, do_executorch: bool, hq: int, wq: int, onnx_opset: int) -> dict:
    vq_dir = out_root / "vqdec"
    _ensure_dir(vq_dir)
    out: dict = {"onnx": None, "coreml": None, "executorch": None, "codebook": codebook_path}
    if not codebook_path:
        return out
    if do_onnx:
        # Try direct function import first
        try:
            from omnicoder.export.onnx_export_vqdec import export_onnx as export_vqdec_onnx  # type: ignore
            onnx_target = vq_dir / "image_vq_decoder.onnx"
            ok = export_vqdec_onnx(codebook=str(codebook_path), onnx=str(onnx_target), hq=int(hq), wq=int(wq))  # type: ignore[arg-type]
            if ok:
                out["onnx"] = str(onnx_target)
            else:
                raise RuntimeError("export_vqdec_onnx returned False")
        except Exception:
            # robust fallback: shell out
            try:
                import sys, subprocess
                onnx_target = vq_dir / "image_vq_decoder.onnx"
                rc = subprocess.run([
                    sys.executable, "-m", "omnicoder.export.onnx_export_vqdec",
                    "--codebook", str(codebook_path),
                    "--onnx", str(onnx_target),
                    "--hq", str(hq), "--wq", str(wq)
                ], check=False).returncode
                if rc == 0:
                    out["onnx"] = str(onnx_target)
            except Exception:
                pass
    # Provider maps next to decoders (when any export requested)
    _write_provider_maps(vq_dir)
    _copy_provider_profiles(vq_dir)
    if do_coreml:
        try:
            import sys, subprocess
            rc = subprocess.run([
                sys.executable, "-m", "omnicoder.export.coreml_export_vqdec",
                "--codebook", str(codebook_path),
                "--out", str(vq_dir / "image_vq_decoder.mlmodel"),
                "--hq", str(hq), "--wq", str(wq)
            ], check=False).returncode
            if rc == 0:
                out["coreml"] = str(vq_dir / "image_vq_decoder.mlmodel")
        except Exception:
            pass
    if do_executorch:
        try:
            import sys, subprocess
            rc = subprocess.run([
                sys.executable, "-m", "omnicoder.export.executorch_export_vqdec",
                "--codebook", str(codebook_path),
                "--out", str(vq_dir / "image_vq_decoder.pte"),
                "--hq", str(hq), "--wq", str(wq)
            ], check=False).returncode
            if rc == 0:
                out["executorch"] = str(vq_dir / "image_vq_decoder.pte")
        except Exception:
            pass
    return out


def _export_vision_backbone(out_root: Path, backend: Optional[str], onnx_opset: int, do_coreml: bool = False, do_executorch: bool = False) -> dict:
    """Export a compact vision backbone (MobileViT/EfficientViT) to ONNX/Core ML/ExecuTorch if toolchains are available."""
    vis_dir = out_root / "vision"
    _ensure_dir(vis_dir)
    out: dict = {"onnx": None, "coreml": None, "executorch": None, "backend": None}
    try:
        import torch
        import timm  # type: ignore
    except Exception:
        # Force-install timm when force-fetch enabled and pip present
        try:
            if os.getenv("OMNICODER_FORCE_FETCH", "0") == "1":
                import sys as _sys, subprocess as _sp
                _sp.run([_sys.executable, "-m", "pip", "install", "timm"], check=False)
                import torch  # type: ignore
                import timm  # type: ignore
            else:
                return out
        except Exception:
            return out
    # Choose a small mobile backbone
    cand_names = []
    if backend and backend.strip():
        cand_names = [backend.strip()]
    else:
        cand_names = [
            "mobilevit_xs",
            "mobilevit_s",
            "efficientvit_lite0",
            "vit_tiny_patch16_224",
        ]
    model = None
    used_name = None
    for name in cand_names:
        try:
            if name in timm.list_models(pretrained=True):
                model = timm.create_model(name, pretrained=True)
                used_name = name
                break
        except Exception:
            continue
    if model is None:
        # Try DINOv2/SigLIP as heavier backbones if force-fetch is on
        if os.getenv("OMNICODER_FORCE_FETCH", "0") == "1":
            try:
                heavy = [
                    "vit_small_patch14_dinov2.lvd142m",
                    "vit_base_patch16_siglip_224",
                ]
                for name in heavy:
                    if name in timm.list_models(pretrained=True):
                        model = timm.create_model(name, pretrained=True)
                        used_name = name
                        break
            except Exception:
                pass
    if model is None:
        return out
    model.eval()
    # Try to prefer forward_features if available
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):  # type: ignore
            try:
                f = getattr(self.m, "forward_features", None)
                if callable(f):
                    return f(x)
            except Exception:
                pass
            return self.m(x)
    wrapper = Wrapper(model)
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = vis_dir / "mobile_vision.onnx"
    try:
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            input_names=["image"], output_names=["features"],
            dynamic_axes={"image": {0: "b", 2: "h", 3: "w"}},
            opset_version=onnx_opset,
        )
        out["onnx"] = str(onnx_path)
        out["backend"] = used_name
        _write_provider_maps(vis_dir)
        _copy_provider_profiles(vis_dir)
    except Exception:
        pass

    # Core ML (best-effort)
    if do_coreml and out.get("onnx") is not None:
        try:
            import coremltools as ct  # type: ignore
            mlp = vis_dir / "mobile_vision.mlmodel"
            traced = torch.jit.trace(wrapper, dummy)
            model = ct.convert(
                traced,
                inputs=[ct.ImageType(name="image", shape=dummy.shape)],
                convert_to="mlprogram",
            )
            model.save(str(mlp))
            out["coreml"] = str(mlp)
        except Exception:
            pass

    # ExecuTorch (best-effort)
    if do_executorch:
        try:
            from torch.export import export as torch_export  # type: ignore
            pte = vis_dir / "mobile_vision.pte"
            prog = torch_export(wrapper, (dummy,))
            with open(pte, "wb") as f:
                f.write(prog.to_pte())
            out["executorch"] = str(pte)
        except Exception:
            try:
                ts = torch.jit.trace(wrapper, dummy, check_trace=False)
                ts_path = vis_dir / "mobile_vision.pt"
                ts.save(str(ts_path))
                out["executorch"] = str(ts_path)
            except Exception:
                pass
    return out


def _download_piper_model(out_root: Path, piper_url: Optional[str]) -> Optional[str]:
    if not piper_url:
        return None
    try:
        import urllib.request
    except Exception:
        return None
    audio_dir = out_root / "audio"
    _ensure_dir(audio_dir)
    dest = audio_dir / Path(piper_url).name
    try:
        print("[Piper] Downloading: {}".format(piper_url))
        urllib.request.urlretrieve(piper_url, dest)
        return str(dest)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Autofetch and export backbones to weights/")
    ap.add_argument("--out_root", type=str, default="weights", help="Root weights directory")
    # Text
    ap.add_argument("--preset", default="mobile_4gb", choices=["mobile_4gb", "mobile_2gb"], help="Mobile text preset")
    ap.add_argument("--seq_len_budget", type=int, default=4096)
    ap.add_argument("--onnx_opset", type=int, default=17)
    ap.add_argument("--quantize_onnx", action="store_true")
    ap.add_argument("--export_executorch", action="store_true")
    # Vision (timm) export
    ap.add_argument("--vision_export_onnx", action="store_true")
    ap.add_argument("--vision_export_coreml", action="store_true")
    ap.add_argument("--vision_export_executorch", action="store_true")
    ap.add_argument("--vision_backend", type=str, default=os.getenv("OMNICODER_VISION_BACKEND", ""), help="timm model name override (e.g., mobilevit_xs)")
    # Image (Stable Diffusion)
    ap.add_argument("--sd_model", type=str, default="", help="HF id for SD, e.g., runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd_local_path", type=str, default="", help="Local SD pipeline path")
    ap.add_argument("--sd_export_onnx", action="store_true")
    ap.add_argument("--sd_export_coreml", action="store_true")
    ap.add_argument("--sd_export_executorch", action="store_true")
    # Image VQ decoder (indices->image) export
    ap.add_argument("--image_vq_codebook", type=str, default="", help="Path to Image VQ-VAE codebook blob (pt)")
    ap.add_argument("--vqdec_export_onnx", action="store_true")
    ap.add_argument("--vqdec_export_coreml", action="store_true")
    ap.add_argument("--vqdec_export_executorch", action="store_true")
    ap.add_argument("--vqdec_hq", type=int, default=14)
    ap.add_argument("--vqdec_wq", type=int, default=14)
    # Video (record reference only)
    ap.add_argument("--video_model", type=str, default="", help="HF id for text-to-video or image-to-video pipeline")
    ap.add_argument("--video_local_path", type=str, default="", help="Local path to video pipeline")
    # Audio (Piper TTS model optional download)
    ap.add_argument("--piper_url", type=str, default="", help="URL to a Piper .onnx model to download (optional)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    _ensure_dir(out_root)

    summary = {
        "text": {},
        "vision": {},
        "image": {},
        "vqdec": {},
        "video": {},
        "audio": {},
    }

    # Text exports
    print("[Text] Exporting decode-step graphs ...")
    summary["text"] = _export_text_decode_step(
        out_root,
        preset=args.preset,
        seq_len_budget=args.seq_len_budget,
        onnx_opset=args.onnx_opset,
        quantize_onnx=args.quantize_onnx,
        export_executorch=args.export_executorch,
    )

    # Vision backbone (optional)
    if args.vision_export_onnx or args.vision_export_coreml or args.vision_export_executorch:
        print("[Vision] Exporting compact vision backbone ...")
        summary["vision"] = _export_vision_backbone(
            out_root,
            backend=args.vision_backend or None,
            onnx_opset=args.onnx_opset,
            do_coreml=bool(args.vision_export_coreml),
            do_executorch=bool(args.vision_export_executorch),
        )
    else:
        summary["vision"] = {"note": "No vision export requested; pass --vision_export_onnx"}

    # Image exports (ONNX/Core ML/ExecuTorch)
    sd_hf = args.sd_model or None
    sd_local = args.sd_local_path or None
    if any([args.sd_export_onnx, args.sd_export_coreml, args.sd_export_executorch]):
        print("[Image] Exporting Stable Diffusion components ...")
        summary["image"] = _export_sd(
            out_root,
            hf_id=sd_hf,
            local_path=sd_local,
            do_onnx=args.sd_export_onnx,
            do_coreml=args.sd_export_coreml,
            do_executorch=args.sd_export_executorch,
            onnx_opset=args.onnx_opset,
        )
    else:
        summary["image"] = {"note": "No SD export requested; use --sd_export_... flags"}

    # Optional: export Core ML decode-step text model when coremltools available
    try:
        import coremltools as _ct  # type: ignore
        # Only attempt when explicitly requested via environment
        import os as _os
        if _os.getenv("OMNICODER_EXPORT_COREML_DECODE", "0") == "1":
            print("[Text] Exporting Core ML MLProgram decode-step ...")
            from .coreml_decode_export import main as _coreml_main  # type: ignore
            # Emulate CLI defaults by calling module main via subprocess to control args, or call directly if simple
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "omnicoder.export.coreml_decode_export", "--out", str(out_root / "text" / "omnicoder_decode_step.mlmodel"), "--preset", args.preset], check=False)
    except Exception:
        pass

    # Image VQ decoder exports
    if args.image_vq_codebook:
        summary["vqdec"] = _export_vqdec(
            out_root,
            codebook_path=args.image_vq_codebook,
            do_onnx=args.vqdec_export_onnx,
            do_coreml=args.vqdec_export_coreml,
            do_executorch=args.vqdec_export_executorch,
            hq=int(args.vqdec_hq), wq=int(args.vqdec_wq), onnx_opset=args.onnx_opset,
        )
    else:
        summary["vqdec"] = {"note": "Provide --image_vq_codebook to export a lightweight indices->image decoder"}

    # Video references
    if args.video_model or args.video_local_path:
        video_dir = out_root / "video"
        _ensure_dir(video_dir)
        (video_dir / "pipeline.txt").write_text(json.dumps({
            "hf_id": args.video_model or None,
            "local_path": args.video_local_path or None,
        }, indent=2))
        summary["video"] = {
            "hf_id": args.video_model or None,
            "local_path": args.video_local_path or None,
        }
    else:
        summary["video"] = {"note": "Provide --video_model or --video_local_path to record a pipeline"}

    # Audio downloads (optional Piper model)
    if args.piper_url:
        print("[Audio] Downloading Piper model ...")
        piper_path = _download_piper_model(out_root, args.piper_url)
        summary["audio"]["piper_model"] = piper_path
    else:
        summary["audio"]["note"] = "EnCodec and Coqui HiFi-GAN load weights internally; Piper model optional."

    # Write overall summary
    (out_root / "backbones_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")
    print(json.dumps(summary, indent=2))

    # Emit unified vocab sidecar mapping JSON (text/image/video/audio ranges)
    try:
        from omnicoder.modeling.multimodal.vocab_map import VocabLayout
        layout = VocabLayout()
        layout.validate()
        side = out_root / 'unified_vocab_map.json'
        side.write_text(json.dumps({
            'text_size': layout.text_size,
            'image_start': layout.image_start, 'image_size': layout.image_size,
            'video_start': layout.video_start, 'video_size': layout.video_size,
            'audio_start': layout.audio_start, 'audio_size': layout.audio_size,
        }, indent=2))
        print(f"[vocab] Wrote unified vocab map sidecar: {side}")
    except Exception as e:
        print(f"[vocab] Skipped unified vocab map: {e}")


if __name__ == "__main__":
    main()



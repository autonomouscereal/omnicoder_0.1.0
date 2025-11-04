from __future__ import annotations
from omnicoder.utils.env_registry import load_dotenv_best_effort
from omnicoder.utils.env_defaults import apply_core_defaults, apply_press_play_defaults, apply_profile

"""
Build a mobile-ready release folder in one command:

Pipeline:
 1) (Optional) Knowledge distillation: teacher (HF) -> student (OmniTransformer)
 2) Export text decode-step ONNX (+ optional int8) and ExecuTorch program
 3) Export Stable Diffusion (ONNX; optional Core ML/ExecuTorch)
 4) Record video pipeline reference (HF/local)
 5) (Optional) Download Piper ONNX model for on-device TTS
 6) Run auto-benchmarks and write a release manifest

This is an orchestrator; it shells out to the project CLIs for robustness.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print(" ", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, env=env)
        return int(proc.returncode)
    except Exception as e:
        print(f"[run] failed: {e}")
        return 1

def _load_dotenv(env_path: str = ".env") -> None:
    try:
        load_dotenv_best_effort((env_path,))
    except Exception:
        pass
def _env_checks() -> None:
    print("[checks] Environment validation...")
    # Python
    import sys as _sys
    print(f"  python={_sys.version.split()[0]}")
    # Torch
    try:
        import torch as _torch  # type: ignore
        print(f"  torch={_torch.__version__} cuda={_torch.cuda.is_available()} devices={_torch.cuda.device_count()}")
    except Exception as e:  # pragma: no cover
        print(f"  [warn] torch not importable: {e}")
    # ONNX Runtime
    try:
        import onnxruntime as _ort  # type: ignore
        print(f"  onnxruntime={_ort.__version__}")
    except Exception:
        print("  [hint] Install ONNX Runtime: pip install onnxruntime (or onnxruntime-gpu)")
    # Diffusers / Optimum
    try:
        import diffusers as _diff  # type: ignore
        print(f"  diffusers={_diff.__version__}")
    except Exception:
        print("  [hint] Install diffusers for image/video: pip install -e .[gen]")
    try:
        import optimum as _opt  # type: ignore
        print(f"  optimum={_opt.__version__}")
    except Exception:
        print("  [hint] Install optimum for ONNX exports: pip install optimum[onnxruntime]")


def main() -> None:
    _load_dotenv()
    try:
        apply_core_defaults(os.environ)  # type: ignore[arg-type]
        apply_press_play_defaults(os.environ)  # type: ignore[arg-type]
        apply_profile(os.environ, "quality")  # type: ignore[arg-type]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Build a mobile-ready release with exports and benchmarks")
    ap.add_argument("--out_root", type=str, default=os.getenv("OMNICODER_OUT_ROOT", "weights/release"))
    # KD
    ap.add_argument("--kd", action="store_true", help="Run knowledge distillation before export", default=(os.getenv("OMNICODER_KD", "0") == "1"))
    ap.add_argument("--kd_data", type=str, default=os.getenv("OMNICODER_KD_DATA", "."))
    ap.add_argument("--kd_steps", type=int, default=int(os.getenv("OMNICODER_KD_STEPS", "200")))
    ap.add_argument("--kd_seq_len", type=int, default=int(os.getenv("OMNICODER_KD_SEQ_LEN", "512")))
    ap.add_argument("--teacher", type=str, default=os.getenv("OMNICODER_KD_TEACHER", "microsoft/phi-2"))
    ap.add_argument("--teacher_device_map", type=str, default=os.getenv("OMNICODER_TEACHER_DEVICE_MAP", ""), help="Pass through to KD teacher device map (e.g., auto, balanced, or explicit mapping)")
    ap.add_argument("--teacher_dtype", type=str, default=os.getenv("OMNICODER_TEACHER_DTYPE", ""), help="Pass through to KD teacher dtype (e.g., fp16, bf16, auto)")
    ap.add_argument("--student_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"), choices=["mobile_4gb","mobile_2gb"]) 
    # Text export
    ap.add_argument("--quantize_onnx", action="store_true", default=(os.getenv("OMNICODER_QUANTIZE_ONNX", "0") == "1"))
    ap.add_argument("--act_scales", type=str, default=os.getenv("OMNICODER_ACT_SCALES", ""), help="Optional JSON with activation scales for Q/DQ guidance")
    ap.add_argument("--export_executorch", action="store_true", default=(os.getenv("OMNICODER_EXPORT_EXECUTORCH", "0") == "1"))
    ap.add_argument("--onnx_opset", type=int, default=int(os.getenv("OMNICODER_ONNX_OPSET", "17")))
    ap.add_argument("--seq_len_budget", type=int, default=int(os.getenv("OMNICODER_SEQ_LEN_BUDGET", "4096")))
    # Image export
    ap.add_argument("--sd_model", type=str, default=os.getenv("OMNICODER_SD_MODEL", "runwayml/stable-diffusion-v1-5"))
    ap.add_argument("--sd_local_path", type=str, default=os.getenv("OMNICODER_SD_LOCAL_PATH", ""))
    ap.add_argument("--sd_export_onnx", action="store_true", default=(os.getenv("OMNICODER_SD_EXPORT_ONNX", "1") == "1"))
    ap.add_argument("--sd_export_coreml", action="store_true", default=(os.getenv("OMNICODER_SD_EXPORT_COREML", "0") == "1"))
    ap.add_argument("--sd_export_executorch", action="store_true", default=(os.getenv("OMNICODER_SD_EXPORT_EXECUTORCH", "0") == "1"))
    # Image VQ decoder export
    ap.add_argument("--image_vq_codebook", type=str, default=os.getenv("OMNICODER_IMAGE_VQ_CODEBOOK", ""), help="Path to Image VQ-VAE codebook (pt)")
    ap.add_argument("--vqdec_export_onnx", action="store_true", default=(os.getenv("OMNICODER_VQDEC_EXPORT_ONNX", "0") == "1"))
    ap.add_argument("--vqdec_export_coreml", action="store_true", default=(os.getenv("OMNICODER_VQDEC_EXPORT_COREML", "0") == "1"))
    ap.add_argument("--vqdec_export_executorch", action="store_true", default=(os.getenv("OMNICODER_VQDEC_EXPORT_EXECUTORCH", "0") == "1"))
    ap.add_argument("--vqdec_hq", type=int, default=int(os.getenv("OMNICODER_VQDEC_HQ", "14")))
    ap.add_argument("--vqdec_wq", type=int, default=int(os.getenv("OMNICODER_VQDEC_WQ", "14")))
    # Video record
    ap.add_argument("--video_model", type=str, default=os.getenv("OMNICODER_VIDEO_MODEL", "stabilityai/text-to-video-sdxl"))
    ap.add_argument("--video_local_path", type=str, default=os.getenv("OMNICODER_VIDEO_LOCAL_PATH", ""))
    # Audio TTS model
    ap.add_argument("--piper_url", type=str, default=os.getenv("OMNICODER_PIPER_URL", ""))
    # Bench
    ap.add_argument("--bench_device", type=str, default=os.getenv("OMNICODER_BENCH_DEVICE", "cpu"))
    ap.add_argument("--bench_image_backend", type=str, default=os.getenv("OMNICODER_BENCH_IMAGE_BACKEND", "diffusers"), choices=["","diffusers","onnx"]) 
    ap.add_argument("--bench_out", type=str, default=os.getenv("OMNICODER_BENCH_OUT", "bench_summary.json"))
    # PTQ preset
    ap.add_argument("--onnx_preset", type=str, default=os.getenv("OMNICODER_ONNX_PRESET", "generic"), choices=["generic","nnapi","coreml","dml"], help="PTQ op coverage preset for ONNX int8")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _env_checks()
    # Prepare persistent caches and deterministic thread settings
    env: dict[str, str] = dict(os.environ)
    try:
        default_hf = str(Path("/models/hf")) if Path("/models").exists() else str(Path("models/hf"))
        env.setdefault("HF_HOME", env.get("HF_HOME", default_hf))
        # Rely on HF_HOME only; TRANSFORMERS_CACHE is deprecated
    except Exception:
        pass
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", env.get("MKL_NUM_THREADS", "1"))
    env.setdefault("TORCH_NUM_THREADS", env.get("TORCH_NUM_THREADS", "1"))

    # 1) KD (optional)
    ckpt = out_root / "omnicoder_student_kd.pt"
    if args.kd:
        print("[1] KD training ...")
        kd_cmd = [
            sys.executable, "-m", "omnicoder.training.distill",
            "--data", args.kd_data,
            "--seq_len", str(args.kd_seq_len),
            "--steps", str(args.kd_steps),
            "--student_mobile_preset", args.student_preset,
            "--teacher", args.teacher,
            "--out", str(ckpt),
            "--log_file", str(out_root / "kd_log.jsonl"),
            "--gradient_checkpointing", "--lora",
        ]
        if args.teacher_device_map.strip():
            kd_cmd += ["--teacher_device_map", args.teacher_device_map]
        if args.teacher_dtype.strip():
            kd_cmd += ["--teacher_dtype", args.teacher_dtype]
        if _run(kd_cmd, env=env) != 0:
            print("[KD] failed; continuing without a student checkpoint.")
    else:
        print("[1] KD skipped by flag --kd not set.")

    # 2–5) Exports via autofetcher
    print("[2] Exporting and staging backbones ...")
    auto_cmd = [
        sys.executable, "-m", "omnicoder.export.autofetch_backbones",
        "--out_root", str(out_root),
        "--preset", args.student_preset,
        "--seq_len_budget", str(args.seq_len_budget),
        "--onnx_opset", str(args.onnx_opset),
        # Enable full controller/value/verifier/bias by default for decode-step
        "--emit_controller",
        "--emit_value",
        "--sfb_bias_input",
    ]
    if args.quantize_onnx:
        auto_cmd.append("--quantize_onnx")
    if args.export_executorch:
        auto_cmd.append("--export_executorch")
    # Keep HRM active during export when requested via env (propagated to autofetch_backbones)
    if os.getenv("OMNICODER_EXPORT_HRM", "0") == "1":
        env["OMNICODER_EXPORT_HRM"] = "1"
    # SD
    auto_cmd += [
        "--sd_model", args.sd_model,
        "--sd_local_path", args.sd_local_path,
        "--sd_export_onnx",
    ]
    if args.sd_export_coreml:
        auto_cmd.append("--sd_export_coreml")
    if args.sd_export_executorch:
        auto_cmd.append("--sd_export_executorch")
    # Video
    if args.video_model:
        auto_cmd += ["--video_model", args.video_model]
    if args.video_local_path:
        auto_cmd += ["--video_local_path", args.video_local_path]
    # VQ decoder
    if args.image_vq_codebook:
        auto_cmd += [
            "--image_vq_codebook", args.image_vq_codebook,
        ]
        if args.vqdec_export_onnx:
            auto_cmd += ["--vqdec_export_onnx"]
        if args.vqdec_export_coreml:
            auto_cmd += ["--vqdec_export_coreml"]
        if args.vqdec_export_executorch:
            auto_cmd += ["--vqdec_export_executorch"]
        auto_cmd += ["--vqdec_hq", str(args.vqdec_hq), "--vqdec_wq", str(args.vqdec_wq)]
    # Audio
    if args.piper_url:
        auto_cmd += ["--piper_url", args.piper_url]
    # Optional: export compact vision backbone
    auto_cmd += ["--vision_export_onnx"]
    env_vis = os.getenv("OMNICODER_VISION_BACKEND", "").strip()
    if env_vis:
        auto_cmd += ["--vision_backend", env_vis]

    rc = _run(auto_cmd, env=env)
    # Export lightweight audio front-end by default for on-device PCM→features
    try:
        mel_out = out_root / "release" / "preprocessors" / "audio_conv.onnx"
        mel_out.parent.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, "-m", "omnicoder.export.mel_export",
            "--out_path", str(mel_out),
            "--opset", str(args.onnx_opset),
            "--n_mels", os.getenv("OMNICODER_MEL_DIM", "80"),
        ], env=env)
    except Exception as e:
        print(f"[warn] audio front-end export skipped: {e}")

    # 6) Benchmarks
    print("[3] Running auto-benchmarks ...")
    bench_cmd = [
        sys.executable, "-m", "omnicoder.eval.auto_benchmark",
        "--device", args.bench_device,
        "--seq_len", "128",
        "--gen_tokens", "128",
        "--preset", args.student_preset,
        "--out", str(out_root / args.bench_out),
    ]
    # If ONNX decode-step was exported to the default location, validate outputs and microbench providers
    onnx_default = out_root / "text" / "omnicoder_decode_step.onnx"
    if onnx_default.exists():
        bench_cmd += ["--validate_onnx", str(onnx_default), "--expect_mtp", "0", "--providers", "CPUExecutionProvider", "--prompt_len", "128"]
    if args.bench_image_backend:
        bench_cmd += ["--image_backend", args.bench_image_backend]
        if args.bench_image_backend == "diffusers":
            bench_cmd += ["--sd_model", args.sd_model]
        else:
            bench_cmd += ["--sd_local_path", str(out_root / "sd_export" / "onnx")]
    _run(bench_cmd, env=env)

    # ExecuTorch NNAPI quant maps sidecar
    try:
        from omnicoder.export.executorch_quant_maps import write_nnapi_maps, write_nnapi_node_maps  # type: ignore
        maps_path = out_root / 'text' / 'nnapi_quant_maps.json'
        write_nnapi_maps(maps_path)
        fused = out_root / 'text' / 'omnicoder_decode_step_fused.onnx'
        base = out_root / 'text' / 'omnicoder_decode_step.onnx'
        onnx_for_nodes = fused if fused.exists() else base
        write_nnapi_node_maps(onnx_for_nodes, out_root / 'text' / 'nnapi_nodes.json')
        print(f"[NNAPI] Wrote quant maps: {maps_path}")
    except Exception as e:
        print(f"[NNAPI] Skipped quant maps: {e}")

    # Release manifest
    manifest = {
        "out_root": str(out_root),
        "student_preset": args.student_preset,
        "student_ckpt": str(ckpt) if ckpt.exists() else None,
        "summary_files": [
            str(out_root / "backbones_summary.json"),
            str(out_root / args.bench_out),
        ],
    }
    (out_root / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("[DONE] Mobile release folder prepared.")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Press Play: one-button end-to-end builder/runner.

This orchestrates, in order:
 1) Environment checks
 2) Optional quick knowledge distillation (teacher -> student)
 3) Export text decode-step ONNX (+ optional per-op PTQ) and ExecuTorch program
 4) (Optional) Export Stable Diffusion backbones (ONNX/Core ML/ExecuTorch)
 5) (Optional) Record video model reference and download a Piper TTS model
 6) Auto-benchmarks (tokens/s, optional image latency) and manifest write
 7) Text inference smoke (native PyTorch and ONNX decode-step)

It reuses existing robust CLI modules to reduce failure modes.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from omnicoder.utils.resources import apply_thread_env_if_auto, audit_env
from omnicoder.utils.env_registry import load_dotenv_best_effort
from omnicoder.utils.env_defaults import apply_core_defaults, apply_run_env_defaults, apply_profile

def _run(cmd: list[str]) -> int:
    print(" ", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    except Exception as e:
        print(f"[run] failed: {e}")
        return 1


def _load_dotenv(env_path: str = ".env") -> None:
    # Backward-compatible shim; prefer centralized loader
    try:
        load_dotenv_best_effort((env_path,))
    except Exception:
        pass

def _env_checks() -> None:
    print("[checks] Environment validation...")
    print(f"  python={sys.version.split()[0]} os={platform.system()} arch={platform.machine()}")
    try:
        import torch  # type: ignore

        print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    except Exception as e:
        print(f"  [warn] torch not importable: {e}")
    try:
        import onnxruntime as ort  # type: ignore

        print(f"  onnxruntime={ort.__version__}")
    except Exception:
        print("  [hint] Install onnxruntime or onnxruntime-gpu for desktop ONNX tests")
    try:
        import diffusers as _diff  # type: ignore

        print(f"  diffusers={_diff.__version__}")
    except Exception:
        print("  [hint] Install diffusers for image/video export and demos: pip install -e .[gen]")


def main() -> None:
    ap = argparse.ArgumentParser(description="One-button end-to-end build, export, and smoke-run")
    # Load .env first so argparse defaults can read OMNICODER_* from environment
    load_dotenv_best_effort((".env", ".env.tuned"))
    # Apply centralized defaults and quality-first profile before parsing args
    try:
        apply_core_defaults(os.environ)  # type: ignore[arg-type]
        apply_run_env_defaults(os.environ)  # type: ignore[arg-type]
        apply_profile(os.environ, "quality")  # type: ignore[arg-type]
    except Exception:
        pass
    # Apply auto resource thread envs early if enabled
    try:
        apply_thread_env_if_auto()
    except Exception:
        pass
    # Audit unknown OMNICODER_* envs
    try:
        unknown = audit_env()
        if unknown:
            print({"env_unknown": unknown[:20], "more": max(0, len(unknown)-20)})
    except Exception:
        pass
    # Allow environment overrides for all key knobs (see .env.example)
    ap.add_argument("--out_root", type=str, default=os.getenv("OMNICODER_OUT_ROOT", "weights/release"))
    # Optional: run a full train → export → bench plan (AlphaGo-style self-play pipelines stitched)
    ap.add_argument("--train", action="store_true", default=(os.getenv("OMNICODER_PRESS_PLAY_TRAIN", "0") == "1"), help="Run training (lets_gooooo) before export/bench")
    ap.add_argument("--train_budget_hours", type=float, default=float(os.getenv("OMNICODER_TRAIN_BUDGET_HOURS", "1")))
    # KD (disabled in press-play; training happens only via lets-gooooo)
    ap.add_argument("--kd", action="store_true", default=False, help="[ignored] KD is disabled in press-play; use lets-gooooo for training")
    ap.add_argument("--kd_steps", type=int, default=int(os.getenv("OMNICODER_KD_STEPS", "200")))
    ap.add_argument("--kd_seq_len", type=int, default=int(os.getenv("OMNICODER_KD_SEQ_LEN", "512")))
    ap.add_argument("--teacher", type=str, default=os.getenv("OMNICODER_KD_TEACHER", "microsoft/phi-2"))
    ap.add_argument("--student_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"), choices=["mobile_4gb", "mobile_2gb"])
    # Text export/PTQ
    ap.add_argument("--quantize_onnx", action="store_true", default=(os.getenv("OMNICODER_QUANTIZE_ONNX", "0") == "1"))
    ap.add_argument("--quantize_onnx_per_op", action="store_true", default=(os.getenv("OMNICODER_QUANTIZE_ONNX_PER_OP", "0") == "1"))
    ap.add_argument("--onnx_preset", type=str, default=os.getenv("OMNICODER_ONNX_PRESET", "generic"), choices=["generic", "nnapi", "coreml", "dml"])
    ap.add_argument("--export_executorch", action="store_true", default=(os.getenv("OMNICODER_EXPORT_EXECUTORCH", "0") == "1"))
    ap.add_argument("--onnx_opset", type=int, default=int(os.getenv("OMNICODER_ONNX_OPSET", "17")))
    ap.add_argument("--seq_len_budget", type=int, default=int(os.getenv("OMNICODER_SEQ_LEN_BUDGET", "4096")))
    # Image/video/audio
    ap.add_argument("--sd_model", type=str, default=os.getenv("OMNICODER_SD_MODEL", "runwayml/stable-diffusion-v1-5"))
    ap.add_argument("--sd_export_onnx", action="store_true", default=(os.getenv("OMNICODER_SD_EXPORT_ONNX", "0") == "1"))
    ap.add_argument("--image_vq_codebook", type=str, default=os.getenv("OMNICODER_IMAGE_VQ_CODEBOOK", ""))
    ap.add_argument("--vision_backend", type=str, default=os.getenv("OMNICODER_VISION_BACKEND", ""))
    ap.add_argument("--image_provider_profile", type=str, default=os.getenv("OMNICODER_IMAGE_PROVIDER_PROFILE", ""))
    ap.add_argument("--vision_provider_profile", type=str, default=os.getenv("OMNICODER_VISION_PROVIDER_PROFILE", ""))
    ap.add_argument("--vqdec_provider_profile", type=str, default=os.getenv("OMNICODER_VQDEC_PROVIDER_PROFILE", ""))
    ap.add_argument("--video_model", type=str, default=os.getenv("OMNICODER_VIDEO_MODEL", "stabilityai/text-to-video-sdxl"))
    ap.add_argument("--piper_url", type=str, default=os.getenv("OMNICODER_PIPER_URL", ""))
    # Optional provider bench for vision/VQ decoders
    ap.add_argument("--vision_bench_providers", type=str, default=os.getenv("OMNICODER_VISION_BENCH_PROVIDERS", ""))
    ap.add_argument("--vision_bench_threshold", type=str, default=os.getenv("OMNICODER_VISION_TPS_THRESHOLDS", ""))
    ap.add_argument("--vqdec_bench_providers", type=str, default=os.getenv("OMNICODER_VQDEC_BENCH_PROVIDERS", ""))
    ap.add_argument("--vqdec_bench_threshold", type=str, default=os.getenv("OMNICODER_VQDEC_TPS_THRESHOLDS", ""))
    # Bench
    ap.add_argument("--bench_device", type=str, default=os.getenv("OMNICODER_BENCH_DEVICE", "cpu"))
    ap.add_argument("--bench_image_backend", type=str, default=os.getenv("OMNICODER_BENCH_IMAGE_BACKEND", ""), choices=["", "diffusers", "onnx"], help="Image bench backend; if empty, auto-detects ONNX export then falls back to diffusers if --sd_model set")
    # Optional: KVQ calibration step to emit weights/kvq_calibration.json for runners/exporters
    ap.add_argument("--run_kv_calibrate", action="store_true", default=(os.getenv("OMNICODER_RUN_KV_CALIBRATE", "0") == "1"))
    ap.add_argument("--kvq_scheme", type=str, default=os.getenv("OMNICODER_KVQ_SCHEME", "u8"), choices=["u8", "nf4"])
    ap.add_argument("--kvq_group", type=int, default=int(os.getenv("OMNICODER_KVQ_GROUP", "64")))
    ap.add_argument("--kvq_prompts_path", type=str, default=os.getenv("OMNICODER_KVQ_PROMPTS_PATH", ""))
    # Optional: Android ADB device smoke (push and run NNAPI decode-step)
    ap.add_argument("--android_adb", action="store_true", default=(os.getenv("OMNICODER_ANDROID_ADB", "0") == "1"))
    ap.add_argument("--android_tps_threshold", type=str, default=os.getenv("OMNICODER_ANDROID_TPS_THRESHOLD", "15.0"))
    # Optional: VQ-VAE autotrain hooks (train codebooks before bundling)
    ap.add_argument("--vq_image_dir", type=str, default=os.getenv("OMNICODER_VQ_IMAGE_DIR", ""), help="Folder of images to train image VQ-VAE codebook")
    ap.add_argument("--vq_image_steps", type=int, default=int(os.getenv("OMNICODER_VQ_IMAGE_STEPS", "5000")))
    ap.add_argument("--vq_video_list", type=str, default=os.getenv("OMNICODER_VQ_VIDEO_LIST", ""), help="Text file with video paths (one per line) for training video VQ-VAE")
    ap.add_argument("--vq_video_samples", type=int, default=int(os.getenv("OMNICODER_VQ_VIDEO_SAMPLES", "4096")))
    ap.add_argument("--vq_audio_dir", type=str, default=os.getenv("OMNICODER_VQ_AUDIO_DIR", ""), help="Folder of wavs to train audio VQ-VAE codebook")
    ap.add_argument("--vq_audio_steps", type=int, default=int(os.getenv("OMNICODER_VQ_AUDIO_STEPS", "20000")))
    # Hierarchical router groups
    ap.add_argument("--moe_group_sizes", type=str, default=os.getenv("OMNICODER_MOE_GROUP_SIZES", ""), help="Comma-separated expert group sizes for hierarchical router (e.g., 4,4)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _env_checks()

    # 0) Optional: training orchestration (GRPO/PPO + ToT+MCTS) before export/bench
    if bool(args.train):
        try:
            print("[0] Running training plan (lets_gooooo) ...")
            _run([
                sys.executable, "-m", "omnicoder.tools.lets_gooooo",
                "--budget_hours", str(args.train_budget_hours),
                "--device", os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"),
                "--out_root", str(Path(args.out_root).parent),
            ])
        except Exception as e:
            print(f"  [warn] training plan skipped: {e}")

    # Persist HF caches and limit threads for reproducibility in subprocesses
    try:
        default_hf = str(Path("/models/hf")) if Path("/models").exists() else str(Path("models/hf"))
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", default_hf))
        # Rely on HF_HOME only; TRANSFORMERS_CACHE is deprecated
    except Exception as e:
        print(f"  [warn] could not set HF caches: {e}")
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
    os.environ.setdefault("TORCH_NUM_THREADS", os.environ.get("TORCH_NUM_THREADS", "1"))

    # 1) Build and export via existing one-shot autofetch tool (includes KD when requested)
    cmd = [
        sys.executable,
        "-m",
        "omnicoder.tools.build_mobile_release",
        "--out_root",
        str(out_root),
        "--onnx_opset",
        str(args.onnx_opset),
        "--seq_len_budget",
        str(args.seq_len_budget),
        "--onnx_preset",
        args.onnx_preset,
        "--student_preset",
        args.student_preset,
    ]
    # Training is disabled in press-play; enforce no-KD regardless of flags
    if bool(args.kd):
        print("[info] KD is disabled in press-play; use lets-gooooo for one-button training")
    run_kd = False
    if run_kd:
        cmd.append("--kd")
        cmd += ["--kd_steps", str(args.kd_steps), "--kd_seq_len", str(args.kd_seq_len), "--teacher", args.teacher]
        if args.moe_group_sizes.strip():
            cmd += ["--moe_group_sizes", args.moe_group_sizes]
        # Router curriculum/env parity: pass through when present
        rc = os.environ.get("OMNICODER_ROUTER_CURRICULUM", "")
        rp = os.environ.get("OMNICODER_ROUTER_PHASE_STEPS", "")
        if rc:
            cmd += ["--router_kind", "auto"]
        if rp and os.environ.get("OMNICODER_ROUTER_SINKHORN_ITERS", ""):
            cmd += ["--router_sinkhorn_iters", os.environ["OMNICODER_ROUTER_SINKHORN_ITERS"]]
        for env_key, flag in [
            ("OMNICODER_AUX_LB_COEF", "--aux_lb_coef"),
            ("OMNICODER_AUX_IMPORTANCE_COEF", "--aux_importance_coef"),
            ("OMNICODER_AUX_LOAD_COEF", "--aux_load_coef"),
            ("OMNICODER_AUX_ZLOSS_COEF", "--aux_zloss_coef"),
        ]:
            if env_key in os.environ:
                cmd += [flag, os.environ[env_key]]
    if args.quantize_onnx:
        cmd.append("--quantize_onnx")
    # build_mobile_release no longer accepts --quantize_onnx_per_op; retain per-op PTQ via onnx_preset and packager fusions
    if args.export_executorch:
        cmd.append("--export_executorch")
    # Add image/video/audio knobs
    if args.sd_model:
        cmd += ["--sd_model", args.sd_model, "--sd_export_onnx"] if args.sd_export_onnx else ["--sd_model", args.sd_model]
    if args.video_model:
        cmd += ["--video_model", args.video_model]
    if args.piper_url:
        cmd += ["--piper_url", args.piper_url]

    print("[1] Building and exporting (this may take a while)...")
    _run(cmd)

    # 1d) Ensure ONNX export-all wrappers exist for downstream runners (end-to-end + standalone)
    try:
        out_all = Path(args.out_root) / "export_all"
        out_all.mkdir(parents=True, exist_ok=True)
        print("[1d] Exporting ONNX (export-all) ...")
        _run([
            sys.executable, "-m", "omnicoder.export.onnx_export",
            "--mobile_preset", args.student_preset,
            "--export_all",
            "--output_dir", str(out_all),
        ])
    except Exception as e:
        print(f"  [warn] export-all skipped: {e}")

    # Unified inference smoke: demonstrate single endpoint producing text by default
    try:
        print("[1c] Unified inference smoke ...")
        from omnicoder.inference.unified import run_unified  # type: ignore
        res = run_unified({"text": "Say 'ready'.", "max_new_tokens": 8}, preset=args.student_preset, device=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"), ckpt=os.getenv("OMNICODER_CKPT", ""))
        print("unified:", {k: (v if isinstance(v, str) else str(v)) for k, v in res.items() if k != "text"})
        if isinstance(res.get("text"), str):
            print("unified.text:", res["text"][:80])
    except Exception as _eus:
        print(f"[warn] unified inference smoke skipped: {_eus}")

    # Microbench and KV prefetch sidecar
    try:
        print("[1b] Running MLA micro-benchmark ...")
        _run([
            sys.executable, "-m", "omnicoder.tools.mla_microbench",
            "--preset", args.student_preset,
            "--device", "cpu",
            "--steps", "256",
            "--out", str(Path(args.out_root) / "mla_microbench.json"),
        ])
    except Exception as e:
        print(f"  [warn] mla_microbench skipped: {e}")
    try:
        print("[1c] Writing KV prefetch predictor sidecar ...")
        decode_onnx = Path(args.out_root) / "text" / "omnicoder_decode_step.onnx"
        if decode_onnx.exists():
            _run([
                sys.executable, "-m", "omnicoder.tools.kv_prefetch_write",
                "--out", str(Path(args.out_root) / "text" / "omnicoder_decode_step.kv_prefetch.json"),
                "--keep_pages", os.getenv("OMNICODER_KV_PREFETCH_KEEP", "2"),
            ])
    except Exception as e:
        print(f"  [warn] kv_prefetch_write skipped: {e}")

    # 1a) Optional: train VQ-VAE codebooks (image/video/audio) if directories/lists provided
    image_codebook_path = None
    try:
        if args.vq_image_dir:
            print("[1a] Training Image VQ-VAE codebook ...")
            img_out = Path(args.out_root) / "vqdec" / "image_vq_codebook.pt"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            rc = _run([
                sys.executable, "-m", "omnicoder.training.vq_train",
                "--data", args.vq_image_dir,
                "--image_size", "224", "--patch", "16", "--emb_dim", "192",
                "--codebook_size", "8192", "--steps", str(args.vq_image_steps),
                "--batch", "32", "--out", str(img_out),
            ])
            if rc == 0:
                image_codebook_path = str(img_out)
    except Exception as e:
        print(f"  [warn] image VQ-VAE training skipped: {e}")

    try:
        if args.vq_video_list:
            print("[1a] Training Video VQ-VAE codebook ...")
            vid_out = Path(args.out_root) / "vqdec" / "video_vq_codebook.pt"
            vid_out.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable, "-m", "omnicoder.training.video_vq_train",
                "--videos", args.vq_video_list, "--resize", "224", "--patch", "16",
                "--emb_dim", "192", "--codebook_size", "8192", "--frames_per_video", "16",
                "--samples", str(args.vq_video_samples), "--out", str(vid_out),
            ])
    except Exception as e:
        print(f"  [warn] video VQ-VAE training skipped: {e}")

    try:
        if args.vq_audio_dir:
            print("[1a] Training Audio VQ-VAE codebook ...")
            aud_out = Path(args.out_root) / "vqdec" / "audio_vq_codebook.pt"
            aud_out.parent.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable, "-m", "omnicoder.training.audio_vq_train",
                "--data", args.vq_audio_dir, "--steps", str(args.vq_audio_steps),
                "--batch", "4", "--segment", "32768", "--codebook_size", "2048",
                "--code_dim", "128", "--out", str(aud_out),
            ])
    except Exception as e:
        print(f"  [warn] audio VQ-VAE training skipped: {e}")

    # 1b) Bundle multimodal decoders via mobile_packager (best-effort)
    try:
        pack_cmd = [
            sys.executable, "-m", "omnicoder.export.mobile_packager",
            "--preset", args.student_preset,
            "--out_dir", str(out_root / "text"),
            "--seq_len_budget", str(args.seq_len_budget),
            "--onnx_preset", args.onnx_preset,
            "--onnx_provider_hint", os.getenv("OMNICODER_ORT_PROVIDER", "CPUExecutionProvider"),
        ]
        if args.quantize_onnx:
            pack_cmd.append("--quantize_onnx")
        if args.export_executorch:
            pack_cmd.append("--export_executorch")
        # Vision
        if args.vision_backend:
            pack_cmd += ["--with_vision", "--vision_backend", args.vision_backend]
            if args.vision_provider_profile:
                pack_cmd += ["--vision_provider_profile", args.vision_provider_profile]
        # SD
        if args.sd_model or args.sd_export_onnx:
            pack_cmd += ["--with_sd", "--sd_model", args.sd_model]
            if args.image_provider_profile:
                pack_cmd += ["--image_provider_profile", args.image_provider_profile]
        # VQDec
        vq_codebook = args.image_vq_codebook or image_codebook_path or ""
        if vq_codebook:
            pack_cmd += ["--with_vqdec", "--image_vq_codebook", vq_codebook]
            if args.vqdec_provider_profile:
                pack_cmd += ["--vqdec_provider_profile", args.vqdec_provider_profile]
        # Piper
        if args.piper_url:
            pack_cmd += ["--with_piper", "--piper_url", args.piper_url]
        # Video reference
        if args.video_model:
            pack_cmd += ["--video_model", args.video_model]
        print("[1b] Bundling multimodal decoders ...")
        _run(pack_cmd)
    except Exception as e:
        print(f"  [warn] Bundling step skipped: {e}")

    # 2) Auto-bench summary
    bench_cmd = [
        sys.executable,
        "-m",
        "omnicoder.eval.auto_benchmark",
        "--device",
        args.bench_device,
        "--seq_len",
        "128",
        "--gen_tokens",
        "128",
        "--preset",
        args.student_preset,
        "--out",
        str(out_root / "bench_summary.json"),
    ]
    # Auto-detect image bench backend if not provided
    bench_backend = args.bench_image_backend
    if not bench_backend:
        try:
            sd_onnx_dir = out_root / "sd_export" / "onnx"
            if sd_onnx_dir.exists():
                bench_backend = "onnx"
            elif args.sd_model:
                bench_backend = "diffusers"
        except Exception:
            bench_backend = ""
    if bench_backend:
        bench_cmd += ["--image_backend", bench_backend]
        if bench_backend == "diffusers" and args.sd_model:
            bench_cmd += ["--sd_model", args.sd_model]
        if bench_backend == "onnx":
            # Prefer the just-exported ONNX folder
            bench_cmd += ["--sd_local_path", str(out_root / "sd_export" / "onnx")]
    bench_json = out_root / "bench_summary.json"
    if bench_json.exists():
        print(f"[2] Skipping auto-benchmarks (already exists: {bench_json})")
    else:
        print("[2] Running auto-benchmarks...")
        _run(bench_cmd)

    # 2b) Optional KV calibration (produces weights/kvq_calibration.json)
    if args.run_kv_calibrate:
        kv_cmd = [
            sys.executable,
            "-m",
            "omnicoder.tools.kv_calibrate",
            "--mobile_preset",
            args.student_preset,
            "--kvq",
            args.kvq_scheme,
            "--group",
            str(args.kvq_group),
            "--out",
            str(Path("weights") / "kvq_calibration.json"),
        ]
        if args.kvq_prompts_path:
            kv_cmd += ["--prompts", args.kvq_prompts_path]
        print("[2b] KV-cache quant calibration ...")
        _run(kv_cmd)

    # 3) Text inference smoke (native + ONNX decode-step)
    print("[3] Text inference (native PyTorch) ...")
    try:
        from omnicoder.inference.generate import build_mobile_model_by_name, generate  # type: ignore
        from omnicoder.inference.gen_config import GenRuntimeConfig  # type: ignore
        from omnicoder.training.simple_tokenizer import get_text_tokenizer  # type: ignore
        import torch  # type: ignore
        model = build_mobile_model_by_name(args.student_preset)
        tok = get_text_tokenizer(prefer_hf=True)
        ids = tok.encode("Hello from Press Play")
        if not isinstance(ids, list) or not ids:
            ids = [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        rc = GenRuntimeConfig(
            draft_ckpt_path=os.environ.get('OMNICODER_DRAFT_CKPT', '').strip() or None,
            draft_preset_name=os.environ.get('OMNICODER_DRAFT_PRESET', 'mobile_2gb').strip() or 'mobile_2gb',
            use_onnx_draft=False,
            onnx_decode_path=None,
            ort_provider=os.environ.get('OMNICODER_ORT_PROVIDER', 'auto'),
            super_verbose=False,
        )
        out_ids = generate(model, input_ids, max_new_tokens=32, runtime_config=rc)
        print(tok.decode(out_ids[0].tolist()))
    except Exception as e:
        print(f"  [warn] native generate smoke failed: {e}")

    onnx_model = out_root / "text" / "omnicoder_decode_step.onnx"
    if onnx_model.exists():
        # Prefer provider hint from environment on all platforms
        provider = os.environ.get("OMNICODER_ORT_PROVIDER", "CPUExecutionProvider")
        print("[4] Text inference (ONNX decode-step) ...")
        _run([
            sys.executable,
            "-m",
            "omnicoder.inference.runtimes.onnx_decode_generate",
            "--model",
            str(onnx_model),
            "--prompt",
            "Hello from ONNX!",
            "--max_new_tokens",
            "32",
            "--provider",
            provider,
        ])

        # Optional: provider TPS canaries on text decode-step
        text_providers = os.environ.get("OMNICODER_TEXT_BENCH_PROVIDERS", "").strip()
        text_thresholds = os.environ.get("OMNICODER_TEXT_TPS_THRESHOLDS", "").strip()
        if text_providers:
            print("[4a] Provider bench (text decode-step) ...")
            kv_sidecar = ""
            try:
                # Prefer a KV paging sidecar if present
                side = onnx_model.with_suffix('.kv_paging.json')
                if side.exists():
                    kv_sidecar = str(side)
            except Exception:
                pass
            cmd = [
                sys.executable, "-m", "omnicoder.inference.runtimes.provider_bench",
                "--model", str(onnx_model),
                "--providers", *text_providers.split(),
                "--prompt_len", "128", "--gen_tokens", "128",
                "--check_fusions", "--require_attention", "--canary_tokens_per_s",
                "--out_json", str(out_root / "text" / "provider_bench.json"),
                "--threshold", text_thresholds,
                *( ["--kv_paging_sidecar", kv_sidecar] if kv_sidecar else [] ),
            ]
            try:
                from pathlib import Path as _P
                thr = _P("profiles") / "provider_thresholds.json"
                if thr.exists():
                    cmd += ["--threshold_json", str(thr)]
            except Exception as e:
                print(f"  [warn] could not add threshold_json: {e}")
            _run(cmd)

        # Optional: emit long-context variants and smoke CPU
        if os.environ.get("OMNICODER_LONGCTX_EXPORT", "0") == "1":
            print("[4c] Long-context decode-step variants (32k/128k) ...")
            longctx_cmd = [
                sys.executable, "-m", "omnicoder.export.onnx_export",
                "--output", str(out_root / "text" / "omnicoder_decode_step.onnx"),
                "--seq_len", "1", "--mobile_preset", args.student_preset, "--decode_step",
                "--emit_longctx_variants", "--yarn",
            ]
            _run(longctx_cmd)
            ctx32k = out_root / "text" / "omnicoder_decode_step_ctx32k.onnx"
            if ctx32k.exists():
                _run([
                    sys.executable, "-m", "omnicoder.inference.runtimes.onnx_decode_generate",
                    "--model", str(ctx32k),
                    "--provider", "CPUExecutionProvider",
                    "--prompt", "Hello longctx", "--max_new_tokens", "16",
                ])

        # Optional: Android ADB NNAPI device smoke
        if args.android_adb:
            print("[4b] Android ADB NNAPI device smoke ...")
            _run([
                sys.executable,
                "-m",
                "omnicoder.tools.android_adb_run",
                "--onnx",
                str(onnx_model),
                "--gen_tokens",
                "128",
                "--prompt_len",
                "128",
                "--tps_threshold",
                str(args.android_tps_threshold),
            ])

    # 4) Optional provider bench for vision/VQ (based on packager manifest)
    bundle_manifest_path = out_root / "mobile_packager_manifest.json"
    bundle = None
    if bundle_manifest_path.exists():
        try:
            bundle = json.loads(bundle_manifest_path.read_text(encoding='utf-8'))
            # Vision bench
            if isinstance(bundle, dict) and isinstance(bundle.get("vision"), dict):
                vis_onnx = bundle["vision"].get("onnx")
                if vis_onnx and args.vision_bench_providers.strip():
                    provs = args.vision_bench_providers.split()
                    print("[4] Provider bench (vision) ...")
                    cmd = [
                        sys.executable, "-m", "omnicoder.inference.runtimes.provider_bench",
                        "--model", str(vis_onnx),
                        "--providers", *provs,
                        "--prompt_len", "128", "--gen_tokens", "128",
                        "--check_fusions", "--canary_tokens_per_s", "--out_json", str(out_root / "vision" / "provider_bench.json"),
                        "--threshold", args.vision_bench_threshold or "",
                    ]
                    try:
                        from pathlib import Path as _P
                        thr = _P("profiles") / "provider_thresholds.json"
                        if thr.exists():
                            cmd += ["--threshold_json", str(thr)]
                    except Exception as e:
                        print(f"  [warn] could not add vqdec threshold_json: {e}")
                    _run(cmd)
            # VQDec bench
            if isinstance(bundle, dict) and isinstance(bundle.get("vqdec"), dict):
                vq_onnx = bundle["vqdec"].get("onnx")
                if vq_onnx and args.vqdec_bench_providers.strip():
                    provs = args.vqdec_bench_providers.split()
                    print("[4] Provider bench (vqdec) ...")
                    cmd = [
                        sys.executable, "-m", "omnicoder.inference.runtimes.provider_bench",
                        "--model", str(vq_onnx),
                        "--providers", *provs,
                        "--prompt_len", "128", "--gen_tokens", "128",
                        "--check_fusions", "--canary_tokens_per_s", "--out_json", str(out_root / "vqdec" / "provider_bench.json"),
                        "--threshold", args.vqdec_bench_threshold or "",
                    ]
                    try:
                        from pathlib import Path as _P
                        thr = _P("profiles") / "provider_thresholds.json"
                        if thr.exists():
                            cmd += ["--threshold_json", str(thr)]
                    except Exception:
                        pass
                    _run(cmd)
        except Exception as e:
            print(f"  [warn] skipping provider bench (bundle): {e}")

    # 5) Write a small press-play manifest (merge bundle manifest if present)
    manifest = {
        "press_play": True,
        "out_root": str(out_root),
        "student_preset": args.student_preset,
        "kd": run_kd,
        "onnx": onnx_model.exists(),
    }
    if bundle_manifest_path.exists():
        try:
            manifest["bundle"] = json.loads(bundle_manifest_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    (out_root / "press_play_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("[DONE] Press Play completed.")
    try:
        # Helpful next steps
        print("Next: export to phone")
        print("  Android (ADB): python -m omnicoder.tools.export_to_phone --platform android")
        print("  iOS (Core ML): python -m omnicoder.tools.export_to_phone --platform ios")
    except Exception as e:
        print(f"  [warn] could not print next steps: {e}")


if __name__ == "__main__":
    main()



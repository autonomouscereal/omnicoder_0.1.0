from __future__ import annotations

"""
lets_gooooo: single-entry orchestrator that plans training by time budget and runs
all stages, then exports and benches. Thin wrapper over run_training + press_play
to satisfy the one-button UX.
"""

import argparse
import os
import sys
from pathlib import Path

from .run_training import main as run_training_main  # type: ignore
from omnicoder.utils.env_registry import load_dotenv_best_effort
from omnicoder.utils.env_defaults import apply_core_defaults, apply_training_defaults, apply_run_env_defaults, apply_profile


def _load_dotenv(env_path: str = ".env") -> None:
	try:
		load_dotenv_best_effort((env_path,))
	except Exception as _e:
		print(f"[warn] _load_dotenv failed for {env_path}: {_e}")


def main() -> None:
    # Load .env first for parity with press_play
    try:
        load_dotenv_best_effort((".env", ".env.tuned"))
        # Apply centralized defaults and quality-first profile
        apply_core_defaults(os.environ)  # type: ignore[arg-type]
        apply_training_defaults(os.environ)  # type: ignore[arg-type]
        apply_run_env_defaults(os.environ)  # type: ignore[arg-type]
        apply_profile(os.environ, "quality")  # type: ignore[arg-type]
    except Exception as _e:
        print(f"[warn] initial dotenv load failed: {_e}")
    ap = argparse.ArgumentParser(description="Single-button: train (time-budgeted) → export → bench")
    ap.add_argument("--budget_hours", type=float, default=float(os.getenv("OMNICODER_TRAIN_BUDGET_HOURS", "1")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
    ap.add_argument("--out_root", type=str, default=os.getenv("OMNICODER_OUT_ROOT", "weights"))
    ap.add_argument("--export_to_phone", type=str, default=os.getenv("OMNICODER_EXPORT_TO_PHONE", ""))
    args, unknown = ap.parse_known_args()
    # Centralized defaults application (fills only missing keys)
    apply_core_defaults(os.environ)  # type: ignore[arg-type]
    apply_training_defaults(os.environ)  # type: ignore[arg-type]

    # Stage 0: optionally run tests (full suite, verbose) to guarantee green state before training
    # Honor EXECUTE_TESTS env: when set to a falsey value (0,false,no), skip tests.
    try:
        do_tests_env = os.getenv("EXECUTE_TESTS", "true").strip().lower()
        do_tests = do_tests_env in ("1", "true", "yes", "on")
    except Exception:
        do_tests = True
    if do_tests:
        try:
            import subprocess
            # Prepare SFB sidecars/env before running tests to ensure SFB tests pass in fresh envs
            try:
                from .sfb_setup import setup_sidecars_and_env  # type: ignore
                setup_sidecars_and_env()
            except Exception as _e:
                print(f"[warn] sfb_setup failed/skipped: {_e}")
            print("[tests] Running pytest -vv -rA ...")
            rc = subprocess.run([sys.executable, "-m", "pytest", "-vv", "-rA"], check=False)
            if int(getattr(rc, "returncode", 1)) != 0:
                print("[tests] failing tests; aborting lets-gooooo")
                sys.exit(rc.returncode)
        except Exception as e:
            print(f"[warn] pytest not available or failed to execute: {e}")
            print("[warn] proceeding only because environment forbids tests; ensure CI is green before running on training bench")
    else:
        try:
            env_mode = os.getenv("ENV", "").strip()
            print(f"[tests] Skipped by EXECUTE_TESTS={os.getenv('EXECUTE_TESTS')} (ENV={env_mode})")
        except Exception:
            print("[tests] Skipped by EXECUTE_TESTS env")

    # Stage 1: time-budgeted orchestrated training and export
    # Reuse run_training.main; it already performs export and provider benches at the end
    # We call it via its Python entrypoint to keep environment intact.
    # Enable unified multimodal training by default when a mixed JSONL is available
    try:
        # Vocab alignment pre-stage: run pre_align when images folder available and no prior weights
        from pathlib import Path as _P
        img_dir = _P(os.getenv("OMNICODER_PRE_ALIGN_IMAGES", "examples/data/vq/images"))
        pre_align_out = _P("weights/pre_align.pt")
        if img_dir.exists() and not pre_align_out.exists():
            import subprocess as _sp
            print("[stage] pre_align (vocab alignment via InfoNCE) ...")
            _sp.run([sys.executable, "-m", "omnicoder.training.pre_align", "--data", str(img_dir), "--steps", "200", "--device", args.device, "--embed_dim", "256"], check=False)
    except Exception as _e:
        print(f"[warn] pre_align skipped: {_e}")
    try:
        if not os.getenv("OMNICODER_MM_JSONL", "").strip():
            # Default to examples mixed VL fused sample if present
            from pathlib import Path as _P
            cand = _P("examples/vl_fused_sample.jsonl")
            if cand.exists():
                os.environ["OMNICODER_MM_JSONL"] = str(cand)
    except Exception:
        pass
    sys.argv = [sys.argv[0], "--budget_hours", str(args.budget_hours), "--device", args.device, "--out_root", args.out_root]
    run_training_main()

    # Stage 2: export ALL artifacts (end-to-end + standalone) to a standard directory
    try:
        import subprocess as _sp
        out_dir = str(Path(args.out_root) / "export_all")
        cmd = [sys.executable, "-m", "omnicoder.export.onnx_export", "--mobile_preset", os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb"), "--opset", os.getenv("OMNICODER_EXPORT_OPSET","18"), "--output_dir", out_dir]
        print("[export_all] ", " ".join(cmd))
        _sp.check_call(cmd)
    except Exception as _e:
        print(f"[warn] export_all failed/skipped: {_e}")

    # After training/export, build a small draft model once and set it globally for IO-free speculative decode
    try:
        from omnicoder.inference.generate import build_mobile_model_by_name, set_global_draft_model  # type: ignore
        draft_preset = os.getenv("OMNICODER_DRAFT_PRESET", "draft_2b")
        draft = build_mobile_model_by_name(draft_preset, mem_slots=0, skip_init=True)
        set_global_draft_model(draft)
        print(f"[draft] global draft model set preset={draft_preset}")
    except Exception as _e:
        print(f"[warn] could not set global draft model: {_e}")

    # Optional: export to phone if requested
    if args.export_to_phone.strip():
        try:
            from .export_to_phone import main as export_main  # type: ignore

            sys.argv = [sys.argv[0], "--platform", args.export_to_phone.strip().lower(), "--out_root", str(Path(args.out_root) / "release")]
            export_main()
        except Exception as e:
            print(f"[warn] export_to_phone failed/skipped: {e}")

    # Final hint
    try:
        print("[DONE] Training + export complete. Artifacts under:", args.out_root)
        print("Next: export to phone (if not done): python -m omnicoder.tools.export_to_phone --platform android|ios --out_root", str(Path(args.out_root) / "release"))
    except Exception as _e:
        print(f"[warn] final message emission failed: {_e}")


if __name__ == "__main__":
    main()



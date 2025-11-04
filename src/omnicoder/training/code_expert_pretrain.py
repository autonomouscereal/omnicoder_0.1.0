from __future__ import annotations

"""
Code expert pretraining wrapper (plumbing + router curriculum for code domain).

This script configures a KD run specialized for code data and enables
router-distribution KD and balanced assignment (Sinkhorn) optionally, to bias
experts toward code sub-tasks without destabilizing routing.

It shells out to `training/distill.py` with appropriate flags so the core loop
remains single-sourced.

Examples:
  python -m omnicoder.training.code_expert_pretrain \
    --data examples/code_eval --teacher bigcode/starcoder2-3b \
    --steps 2000 --device cuda --student_mobile_preset mobile_4gb
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def _resolve_code_teacher() -> str:
    # Try profiles/teachers.json → mobile_4gb preset → code key
    try:
        data = json.loads(Path("profiles/teachers.json").read_text(encoding="utf-8"))
        # Prefer explicit mobile preset mapping, else default.code
        preset = os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb")
        if isinstance(data, dict):
            mp = data.get("mobile_presets", {})
            if isinstance(mp, dict) and preset in mp and isinstance(mp[preset], dict):
                code = mp[preset].get("code", "")
                if code:
                    return str(code)
            default = data.get("default", {})
            if isinstance(default, dict) and default.get("code"):
                return str(default["code"])
    except Exception:
        pass
    return os.getenv("OMNICODER_KD_CODE_TEACHER", "bigcode/starcoder2-3b")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretrain code experts (KD) with router curriculum")
    ap.add_argument("--data", type=str, default=os.getenv("OMNICODER_DATA_CODE", "examples/code_eval"))
    ap.add_argument("--teacher", type=str, default="", help="HF id of teacher; resolved from profiles if empty")
    ap.add_argument("--steps", type=int, default=int(os.getenv("OMNICODER_CODE_STEPS", "2000")))
    ap.add_argument("--seq_len", type=int, default=int(os.getenv("OMNICODER_CODE_SEQ", "512")))
    ap.add_argument("--batch_size", type=int, default=int(os.getenv("OMNICODER_CODE_BATCH", "2")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
    ap.add_argument("--student_mobile_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"))
    ap.add_argument("--out", type=str, default=os.getenv("OMNICODER_CODE_OUT", "weights/omnicoder_student_code.pt"))
    # Router curriculum knobs (best-effort)
    ap.add_argument("--expert_route_kd", action="store_true", default=True)
    ap.add_argument("--sinkhorn_iters", type=int, default=int(os.getenv("OMNICODER_CODE_SINKHORN_ITERS", "8")))
    ap.add_argument("--sinkhorn_tau", type=float, default=float(os.getenv("OMNICODER_CODE_SINKHORN_TAU", "0.7")))
    ap.add_argument("--moe_group_sizes", type=str, default=os.getenv("OMNICODER_MOE_GROUP_SIZES", ""))
    ap.add_argument("--lora", action="store_true", default=(os.getenv("OMNICODER_CODE_LORA", "1") == "1"))
    ap.add_argument("--lora_r", type=int, default=int(os.getenv("OMNICODER_CODE_LORA_R", "16")))
    ap.add_argument("--lora_alpha", type=int, default=int(os.getenv("OMNICODER_CODE_LORA_ALPHA", "32")))
    ap.add_argument("--lora_dropout", type=float, default=float(os.getenv("OMNICODER_CODE_LORA_DROPOUT", "0.05")))
    args = ap.parse_args()

    teacher = args.teacher or _resolve_code_teacher()

    Path("weights").mkdir(exist_ok=True)
    cmd = [
        os.environ.get("PYTHON", "python"), "-m", "omnicoder.training.distill",
        "--data", args.data,
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--steps", str(args.steps),
        "--device", args.device,
        "--teacher", teacher,
        "--student_mobile_preset", args.student_mobile_preset,
        "--out", args.out,
        "--alpha_kd", os.getenv("OMNICODER_KD_ALPHA", "0.9"),
        "--kl_temp", os.getenv("OMNICODER_KD_TEMP", "1.5"),
    ]
    if args.expert_route_kd:
        cmd += ["--expert_route_kd", "--router_sinkhorn_iters", str(int(args.sinkhorn_iters)), "--router_sinkhorn_tau", str(float(args.sinkhorn_tau))]
    if args.moe_group_sizes:
        cmd += ["--moe_group_sizes", args.moe_group_sizes]
    if args.lora:
        cmd += ["--lora", "--lora_r", str(int(args.lora_r)), "--lora_alpha", str(int(args.lora_alpha)), "--lora_dropout", str(float(args.lora_dropout))]

    print("[code_expert_pretrain] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)
    print(f"[code_expert_pretrain] done. out={args.out}")


if __name__ == "__main__":
    main()



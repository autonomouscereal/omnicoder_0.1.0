"""
Draft training wrapper: trains a compact student (~0.5–1B) via KD
and auto-exports a one-step ONNX draft model for speculative decoding.

This wrapper shells out to the existing KD script and ONNX exporter
to keep coupling low.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a compact draft model via KD and export its ONNX one-step")
    ap.add_argument("--data", type=str, default="./examples/code_eval", help="KD data folder or JSONL")
    ap.add_argument("--data_is_jsonl", action="store_true", default=False)
    ap.add_argument("--teacher", type=str, default="microsoft/phi-2")
    ap.add_argument("--teacher_device_map", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--teacher_dtype", type=str, default="auto", help="auto|fp16|bf16|fp32")
    ap.add_argument("--student_mobile_preset", type=str, default="mobile_2gb", help="Preset approximating a ~1B draft")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_ckpt", type=str, default="weights/omnicoder_draft_kd.pt")
    ap.add_argument("--out_onnx", type=str, default="weights/text/draft_decode_step.onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--bench_accept", action="store_true", default=True)
    ap.add_argument("--bench_out", type=str, default="weights/draft_acceptance.json")
    # LoRA options (pass-through to KD)
    ap.add_argument("--lora", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    Path("weights").mkdir(exist_ok=True)
    Path("weights/text").mkdir(parents=True, exist_ok=True)

    # 1) KD training
    kd_cmd = [
        os.environ.get("PYTHON", "python"), "-m", "omnicoder.training.distill",
        "--data", args.data,
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--steps", str(args.steps),
        "--device", args.device,
        "--teacher", args.teacher,
        "--teacher_device_map", args.teacher_device_map,
        "--teacher_dtype", args.teacher_dtype,
        "--student_mobile_preset", args.student_mobile_preset,
        "--gradient_checkpointing",
        "--lora" if args.lora else "",
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--log_file", "weights/kd_draft_log.jsonl",
        "--out", args.out_ckpt,
    ]
    if args.data_is_jsonl:
        kd_cmd.insert(5, "--data_is_jsonl")
    print("[draft_train] running:", " ".join(kd_cmd), flush=True)
    subprocess.run(kd_cmd, check=False)

    # 2) Export ONNX one-step draft
    onnx_cmd = [
        os.environ.get("PYTHON", "python"), "-m", "omnicoder.export.onnx_export",
        "--output", args.out_onnx,
        "--seq_len", "1",
        "--mobile_preset", args.student_mobile_preset,
        "--opset", str(args.opset),
        "--decode_step",
    ]
    print("[draft_train] exporting draft ONNX:", " ".join(onnx_cmd), flush=True)
    subprocess.run(onnx_cmd, check=False)
    # 3) Optional acceptance tuning curve
    if args.bench_accept:
        try:
            bench_cmd = [
                os.environ.get("PYTHON", "python"), "-m", "omnicoder.tools.bench_acceptance",
                "--model", args.out_onnx,
                "--mobile_preset", args.student_mobile_preset,
                "--draft_preset", args.student_mobile_preset,
                "--tune_threshold", "--write_profiles",
                "--out_json", args.bench_out,
            ]
            print("[draft_train] benchmarking acceptance:", " ".join(bench_cmd), flush=True)
            subprocess.run(bench_cmd, check=False)
            # Auto-update profiles/acceptance_thresholds.json when recommended
            try:
                pth = Path(args.bench_out)
                if pth.exists():
                    data = json.loads(pth.read_text(encoding='utf-8'))
                    raw_thr = data.get('recommended_threshold', 0.0) if isinstance(data, dict) else 0.0
                    thr = float(raw_thr) if raw_thr is not None else 0.0
                    tps = float(data.get('tokens_per_second', 0.0)) if isinstance(data, dict) else 0.0
                    tps_draft = data.get('tokens_per_second_draft', None)
                    print(f"[draft_train] bench TPS: base={tps:.3f} draft={tps_draft if tps_draft is not None else 'n/a'} delta={data.get('tps_delta', None)}")
                    if thr > 0.0:
                        prof = Path('profiles')
                        prof.mkdir(parents=True, exist_ok=True)
                        acc_path = prof / 'acceptance_thresholds.json'
                        cur = {}
                        if acc_path.exists():
                            try:
                                cur = json.loads(acc_path.read_text(encoding='utf-8'))
                            except Exception:
                                cur = {}
                        # Update default keys: use student preset as both draft and student keys
                        cur_key = os.environ.get('OMNICODER_DRAFT_PRESET', args.student_mobile_preset)
                        cur[str(cur_key)] = thr
                        cur[str(args.student_mobile_preset)] = thr
                        acc_path.write_text(json.dumps(cur, indent=2), encoding='utf-8')
                        print(f"[draft_train] updated {acc_path} with {cur_key}={thr}")
            except Exception as e:
                print(f"[draft_train] acceptance thresholds update skipped: {e}")
        except Exception:
            pass
    print(f"[draft_train] done. ckpt={args.out_ckpt} onnx={args.out_onnx}")


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


OPTIONAL_DEPS = ("torch", "transformers", "datasets", "trl", "peft")
FOUR_BIT_DEPS = ("bitsandbytes",)


def dep_status(load_in_4bit: bool = False) -> dict[str, bool]:
    deps = {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_DEPS}
    if load_in_4bit:
        for name in FOUR_BIT_DEPS:
            deps[name] = importlib.util.find_spec(name) is not None
    return deps


def missing_deps(load_in_4bit: bool = False) -> list[str]:
    status = dep_status(load_in_4bit)
    return [name for name, ok in status.items() if not ok]


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
        if limit and len(rows) >= limit:
            break
    return rows


def normalize_sft_rows(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    from omnicoder.training import local_hf_trainer_bridge_2026

    rows, _, _ = local_hf_trainer_bridge_2026.normalize_sft_rows(path, limit, require_train_bucket=True)
    return rows


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 QLoRA bridge SFT trainer")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--train_jsonl", required=False)
    ap.add_argument("--eval_jsonl", default=None)
    ap.add_argument("--out_dir", default="weights/sft_qlora_2026")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="all-linear")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--bnb_4bit_quant_type", default="nf4")
    ap.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    ap.add_argument("--compute_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--assistant_only_loss", action="store_true")
    ap.add_argument("--check_deps", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    return ap


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    missing = missing_deps(args.load_in_4bit)
    manifest_path = args.manifest or str(Path(args.out_dir) / "omnicoder_sft_qlora_manifest.json")
    base_manifest = {
        "schema": "omnicoder2026_sft_qlora_bridge_v1",
        "model": args.model,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "out_dir": args.out_dir,
        "deps": dep_status(args.load_in_4bit),
        "load_in_4bit": bool(args.load_in_4bit),
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "target_modules": args.target_modules,
        },
        "training": {
            "max_seq_len": int(args.max_seq_len),
            "max_steps": int(args.max_steps),
            "learning_rate": float(args.learning_rate),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "packing": bool(args.packing),
            "assistant_only_loss": bool(args.assistant_only_loss),
        },
        "status": "created",
    }
    if missing:
        base_manifest["status"] = "missing_dependencies"
        base_manifest["missing_dependencies"] = missing
        base_manifest["install_hint"] = "Install/pin transformers, datasets, trl, peft, and bitsandbytes for Linux QLoRA runs."
        write_manifest(manifest_path, base_manifest)
        raise SystemExit(json.dumps(base_manifest, ensure_ascii=True))
    if args.check_deps:
        base_manifest["status"] = "deps_ok"
        write_manifest(manifest_path, base_manifest)
        return base_manifest
    if not args.train_jsonl:
        base_manifest["status"] = "missing_train_jsonl"
        write_manifest(manifest_path, base_manifest)
        raise SystemExit(json.dumps(base_manifest, ensure_ascii=True))
    rows = normalize_sft_rows(args.train_jsonl, args.limit)
    if not rows:
        base_manifest["status"] = "empty_dataset"
        write_manifest(manifest_path, base_manifest)
        raise SystemExit(json.dumps(base_manifest, ensure_ascii=True))
    base_manifest["train_records"] = len(rows)
    if args.dry_run:
        base_manifest["status"] = "dry_run_ok"
        write_manifest(manifest_path, base_manifest)
        return base_manifest

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.compute_dtype]
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
            bnb_4bit_compute_dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **model_kwargs)
    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )
    train_dataset = Dataset.from_list(rows)
    eval_dataset = Dataset.from_list(normalize_sft_rows(args.eval_jsonl, args.limit)) if args.eval_jsonl else None
    cfg = SFTConfig(
        output_dir=args.out_dir,
        max_length=int(args.max_seq_len),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        logging_steps=int(args.logging_steps),
        packing=bool(args.packing),
        assistant_only_loss=bool(args.assistant_only_loss),
        fp16=args.compute_dtype == "fp16",
        bf16=args.compute_dtype == "bf16",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(args.out_dir)
    base_manifest["status"] = "completed"
    base_manifest["train_result"] = result.metrics
    write_manifest(manifest_path, base_manifest)
    return base_manifest


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        payload = run_train(args)
        print(json.dumps(payload, ensure_ascii=True))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            raise SystemExit(2)
        raise


if __name__ == "__main__":
    main()

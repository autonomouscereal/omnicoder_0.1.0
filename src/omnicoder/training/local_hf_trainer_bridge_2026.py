from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


BASE_DEPS = ("torch", "transformers", "datasets", "trl", "peft")
UNSLOTH_DEPS = BASE_DEPS + ("unsloth",)
DEFAULT_PROTECTED_GPUS = "0,4,6"
TRAIN_SPLITS = {"train", "training"}
REJECT_SPLITS = {"eval", "evaluation", "test", "holdout", "eval_holdout", "reportable", "validation", "valid"}


def find_dep(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dep_status(backend: str) -> dict[str, bool]:
    deps = UNSLOTH_DEPS if backend == "unsloth" else BASE_DEPS
    return {name: find_dep(name) for name in deps}


def dep_versions(backend: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in (UNSLOTH_DEPS if backend == "unsloth" else BASE_DEPS):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def missing_deps(backend: str, load_in_4bit: bool) -> list[str]:
    status = dep_status(backend)
    missing = [name for name, ok in status.items() if not ok]
    if load_in_4bit and not find_dep("bitsandbytes"):
        missing.append("bitsandbytes")
    return missing


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def nested_value(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def row_split_markers(obj: dict[str, Any]) -> set[str]:
    markers: set[str] = set()
    for key in ("split", "bucket", "namespace", "data_split", "dataset_split", "training_bucket", "use_policy"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            markers.add(value.strip().lower())
    for path in (
        ("metadata", "split"),
        ("metadata", "bucket"),
        ("metadata", "namespace"),
        ("lineage", "split"),
        ("lineage", "bucket"),
        ("lineage", "namespace"),
    ):
        value = nested_value(obj, path)
        if isinstance(value, str) and value.strip():
            markers.add(value.strip().lower())
    return markers


def rejection_reason(obj: dict[str, Any], require_train_bucket: bool) -> str | None:
    markers = row_split_markers(obj)
    if markers & REJECT_SPLITS:
        return "non_train_split"
    if require_train_bucket and markers and not (markers & TRAIN_SPLITS):
        return "non_train_bucket"
    if obj.get("synthetic_seed_only") is True:
        return "synthetic_seed_only"
    if obj.get("secret_rejected") is True or nested_value(obj, ("metadata", "secret_rejected")) is True:
        return "secret_rejected"
    contamination = obj.get("contamination") or obj.get("contamination_report") or {}
    if isinstance(contamination, dict) and contamination.get("contaminated") is True:
        return "contaminated"
    status = str(obj.get("contamination_status") or nested_value(obj, ("metadata", "contamination_status")) or "").lower()
    if status in {"contaminated", "rejected", "benchmark_leak"}:
        return "contaminated"
    if obj.get("reportable_task") is True or nested_value(obj, ("metadata", "reportable_task")) is True:
        return "reportable_task"
    return None


def normalize_messages(messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(messages, list):
        return None
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "user").lower()
        if role == "human":
            role = "user"
        elif role in {"gpt", "bot"}:
            role = "assistant"
        content = message.get("content")
        if content is None:
            content = message.get("value")
        if isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content})
    return normalized or None


def normalize_sft_obj(obj: dict[str, Any], require_train_bucket: bool) -> tuple[dict[str, Any] | None, str | None]:
    reason = rejection_reason(obj, require_train_bucket)
    if reason:
        return None, reason
    messages = normalize_messages(obj.get("messages")) or normalize_messages(obj.get("conversations"))
    if messages:
        return {"messages": messages, "metadata": obj.get("metadata", {})}, None
    input_json = obj.get("input_json") if isinstance(obj.get("input_json"), dict) else {}
    target_json = obj.get("target_json") if isinstance(obj.get("target_json"), dict) else {}
    if input_json or target_json:
        input_messages = normalize_messages(input_json.get("messages")) or normalize_messages(input_json.get("conversations"))
        target_messages = normalize_messages(target_json.get("messages")) or normalize_messages(target_json.get("conversations"))
        target_content = (
            target_json.get("content")
            or target_json.get("text")
            or target_json.get("completion")
            or target_json.get("response")
            or target_json.get("answer")
            or target_json.get("output")
        )
        input_content = (
            input_json.get("content")
            or input_json.get("text")
            or input_json.get("prompt")
            or input_json.get("instruction")
            or input_json.get("question")
        )
        if input_messages and isinstance(target_content, str) and target_content.strip():
            return {
                "messages": input_messages + [{"role": "assistant", "content": target_content}],
                "metadata": obj.get("metadata", {}),
            }, None
        if input_messages and target_messages:
            return {"messages": input_messages + target_messages, "metadata": obj.get("metadata", {})}, None
        if isinstance(input_content, str) and input_content.strip() and isinstance(target_content, str) and target_content.strip():
            return {"prompt": input_content, "completion": target_content, "metadata": obj.get("metadata", {})}, None
    prompt = obj.get("prompt")
    completion = obj.get("completion") or obj.get("response") or obj.get("answer") or obj.get("output")
    if prompt is not None and completion is not None:
        return {"prompt": str(prompt), "completion": str(completion), "metadata": obj.get("metadata", {})}, None
    instruction = obj.get("instruction") or obj.get("Instruction")
    output = obj.get("output") or obj.get("Output")
    if instruction is not None and output is not None:
        input_text = obj.get("input") or obj.get("Input") or ""
        prompt_text = str(instruction)
        if input_text:
            prompt_text = f"{prompt_text}\n\n{input_text}"
        return {"prompt": prompt_text, "completion": str(output), "metadata": obj.get("metadata", {})}, None
    text = obj.get("text") or obj.get("content")
    if isinstance(text, str) and text.strip():
        return {"text": text, "metadata": obj.get("metadata", {})}, None
    return None, "unsupported_schema"


def iter_jsonl(path: str | Path, limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    total = 0
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        total += 1
        if limit and total > limit:
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rejected["parse_error"] = rejected.get("parse_error", 0) + 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            rejected["non_object"] = rejected.get("non_object", 0) + 1
    return rows, rejected, total


def normalize_sft_rows(path: str | Path, limit: int = 0, require_train_bucket: bool = True) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    raw_rows, rejected, total = iter_jsonl(path, limit)
    normalized: list[dict[str, Any]] = []
    for obj in raw_rows:
        row, reason = normalize_sft_obj(obj, require_train_bucket)
        if row is None:
            rejected[str(reason or "rejected")] = rejected.get(str(reason or "rejected"), 0) + 1
        else:
            normalized.append(row)
    return normalized, rejected, total


def inspect_sft_dataset(path: str | Path | None, limit: int = 0, require_train_bucket: bool = True) -> dict[str, Any]:
    if not path:
        return {"exists": False, "records": 0, "accepted_records": 0, "rejected": {"missing_path": 1}}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(path), "records": 0, "accepted_records": 0, "rejected": {"missing_path": 1}}
    rows, rejected, total = normalize_sft_rows(p, limit, require_train_bucket=require_train_bucket)
    formats: dict[str, int] = {}
    chars = 0
    for row in rows:
        if "messages" in row:
            formats["messages"] = formats.get("messages", 0) + 1
            chars += sum(len(str(msg.get("content") or "")) for msg in row.get("messages", []))
        elif "prompt" in row:
            formats["prompt_completion"] = formats.get("prompt_completion", 0) + 1
            chars += len(str(row.get("prompt") or "")) + len(str(row.get("completion") or ""))
        elif "text" in row:
            formats["text"] = formats.get("text", 0) + 1
            chars += len(str(row.get("text") or ""))
    return {
        "exists": True,
        "path": str(p),
        "records_seen": total,
        "accepted_records": len(rows),
        "rejected": rejected,
        "formats": formats,
        "estimated_tokens": chars // 4,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def materialize_filtered_sft_files(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[Path, Path | None]:
    dataset_dir = Path(args.out_dir) / "datasets"
    train_rows, train_rejected, _ = normalize_sft_rows(
        args.train_jsonl,
        args.limit,
        require_train_bucket=not args.allow_nontrain_rows,
    )
    train_path = dataset_dir / "train_sft_filtered.jsonl"
    write_jsonl(train_path, train_rows)
    eval_path = None
    if args.eval_jsonl:
        eval_rows, _, _ = normalize_sft_rows(args.eval_jsonl, args.eval_limit, require_train_bucket=False)
        if eval_rows:
            eval_path = dataset_dir / "eval_sft_filtered.jsonl"
            write_jsonl(eval_path, eval_rows)
    manifest["filtered_datasets"] = {
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path) if eval_path else None,
        "train_records": len(train_rows),
        "train_rejected": train_rejected,
    }
    return train_path, eval_path


def target_modules(value: str) -> str | list[str]:
    if value == "all-linear":
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_gpu_ids(value: str | None) -> set[str]:
    ids = set()
    for part in parse_csv_set(value):
        if part.lower() in {"none", "cpu", "-1"}:
            continue
        ids.add(part)
    return ids


def gpu_guard(args: argparse.Namespace) -> dict[str, Any]:
    protected = parse_gpu_ids(args.protected_gpus or os.getenv("OMNICODER_PROTECTED_GPUS", DEFAULT_PROTECTED_GPUS))
    requested = parse_gpu_ids(args.host_gpu_ids or os.getenv("OMNICODER_HOST_GPU_IDS", ""))
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not requested and visible:
        requested = parse_gpu_ids(visible)
    overlap = sorted(protected & requested)
    return {
        "protected_gpus": sorted(protected),
        "requested_host_gpus": sorted(requested),
        "cuda_visible_devices": visible,
        "ok": bool(args.allow_protected_gpus or not overlap),
        "overlap": overlap,
    }


def benchmark_command(args: argparse.Namespace) -> dict[str, Any]:
    runner = (
        f"{sys.executable} -m omnicoder.inference.local_hf_adapter_runner_2026 "
        f"--model {args.model} --adapter {Path(args.out_dir).as_posix()}"
    )
    return {
        "backend": "checkpoint-runner",
        "checkpoint_runner_command": runner,
        "checkpoint_path": str(Path(args.out_dir)),
        "note": "Use only for adapter-sized local HF/Unsloth artifacts; the native 20B checkpoint remains benchmarked through the dense pipeline.",
    }


def build_manifest(args: argparse.Namespace, status: str = "created") -> dict[str, Any]:
    backend = str(args.backend)
    require_train_bucket = not bool(args.allow_nontrain_rows)
    train = inspect_sft_dataset(args.train_jsonl, args.limit, require_train_bucket=require_train_bucket)
    eval_data = inspect_sft_dataset(args.eval_jsonl, args.eval_limit, require_train_bucket=False) if args.eval_jsonl else None
    gpu = gpu_guard(args)
    missing = missing_deps(backend, bool(args.load_in_4bit))
    return {
        "schema": "omnicoder.local_hf_trainer_bridge_2026.v1",
        "command": args.command,
        "backend": backend,
        "status": status,
        "model": args.model,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl,
        "out_dir": args.out_dir,
        "dataset": train,
        "eval_dataset": eval_data,
        "deps": dep_status(backend),
        "versions": dep_versions(backend),
        "missing_dependencies": missing,
        "gpu_guard": gpu,
        "safety": {
            "require_train_bucket": require_train_bucket,
            "protected_gpu_policy": "fail closed unless --allow-protected-gpus is explicit",
            "intended_use": "isolated local HF/Unsloth sidecar; not native Omnicoder 20B checkpoint training",
        },
        "training": {
            "max_seq_len": int(args.max_seq_len),
            "max_steps": int(args.max_steps),
            "learning_rate": float(args.learning_rate),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "save_steps": int(args.save_steps),
            "eval_steps": int(args.eval_steps),
            "logging_steps": int(args.logging_steps),
            "packing": bool(args.packing),
            "assistant_only_loss": bool(args.assistant_only_loss),
            "load_in_4bit": bool(args.load_in_4bit),
            "dtype": args.dtype,
            "lora": {
                "r": int(args.lora_r),
                "alpha": int(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "target_modules": target_modules(args.target_modules),
            },
            "unsloth": {
                "tiled_mlp": bool(args.unsloth_tiled_mlp),
                "gradient_checkpointing": args.unsloth_gradient_checkpointing,
            },
        },
        "benchmark": benchmark_command(args),
        "deployment": {
            "gguf_quantization": args.gguf_quantization,
            "save_gguf": bool(args.save_gguf),
            "chat_template_warning": "Keep the same chat template during llama.cpp, LM Studio, or Ollama deployment.",
        },
    }


def live_unsloth_sft(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    materialize_filtered_sft_files(args, manifest)
    train_rows, _, _ = normalize_sft_rows(args.train_jsonl, args.limit, require_train_bucket=not args.allow_nontrain_rows)
    eval_rows: list[dict[str, Any]] | None = None
    if args.eval_jsonl:
        eval_rows, _, _ = normalize_sft_rows(args.eval_jsonl, args.eval_limit, require_train_bucket=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "auto": None}[args.dtype]
    model_kwargs: dict[str, Any] = {
        "model_name": args.model,
        "max_seq_length": int(args.max_seq_len),
        "dtype": dtype,
        "load_in_4bit": bool(args.load_in_4bit),
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.unsloth_tiled_mlp:
        model_kwargs["unsloth_tiled_mlp"] = True
    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(args.lora_r),
        target_modules=target_modules(args.target_modules),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        use_gradient_checkpointing=args.unsloth_gradient_checkpointing,
        random_state=int(args.seed),
    )
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
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
        fp16=args.dtype == "fp16",
        bf16=args.dtype == "bf16",
        report_to="none",
    )
    trainer_kwargs = {
        "model": model,
        "args": cfg,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
    }
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(args.out_dir)
    if args.save_gguf:
        model.save_pretrained_gguf(args.out_dir, tokenizer, quantization_method=args.gguf_quantization)
    manifest["status"] = "completed"
    manifest["train_result"] = result.metrics
    return manifest


def live_trl_sft(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    from omnicoder.training import sft_qlora_2026

    train_path, eval_path = materialize_filtered_sft_files(args, manifest)
    trl_args = argparse.Namespace(
        model=args.model,
        train_jsonl=str(train_path),
        eval_jsonl=str(eval_path) if eval_path else None,
        out_dir=args.out_dir,
        manifest=args.manifest,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        compute_dtype="fp16" if args.dtype == "auto" else args.dtype,
        packing=args.packing,
        assistant_only_loss=args.assistant_only_loss,
        check_deps=False,
        dry_run=False,
        limit=args.limit,
    )
    result = sft_qlora_2026.run_train(trl_args)
    manifest["status"] = result.get("status", "completed")
    manifest["trainer_manifest"] = result
    return manifest


def execute(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    manifest = build_manifest(args)
    manifest_path = Path(args.manifest or Path(args.out_dir) / "local_hf_trainer_manifest.json")
    dataset = manifest["dataset"]
    if args.command == "inspect-dataset":
        manifest["status"] = "dataset_inspected"
        write_json(manifest_path, manifest)
        return 0, manifest
    if not dataset.get("exists"):
        manifest["status"] = "missing_train_jsonl"
        write_json(manifest_path, manifest)
        return 2, manifest
    if int(dataset.get("accepted_records") or 0) <= 0:
        manifest["status"] = "empty_training_dataset"
        write_json(manifest_path, manifest)
        return 2, manifest
    if not manifest["gpu_guard"]["ok"]:
        manifest["status"] = "protected_gpu_overlap"
        write_json(manifest_path, manifest)
        return 2, manifest
    if args.check_deps:
        manifest["status"] = "deps_ok" if not manifest["missing_dependencies"] else "missing_dependencies"
        write_json(manifest_path, manifest)
        return (0 if not manifest["missing_dependencies"] or args.allow_missing_backend else 2), manifest
    if args.dry_run:
        manifest["status"] = "dry_run_ok" if not manifest["missing_dependencies"] else "dry_run_ok_missing_backend"
        write_json(manifest_path, manifest)
        return 0, manifest
    if manifest["missing_dependencies"] and not args.allow_missing_backend:
        manifest["status"] = "missing_dependencies"
        manifest["install_hint"] = "Install the optional local trainer backend in an isolated Linux venv/container, for example: uv pip install unsloth --torch-backend=auto."
        write_json(manifest_path, manifest)
        return 2, manifest
    if args.backend == "unsloth":
        manifest = live_unsloth_sft(args, manifest)
    else:
        manifest = live_trl_sft(args, manifest)
    write_json(manifest_path, manifest)
    return 0, manifest


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["unsloth", "trl"], default="unsloth")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-jsonl", "--train_jsonl", dest="train_jsonl", required=True)
    parser.add_argument("--eval-jsonl", "--eval_jsonl", dest="eval_jsonl", default=None)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="weights/local_hf_trainer_2026")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-seq-len", "--max_seq_len", dest="max_seq_len", type=int, default=4096)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=1000)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", "--per_device_train_batch_size", dest="per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", "--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps", type=int, default=100)
    parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps", type=int, default=100)
    parser.add_argument("--logging-steps", "--logging_steps", dest="logging_steps", type=int, default=10)
    parser.add_argument("--lora-r", "--lora_r", dest="lora_r", type=int, default=16)
    parser.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", "--lora_dropout", dest="lora_dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", "--target_modules", dest="target_modules", default="all-linear")
    parser.add_argument("--load-in-4bit", "--load_in_4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--assistant-only-loss", "--assistant_only_loss", dest="assistant_only_loss", action="store_true")
    parser.add_argument("--unsloth-tiled-mlp", "--unsloth_tiled_mlp", dest="unsloth_tiled_mlp", action="store_true")
    parser.add_argument("--unsloth-gradient-checkpointing", "--unsloth_gradient_checkpointing", dest="unsloth_gradient_checkpointing", default="unsloth")
    parser.add_argument("--device-map", "--device_map", dest="device_map", default="")
    parser.add_argument("--host-gpu-ids", "--host_gpu_ids", dest="host_gpu_ids", default="")
    parser.add_argument("--protected-gpus", "--protected_gpus", dest="protected_gpus", default=DEFAULT_PROTECTED_GPUS)
    parser.add_argument("--allow-protected-gpus", "--allow_protected_gpus", dest="allow_protected_gpus", action="store_true")
    parser.add_argument("--allow-nontrain-rows", "--allow_nontrain_rows", dest="allow_nontrain_rows", action="store_true")
    parser.add_argument("--allow-missing-backend", "--allow_missing_backend", dest="allow_missing_backend", action="store_true")
    parser.add_argument("--check-deps", "--check_deps", dest="check_deps", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-limit", "--eval_limit", dest="eval_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-gguf", "--save_gguf", dest="save_gguf", action="store_true")
    parser.add_argument("--gguf-quantization", "--gguf_quantization", dest="gguf_quantization", default="q4_k_m")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local HF/Unsloth sidecar trainer for Omnicoder curated datasets")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-dataset", help="Validate and summarize a local JSONL dataset without loading a model")
    add_common_args(inspect)
    sft = sub.add_parser("sft", help="Run or dry-run local SFT through TRL or Unsloth")
    add_common_args(sft)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, manifest = execute(args)
    stream = sys.stdout if code == 0 else sys.stderr
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True), file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

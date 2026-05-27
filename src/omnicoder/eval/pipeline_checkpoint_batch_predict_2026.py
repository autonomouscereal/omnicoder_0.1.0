from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
from omnicoder.eval import reportable_prediction_harness_2026 as harness
from omnicoder.modeling.omnicoder2026 import OmniCoder2026Config
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.training.pipeline_pretrain_2026_dense import (
    OmniCoder2026PipelineShard,
    autocast_context,
    load_checkpoint_shard,
    rank_device,
    shard_spec,
    stage_ranges,
    validate_target_device_placement,
)
from omnicoder.training.pretrain_2026_dense import _dtype_from_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


BACKEND_NAME = "pipeline_checkpoint_batch_predict_2026"
SUMMARY_SCHEMA = "omnicoder.pipeline_checkpoint_batch_predict_2026.summary.v1"
EXPECTED_SHARDS = 3
CONFIG_FIELDS = {field.name for field in fields(OmniCoder2026Config)}
TEXT_RANGE = DEFAULT_LEDGER.as_config_ranges()["text"]


class BatchPredictError(ValueError):
    """Raised for input, checkpoint, or decode failures that must fail closed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BatchPredictError(f"{path} must contain one JSON object")
    return payload


def _rank_files(checkpoint: Path) -> list[Path]:
    return sorted(path for path in checkpoint.glob("rank*.pt") if path.is_file())


def _checkpoint_dir(value: str | Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise BatchPredictError("--checkpoint is required")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists() or not path.is_dir():
        raise BatchPredictError(f"pipeline checkpoint must be a sharded checkpoint directory, got: {path}")
    return path


def _load_manifest(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / ".complete.json"
    if not manifest_path.exists():
        raise BatchPredictError(f"pipeline checkpoint is missing manifest.json: {checkpoint}")
    if not complete_path.exists():
        raise BatchPredictError(f"pipeline checkpoint is missing .complete.json: {checkpoint}")
    manifest = _read_json(manifest_path)
    rank_files = _rank_files(checkpoint)
    if len(rank_files) != EXPECTED_SHARDS:
        raise BatchPredictError(
            f"batch predictor expects exactly {EXPECTED_SHARDS} pipeline shards, found {len(rank_files)} in {checkpoint}"
        )
    world_size = int(manifest.get("world_size") or len(rank_files))
    if world_size != EXPECTED_SHARDS:
        raise BatchPredictError(
            f"batch predictor expects manifest world_size={EXPECTED_SHARDS}, got {world_size}"
        )
    expected = [checkpoint / f"rank{rank:05d}.pt" for rank in range(EXPECTED_SHARDS)]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise BatchPredictError(f"pipeline checkpoint rank files are not contiguous; missing: {missing}")
    marker_missing = [
        f"rank{rank:05d}.pt.complete.json"
        for rank in range(EXPECTED_SHARDS)
        if not (checkpoint / f"rank{rank:05d}.pt.complete.json").exists()
    ]
    if marker_missing:
        raise BatchPredictError(f"pipeline checkpoint is missing rank completion markers: {marker_missing}")
    return manifest


def _checkpoint_train_args(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            train_args = manifest.get("train_args")
            if isinstance(train_args, dict):
                return train_args
        except Exception:
            pass
    first_rank = path / "rank00000.pt"
    if not first_rank.exists():
        return {}
    try:
        payload = torch.load(first_rank, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    return train_args if isinstance(train_args, dict) else {}


def _placement_counts_from_train_args(train_args: dict[str, Any]) -> str:
    counts = str(train_args.get("placement_layer_counts") or "").strip()
    if counts:
        return counts
    raw_ranges = str(train_args.get("pipeline_stage_ranges") or "").strip()
    if not raw_ranges:
        return ""
    parsed: list[str] = []
    for segment in raw_ranges.split(","):
        segment = segment.strip()
        if not segment:
            continue
        start_s, end_s = segment.split(":", 1)
        parsed.append(str(int(end_s) - int(start_s)))
    return ",".join(parsed)


def _first_rank_payload(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint / "rank00000.pt", map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise BatchPredictError(f"{checkpoint / 'rank00000.pt'} is not a checkpoint payload")
    return payload


def _checkpoint_kwargs(checkpoint: Path, preset_name: str) -> tuple[dict[str, Any], str]:
    preset = get_omnicoder2026_preset(preset_name)
    kwargs = preset_to_model_kwargs(preset)
    payload = _first_rank_payload(checkpoint)
    saved_name = ""
    for key in ("preset", "config", "model_config"):
        value = payload.get(key)
        if value is not None and not isinstance(value, dict) and hasattr(value, "__dict__"):
            value = value.__dict__
        if isinstance(value, dict):
            kwargs.update({name: item for name, item in value.items() if name in CONFIG_FIELDS})
            if key == "preset":
                saved_name = str(value.get("name") or "")
    state = payload.get("model_state_dict")
    if isinstance(state, dict):
        embed = state.get("embed.weight")
        if isinstance(embed, torch.Tensor) and embed.ndim == 2:
            kwargs["vocab_size"] = int(embed.shape[0])
            kwargs["d_model"] = int(embed.shape[1])
    if isinstance(kwargs.get("layer_pattern"), list):
        kwargs["layer_pattern"] = tuple(kwargs["layer_pattern"])
    # Pipeline checkpoints put embed on rank 0 and lm_head on the final rank.
    kwargs["tie_embeddings"] = False
    return kwargs, saved_name or preset.name


def _set_fake_quant_env(fake_quant_chunk_rows: int, fake_quant_max_full_elements: int) -> None:
    if int(fake_quant_chunk_rows or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_CHUNK_ROWS"] = str(int(fake_quant_chunk_rows))
    if int(fake_quant_max_full_elements or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS"] = str(int(fake_quant_max_full_elements))


def _init_process_group(args: argparse.Namespace) -> None:
    if dist.is_initialized():
        return
    backend = str(args.dist_backend or "auto").lower()
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    timeout_seconds = int(
        getattr(args, "dist_timeout_seconds", 0)
        or os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200")
    )
    dist.init_process_group(backend=backend, timeout=_dt.timedelta(seconds=max(1, timeout_seconds)))


def _build_shard(args: argparse.Namespace) -> tuple[OmniCoder2026PipelineShard, torch.device, int, int, str]:
    _init_process_group(args)
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    if world_size != EXPECTED_SHARDS:
        raise BatchPredictError(f"batch predictor expects {EXPECTED_SHARDS} distributed ranks, got {world_size}")
    device = rank_device(rank, args.rank_device_map)
    checkpoint = _checkpoint_dir(args.checkpoint)
    _load_manifest(checkpoint)
    _set_fake_quant_env(int(args.fake_quant_chunk_rows or 0), int(args.fake_quant_max_full_elements or 0))
    train_args = _checkpoint_train_args(checkpoint)
    kwargs, saved_preset_name = _checkpoint_kwargs(checkpoint, args.preset)
    if bool(args.fake_quant) or bool(train_args.get("fake_quant")):
        kwargs["fake_quant"] = True
    cfg = OmniCoder2026Config(**kwargs)
    placement_counts = str(args.placement_layer_counts or "").strip() or _placement_counts_from_train_args(train_args)
    ranges = stage_ranges(int(cfg.n_layers), placement_counts)
    if len(ranges) != world_size:
        raise BatchPredictError(f"world_size={world_size} must match pipeline stages {ranges}")
    spec = shard_spec(rank, ranges)
    resume_args = SimpleNamespace(
        require_target_contract=bool(args.require_target_contract),
        allow_p40_target_contract_eval=bool(args.allow_p40_target_contract_eval),
        placement_layer_counts=placement_counts,
        pipeline_stage_ranges=str(train_args.get("pipeline_stage_ranges") or ""),
        pipeline_microbatches=str(train_args.get("pipeline_microbatches") or ""),
        pipeline_schedule=str(train_args.get("pipeline_schedule") or ""),
        fake_quant=bool(kwargs.get("fake_quant")),
    )
    validate_target_device_placement(resume_args, ranges, spec, device)
    dtype_name = str(args.init_dtype or "auto").lower()
    if dtype_name == "auto":
        dtype_name = str(args.precision or "fp32").lower()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(_dtype_from_name(dtype_name))
    try:
        with torch.device(device):
            shard = OmniCoder2026PipelineShard(cfg, spec, checkpoint_blocks=False).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
    load_checkpoint_shard(
        checkpoint,
        shard,
        optimizer=None,
        preset=SimpleNamespace(name=saved_preset_name),
        args=resume_args,
    )
    state = shard.state_dict()
    if spec.has_embed:
        embed = state.get("embed.weight")
        if not isinstance(embed, torch.Tensor) or embed.shape != (int(cfg.vocab_size), int(cfg.d_model)):
            raise BatchPredictError("loaded rank0 checkpoint has invalid embed.weight shape")
        if not torch.isfinite(embed.float()).all() or float(embed.float().std(unbiased=False)) <= 0.0:
            raise BatchPredictError("loaded rank0 embed.weight is non-finite or zero-variance")
    if spec.has_head:
        head = state.get("lm_head.weight")
        if not isinstance(head, torch.Tensor) or head.shape != (int(cfg.vocab_size), int(cfg.d_model)):
            raise BatchPredictError("loaded final-rank checkpoint has invalid lm_head.weight shape")
        if not torch.isfinite(head.float()).all() or float(head.float().std(unbiased=False)) <= 0.0:
            raise BatchPredictError("loaded final-rank lm_head.weight is non-finite or zero-variance")
    shard.eval()
    return shard, device, int(cfg.d_model), int(cfg.vocab_size), saved_preset_name


def _broadcast_task_header(device: torch.device, active: bool | None, max_new_tokens: int = 0) -> tuple[bool, int]:
    rank = int(dist.get_rank())
    if rank == 0:
        payload = torch.tensor(
            [1 if active else 0, int(max_new_tokens)],
            dtype=torch.long,
            device=device,
        )
    else:
        payload = torch.empty((2,), dtype=torch.long, device=device)
    dist.broadcast(payload, src=0)
    return bool(int(payload[0].item())), int(payload[1].item())


def _broadcast_ids(batch: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    rank = int(dist.get_rank())
    meta = torch.tensor([int(batch.shape[1]) if batch is not None else 0], dtype=torch.long, device=device)
    dist.broadcast(meta, src=0)
    length = int(meta[0].item())
    if length <= 0:
        raise BatchPredictError("cannot decode an empty prompt")
    if rank != 0:
        batch = torch.empty((1, length), dtype=torch.long, device=device)
    else:
        batch = batch.to(device, non_blocking=True)
    dist.broadcast(batch, src=0)
    return batch


def _broadcast_stop(device: torch.device, stop: bool | None) -> bool:
    rank = int(dist.get_rank())
    if rank == 0:
        payload = torch.tensor([1 if stop else 0], dtype=torch.long, device=device)
    else:
        payload = torch.empty((1,), dtype=torch.long, device=device)
    dist.broadcast(payload, src=0)
    return bool(int(payload[0].item()))


def _pipeline_next_token(
    shard: OmniCoder2026PipelineShard,
    batch: torch.Tensor,
    *,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    vocab_size: int,
    precision: str,
) -> int:
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    length = int(batch.shape[1])
    with torch.no_grad(), autocast_context(device, precision):
        if world_size == 1:
            hidden = shard(batch)
            logits = shard.lm_head(hidden[:, -1:, :]).float()
            token = _select_text_token(logits[:, -1, :], vocab_size)
            dist.broadcast(token, src=0)
            return int(token.detach().cpu().item())
        if rank == 0:
            hidden = shard(batch)
            dist.send(hidden.contiguous(), dst=1)
            token = torch.empty((1,), dtype=torch.long, device=device)
            dist.broadcast(token, src=world_size - 1)
            return int(token.detach().cpu().item())
        hidden = torch.empty((1, length, d_model), dtype=hidden_dtype, device=device)
        dist.recv(hidden, src=rank - 1)
        hidden = shard(hidden)
        if rank < world_size - 1:
            dist.send(hidden.contiguous(), dst=rank + 1)
            token = torch.empty((1,), dtype=torch.long, device=device)
            dist.broadcast(token, src=world_size - 1)
            return int(token.detach().cpu().item())
        logits = shard.lm_head(hidden[:, -1:, :]).float()
        token = _select_text_token(logits[:, -1, :], vocab_size)
        dist.broadcast(token, src=rank)
        return int(token.detach().cpu().item())


def _select_text_token(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    lo, hi = TEXT_RANGE
    hi = min(int(hi), int(vocab_size), int(logits.shape[-1]))
    if hi <= int(lo):
        raise BatchPredictError(f"text ledger range {TEXT_RANGE} is outside vocab_size={vocab_size}")
    masked = logits.float().clone()
    masked[..., : int(lo)] = float("-inf")
    masked[..., hi:] = float("-inf")
    return torch.argmax(masked, dim=-1).to(dtype=torch.long)


def _decode_rank0(
    shard: OmniCoder2026PipelineShard,
    prompt: str,
    *,
    tokenizer: Any,
    eos_id: int | None,
    max_new_tokens: int,
    max_prompt_tokens: int,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    vocab_size: int,
    precision: str,
) -> tuple[str, int]:
    ids = [int(item) for item in tokenizer.encode(prompt)]
    if not ids:
        raise BatchPredictError("tokenizer produced no prompt tokens")
    if len(ids) > int(max_prompt_tokens):
        raise BatchPredictError(
            f"prompt token length {len(ids)} exceeds --max-prompt-tokens={int(max_prompt_tokens)}"
        )
    generated = [int(item) % max(2, int(vocab_size)) for item in ids]
    new_tokens: list[int] = []
    for _ in range(int(max_new_tokens)):
        batch = torch.tensor([generated], dtype=torch.long, device=device)
        batch = _broadcast_ids(batch, device)
        next_token = _pipeline_next_token(
            shard,
            batch,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            vocab_size=vocab_size,
            precision=precision,
        )
        token_id = int(next_token) % max(2, int(vocab_size))
        generated.append(token_id)
        new_tokens.append(token_id)
        should_stop = isinstance(eos_id, int) and int(next_token) == int(eos_id)
        _broadcast_stop(device, should_stop)
        if should_stop:
            break
    text = tokenizer.decode(new_tokens).strip()
    if not text:
        text = "__OMNICODER_EMPTY_DECODE__"
    return text, len(new_tokens)


def _decode_nonzero(
    shard: OmniCoder2026PipelineShard,
    *,
    max_new_tokens: int,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
) -> None:
    for _ in range(int(max_new_tokens)):
        batch = _broadcast_ids(None, device)
        _pipeline_next_token(
            shard,
            batch,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            vocab_size=int(getattr(shard, "cfg").vocab_size),
            precision=precision,
        )
        if _broadcast_stop(device, None):
            break


def _generation_config(args: argparse.Namespace, checkpoint: Path) -> harness.GenerateConfig:
    return harness.GenerateConfig(
        backend=BACKEND_NAME,
        model=str(args.model or checkpoint),
        max_output_tokens=int(args.max_output_tokens),
        temperature=0.0,
        timeout_seconds=0,
        base_url="",
        api_key_env="",
        checkpoint_runner="",
        checkpoint_path=str(checkpoint),
    )


def _prediction_row(
    task: harness.TaskRecord,
    cfg: harness.GenerateConfig,
    output_field: str,
    text: str,
    *,
    latency_seconds: float,
    generated_tokens: int,
    checkpoint: Path,
    allow_rejected_model_output: bool = False,
) -> dict[str, Any]:
    if output_field == "artifact_path":
        raise BatchPredictError(
            f"{task.source_path}:{task.source_line}: text-only checkpoint decode cannot produce artifact_path"
        )
    row = {
        "schema": harness.PREDICTION_SCHEMA,
        "schema_version": harness.SCHEMA_VERSION,
        "created_at": harness.utc_now(),
        "benchmark_id": task.benchmark_id,
        "task_id": task.task_id,
        "task_revision": str(task.row.get("task_revision") or task.row.get("dataset_revision") or "unknown"),
        "dataset_revision": str(task.row.get("dataset_revision") or task.row.get("task_revision") or "unknown"),
        "snapshot_id": str(
            task.row.get("snapshot_id")
            or task.row.get("official_snapshot_id")
            or task.row.get("authorized_snapshot_id")
            or task.row.get("snapshot_sha256")
            or ""
        ),
        "model": cfg.model,
        "backend": BACKEND_NAME,
        "source_task_path": str(task.source_path),
        "source_line": task.source_line,
        "task_row_sha256": harness.stable_hash(task.row),
        "task_file_sha256": harness.file_sha256(task.source_path),
        "request_sha256": harness.stable_hash(harness.model_request(task, cfg)),
        "latency_seconds": round(float(latency_seconds), 6),
        "generation_metadata": {
            "checkpoint_path": str(checkpoint),
            "generated_tokens": int(generated_tokens),
            "decode": "greedy",
            "temperature": 0.0,
        },
    }
    row[output_field] = text
    rejected_outputs = harness.decode_sanity_rejections(row)
    if rejected_outputs:
        row["prediction_quality_status"] = "rejected_model_output"
        row["prediction_quality_reasons"] = rejected_outputs
        row["generation_metadata"]["output_quality_status"] = "rejected_model_output"
        row["generation_metadata"]["output_quality_reasons"] = rejected_outputs
    row["prediction_id"] = harness.stable_hash({key: value for key, value in row.items() if key != "prediction_id"})
    harness.validate_prediction_row(row, allow_rejected_model_output=True)
    return row


def _skipped_prediction_row(
    task: harness.TaskRecord,
    cfg: harness.GenerateConfig,
    output_field: str,
    *,
    reason: str,
    prompt_tokens: int,
    max_prompt_tokens: int,
    checkpoint: Path,
) -> dict[str, Any]:
    row = {
        "schema": harness.PREDICTION_SCHEMA,
        "schema_version": harness.SCHEMA_VERSION,
        "created_at": harness.utc_now(),
        "benchmark_id": task.benchmark_id,
        "task_id": task.task_id,
        "task_revision": str(task.row.get("task_revision") or task.row.get("dataset_revision") or "unknown"),
        "dataset_revision": str(task.row.get("dataset_revision") or task.row.get("task_revision") or "unknown"),
        "snapshot_id": str(
            task.row.get("snapshot_id")
            or task.row.get("official_snapshot_id")
            or task.row.get("authorized_snapshot_id")
            or task.row.get("snapshot_sha256")
            or ""
        ),
        "model": cfg.model,
        "backend": BACKEND_NAME,
        "source_task_path": str(task.source_path),
        "source_line": task.source_line,
        "task_row_sha256": harness.stable_hash(task.row),
        "task_file_sha256": harness.file_sha256(task.source_path),
        "request_sha256": harness.stable_hash(harness.model_request(task, cfg)),
        "latency_seconds": 0.0,
        "generation_metadata": {
            "checkpoint_path": str(checkpoint),
            "generated_tokens": 0,
            "decode": "skipped",
            "temperature": 0.0,
            "skipped": True,
            "skip_reason": reason,
            "prompt_tokens": int(prompt_tokens),
            "max_prompt_tokens": int(max_prompt_tokens),
        },
    }
    row[output_field] = f"__OMNICODER_SKIPPED__:{reason}:prompt_tokens={int(prompt_tokens)}"
    row["prediction_id"] = harness.stable_hash({key: value for key, value in row.items() if key != "prediction_id"})
    harness.validate_prediction_row(row)
    return row


def _task_counts(tasks: list[harness.TaskRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.benchmark_id] = counts.get(task.benchmark_id, 0) + 1
    return counts


def _summary(
    args: argparse.Namespace,
    checkpoint: Path,
    task_paths: list[Path],
    tasks: list[harness.TaskRecord],
    rows: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
    saved_preset_name: str,
    prediction_sha256: str,
) -> dict[str, Any]:
    total_generated = 0
    skipped = 0
    skipped_by_reason: dict[str, int] = {}
    for row in rows:
        metadata = row.get("generation_metadata")
        if isinstance(metadata, dict):
            total_generated += int(metadata.get("generated_tokens") or 0)
            if bool(metadata.get("skipped")):
                skipped += 1
                reason = str(metadata.get("skip_reason") or "unknown")
                skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
    return {
        "status": "ok",
        "schema": SUMMARY_SCHEMA,
        "schema_version": harness.SCHEMA_VERSION,
        "created_at": harness.utc_now(),
        "backend": BACKEND_NAME,
        "checkpoint": str(checkpoint),
        "model": str(args.model or checkpoint),
        "preset": str(args.preset),
        "checkpoint_preset": saved_preset_name,
        "tasks": [str(path) for path in task_paths],
        "predictions": str(harness.resolve_path(args.out)),
        "records": len(rows),
        "authorized_tasks": len(tasks),
        "task_mode": "local_public_dev" if bool(args.allow_local_dev_tasks) else "authorized_reportable",
        "official_score": False,
        "backend_counts": {BACKEND_NAME: len(rows)},
        "by_benchmark": _task_counts(tasks),
        "skipped": {
            "records": skipped,
            "by_reason": skipped_by_reason,
        },
        "prediction_sha256": prediction_sha256,
        "distributed": {
            "world_size": int(dist.get_world_size()) if dist.is_initialized() else EXPECTED_SHARDS,
            "pipeline_stage": True,
            "expected_shards": EXPECTED_SHARDS,
            "rank_device_map": str(args.rank_device_map or ""),
            "placement_layer_counts": str(args.placement_layer_counts or ""),
        },
        "decode": {
            "mode": "greedy",
            "temperature": 0.0,
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_output_tokens": int(args.max_output_tokens),
            "generated_tokens": total_generated,
        },
        "runtime": {
            "precision": str(args.precision),
            "init_dtype": str(args.init_dtype),
            "fake_quant": bool(args.fake_quant),
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "records_per_second": round(float(len(rows)) / max(1.0e-6, float(elapsed_seconds)), 6),
        },
    }


def _run_rank0_batch(
    args: argparse.Namespace,
    shard: OmniCoder2026PipelineShard,
    device: torch.device,
    d_model: int,
    vocab_size: int,
    saved_preset_name: str,
) -> dict[str, Any]:
    checkpoint = _checkpoint_dir(args.checkpoint)
    task_paths = harness.task_paths(list(args.tasks or []))
    tasks = harness.load_tasks(task_paths, allow_local_dev=bool(args.allow_local_dev_tasks))
    tokenizer = get_text_tokenizer(prefer_hf=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos = int(eos_id) if isinstance(eos_id, int) else None
    cfg = _generation_config(args, checkpoint)
    hidden_dtype_name = str(args.init_dtype if str(args.init_dtype or "auto").lower() != "auto" else args.precision)
    hidden_dtype = _dtype_from_name(hidden_dtype_name)
    rows: list[dict[str, Any]] = []
    out_path = harness.resolve_path(args.out)
    if out_path.exists() and not bool(args.force):
        raise BatchPredictError(f"output already exists; pass --force to overwrite: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress_tasks = int(args.progress_tasks or 0)
    rejected_failure = ""
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks, 1):
            output_field = harness.output_field_for_task(task)
            task_started = time.perf_counter()
            prompt = harness.prompt_from_task(task.row)
            prompt_tokens = len(tokenizer.encode(prompt))
            skipped_reason = ""
            if prompt_tokens > int(args.max_prompt_tokens):
                _broadcast_task_header(device, True, 0)
                row = _skipped_prediction_row(
                    task,
                    cfg,
                    output_field,
                    reason="prompt_over_max_prompt_tokens",
                    prompt_tokens=prompt_tokens,
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    checkpoint=checkpoint,
                )
                skipped_reason = "prompt_over_max_prompt_tokens"
            else:
                _broadcast_task_header(device, True, int(args.max_output_tokens))
                text, generated_tokens = _decode_rank0(
                    shard,
                    prompt,
                    tokenizer=tokenizer,
                    eos_id=eos,
                    max_new_tokens=int(args.max_output_tokens),
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    device=device,
                    hidden_dtype=hidden_dtype,
                    d_model=d_model,
                    vocab_size=vocab_size,
                    precision=str(args.precision),
                )
                row = _prediction_row(
                    task,
                    cfg,
                    output_field,
                    text,
                    latency_seconds=time.perf_counter() - task_started,
                    generated_tokens=generated_tokens,
                    checkpoint=checkpoint,
                    allow_rejected_model_output=bool(args.allow_rejected_model_output),
                )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":")) + "\n")
            handle.flush()
            if row.get("prediction_quality_status") == "rejected_model_output" and not bool(args.allow_rejected_model_output):
                rejected_failure = (
                    f"{task.source_path}:{task.source_line}: greedy decode failed sanity gate: "
                    f"{row.get('prediction_quality_reasons')}"
                )
                break
            if progress_tasks and (index % progress_tasks) == 0:
                elapsed = max(1.0e-6, time.perf_counter() - started)
                event = {
                    "event": "pipeline_checkpoint_batch_predict_progress",
                    "records": index,
                    "total": len(tasks),
                    "elapsed_sec": round(elapsed, 3),
                    "records_per_sec": round(index / elapsed, 6),
                    "benchmark_id": task.benchmark_id,
                    "task_id": task.task_id,
                }
                if skipped_reason:
                    event.update(
                        {
                            "skipped": True,
                            "skip_reason": skipped_reason,
                            "prompt_tokens": int(prompt_tokens),
                            "max_prompt_tokens": int(args.max_prompt_tokens),
                        }
                    )
                print(
                    json.dumps(
                        event,
                        sort_keys=True,
                    ),
                    flush=True,
                )
    _broadcast_task_header(device, False, 0)
    if rejected_failure:
        raise BatchPredictError(rejected_failure)
    prediction_sha256 = harness.file_sha256(out_path)
    summary = _summary(
        args,
        checkpoint,
        task_paths,
        tasks,
        rows,
        elapsed_seconds=time.perf_counter() - started,
        saved_preset_name=saved_preset_name,
        prediction_sha256=prediction_sha256,
    )
    if args.summary:
        harness.write_json(harness.resolve_path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)
    return summary


def _run_nonzero_batch(
    args: argparse.Namespace,
    shard: OmniCoder2026PipelineShard,
    device: torch.device,
    d_model: int,
) -> None:
    hidden_dtype_name = str(args.init_dtype if str(args.init_dtype or "auto").lower() != "auto" else args.precision)
    hidden_dtype = _dtype_from_name(hidden_dtype_name)
    while True:
        active, max_new_tokens = _broadcast_task_header(device, None)
        if not active:
            break
        _decode_nonzero(
            shard,
            max_new_tokens=max_new_tokens,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            precision=str(args.precision),
        )


def _worker_main(args: argparse.Namespace) -> int:
    shard, device, d_model, vocab_size, saved_preset_name = _build_shard(args)
    rank = int(dist.get_rank())
    if rank == 0:
        _run_rank0_batch(args, shard, device, d_model, vocab_size, saved_preset_name)
    else:
        _run_nonzero_batch(args, shard, device, d_model)
    return 0


def _torchrun_world_size(checkpoint: Path, manifest: dict[str, Any], explicit: int) -> int:
    nproc = int(explicit or 0)
    if nproc <= 0:
        nproc = int(manifest.get("world_size") or len(_rank_files(checkpoint)))
    if nproc != EXPECTED_SHARDS:
        raise BatchPredictError(f"--nproc-per-node must be {EXPECTED_SHARDS} for this batch predictor, got {nproc}")
    return nproc


def _preflight_parent(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    checkpoint = _checkpoint_dir(args.checkpoint)
    manifest = _load_manifest(checkpoint)
    nproc = _torchrun_world_size(checkpoint, manifest, int(args.nproc_per_node or 0))
    harness.task_paths(list(args.tasks or []))
    out = harness.resolve_path(args.out)
    if out.exists() and not bool(args.force):
        raise BatchPredictError(f"output already exists; pass --force to overwrite: {out}")
    summary = str(args.summary or "").strip()
    if summary:
        summary_path = harness.resolve_path(summary)
        if summary_path.exists() and not bool(args.force):
            raise BatchPredictError(f"summary already exists; pass --force to overwrite: {summary_path}")
    return checkpoint, manifest, nproc


def _parent_main(args: argparse.Namespace) -> int:
    _checkpoint, _manifest, nproc = _preflight_parent(args)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "--max_restarts",
        "0",
        "-m",
        "omnicoder.eval.pipeline_checkpoint_batch_predict_2026",
        "--worker",
        "--checkpoint",
        str(args.checkpoint),
        "--out",
        str(args.out),
        "--preset",
        str(args.preset),
        "--rank-device-map",
        str(args.rank_device_map or ""),
        "--placement-layer-counts",
        str(args.placement_layer_counts or ""),
        "--precision",
        str(args.precision),
        "--init-dtype",
        str(args.init_dtype),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--dist-backend",
        str(args.dist_backend),
        "--dist-timeout-seconds",
        str(args.dist_timeout_seconds),
        "--progress-tasks",
        str(args.progress_tasks),
        "--fake-quant-chunk-rows",
        str(args.fake_quant_chunk_rows),
        "--fake-quant-max-full-elements",
        str(args.fake_quant_max_full_elements),
    ]
    if str(args.summary or "").strip():
        cmd.extend(["--summary", str(args.summary)])
    if str(args.model or "").strip():
        cmd.extend(["--model", str(args.model)])
    for task in args.tasks or []:
        cmd.extend(["--tasks", str(task)])
    if bool(args.fake_quant):
        cmd.append("--fake-quant")
    if bool(args.require_target_contract):
        cmd.append("--require-target-contract")
    if bool(args.allow_p40_target_contract_eval):
        cmd.append("--allow-p40-target-contract-eval")
    if bool(args.allow_local_dev_tasks):
        cmd.append("--allow-local-dev-tasks")
    if bool(args.allow_one_token_canary):
        cmd.append("--allow-one-token-canary")
    if bool(args.allow_rejected_model_output):
        cmd.append("--allow-rejected-model-output")
    if bool(args.force):
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=str(repo_root()), check=False)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent distributed batch predictions for 3-shard Omnicoder2026 pipeline checkpoints"
    )
    parser.add_argument("--worker", action="store_true", help="Internal torchrun worker mode")
    parser.add_argument("--checkpoint", required=True, help="Complete 3-shard pipeline checkpoint directory")
    parser.add_argument("--tasks", action="append", required=True, help="Reportable/public-dev task JSONL file or directory; repeatable")
    parser.add_argument("--out", required=True, help="Prediction JSONL output path")
    parser.add_argument("--summary", default="", help="Optional summary JSON path")
    parser.add_argument("--model", default="", help="Model label written into prediction rows; defaults to --checkpoint")
    parser.add_argument("--preset", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRESET", "omnicoder2026_20b_1m"))
    parser.add_argument("--nproc", "--nproc-per-node", "--nproc_per_node", dest="nproc_per_node", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_NPROC_PER_NODE", "0") or 0))
    parser.add_argument("--rank-map", "--rank-device-map", "--rank_device_map", dest="rank_device_map", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_RANK_DEVICE_MAP", ""))
    parser.add_argument("--placement", "--placement-layer-counts", "--placement_layer_counts", dest="placement_layer_counts", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PLACEMENT_LAYER_COUNTS", ""))
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRECISION", "fp16"))
    parser.add_argument("--init-dtype", "--init_dtype", dest="init_dtype", choices=["auto", "fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_BATCH_INIT_DTYPE", "auto"))
    parser.add_argument("--max-prompt-tokens", "--max_prompt_tokens", dest="max_prompt_tokens", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_MAX_PROMPT_TOKENS", "4096") or 4096))
    parser.add_argument("--max-output-tokens", "--max_output_tokens", "--max-new-tokens", "--max_new_tokens", dest="max_output_tokens", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_MAX_OUTPUT_TOKENS", "256") or 256))
    parser.add_argument("--allow-one-token-canary", "--allow_one_token_canary", dest="allow_one_token_canary", action="store_true", help="Explicitly allow <=1 output token canary runs. Real benchmark/eval runs should not use this.")
    parser.add_argument("--allow-rejected-model-output", "--allow_rejected_model_output", dest="allow_rejected_model_output", action="store_true", help="Debug-only: write rejected/junk model outputs instead of failing nonzero.")
    parser.add_argument("--fake-quant", "--fake_quant", dest="fake_quant", action="store_true")
    parser.add_argument("--fake-quant-chunk-rows", "--fake_quant_chunk_rows", dest="fake_quant_chunk_rows", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_FAKE_QUANT_CHUNK_ROWS", "0") or 0))
    parser.add_argument("--fake-quant-max-full-elements", "--fake_quant_max_full_elements", dest="fake_quant_max_full_elements", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_FAKE_QUANT_MAX_FULL_ELEMENTS", "0") or 0))
    parser.add_argument("--dist-backend", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_DIST_BACKEND", "auto"))
    parser.add_argument("--dist-timeout-seconds", "--dist_timeout_seconds", dest="dist_timeout_seconds", type=int, default=int(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200") or 7200))
    parser.add_argument("--progress-tasks", "--progress_tasks", dest="progress_tasks", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_PROGRESS_TASKS", "1") or 1))
    parser.add_argument("--require-target-contract", "--require_target_contract", dest="require_target_contract", action="store_true")
    parser.add_argument("--allow-p40-target-contract-eval", "--allow_p40_target_contract_eval", dest="allow_p40_target_contract_eval", action="store_true")
    parser.add_argument(
        "--allow-local-dev-tasks",
        "--allow_local_dev_tasks",
        dest="allow_local_dev_tasks",
        action="store_true",
        help="Accept reportable=false public-dev/local-regression task rows; outputs remain non-reportable.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing --out/--summary")
    return parser


def _running_under_torchrun() -> bool:
    return bool(os.getenv("RANK")) and bool(os.getenv("WORLD_SIZE"))


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_prompt_tokens) <= 0:
        raise BatchPredictError("--max-prompt-tokens must be positive")
    if int(args.max_output_tokens) <= 0:
        raise BatchPredictError("--max-output-tokens must be positive")
    if int(args.max_output_tokens) <= 1 and not bool(args.allow_one_token_canary):
        raise BatchPredictError("--max-output-tokens <= 1 is a canary-only setting; pass --allow-one-token-canary only for explicit non-reportable smoke runs")
    if int(args.dist_timeout_seconds) <= 0:
        raise BatchPredictError("--dist-timeout-seconds must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        if bool(args.worker) or _running_under_torchrun():
            return _worker_main(args)
        return _parent_main(args)
    except (BatchPredictError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        payload = {"status": "error", "error": str(exc), "runner": BACKEND_NAME}
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

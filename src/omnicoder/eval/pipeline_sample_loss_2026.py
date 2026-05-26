from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
from omnicoder.eval.sample_loss_2026 import (
    _add_loss,
    _candidate_data_files,
    _chunks,
    _finalize,
    _json_default,
    _new_bucket,
    _read_jsonl,
    _record_ids,
    _record_modality,
)
from omnicoder.modeling.omnicoder2026 import OmniCoder2026Config
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


def _set_fake_quant_env(fake_quant_chunk_rows: int, fake_quant_max_full_elements: int) -> None:
    if int(fake_quant_chunk_rows or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_CHUNK_ROWS"] = str(int(fake_quant_chunk_rows))
    if int(fake_quant_max_full_elements or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS"] = str(int(fake_quant_max_full_elements))


def _checkpoint_train_args(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    train_args = manifest.get("train_args")
    return train_args if isinstance(train_args, dict) else {}


def _build_shard(args: argparse.Namespace) -> tuple[OmniCoder2026PipelineShard, torch.device, int, int]:
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_seconds = int(getattr(args, "dist_timeout_seconds", 0) or os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200"))
        dist.init_process_group(backend=backend, timeout=_dt.timedelta(seconds=max(1, timeout_seconds)))
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    device = rank_device(rank, args.rank_device_map)
    _set_fake_quant_env(int(args.fake_quant_chunk_rows or 0), int(args.fake_quant_max_full_elements or 0))
    train_args = _checkpoint_train_args(args.checkpoint)
    preset = get_omnicoder2026_preset(args.preset)
    kwargs = preset_to_model_kwargs(preset)
    kwargs["fake_quant"] = bool(args.fake_quant or train_args.get("fake_quant"))
    kwargs["tie_embeddings"] = False
    cfg = OmniCoder2026Config(**kwargs)
    ranges = (
        stage_ranges(int(cfg.n_layers), str(args.placement_layer_counts))
        if str(args.placement_layer_counts or "").strip()
        else stage_ranges(int(cfg.n_layers), str(train_args.get("placement_layer_counts") or ""))
    )
    if len(ranges) != world_size:
        raise ValueError(f"world_size={world_size} must match pipeline ranges {ranges}")
    spec = shard_spec(rank, ranges)
    validate_target_device_placement(args, ranges, spec, device)
    init_dtype_name = str(args.init_dtype or "auto").lower()
    if init_dtype_name == "auto":
        init_dtype_name = str(args.precision or "fp32").lower()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(_dtype_from_name(init_dtype_name))
    try:
        with torch.device(device):
            shard = OmniCoder2026PipelineShard(cfg, spec, checkpoint_blocks=False).to(device)
    finally:
        torch.set_default_dtype(old_dtype)
    load_checkpoint_shard(args.checkpoint, shard, optimizer=None, preset=preset, args=None)
    shard.eval()
    return shard, device, int(cfg.d_model), int(cfg.vocab_size)


def _broadcast_batch(batch: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    rank = int(dist.get_rank())
    meta = torch.tensor([1 if batch is not None else 0, int(batch.shape[1]) if batch is not None else 0], dtype=torch.long, device=device)
    dist.broadcast(meta, src=0)
    if int(meta[0].item()) == 0:
        return None
    if rank != 0:
        batch = torch.empty((1, int(meta[1].item())), dtype=torch.long, device=device)
    else:
        batch = batch.to(device, non_blocking=True)
    dist.broadcast(batch, src=0)
    return batch


def _pipeline_loss(
    shard: OmniCoder2026PipelineShard,
    batch: torch.Tensor,
    *,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
    lm_loss_chunk_tokens: int,
) -> float:
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    length = int(batch.shape[1])
    with torch.no_grad(), autocast_context(device, precision):
        if world_size == 1:
            hidden = shard(batch)
        elif rank == 0:
            hidden = shard(batch)
            dist.send(hidden.contiguous(), dst=1)
            loss_tensor = torch.empty((), dtype=torch.float32, device=device)
            dist.broadcast(loss_tensor, src=world_size - 1)
            return float(loss_tensor.cpu())
        else:
            hidden = torch.empty((1, length, d_model), dtype=hidden_dtype, device=device)
            dist.recv(hidden, src=rank - 1)
            hidden = shard(hidden)
            if rank < world_size - 1:
                dist.send(hidden.contiguous(), dst=rank + 1)
                loss_tensor = torch.empty((), dtype=torch.float32, device=device)
                dist.broadcast(loss_tensor, src=world_size - 1)
                return float(loss_tensor.cpu())
        shifted_hidden = hidden[:, :-1, :]
        shifted_labels = batch[:, 1:].to(hidden.device, non_blocking=True)
        loss_tensor = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, shifted_hidden.shape[1], max(1, int(lm_loss_chunk_tokens))):
            end = min(shifted_hidden.shape[1], start + int(lm_loss_chunk_tokens))
            logits = shard.lm_head(shifted_hidden[:, start:end, :]).float()
            loss_tensor = loss_tensor + F.cross_entropy(
                logits.transpose(1, 2),
                shifted_labels[:, start:end],
                reduction="sum",
            )
        dist.broadcast(loss_tensor, src=world_size - 1)
        return float(loss_tensor.cpu())


def evaluate(args: argparse.Namespace) -> dict[str, Any] | None:
    shard, device, d_model, vocab_size = _build_shard(args)
    rank = int(dist.get_rank())
    files = _candidate_data_files(args.data, args.data_dir, exclude_aggregate_jsonl=bool(args.exclude_aggregate_jsonl))
    if not files:
        raise SystemExit("no JSONL files found; pass --data-dir or repeated --data")
    tokenizer = get_text_tokenizer(prefer_hf=True) if rank == 0 else None
    hidden_dtype_name = str(args.init_dtype if str(args.init_dtype or "auto").lower() != "auto" else args.precision)
    hidden_dtype = _dtype_from_name(hidden_dtype_name)
    overall = _new_bucket()
    by_modality: dict[str, dict[str, Any]] = {}
    file_results: list[dict[str, Any]] = []
    if rank == 0:
        records_seen = 0
        chunks_seen = 0
        progress_records = max(0, int(getattr(args, "progress_records", 0) or 0))
        started_at = time.time()
        for path in files:
            file_bucket = _new_bucket(str(path))
            for record in _read_jsonl(path, int(args.max_records_per_file)):
                modality = _record_modality(record, path.stem)
                ids = _record_ids(record, tokenizer, vocab_size)
                if len(ids) < 2:
                    continue
                records_seen += 1
                file_bucket["records"] += 1
                overall["records"] += 1
                modality_bucket = by_modality.setdefault(modality, _new_bucket())
                modality_bucket["records"] += 1
                file_modality = file_bucket["modalities"].setdefault(modality, _new_bucket())
                file_modality["records"] += 1
                for chunk in _chunks(ids, int(args.seq_len)):
                    batch = torch.tensor([chunk], dtype=torch.long, device=device)
                    batch = _broadcast_batch(batch, device)
                    if batch is None:
                        raise RuntimeError("rank 0 unexpectedly received evaluation stop marker")
                    token_count = max(0, len(chunk) - 1)
                    loss_sum = _pipeline_loss(
                        shard,
                        batch,
                        device=device,
                        hidden_dtype=hidden_dtype,
                        d_model=d_model,
                        precision=str(args.precision),
                        lm_loss_chunk_tokens=int(args.lm_loss_chunk_tokens),
                    )
                    _add_loss(file_bucket, loss_sum, token_count)
                    _add_loss(file_modality, loss_sum, token_count)
                    _add_loss(modality_bucket, loss_sum, token_count)
                    _add_loss(overall, loss_sum, token_count)
                    chunks_seen += 1
                if progress_records and (records_seen % progress_records) == 0:
                    elapsed = max(1e-6, time.time() - started_at)
                    print(
                        json.dumps(
                            {
                                "event": "pipeline_sample_loss_progress",
                                "records": records_seen,
                                "chunks": chunks_seen,
                                "elapsed_sec": round(elapsed, 3),
                                "records_per_sec": round(records_seen / elapsed, 6),
                                "current_file": str(path),
                                "modality": modality,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            _finalize(file_bucket)
            file_results.append(file_bucket)
        _broadcast_batch(None, device)
        _finalize(overall)
        for bucket in by_modality.values():
            _finalize(bucket)
        return {
            "schema": "omnicoder.pipeline_sample_loss_2026.v1",
            "checkpoint": str(args.checkpoint),
            "profile": str(args.preset),
            "seq_len": int(args.seq_len),
            "max_records_per_file": int(args.max_records_per_file),
            "distributed": {"world_size": int(dist.get_world_size()), "pipeline_stage": True},
            "files": file_results,
            "modalities": by_modality,
            "overall": overall,
        }
    while True:
        batch = _broadcast_batch(None, device)
        if batch is None:
            break
        _pipeline_loss(
            shard,
            batch,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            precision=str(args.precision),
            lm_loss_chunk_tokens=int(args.lm_loss_chunk_tokens),
        )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed sample-loss eval for Omnicoder2026 pipeline-stage sharded checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--exclude-aggregate-jsonl", action="store_true")
    parser.add_argument("--preset", default="omnicoder2026_20b_1m")
    parser.add_argument("--rank_device_map", default="")
    parser.add_argument("--placement_layer_counts", default="")
    parser.add_argument("--dist-timeout-seconds", "--dist_timeout_seconds", dest="dist_timeout_seconds", type=int, default=int(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200")))
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=1024)
    parser.add_argument("--max-records-per-file", "--max_records_per_file", dest="max_records_per_file", type=int, default=32)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--init-dtype", "--init_dtype", dest="init_dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--fake_quant", action="store_true")
    parser.add_argument("--fake-quant-chunk-rows", "--fake_quant_chunk_rows", dest="fake_quant_chunk_rows", type=int, default=0)
    parser.add_argument("--fake-quant-max-full-elements", "--fake_quant_max_full_elements", dest="fake_quant_max_full_elements", type=int, default=0)
    parser.add_argument("--lm-loss-chunk-tokens", "--lm_loss_chunk_tokens", dest="lm_loss_chunk_tokens", type=int, default=int(os.getenv("OMNICODER2026_LM_LOSS_CHUNK_TOKENS", "128") or 128))
    parser.add_argument("--progress-records", "--progress_records", dest="progress_records", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_EVAL_PROGRESS_RECORDS", "4") or 4))
    parser.add_argument("--require_target_contract", action="store_true")
    parser.add_argument("--allow-p40-target-contract-eval", "--allow_p40_target_contract_eval", dest="allow_p40_target_contract_eval", action="store_true")
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = evaluate(args)
        if result is not None:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
            print(json.dumps(result, sort_keys=True, default=_json_default), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

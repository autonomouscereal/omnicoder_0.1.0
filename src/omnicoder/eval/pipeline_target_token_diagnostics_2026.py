from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from omnicoder.eval.pipeline_checkpoint_batch_predict_2026 import (
    _build_shard,
    _checkpoint_dir,
    _load_manifest,
    _torchrun_world_size,
)
from omnicoder.eval.sample_loss_2026 import (
    _candidate_data_files,
    _read_jsonl,
    _record_modality,
)
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.tokenization.text_range_2026 import effective_text_token_range, tokenizer_vocab_size
from omnicoder.training.pipeline_pretrain_2026_dense import (
    autocast_context,
    record_ids_labels_weight,
)
from omnicoder.training.pretrain_2026_dense import _dtype_from_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


SCHEMA = "omnicoder.pipeline_target_token_diagnostics_2026.v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _sanitize_ids_labels(ids: list[int], labels: list[int], vocab_size: int) -> tuple[list[int], list[int]]:
    cleaned_ids: list[int] = []
    cleaned_labels: list[int] = []
    for token, label in zip(ids, labels):
        token_id = int(token)
        if token_id < 0 or token_id >= int(vocab_size):
            token_id = 1
        cleaned_ids.append(token_id)
        try:
            label_id = int(label)
        except Exception:
            label_id = -100
        cleaned_labels.append(token_id if label_id >= 0 and token_id != 0 else -100)
    if len(cleaned_labels) < len(cleaned_ids):
        cleaned_labels.extend([-100] * (len(cleaned_ids) - len(cleaned_labels)))
    return cleaned_ids, cleaned_labels


def _decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return ""


def _ledger_range_for_token(token_id: int, fallback_range: tuple[int, int] | None = None) -> tuple[int, int] | None:
    try:
        token_range = DEFAULT_LEDGER.lookup(int(token_id))
        return int(token_range.begin), int(token_range.end)
    except Exception:
        return fallback_range


def _ledger_family_for_token(token_id: int) -> str:
    try:
        return str(DEFAULT_LEDGER.lookup(int(token_id)).name)
    except Exception:
        return "unknown"


def _topk(logits: torch.Tensor, tokenizer: Any, k: int, token_range: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    scores = logits.float()
    if token_range is not None:
        lo, hi = int(token_range[0]), min(int(token_range[1]), int(scores.numel()))
        masked = scores.new_full(scores.shape, float("-inf"))
        masked[lo:hi] = scores[lo:hi]
        scores = masked
    count = max(1, min(int(k), int(scores.numel())))
    values, ids = torch.topk(scores, k=count)
    return [
        {"token_id": int(token_id), "logit": float(value), "decoded": _decode_one(tokenizer, int(token_id))}
        for token_id, value in zip(ids.detach().cpu().tolist(), values.detach().cpu().tolist())
    ]


def _token_rank(logits: torch.Tensor, target_id: int, token_range: tuple[int, int] | None = None) -> int:
    scores = logits.float()
    target = int(target_id)
    if token_range is not None:
        lo, hi = int(token_range[0]), min(int(token_range[1]), int(scores.numel()))
        if target < lo or target >= hi:
            return int(scores.numel())
        scores = scores[lo:hi]
        target_score = scores[target - lo]
    else:
        target_score = scores[target]
    return int(torch.count_nonzero(scores > target_score).detach().cpu().item()) + 1


def _ce(logits: torch.Tensor, target_id: int) -> float:
    target = torch.tensor([int(target_id)], dtype=torch.long, device=logits.device)
    return float(F.cross_entropy(logits.float().view(1, -1), target, reduction="mean").detach().cpu())


def _broadcast_batch_labels(
    batch: torch.Tensor | None,
    labels: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    rank = int(dist.get_rank())
    active = batch is not None and labels is not None
    length = int(batch.shape[1]) if active else 0
    meta = torch.tensor([1 if active else 0, length], dtype=torch.long, device=device)
    dist.broadcast(meta, src=0)
    if int(meta[0].item()) == 0:
        return None, None
    if rank != 0:
        batch = torch.empty((1, int(meta[1].item())), dtype=torch.long, device=device)
        labels = torch.empty((1, int(meta[1].item())), dtype=torch.long, device=device)
    else:
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    dist.broadcast(batch, src=0)
    dist.broadcast(labels, src=0)
    return batch, labels


def _broadcast_ids(batch: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    rank = int(dist.get_rank())
    active = batch is not None
    length = int(batch.shape[1]) if active else 0
    meta = torch.tensor([1 if active else 0, length], dtype=torch.long, device=device)
    dist.broadcast(meta, src=0)
    if int(meta[0].item()) == 0:
        return None
    if rank != 0:
        batch = torch.empty((1, int(meta[1].item())), dtype=torch.long, device=device)
    else:
        batch = batch.to(device, non_blocking=True)
    dist.broadcast(batch, src=0)
    return batch


def _pipeline_hidden(
    shard: Any,
    batch: torch.Tensor,
    *,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
) -> torch.Tensor | None:
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    length = int(batch.shape[1])
    with torch.no_grad(), autocast_context(device, precision):
        if world_size == 1:
            return shard(batch)
        if rank == 0:
            hidden = shard(batch)
            dist.send(hidden.contiguous(), dst=1)
            return None
        hidden = torch.empty((1, length, d_model), dtype=hidden_dtype, device=device)
        dist.recv(hidden, src=rank - 1)
        hidden = shard(hidden)
        if rank < world_size - 1:
            dist.send(hidden.contiguous(), dst=rank + 1)
            return None
        return hidden


def _pipeline_logits(
    shard: Any,
    batch: torch.Tensor,
    *,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
) -> torch.Tensor | None:
    hidden = _pipeline_hidden(
        shard,
        batch,
        device=device,
        hidden_dtype=hidden_dtype,
        d_model=d_model,
        precision=precision,
    )
    if hidden is None:
        return None
    with torch.no_grad(), autocast_context(device, precision):
        return shard.lm_head(hidden[:, :-1, :]).float()


def _pipeline_last_logits(
    shard: Any,
    batch: torch.Tensor,
    *,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
) -> torch.Tensor | None:
    hidden = _pipeline_hidden(
        shard,
        batch,
        device=device,
        hidden_dtype=hidden_dtype,
        d_model=d_model,
        precision=precision,
    )
    if hidden is None:
        return None
    with torch.no_grad(), autocast_context(device, precision):
        return shard.lm_head(hidden[:, -1, :]).float()


def _diagnose_on_final_rank(
    *,
    record_meta: dict[str, Any],
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    prefix_logits: torch.Tensor | None,
    tokenizer: Any,
    text_range: tuple[int, int],
    top_k: int,
    max_positions: int,
) -> dict[str, Any] | None:
    if int(dist.get_rank()) != int(dist.get_world_size()) - 1:
        return None
    if logits is None:
        raise RuntimeError("final rank did not receive full-sequence logits")
    shifted_labels = labels[:, 1:].to(logits.device)
    valid_positions = torch.nonzero(shifted_labels[0].ge(0), as_tuple=False).flatten()
    token_losses: list[float] = []
    target_ranks: list[int] = []
    target_text_ranks: list[int] = []
    rows: list[dict[str, Any]] = []
    for shifted_pos in valid_positions.detach().cpu().tolist():
        pos = int(shifted_pos)
        target_id = int(shifted_labels[0, pos].item())
        row_logits = logits[0, pos, :]
        loss = _ce(row_logits, target_id)
        ledger_range = _ledger_range_for_token(target_id, text_range)
        ledger_family = _ledger_family_for_token(target_id)
        rank = _token_rank(row_logits, target_id, ledger_range)
        text_rank = _token_rank(row_logits, target_id, text_range)
        token_losses.append(loss)
        target_ranks.append(rank)
        target_text_ranks.append(text_rank)
        if len(rows) < int(max_positions):
            rows.append(
                {
                    "token_index": pos + 1,
                    "predict_from_index": pos,
                    "target_token_id": target_id,
                    "target_decoded": _decode_one(tokenizer, target_id),
                    "target_ledger_family": ledger_family,
                    "loss": loss,
                    "target_rank_ledger_range": rank,
                    "target_rank_text_range": text_rank,
                    "top_ledger_tokens": _topk(row_logits, tokenizer, int(top_k), ledger_range),
                    "top_text_tokens": _topk(row_logits, tokenizer, int(top_k), text_range),
                }
            )
    first_target_index = int(valid_positions[0].item()) + 1 if valid_positions.numel() else None
    prefix_report: dict[str, Any] = {"status": "skipped", "reason": "no_target_tokens"}
    if first_target_index is not None:
        first_target_id = int(shifted_labels[0, int(first_target_index) - 1].item())
        full_logits = logits[0, int(first_target_index) - 1, :]
        first_ledger_range = _ledger_range_for_token(first_target_id, text_range)
        first_ledger_family = _ledger_family_for_token(first_target_id)
        prefix_report = {
            "status": "missing_prefix_logits",
            "first_target_index": int(first_target_index),
            "target_token_id": first_target_id,
            "target_decoded": _decode_one(tokenizer, first_target_id),
            "target_ledger_family": first_ledger_family,
        }
        if prefix_logits is not None:
            prefix_vec = prefix_logits[0]
            full_top = _topk(full_logits, tokenizer, int(top_k), first_ledger_range)
            prefix_top = _topk(prefix_vec, tokenizer, int(top_k), first_ledger_range)
            full_text_top = _topk(full_logits, tokenizer, int(top_k), text_range)
            prefix_text_top = _topk(prefix_vec, tokenizer, int(top_k), text_range)
            target_logit_delta = float((full_logits[first_target_id] - prefix_vec[first_target_id]).detach().cpu())
            prefix_report = {
                "status": "ok",
                "first_target_index": int(first_target_index),
                "target_token_id": first_target_id,
                "target_decoded": _decode_one(tokenizer, first_target_id),
                "target_ledger_family": first_ledger_family,
                "full_sequence": {
                    "loss": _ce(full_logits, first_target_id),
                    "target_rank_ledger_range": _token_rank(full_logits, first_target_id, first_ledger_range),
                    "target_rank_text_range": _token_rank(full_logits, first_target_id, text_range),
                    "top_ledger_tokens": full_top,
                    "top_text_tokens": full_text_top,
                    "selected_token_id": int(full_top[0]["token_id"]) if full_top else None,
                    "selected_decoded": str(full_top[0]["decoded"]) if full_top else "",
                },
                "prefix_only": {
                    "loss": _ce(prefix_vec, first_target_id),
                    "target_rank_ledger_range": _token_rank(prefix_vec, first_target_id, first_ledger_range),
                    "target_rank_text_range": _token_rank(prefix_vec, first_target_id, text_range),
                    "top_ledger_tokens": prefix_top,
                    "top_text_tokens": prefix_text_top,
                    "selected_token_id": int(prefix_top[0]["token_id"]) if prefix_top else None,
                    "selected_decoded": str(prefix_top[0]["decoded"]) if prefix_top else "",
                },
                "target_logit_full_minus_prefix": target_logit_delta,
                "top1_matches_target": bool(prefix_top and int(prefix_top[0]["token_id"]) == int(first_target_id)),
                "top1_full_prefix_match": bool(
                    full_top and prefix_top and int(full_top[0]["token_id"]) == int(prefix_top[0]["token_id"])
                ),
            }
    target_count = len(token_losses)
    mean_loss = float(sum(token_losses) / target_count) if target_count else math.nan
    mean_rank = float(sum(target_ranks) / target_count) if target_count else math.nan
    mean_text_rank = float(sum(target_text_ranks) / target_count) if target_count else math.nan
    return {
        **record_meta,
        "target_token_count": int(target_count),
        "target_loss_mean": mean_loss,
        "target_loss_max": float(max(token_losses)) if token_losses else math.nan,
        "target_ppl_mean": float(math.exp(min(20.0, mean_loss))) if token_losses else math.nan,
        "target_rank_mean_ledger_range": mean_rank,
        "target_rank_max_ledger_range": int(max(target_ranks)) if target_ranks else None,
        "target_rank_mean_text_range": mean_text_rank,
        "target_rank_max_text_range": int(max(target_text_ranks)) if target_text_ranks else None,
        "first_target": prefix_report,
        "positions": rows,
    }


def _rank0_collect_final_diagnostic(
    *,
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    prefix_logits: torch.Tensor | None,
    tokenizer: Any,
    text_range: tuple[int, int],
    top_k: int,
    max_positions: int,
) -> dict[str, Any]:
    world_size = int(dist.get_world_size())
    if world_size == 1:
        result = _diagnose_on_final_rank(
            record_meta={},
            logits=logits,
            labels=labels,
            prefix_logits=prefix_logits,
            tokenizer=tokenizer,
            text_range=text_range,
            top_k=top_k,
            max_positions=max_positions,
        )
    else:
        objects: list[Any] = [None]
        dist.broadcast_object_list(objects, src=world_size - 1)
        result = objects[0]
    if not isinstance(result, dict):
        raise RuntimeError("final rank did not return a diagnostic record")
    return result


def _bucket_add(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["records"] = int(bucket.get("records") or 0) + 1
    target_count = int(record.get("target_token_count") or 0)
    bucket["target_tokens"] = int(bucket.get("target_tokens") or 0) + target_count
    if target_count:
        bucket["loss_weighted_sum"] = float(bucket.get("loss_weighted_sum") or 0.0) + float(record["target_loss_mean"]) * target_count
        bucket["rank_weighted_sum"] = float(bucket.get("rank_weighted_sum") or 0.0) + float(record["target_rank_mean_text_range"]) * target_count
        bucket["ledger_rank_weighted_sum"] = float(bucket.get("ledger_rank_weighted_sum") or 0.0) + float(
            record.get("target_rank_mean_ledger_range") or 0.0
        ) * target_count
    first = record.get("first_target") if isinstance(record.get("first_target"), dict) else {}
    if first.get("status") == "ok":
        bucket["first_targets"] = int(bucket.get("first_targets") or 0) + 1
        if bool(first.get("top1_matches_target")):
            bucket["first_target_top1"] = int(bucket.get("first_target_top1") or 0) + 1
        prefix = first.get("prefix_only") if isinstance(first.get("prefix_only"), dict) else {}
        full = first.get("full_sequence") if isinstance(first.get("full_sequence"), dict) else {}
        if prefix:
            bucket["first_prefix_loss_sum"] = float(bucket.get("first_prefix_loss_sum") or 0.0) + float(prefix.get("loss") or 0.0)
        if full:
            bucket["first_full_loss_sum"] = float(bucket.get("first_full_loss_sum") or 0.0) + float(full.get("loss") or 0.0)


def _bucket_finalize(bucket: dict[str, Any]) -> None:
    tokens = int(bucket.get("target_tokens") or 0)
    first = int(bucket.get("first_targets") or 0)
    if tokens:
        bucket["target_loss_mean"] = float(bucket.get("loss_weighted_sum") or 0.0) / float(tokens)
        bucket["target_rank_mean_text_range"] = float(bucket.get("rank_weighted_sum") or 0.0) / float(tokens)
        bucket["target_rank_mean_ledger_range"] = float(bucket.get("ledger_rank_weighted_sum") or 0.0) / float(tokens)
        bucket["target_ppl_mean"] = float(math.exp(min(20.0, bucket["target_loss_mean"])))
    if first:
        bucket["first_target_top1_rate"] = float(bucket.get("first_target_top1") or 0) / float(first)
        bucket["first_prefix_loss_mean"] = float(bucket.get("first_prefix_loss_sum") or 0.0) / float(first)
        bucket["first_full_loss_mean"] = float(bucket.get("first_full_loss_sum") or 0.0) / float(first)
    for key in ("loss_weighted_sum", "rank_weighted_sum", "ledger_rank_weighted_sum", "first_prefix_loss_sum", "first_full_loss_sum"):
        bucket.pop(key, None)


def evaluate(args: argparse.Namespace) -> dict[str, Any] | None:
    if not dist.is_initialized():
        backend = str(args.dist_backend or "auto").lower()
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=_dt.timedelta(seconds=max(1, int(args.dist_timeout_seconds))),
        )
    shard, device, d_model, vocab_size, saved_preset_name = _build_shard(args)
    rank = int(dist.get_rank())
    tokenizer = get_text_tokenizer(prefer_hf=True)
    text_range = effective_text_token_range(tokenizer=tokenizer, model_vocab_size=vocab_size)
    hidden_dtype_name = str(args.init_dtype if str(args.init_dtype or "auto").lower() != "auto" else args.precision)
    hidden_dtype = _dtype_from_name(hidden_dtype_name)

    records: list[dict[str, Any]] = []
    modality_buckets: dict[str, dict[str, Any]] = {}
    overall: dict[str, Any] = {"records": 0, "target_tokens": 0, "first_targets": 0, "first_target_top1": 0}
    started = time.perf_counter()

    if rank == 0:
        files = _candidate_data_files(args.data, args.data_dir, exclude_aggregate_jsonl=bool(args.exclude_aggregate_jsonl))
        if not files:
            raise SystemExit("no JSONL files found; pass --data-dir or repeated --data")
        for path in files:
            for line_index, record in enumerate(_read_jsonl(path, int(args.max_records_per_file)), start=1):
                if not isinstance(record, dict):
                    continue
                ids, labels, _weight = record_ids_labels_weight(record, tokenizer)
                ids, labels = _sanitize_ids_labels(ids, labels, int(vocab_size))
                if len(ids) < 2:
                    continue
                if len(ids) > int(args.seq_len):
                    ids = ids[: int(args.seq_len)]
                    labels = labels[: int(args.seq_len)]
                if not any(int(label) >= 0 for label in labels[1:]):
                    continue
                modality = _record_modality(record, path.stem)
                batch = torch.tensor([ids], dtype=torch.long, device=device)
                label_batch = torch.tensor([labels], dtype=torch.long, device=device)
                batch, label_batch = _broadcast_batch_labels(batch, label_batch, device)
                if batch is None or label_batch is None:
                    raise RuntimeError("unexpected stop marker on rank 0")
                logits = _pipeline_logits(
                    shard,
                    batch,
                    device=device,
                    hidden_dtype=hidden_dtype,
                    d_model=d_model,
                    precision=str(args.precision),
                )
                first_label = next(index for index, label in enumerate(labels[1:], start=1) if int(label) >= 0)
                prefix_ids = ids[:first_label]
                prefix_batch = torch.tensor([prefix_ids], dtype=torch.long, device=device) if prefix_ids else None
                prefix_batch = _broadcast_ids(prefix_batch, device)
                prefix_logits = None
                if prefix_batch is not None:
                    prefix_logits = _pipeline_last_logits(
                        shard,
                        prefix_batch,
                        device=device,
                        hidden_dtype=hidden_dtype,
                        d_model=d_model,
                        precision=str(args.precision),
                    )
                meta = {
                    "source_path": str(path),
                    "source_line": int(line_index),
                    "modality": modality,
                    "token_count": int(len(ids)),
                    "prompt_token_count_before_first_target": int(first_label),
                }
                result = _rank0_collect_final_diagnostic(
                    logits=logits,
                    labels=label_batch,
                    prefix_logits=prefix_logits,
                    tokenizer=tokenizer,
                    text_range=text_range,
                    top_k=int(args.top_k),
                    max_positions=int(args.max_positions),
                )
                result.update(meta)
                records.append(result)
                bucket = modality_buckets.setdefault(modality, {"records": 0, "target_tokens": 0, "first_targets": 0, "first_target_top1": 0})
                _bucket_add(bucket, result)
                _bucket_add(overall, result)
                if int(args.progress_records or 0) and (len(records) % int(args.progress_records)) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "pipeline_target_token_diagnostics_progress",
                                "records": len(records),
                                "modality": modality,
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        _broadcast_batch_labels(None, None, device)
        for bucket in modality_buckets.values():
            _bucket_finalize(bucket)
        _bucket_finalize(overall)
        return {
            "schema": SCHEMA,
            "status": "ok",
            "training_invoked": False,
            "checkpoint": str(args.checkpoint),
            "preset": str(args.preset),
            "checkpoint_preset": str(saved_preset_name),
            "seq_len": int(args.seq_len),
            "model_vocab_size": int(vocab_size),
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "vocab_size": tokenizer_vocab_size(tokenizer),
            },
            "text_range": [int(text_range[0]), int(text_range[1])],
            "overall": overall,
            "modalities": modality_buckets,
            "records": records,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }

    while True:
        batch, labels = _broadcast_batch_labels(None, None, device)
        if batch is None or labels is None:
            break
        logits = _pipeline_logits(
            shard,
            batch,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            precision=str(args.precision),
        )
        shifted_labels = labels[:, 1:]
        valid = torch.nonzero(shifted_labels[0].ge(0), as_tuple=False).flatten()
        prefix_batch = None
        if valid.numel():
            first_label = int(valid[0].item()) + 1
            if rank == 0:
                raise RuntimeError("rank 0 should not be in worker loop")
        prefix_batch = _broadcast_ids(None, device)
        prefix_logits = None
        if prefix_batch is not None:
            prefix_logits = _pipeline_last_logits(
                shard,
                prefix_batch,
                device=device,
                hidden_dtype=hidden_dtype,
                d_model=d_model,
                precision=str(args.precision),
            )
        result = _diagnose_on_final_rank(
            record_meta={},
            logits=logits,
            labels=labels,
            prefix_logits=prefix_logits,
            tokenizer=tokenizer,
            text_range=text_range,
            top_k=int(args.top_k),
            max_positions=int(args.max_positions),
        )
        objects: list[Any] = [result]
        dist.broadcast_object_list(objects, src=dist.get_world_size() - 1)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-target-token CE/rank diagnostics for sharded Omnicoder pipeline checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--exclude-aggregate-jsonl", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRESET", "omnicoder2026_20b_1m"))
    parser.add_argument("--rank-device-map", "--rank_device_map", dest="rank_device_map", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_RANK_DEVICE_MAP", ""))
    parser.add_argument("--placement-layer-counts", "--placement_layer_counts", dest="placement_layer_counts", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PLACEMENT_LAYER_COUNTS", ""))
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRECISION", "fp16"))
    parser.add_argument("--init-dtype", "--init_dtype", dest="init_dtype", choices=["auto", "fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_BATCH_INIT_DTYPE", "auto"))
    parser.add_argument("--dist-backend", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_DIST_BACKEND", "auto"))
    parser.add_argument("--dist-timeout-seconds", "--dist_timeout_seconds", dest="dist_timeout_seconds", type=int, default=int(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200") or 7200))
    parser.add_argument("--expected-world-size", "--expected_world_size", "--nproc", "--nproc-per-node", "--nproc_per_node", dest="expected_world_size", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_EXPECTED_WORLD_SIZE", "0") or 0))
    parser.add_argument("--seq-len", "--seq_len", dest="seq_len", type=int, default=1024)
    parser.add_argument("--max-records-per-file", "--max_records_per_file", dest="max_records_per_file", type=int, default=0)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=8)
    parser.add_argument("--max-positions", "--max_positions", dest="max_positions", type=int, default=12)
    parser.add_argument("--progress-records", "--progress_records", dest="progress_records", type=int, default=1)
    parser.add_argument("--fake-quant", "--fake_quant", dest="fake_quant", action="store_true")
    parser.add_argument("--fake-quant-chunk-rows", "--fake_quant_chunk_rows", dest="fake_quant_chunk_rows", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_FAKE_QUANT_CHUNK_ROWS", "0") or 0))
    parser.add_argument("--fake-quant-max-full-elements", "--fake_quant_max_full_elements", dest="fake_quant_max_full_elements", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_FAKE_QUANT_MAX_FULL_ELEMENTS", "0") or 0))
    parser.add_argument("--require-target-contract", "--require_target_contract", dest="require_target_contract", action="store_true")
    parser.add_argument("--allow-p40-target-contract-eval", "--allow_p40_target_contract_eval", dest="allow_p40_target_contract_eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = _checkpoint_dir(args.checkpoint)
    manifest = _load_manifest(checkpoint, expected_world_size=int(args.expected_world_size or 0))
    expected_world_size = _torchrun_world_size(checkpoint, manifest, int(args.expected_world_size or 0))
    actual_world_size = int(os.getenv("WORLD_SIZE", "1"))
    if actual_world_size != expected_world_size:
        raise SystemExit(f"run under torchrun with nproc_per_node={expected_world_size}")
    args.checkpoint = str(checkpoint)
    try:
        result = evaluate(args)
        if result is not None:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=True, sort_keys=True, default=_json_default), flush=True)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

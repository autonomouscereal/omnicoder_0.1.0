from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from omnicoder.eval.pipeline_checkpoint_batch_predict_2026 import (
    _build_shard,
    _broadcast_ids,
    _checkpoint_dir,
    _load_manifest,
    _torchrun_world_size,
)
from omnicoder.tokenization.text_range_2026 import effective_text_token_range, tokenizer_vocab_size
from omnicoder.training.pipeline_pretrain_2026_dense import autocast_context
from omnicoder.training.pretrain_2026_dense import _dtype_from_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _tensor_stats(tensor: torch.Tensor, *, sample_limit: int = 1_000_000) -> dict[str, Any]:
    data = tensor.detach()
    flat = data.reshape(-1)
    stride = max(1, int(flat.numel()) // max(1, int(sample_limit)))
    sample = flat[::stride].float()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "numel": int(data.numel()),
        "sample_count": int(sample.numel()),
        "finite_sample": bool(torch.isfinite(sample).all().item()),
        "mean_sample": float(sample.mean().cpu()) if sample.numel() else 0.0,
        "std_sample": float(sample.std(unbiased=False).cpu()) if sample.numel() else 0.0,
        "absmax_sample": float(sample.abs().max().cpu()) if sample.numel() else 0.0,
    }


def _local_rank_report(shard: Any, device: torch.device) -> dict[str, Any]:
    spec = getattr(shard, "spec")
    state = shard.state_dict()
    tensors: dict[str, Any] = {}
    for name in ("embed.weight", "lm_head.weight"):
        value = state.get(name)
        if isinstance(value, torch.Tensor):
            tensors[name] = _tensor_stats(value)
    block_tensors = sum(1 for key in state if key.startswith("blocks."))
    return {
        "rank": int(dist.get_rank()),
        "device": str(device),
        "stage_index": int(spec.stage_index),
        "layer_start": int(spec.layer_start),
        "layer_end": int(spec.layer_end),
        "has_embed": bool(spec.has_embed),
        "has_head": bool(spec.has_head),
        "block_tensor_count": int(block_tensors),
        "tensors": tensors,
    }


def _topk_payload(logits: torch.Tensor, tokenizer: Any, top_k: int, text_range: tuple[int, int]) -> dict[str, Any]:
    logits = logits.float()
    k = max(1, min(int(top_k), int(logits.numel())))
    raw_values, raw_ids = torch.topk(logits, k=k)
    lo, hi = text_range
    text_logits = logits[int(lo) : int(hi)]
    text_k = max(1, min(k, int(text_logits.numel())))
    text_values, text_rel_ids = torch.topk(text_logits, k=text_k)
    text_ids = text_rel_ids + int(lo)

    def decode_one(token_id: int) -> str:
        try:
            return str(tokenizer.decode([int(token_id)]))
        except Exception:
            return ""

    return {
        "raw_topk": [
            {"token_id": int(tid), "logit": float(val), "decoded": decode_one(int(tid))}
            for tid, val in zip(raw_ids.detach().cpu().tolist(), raw_values.detach().cpu().tolist())
        ],
        "text_topk": [
            {"token_id": int(tid), "logit": float(val), "decoded": decode_one(int(tid))}
            for tid, val in zip(text_ids.detach().cpu().tolist(), text_values.detach().cpu().tolist())
        ],
    }


def _next_token_probe(
    shard: Any,
    batch: torch.Tensor,
    *,
    tokenizer: Any,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
    text_range: tuple[int, int],
    top_k: int,
) -> tuple[int, dict[str, Any]]:
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    length = int(batch.shape[1])
    details: dict[str, Any] | None = None
    with torch.no_grad(), autocast_context(device, precision):
        if rank == 0:
            hidden = shard(batch)
            dist.send(hidden.contiguous(), dst=1)
            token = torch.empty((1,), dtype=torch.long, device=device)
            dist.broadcast(token, src=world_size - 1)
        else:
            hidden = torch.empty((1, length, d_model), dtype=hidden_dtype, device=device)
            dist.recv(hidden, src=rank - 1)
            hidden = shard(hidden)
            if rank < world_size - 1:
                dist.send(hidden.contiguous(), dst=rank + 1)
                token = torch.empty((1,), dtype=torch.long, device=device)
                dist.broadcast(token, src=world_size - 1)
            else:
                logits = shard.lm_head(hidden[:, -1:, :]).float()[0, 0]
                topk = _topk_payload(logits, tokenizer, top_k, text_range)
                next_id = int(topk["text_topk"][0]["token_id"])
                token = torch.tensor([next_id], dtype=torch.long, device=device)
                details = {
                    "rank": rank,
                    "selected_token_id": next_id,
                    "selected_decoded": topk["text_topk"][0]["decoded"],
                    "text_range": [int(text_range[0]), int(text_range[1])],
                    **topk,
                }
                dist.broadcast(token, src=rank)
    objects: list[Any] = [details]
    dist.broadcast_object_list(objects, src=world_size - 1)
    return int(token.detach().cpu().item()), objects[0]


def _run_worker(args: argparse.Namespace) -> dict[str, Any] | None:
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
    local_report = _local_rank_report(shard, device)
    rank_reports: list[Any] = [None for _ in range(int(dist.get_world_size()))]
    dist.all_gather_object(rank_reports, local_report)
    hidden_dtype_name = str(args.init_dtype if str(args.init_dtype or "auto").lower() != "auto" else args.precision)
    hidden_dtype = _dtype_from_name(hidden_dtype_name)
    prompt_ids = [int(item) for item in tokenizer.encode(str(args.prompt))]
    bad_ids = [item for item in prompt_ids if item < 0 or item >= vocab_size]
    if bad_ids:
        raise ValueError(f"tokenizer produced ids outside model vocab: examples={bad_ids[:8]} vocab_size={vocab_size}")
    generated = list(prompt_ids)
    token_steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(int(args.max_new_tokens)):
        batch = torch.tensor([generated], dtype=torch.long, device=device) if rank == 0 else None
        batch = _broadcast_ids(batch, device)
        next_id, details = _next_token_probe(
            shard,
            batch,
            tokenizer=tokenizer,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            precision=str(args.precision),
            text_range=text_range,
            top_k=int(args.top_k),
        )
        if rank == 0:
            generated.append(next_id)
            token_steps.append({"step": step + 1, **details})
    if rank != 0:
        return None
    new_ids = generated[len(prompt_ids) :]
    return {
        "schema": "omnicoder.pipeline_token_topk_probe_2026.v1",
        "status": "ok",
        "training_invoked": False,
        "checkpoint": str(args.checkpoint),
        "preset": str(args.preset),
        "checkpoint_preset": saved_preset_name,
        "rank_reports": rank_reports,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer_vocab_size(tokenizer),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        "model_vocab_size": int(vocab_size),
        "text_range": [int(text_range[0]), int(text_range[1])],
        "prompt": str(args.prompt),
        "prompt_token_ids": prompt_ids,
        "prompt_roundtrip": tokenizer.decode(prompt_ids),
        "generated_token_ids": new_ids,
        "generated_text": tokenizer.decode(new_ids),
        "full_text": tokenizer.decode(generated),
        "steps": token_steps,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-training top-k/token mapping probe for sharded Omnicoder pipeline checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="Write a tiny Python add function.")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=8)
    parser.add_argument("--preset", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRESET", "omnicoder2026_20b_1m"))
    parser.add_argument("--rank-device-map", "--rank_device_map", dest="rank_device_map", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_RANK_DEVICE_MAP", ""))
    parser.add_argument("--placement-layer-counts", "--placement_layer_counts", dest="placement_layer_counts", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PLACEMENT_LAYER_COUNTS", ""))
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRECISION", "fp16"))
    parser.add_argument("--init-dtype", "--init_dtype", dest="init_dtype", choices=["auto", "fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_BATCH_INIT_DTYPE", "auto"))
    parser.add_argument("--dist-backend", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_DIST_BACKEND", "auto"))
    parser.add_argument("--dist-timeout-seconds", "--dist_timeout_seconds", dest="dist_timeout_seconds", type=int, default=int(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200") or 7200))
    parser.add_argument("--expected-world-size", "--expected_world_size", "--nproc", "--nproc-per-node", "--nproc_per_node", dest="expected_world_size", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_EXPECTED_WORLD_SIZE", "0") or 0))
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
        result = _run_worker(args)
        if result is not None:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
from omnicoder.modeling.omnicoder2026 import OmniCoder2026Block, OmniCoder2026Config, QuantAwareLinear, RMSNorm
from omnicoder.training.pretrain_2026_dense import (
    TARGET_PRESET,
    _ids_from_record,
    _atomic_torch_save,
    _dtype_from_name,
    _is_probe_name,
    _text_from_record,
    _restore_rng_state,
    _rng_state,
    _sha256_file,
    _write_log,
)
from omnicoder.training.simple_tokenizer import get_text_tokenizer

PIPELINE_LOW_MEMORY_OPTIMIZER_NOTE = (
    "PipelineStage training uses a delayed per-rank low-memory update after "
    "the schedule drains all microbatches. It intentionally does not use "
    "post-accumulate hooks because those can step before every pipeline "
    "microbatch has contributed to the global batch."
)


@dataclass(frozen=True)
class PipelineShardSpec:
    stage_index: int
    num_stages: int
    layer_start: int
    layer_end: int
    has_embed: bool
    has_head: bool


def parse_stage_ranges(raw: str, n_layers: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        start_s, end_s = part.split(":", 1)
        ranges.append((int(start_s), int(end_s)))
    if not ranges:
        if int(n_layers) == 64:
            return [(0, 16), (16, 32), (32, 64)]
        if int(n_layers) < 3:
            raise ValueError("pipeline probe requires at least 3 layers")
        first = max(1, int(n_layers) // 3)
        second = max(1, int(n_layers) // 3)
        ranges = [(0, first), (first, first + second), (first + second, int(n_layers))]
    if ranges[0][0] != 0 or ranges[-1][1] != n_layers:
        raise ValueError(f"stage ranges must cover [0,{n_layers}); got {ranges}")
    for prev, current in zip(ranges, ranges[1:], strict=False):
        if prev[1] != current[0]:
            raise ValueError(f"stage ranges must be contiguous; got {ranges}")
    return ranges


def stage_ranges(n_layers: int, placement_layer_counts: str = "") -> list[tuple[int, int]]:
    counts = [int(part.strip()) for part in str(placement_layer_counts or "").split(",") if part.strip()]
    if counts:
        if not counts or any(count <= 0 for count in counts):
            raise ValueError(f"placement_layer_counts must contain positive counts, got {counts}")
        if sum(counts) != int(n_layers):
            raise ValueError(f"placement_layer_counts must sum to {n_layers}, got {counts}")
        ranges: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            end = start + count
            ranges.append((start, end))
            start = end
        return ranges
    return parse_stage_ranges("", int(n_layers))


def shard_spec(rank: int, ranges: list[tuple[int, int]]) -> PipelineShardSpec:
    if rank < 0 or rank >= len(ranges):
        raise ValueError(f"rank {rank} outside stage ranges {ranges}")
    start, end = ranges[rank]
    return PipelineShardSpec(
        stage_index=rank,
        num_stages=len(ranges),
        layer_start=start,
        layer_end=end,
        has_embed=rank == 0,
        has_head=rank == len(ranges) - 1,
    )


class OmniCoder2026PipelineShard(nn.Module):
    """One pipeline stage with checkpoint-compatible local module names."""

    def __init__(self, cfg: OmniCoder2026Config, spec: PipelineShardSpec, *, checkpoint_blocks: bool = False):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model) if spec.has_embed else nn.Identity()
        pattern = list(cfg.layer_pattern)
        blocks: list[nn.Module] = []
        for index in range(int(cfg.n_layers)):
            if spec.layer_start <= index < spec.layer_end:
                blocks.append(OmniCoder2026Block(cfg, pattern[index % len(pattern)]))
            else:
                blocks.append(nn.Identity())
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps) if spec.has_head else nn.Identity()
        self.lm_head = (
            QuantAwareLinear(cfg.d_model, cfg.vocab_size, bias=False, fake_quant=False, group_size=cfg.fake_quant_group_size)
            if spec.has_head
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.has_embed:
            if x.dtype != torch.long:
                x = x.to(dtype=torch.long)
            x = x.remainder(int(self.cfg.vocab_size))
            x = self.embed(x)
        for index in range(self.spec.layer_start, self.spec.layer_end):
            block = self.blocks[index]
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        if self.spec.has_head:
            x = self.norm(x)
        return x

    def chunked_lm_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        chunk_tokens: int = 128,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.spec.has_head:
            raise RuntimeError("LM loss can only be computed on the final pipeline stage")
        if labels.device != hidden.device:
            labels = labels.to(hidden.device, non_blocking=True)
        shifted_hidden = hidden[:, :-1, :]
        shifted_labels = labels[:, 1:]
        if sample_weights is not None:
            weights = sample_weights.to(hidden.device, non_blocking=True).to(dtype=torch.float32).reshape(-1)
            if weights.numel() != shifted_hidden.shape[0]:
                raise ValueError(f"sample_weights batch mismatch: weights={weights.numel()} batch={shifted_hidden.shape[0]}")
            per_sample_sum = hidden.new_zeros((shifted_hidden.shape[0],), dtype=torch.float32)
            per_sample_tokens = hidden.new_zeros((shifted_hidden.shape[0],), dtype=torch.float32)
            for start in range(0, shifted_hidden.shape[1], max(1, int(chunk_tokens))):
                end = min(shifted_hidden.shape[1], start + int(chunk_tokens))
                logits = self.lm_head(shifted_hidden[:, start:end, :])
                token_losses = F.cross_entropy(logits.transpose(1, 2), shifted_labels[:, start:end], reduction="none").float()
                mask = shifted_labels[:, start:end].ne(0).float()
                per_sample_sum = per_sample_sum + (token_losses * mask).sum(dim=1)
                per_sample_tokens = per_sample_tokens + mask.sum(dim=1)
            per_sample = per_sample_sum / per_sample_tokens.clamp_min(1.0)
            return (per_sample * weights).mean().to(dtype=hidden.dtype)
        total_tokens = max(1, int(shifted_labels.numel()))
        loss_sum = hidden.new_zeros(())
        for start in range(0, shifted_hidden.shape[1], max(1, int(chunk_tokens))):
            end = min(shifted_hidden.shape[1], start + int(chunk_tokens))
            logits = self.lm_head(shifted_hidden[:, start:end, :])
            loss_sum = loss_sum + F.cross_entropy(logits.transpose(1, 2), shifted_labels[:, start:end], reduction="sum")
        return loss_sum / float(total_tokens)

    def local_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if not key.endswith("._metadata")}


def load_full_checkpoint_shard(path: str | Path, shard: OmniCoder2026PipelineShard) -> tuple[int, float | None]:
    return load_checkpoint_shard(path, shard)


def _compact_json(value: Any, limit: int = 12000) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)[:limit]


def _reward_value(record: dict[str, Any]) -> float:
    for key in ("reward", "score"):
        value = record.get(key)
        if value is not None:
            try:
                return max(-1.0, min(1.0, float(value)))
            except Exception:
                pass
    verifier = record.get("verifier")
    if isinstance(verifier, dict) and verifier.get("reward") is not None:
        try:
            return max(-1.0, min(1.0, float(verifier["reward"])))
        except Exception:
            pass
    try:
        return max(-1.0, min(1.0, float(record.get("quality_score"))))
    except Exception:
        return 0.5


def _pipeline_record_to_text_and_weight(record: dict[str, Any]) -> tuple[str, float]:
    kind = str(record.get("training_kind") or "").lower()
    prompt = str(record.get("prompt") or "")
    reward = _reward_value(record)
    if kind.endswith("_rlvr"):
        target = {
            "verifier": record.get("verifier", {}),
            "environment": record.get("environment", {}),
            "reward": record.get("reward", reward),
            "reward_components": record.get("reward_components", {}),
            "tool_calls": record.get("tool_calls", []),
            "tool_results": record.get("tool_results", []),
        }
        text = f"user: {prompt}\nassistant: {_compact_json(target)}"
        weight = 0.75 + max(0.0, reward) * 1.25
    elif kind == "tool_preference" or {"prompt", "chosen", "rejected"} <= set(record):
        text = f"user: {prompt}\nassistant: {record.get('chosen', '')}"
        weight = 1.25 + max(0.0, reward) * 0.5
    elif kind == "tool_reward":
        target = {
            "tool_calls": record.get("tool_calls", []),
            "tool_results": record.get("tool_results", []),
            "reward": reward,
            "reward_components": record.get("reward_components", {}),
        }
        text = f"user: {prompt}\nassistant: {_compact_json(target)}"
        weight = 0.5 + max(0.0, reward) * 1.5
    elif kind == "tool_safety_negative":
        text = f"user: {prompt}\nassistant: {record.get('chosen', 'Refuse unsafe tool use and protect credentials.')}"
        weight = 1.5
    else:
        text = _text_from_record(record)
        weight = 1.0 + max(0.0, reward) * 0.25
    return text.strip(), max(0.05, min(2.5, float(weight)))


class WeightedTextJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tokenizer: Any, seq_len: int, max_records: int = 0, vocab_size: int = 0):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.samples: list[tuple[list[int], float]] = []
        limit = int(max_records) if int(max_records) > 0 else None
        p = Path(path)
        paths = sorted(p.rglob("*.jsonl")) + sorted(p.rglob("*.txt")) if p.is_dir() else [p]
        for src in paths:
            if limit is not None and len(self.samples) >= limit:
                break
            self._load_path(src, limit)
        if not self.samples:
            self.samples.append(([1] * self.seq_len, 0.05))

    def _sanitize_id(self, value: int) -> int:
        token = int(value)
        if token < 0:
            return 0
        if self.vocab_size > 0 and token >= self.vocab_size:
            return 1
        return token

    def _load_path(self, path: Path, limit: int | None) -> None:
        if not path.exists():
            return
        if path.suffix.lower() == ".txt":
            self._append_text(path.read_text(encoding="utf-8", errors="ignore"), 1.0, limit)
            return
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if limit is not None and len(self.samples) >= limit:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                obj = {"text": line}
            if not isinstance(obj, dict):
                obj = {"text": str(obj)}
            _, weight = _pipeline_record_to_text_and_weight(obj)
            ids = _ids_from_record(obj)
            if ids:
                self._append_ids(ids, weight, limit)
            else:
                text, weight = _pipeline_record_to_text_and_weight(obj)
                self._append_text(text, weight, limit)

    def _append_text(self, text: str, weight: float, limit: int | None) -> None:
        if limit is not None and len(self.samples) >= limit:
            return
        ids = [int(x) for x in self.tokenizer.encode(text)]
        self._append_ids(ids, weight, limit)

    def _append_ids(self, ids: list[int], weight: float, limit: int | None) -> None:
        if limit is not None and len(self.samples) >= limit:
            return
        cleaned = [self._sanitize_id(x) for x in ids]
        if len(cleaned) < 2:
            return
        for start in range(0, len(cleaned), max(1, self.seq_len)):
            chunk = cleaned[start:start + self.seq_len]
            if len(chunk) < 2:
                continue
            self.samples.append((chunk, float(weight)))
            if limit is not None and len(self.samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, weight = self.samples[idx]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        return torch.tensor(ids[: self.seq_len], dtype=torch.long), torch.tensor(float(weight), dtype=torch.float32)


def _validate_resume_payload(
    checkpoint: dict[str, Any],
    *,
    preset: object | None,
    args: argparse.Namespace | None,
    sharded: bool,
) -> None:
    saved_preset = checkpoint.get("preset") if isinstance(checkpoint.get("preset"), dict) else {}
    saved_name = str(saved_preset.get("name") or "")
    current_name = str(getattr(preset, "name", "")) if preset is not None else ""
    if saved_name and current_name and saved_name != current_name:
        raise ValueError(f"resume checkpoint preset mismatch: checkpoint={saved_name!r} current={current_name!r}")
    if args is None:
        return
    if bool(getattr(args, "require_target_contract", False)) and saved_name and saved_name != TARGET_PRESET:
        raise ValueError(f"target contract resume requires {TARGET_PRESET!r}, got checkpoint preset {saved_name!r}")
    train_args = checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {}
    for key in ("pipeline_stage_ranges", "placement_layer_counts", "pipeline_microbatches", "pipeline_schedule", "fake_quant"):
        saved = train_args.get(key)
        if saved is None or saved == "":
            continue
        current = getattr(args, key, None)
        if key == "pipeline_stage_ranges" and str(getattr(args, "placement_layer_counts", "") or "").strip():
            continue
        if str(saved) != str(current):
            raise ValueError(f"resume checkpoint {key} mismatch: checkpoint={saved!r} current={current!r}")
    if sharded:
        saved_world = checkpoint.get("world_size")
        if saved_world is not None and int(saved_world) != int(dist.get_world_size()):
            raise ValueError(f"resume world_size mismatch: checkpoint={saved_world} current={dist.get_world_size()}")


def load_checkpoint_shard(
    path: str | Path,
    shard: OmniCoder2026PipelineShard,
    optimizer: Any | None = None,
    *,
    preset: object | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[int, float | None]:
    if not path:
        return 0, None
    source = Path(path)
    sharded = source.is_dir()
    if sharded:
        manifest_path = source / "manifest.json"
        complete_path = source / ".complete.json"
        rank_complete = source / f"rank{int(dist.get_rank()):05d}.pt.complete.json"
        if not complete_path.exists() or not manifest_path.exists() or not rank_complete.exists():
            raise ValueError(
                f"incomplete sharded checkpoint {source}: expected manifest, directory complete marker, "
                f"and rank marker {rank_complete.name}"
            )
    checkpoint_path = source / f"rank{int(dist.get_rank()):05d}.pt" if sharded else source
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {checkpoint_path} has no model_state_dict")
    _validate_resume_payload(checkpoint, preset=preset, args=args, sharded=sharded)
    current = shard.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for key, tensor in current.items():
        if key in state and tuple(state[key].shape) == tuple(tensor.shape):
            filtered[key] = state[key]
        elif key == "lm_head.weight" and "embed.weight" in state and tuple(state["embed.weight"].shape) == tuple(tensor.shape):
            filtered[key] = state["embed.weight"]
        else:
            missing.append(key)
    if missing:
        raise ValueError(f"checkpoint {checkpoint_path} is missing local shard tensors: {missing[:8]}")
    shard.load_state_dict(filtered, strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state") or {})
    return int(checkpoint.get("global_step") or 0), checkpoint.get("last_loss")


def save_sharded_checkpoint(
    path: str | Path,
    shard: OmniCoder2026PipelineShard,
    *,
    preset: object,
    args: argparse.Namespace,
    optimizer: Any | None,
    global_step: int,
    last_loss: float | None,
) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        for stale in (target / ".complete.json", target / "manifest.json"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
    dist.barrier()
    rank_path = target / f"rank{int(dist.get_rank()):05d}.pt"
    try:
        Path(str(rank_path) + ".complete.json").unlink()
    except FileNotFoundError:
        pass
    payload = {
        "format": "omnicoder2026_pipeline_stage_checkpoint_v2",
        "rank": int(dist.get_rank()),
        "world_size": int(dist.get_world_size()),
        "model_state_dict": shard.local_state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None,
        "rng_state": _rng_state(),
        "global_step": int(global_step),
        "last_loss": last_loss,
        "preset": preset.__dict__,
        "data": {
            "path": args.data,
            "sha256": _sha256_file(args.data) if Path(args.data).exists() else None,
            "manifest": getattr(args, "data_manifest", None),
        },
        "train_args": {
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "pipeline_stage_ranges": str(args.pipeline_stage_ranges),
            "placement_layer_counts": str(getattr(args, "placement_layer_counts", "") or ""),
            "pipeline_microbatches": int(args.pipeline_microbatches),
            "n_microbatches": int(args.pipeline_microbatches),
            "pipeline_schedule": str(args.pipeline_schedule),
            "schedule": str(args.pipeline_schedule),
            "fake_quant": bool(args.fake_quant),
            "optimizer": str(args.optimizer),
            "optimizer_in_backward": bool(getattr(args, "optimizer_in_backward", False)),
            "optimizer_in_backward_update": str(getattr(args, "optimizer_in_backward_update", "")),
            "optimizer_in_backward_grad_clip": float(getattr(args, "optimizer_in_backward_grad_clip", 0.0) or 0.0),
            "optimizer_in_backward_adafactor_chunk_rows": int(getattr(args, "optimizer_in_backward_adafactor_chunk_rows", 0) or 0),
        },
        "spec": shard.spec.__dict__,
        "notes": {"pipeline_low_memory_optimizer": PIPELINE_LOW_MEMORY_OPTIMIZER_NOTE},
    }
    _atomic_torch_save(payload, rank_path)
    dist.barrier()
    if dist.get_rank() == 0:
        manifest = {
            "format": "omnicoder2026_pipeline_stage_checkpoint_v2",
            "checkpoint_dir": str(target),
            "world_size": int(dist.get_world_size()),
            "rank_files": [f"rank{rank:05d}.pt" for rank in range(int(dist.get_world_size()))],
            "global_step": int(global_step),
            "last_loss": last_loss,
            "preset": preset.__dict__,
            "train_args": payload["train_args"],
            "note": "Per-stage pipeline checkpoint. Use merge tooling before GGUF/export.",
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        (target / ".complete.json").write_text(json.dumps({"status": "complete", **manifest}, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    dist.barrier()


def rank_device(rank: int, raw_map: str) -> torch.device:
    parts = [part.strip() for part in str(raw_map or "").split(",") if part.strip()]
    if parts and rank >= len(parts):
        raise ValueError(f"rank_device_map has {len(parts)} entries for rank {rank}: {raw_map!r}")
    index = int(parts[rank]) if parts else int(os.getenv("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} outside visible device_count={torch.cuda.device_count()}")
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cpu")


def validate_target_device_placement(args: argparse.Namespace, ranges: list[tuple[int, int]], spec: PipelineShardSpec, device: torch.device) -> None:
    if not bool(getattr(args, "require_target_contract", False)) or device.type != "cuda":
        return
    props = torch.cuda.get_device_properties(device)
    local = {
        "rank": int(dist.get_rank()),
        "device_index": int(device.index or 0),
        "name": str(props.name),
        "total_memory": int(props.total_memory),
        "layers": int(spec.layer_end - spec.layer_start),
        "has_head": bool(spec.has_head),
    }
    reports: list[dict[str, Any] | None] = [None for _ in range(int(dist.get_world_size()))]
    dist.all_gather_object(reports, local)
    complete = [item for item in reports if isinstance(item, dict)]
    if len(complete) != int(dist.get_world_size()):
        raise ValueError(f"incomplete device placement report: {reports!r}")
    p40_ranks = [item for item in complete if "P40" in str(item.get("name", "")).upper()]
    if p40_ranks:
        raise ValueError(f"target-contract pipeline may not include P40 devices: {p40_ranks!r}")
    max_layers = max(int(item["layers"]) for item in complete)
    max_memory = max(int(item["total_memory"]) for item in complete)
    heavy_ranks = [item for item in complete if int(item["layers"]) == max_layers]
    misplaced = [item for item in heavy_ranks if int(item["total_memory"]) < max_memory]
    if misplaced:
        raise ValueError(
            "target-contract pipeline places the largest layer/head shard on a non-largest visible GPU. "
            f"reports={complete!r}; ranges={ranges!r}. Expose host GPUs in fast-card order, "
            "for example Docker --gpus '\"device=0,4,6\"' with --rank_device_map 0,1,2."
        )


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: str):
    key = str(precision or "fp32").lower()
    if device.type == "cuda" and key in {"fp16", "bf16"}:
        yield_context = torch.autocast(device_type="cuda", dtype=_dtype_from_name(key))
    else:
        yield_context = contextlib.nullcontext()
    with yield_context:
        yield


class PipelineLowMemoryAdafactor:
    """Per-rank delayed low-memory optimizer for PipelineStage schedules."""

    def __init__(self, params: list[torch.nn.Parameter], args: argparse.Namespace):
        self.params = [param for param in params if param.requires_grad]
        self.lr = float(args.lr)
        self.chunk_rows = max(1, int(getattr(args, "optimizer_in_backward_adafactor_chunk_rows", 256) or 256))
        self.clip_threshold = float(getattr(args, "optimizer_in_backward_adafactor_clip_threshold", 1.0) or 1.0)
        self.decay_rate = float(getattr(args, "optimizer_in_backward_adafactor_decay_rate", -0.8) or -0.8)
        self.eps1 = float(getattr(args, "optimizer_in_backward_adafactor_eps1", 1.0e-30) or 1.0e-30)
        self.fallback_clip = float(getattr(args, "optimizer_in_backward_grad_clip", 1.0) or 0.0)
        self.clip_mode = str(getattr(args, "optimizer_in_backward_clip_mode", "rms") or "rms").lower()
        self.states: dict[int, dict[str, Any]] = {}
        self.param_groups = [{"params": self.params, "lr": self.lr}]
        self.post_accumulate = bool(getattr(args, "optimizer_in_backward", False)) and int(getattr(args, "pipeline_microbatches", 1) or 1) == 1
        self.handles: list[Any] = []
        if self.post_accumulate:
            sample = self.params[0] if self.params else None
            if sample is None or not hasattr(sample, "register_post_accumulate_grad_hook"):
                raise RuntimeError("optimizer-in-backward requires Tensor.register_post_accumulate_grad_hook")
            for param in self.params:
                self.handles.append(param.register_post_accumulate_grad_hook(self._hook_step))

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def _fallback_step(self, param: torch.nn.Parameter, grad: torch.Tensor) -> None:
        if grad.is_sparse:
            param.add_(grad, alpha=-self.lr)
            return
        if grad.is_floating_point():
            limit = self.fallback_clip if self.fallback_clip > 0 else 1.0
            torch.nan_to_num_(grad, nan=0.0, posinf=limit, neginf=-limit)
            if self.fallback_clip > 0 and self.clip_mode == "rms":
                norm = torch.linalg.vector_norm(grad.detach())
                target = self.fallback_clip * math.sqrt(max(1, int(grad.numel())))
                scale = torch.clamp(
                    torch.as_tensor(target, device=grad.device, dtype=norm.dtype) / norm.clamp_min(1.0e-12),
                    max=1.0,
                )
                grad.mul_(scale.to(dtype=grad.dtype))
            elif self.fallback_clip > 0:
                grad.clamp_(min=-self.fallback_clip, max=self.fallback_clip)
        param.add_(grad, alpha=-self.lr)

    def _step_param(self, param: torch.nn.Parameter) -> None:
        grad = param.grad
        if grad is None:
            return
        if grad.is_sparse or grad.ndim != 2:
            self._fallback_step(param, grad)
            param.grad = None
            return
        rows, cols = int(grad.shape[0]), int(grad.shape[1])
        state = self.states.setdefault(
            id(param),
            {
                "step": 0,
                "row": torch.zeros(rows, device=param.device, dtype=torch.float32),
                "col": torch.zeros(cols, device=param.device, dtype=torch.float32),
            },
        )
        row_state = state["row"]
        col_state = state["col"]
        if tuple(row_state.shape) != (rows,) or tuple(col_state.shape) != (cols,):
            state["row"] = torch.zeros(rows, device=param.device, dtype=torch.float32)
            state["col"] = torch.zeros(cols, device=param.device, dtype=torch.float32)
            row_state = state["row"]
            col_state = state["col"]
        state["step"] = int(state["step"]) + 1
        beta2 = 1.0 - (float(state["step"]) ** self.decay_rate)
        one_minus_beta2 = 1.0 - beta2
        assert isinstance(row_state, torch.Tensor) and isinstance(col_state, torch.Tensor)
        torch.nan_to_num_(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        col_sum = torch.zeros_like(col_state)
        for start in range(0, rows, self.chunk_rows):
            end = min(rows, start + self.chunk_rows)
            g2 = grad[start:end].detach().to(torch.float32, copy=True)
            g2.square_().add_(self.eps1)
            row_state[start:end].mul_(beta2).add_(g2.mean(dim=1), alpha=one_minus_beta2)
            col_sum.add_(g2.sum(dim=0))
        col_state.mul_(beta2).add_(col_sum.div_(max(1, rows)), alpha=one_minus_beta2)
        row_mean = row_state.mean().clamp_min(self.eps1)
        col_factor = col_state.clamp_min(self.eps1).rsqrt()
        update_sq = torch.zeros((), device=param.device, dtype=torch.float32)
        for start in range(0, rows, self.chunk_rows):
            end = min(rows, start + self.chunk_rows)
            update = grad[start:end].detach().to(torch.float32, copy=True)
            row_factor = (row_state[start:end].clamp_min(self.eps1) / row_mean).rsqrt()
            update.mul_(row_factor[:, None]).mul_(col_factor[None, :])
            update_sq.add_(update.square_().sum())
        update_rms = torch.sqrt(update_sq / max(1, grad.numel()))
        denom = torch.clamp(update_rms / max(1.0e-12, self.clip_threshold), min=1.0)
        for start in range(0, rows, self.chunk_rows):
            end = min(rows, start + self.chunk_rows)
            update = grad[start:end].detach().to(torch.float32, copy=True)
            row_factor = (row_state[start:end].clamp_min(self.eps1) / row_mean).rsqrt()
            update.mul_(row_factor[:, None]).mul_(col_factor[None, :])
            update.mul_(self.lr).div_(denom)
            param.data[start:end].add_(update.to(dtype=param.dtype), alpha=-1.0)
        param.grad = None

    def _hook_step(self, param: torch.nn.Parameter) -> None:
        with torch.no_grad():
            self._step_param(param)

    def step(self) -> None:
        if self.post_accumulate:
            return
        with torch.no_grad():
            for param in self.params:
                self._step_param(param)

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

    def state_dict(self) -> dict[str, Any]:
        state_items: list[dict[str, Any] | None] = []
        for param in self.params:
            state = self.states.get(id(param))
            if not state:
                state_items.append(None)
                continue
            row = state.get("row")
            col = state.get("col")
            state_items.append(
                {
                    "step": int(state.get("step") or 0),
                    "row": row.detach().cpu() if isinstance(row, torch.Tensor) else None,
                    "col": col.detach().cpu() if isinstance(col, torch.Tensor) else None,
                }
            )
        return {
            "format": "omnicoder2026_pipeline_lowmem_adafactor_v1",
            "lr": self.lr,
            "chunk_rows": self.chunk_rows,
            "clip_threshold": self.clip_threshold,
            "decay_rate": self.decay_rate,
            "eps1": self.eps1,
            "post_accumulate": self.post_accumulate,
            "states": state_items,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        states = payload.get("states") if isinstance(payload, dict) else None
        if not isinstance(states, list):
            return
        for param, item in zip(self.params, states, strict=False):
            if not isinstance(item, dict):
                continue
            row = item.get("row")
            col = item.get("col")
            if not isinstance(row, torch.Tensor) or not isinstance(col, torch.Tensor):
                continue
            if param.ndim != 2 or tuple(row.shape) != (param.shape[0],) or tuple(col.shape) != (param.shape[1],):
                continue
            self.states[id(param)] = {
                "step": int(item.get("step") or 0),
                "row": row.to(device=param.device, dtype=torch.float32),
                "col": col.to(device=param.device, dtype=torch.float32),
            }


def build_optimizer(args: argparse.Namespace, shard: nn.Module) -> Any:
    params = [param for param in shard.parameters() if param.requires_grad]
    update_mode = str(getattr(args, "optimizer_in_backward_update", "") or "").lower()
    if bool(getattr(args, "optimizer_in_backward", False)) or update_mode in {"lowmem_adafactor", "chunked_adafactor"}:
        if update_mode not in {"", "lowmem_adafactor", "chunked_adafactor"}:
            raise ValueError(f"Pipeline low-memory mode only supports lowmem_adafactor/chunked_adafactor, got {update_mode!r}")
        return PipelineLowMemoryAdafactor(params, args)
    if str(args.optimizer).lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor

            return Adafactor(params, lr=float(args.lr), relative_step=False, scale_parameter=False, warmup_init=False, weight_decay=0.0)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimental 2026 dense pipeline trainer.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--log_file", default="")
    parser.add_argument("--data_manifest", default="")
    parser.add_argument("--preset", default="ledger_probe")
    parser.add_argument("--rank_device_map", default="")
    parser.add_argument("--pipeline_stage_ranges", default="")
    parser.add_argument("--placement_layer_counts", default="")
    parser.add_argument("--pipeline_schedule", default="1f1b", choices=["1f1b", "gpipe"])
    parser.add_argument("--schedule", dest="pipeline_schedule", choices=["1f1b", "gpipe"])
    parser.add_argument("--pipeline_microbatches", type=int, default=2)
    parser.add_argument("--n_microbatches", dest="pipeline_microbatches", type=int)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--init_dtype", default="auto")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--optimizer_in_backward", action="store_true")
    parser.add_argument("--optimizer_in_backward_update", default="lowmem_adafactor", choices=["lowmem_adafactor", "chunked_adafactor"])
    parser.add_argument("--optimizer_in_backward_grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer_in_backward_clip_mode", default="rms", choices=["rms", "clamp"])
    parser.add_argument("--optimizer_in_backward_adafactor_chunk_rows", type=int, default=256)
    parser.add_argument("--optimizer_in_backward_adafactor_clip_threshold", type=float, default=1.0)
    parser.add_argument("--optimizer_in_backward_adafactor_decay_rate", type=float, default=-0.8)
    parser.add_argument("--optimizer_in_backward_adafactor_eps1", type=float, default=1.0e-30)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--fake_quant", action="store_true")
    parser.add_argument("--fake_quant_chunk_rows", type=int, default=0)
    parser.add_argument("--fake_quant_max_full_elements", type=int, default=0)
    parser.add_argument("--lm_loss_chunk_tokens", type=int, default=int(os.getenv("OMNICODER2026_LM_LOSS_CHUNK_TOKENS", "128") or 128))
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--require_target_contract", action="store_true")
    parser.add_argument("--allow_probe", action="store_true")
    parser.add_argument("--debug_events", action="store_true")
    args = parser.parse_args(argv)

    if int(args.fake_quant_chunk_rows or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_CHUNK_ROWS"] = str(int(args.fake_quant_chunk_rows))
    if int(args.fake_quant_max_full_elements or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS"] = str(int(args.fake_quant_max_full_elements))
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    device = rank_device(rank, args.rank_device_map)

    preset = get_omnicoder2026_preset(args.preset)
    if bool(args.require_target_contract) and _is_probe_name(preset.name) and not bool(args.allow_probe):
        raise ValueError(
            f"Refusing to train verifier preset {preset.name!r} for a target-contract run. "
            f"Pass --allow_probe only for explicit validation runs, or use --preset {TARGET_PRESET}."
        )
    kwargs = preset_to_model_kwargs(preset)
    kwargs["fake_quant"] = bool(args.fake_quant)
    # Pipeline split puts embed on stage 0 and head on the final stage, so keep
    # weights untied in this lane and initialize lm_head from embed on load.
    kwargs["tie_embeddings"] = False
    cfg = OmniCoder2026Config(**kwargs)
    seq_len = int(args.seq_len or preset.train_seq_len)
    pipeline_microbatches = int(args.pipeline_microbatches)
    batch_size = int(args.batch_size)
    if pipeline_microbatches < 1:
        raise ValueError("--pipeline_microbatches must be >= 1")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if pipeline_microbatches > batch_size:
        raise ValueError(f"--pipeline_microbatches={pipeline_microbatches} cannot exceed --batch_size={batch_size}")
    if batch_size % pipeline_microbatches != 0:
        raise ValueError(f"--batch_size={batch_size} must be divisible by --pipeline_microbatches={pipeline_microbatches}")
    ranges = stage_ranges(int(cfg.n_layers), str(args.placement_layer_counts)) if str(args.placement_layer_counts or "").strip() else parse_stage_ranges(str(args.pipeline_stage_ranges), int(cfg.n_layers))
    if len(ranges) != world_size:
        raise ValueError(f"world_size={world_size} must match pipeline stages {ranges}")
    spec = shard_spec(rank, ranges)
    validate_target_device_placement(args, ranges, spec, device)
    init_dtype_name = str(args.init_dtype or "auto").lower()
    if init_dtype_name == "auto":
        init_dtype_name = str(args.precision or "fp32").lower()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(_dtype_from_name(init_dtype_name))
    try:
        with torch.device(device):
            shard = OmniCoder2026PipelineShard(cfg, spec, checkpoint_blocks=bool(args.activation_checkpointing)).to(device)
    finally:
        torch.set_default_dtype(old_dtype)

    start_step, last_loss = 0, None
    shard.train()
    microbatch_size = batch_size // pipeline_microbatches
    example_input = torch.zeros((microbatch_size, seq_len), dtype=torch.long, device=device)
    if not spec.has_embed:
        example_input = torch.zeros((example_input.shape[0], seq_len, cfg.d_model), dtype=_dtype_from_name(init_dtype_name), device=device)

    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

    with torch.no_grad(), autocast_context(device, str(args.precision)):
        example_output = shard(example_input)
    stage = PipelineStage(
        shard,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=(example_input,),
        output_args=example_output,
    )
    optimizer = build_optimizer(args, shard)
    if args.resume:
        start_step, last_loss = load_checkpoint_shard(args.resume, shard, optimizer, preset=preset, args=args)

    def _unused_nonfinal_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Enable pipeline backward plumbing on non-final ranks.

        PyTorch's single-stage schedules only build backward send/recv handling
        when a loss function is present. The schedule calls the loss function
        only on the final stage, but non-final stages still need a non-None
        loss_fn so gradients flow instead of rank 0 finishing early while later
        ranks wait forever.
        """

        return output.sum() * 0.0

    current_sample_weights: dict[str, torch.Tensor] = {}
    loss_fn = (
        lambda hidden, labels: shard.chunked_lm_loss(
            hidden,
            labels,
            int(args.lm_loss_chunk_tokens),
            current_sample_weights.get("weights"),
        )
    ) if spec.has_head else _unused_nonfinal_loss
    if args.pipeline_schedule == "1f1b" and pipeline_microbatches == 1:
        if rank == 0:
            _write_log(args.log_file, {"event": "pipeline_schedule_auto_gpipe", "reason": "1f1b_with_one_microbatch_has_no_overlap_and_is_unstable_on_this_torch_nccl_build"})
        args.pipeline_schedule = "gpipe"

    if args.pipeline_schedule == "gpipe":
        schedule = ScheduleGPipe(stage, n_microbatches=pipeline_microbatches, loss_fn=loss_fn)
    else:
        schedule = Schedule1F1B(stage, n_microbatches=pipeline_microbatches, loss_fn=loss_fn)

    tokenizer = get_text_tokenizer(prefer_hf=True) if rank == 0 else None
    data = WeightedTextJsonlDataset(args.data, tokenizer, seq_len=seq_len, max_records=args.max_records, vocab_size=int(getattr(preset, "vocab_size", 0) or 0)) if rank == 0 else None
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True) if rank == 0 else None
    it = iter(loader) if loader is not None else None
    def debug_event(message: str) -> None:
        if bool(args.debug_events):
            print(json.dumps({"event": "pipeline_debug", "rank": rank, "message": message}, ensure_ascii=True), flush=True)

    for local_step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        if rank == 0:
            debug_event("rank0_fetch_batch_start")
            try:
                batch_item = next(it)  # type: ignore[arg-type]
            except StopIteration:
                it = iter(loader)  # type: ignore[arg-type]
                batch_item = next(it)
            batch, batch_weights = batch_item
            batch = batch.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True).float()
            debug_event("rank0_fetch_batch_done")
        else:
            batch = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
            batch_weights = torch.empty((batch_size,), dtype=torch.float32, device=device)
        debug_event("broadcast_start")
        dist.broadcast(batch, src=0)
        dist.broadcast(batch_weights, src=0)
        current_sample_weights["weights"] = batch_weights
        debug_event("broadcast_done")
        with autocast_context(device, str(args.precision)):
            if rank == 0:
                debug_event("schedule_step_rank0_start")
                schedule.step(batch, target=batch, losses=losses)
                debug_event("schedule_step_rank0_done")
            else:
                debug_event("schedule_step_nonzero_start")
                schedule.step(target=batch, losses=losses)
                debug_event("schedule_step_nonzero_done")
        optimizer.step()
        debug_event("optimizer_step_done")
        if spec.has_head and losses:
            loss_value = float(torch.stack([loss.detach().float() for loss in losses]).mean().cpu())
            last_loss = loss_value
        loss_tensor = torch.tensor(float(last_loss) if last_loss is not None else -1.0, device=device)
        dist.broadcast(loss_tensor, src=world_size - 1)
        if rank == 0:
            global_step = start_step + local_step + 1
            _write_log(args.log_file, {"step": global_step, "local_step": local_step + 1, "loss": float(loss_tensor.cpu()), "preset": preset.name, "seq_len": seq_len, "distributed": "pipeline", "world_size": world_size, "pipeline_schedule": args.pipeline_schedule, "pipeline_microbatches": pipeline_microbatches, "microbatch_size": microbatch_size, "sample_weight_mean": float(batch_weights.detach().mean().cpu()), "optimizer": str(args.optimizer), "optimizer_in_backward": bool(args.optimizer_in_backward), "optimizer_in_backward_update": str(args.optimizer_in_backward_update)})
        if int(args.save_interval) > 0 and (start_step + local_step + 1) % int(args.save_interval) == 0:
            save_sharded_checkpoint(Path(args.out).with_name(f"{Path(args.out).stem}.step{start_step + local_step + 1}"), shard, preset=preset, args=args, optimizer=optimizer, global_step=start_step + local_step + 1, last_loss=float(loss_tensor.cpu()))

    save_sharded_checkpoint(args.out, shard, preset=preset, args=args, optimizer=optimizer, global_step=start_step + int(args.steps), last_loss=float(loss_tensor.cpu()))
    if rank == 0:
        _write_log(args.log_file, {"status": "ok", "out": args.out, "last_loss": float(loss_tensor.cpu()), "global_step": start_step + int(args.steps), "distributed": "pipeline", "world_size": world_size})
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

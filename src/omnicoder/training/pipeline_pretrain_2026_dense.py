from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.profiler import record_function

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
from omnicoder.model_contract_2026 import validate_target_contract_preset
from omnicoder.modeling.omnicoder2026 import (
    AdaptiveLatentReasoner,
    OmniCoder2026Block,
    OmniCoder2026Config,
    QuantAwareLinear,
    RMSNorm,
    reset_omnicoder2026_parameters,
)
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.training.pretrain_2026_dense import (
    TARGET_PRESET,
    _ids_from_record,
    _atomic_torch_save,
    _dtype_from_name,
    _is_probe_name,
    _text_from_record,
    _row_trainability_rejection_reason,
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
CHECKPOINT_ATTEMPT_MARKER = ".checkpoint_save_attempt.json"
TRAIN_DIAGNOSTICS_SCHEMA = "omnicoder.train_diagnostics_2026.v1"
STEP_TIMING_SCHEMA = "omnicoder.pipeline_step_timing_2026.v1"
CHECKPOINT_IO_SCHEMA = "omnicoder.pipeline_checkpoint_io_2026.v1"
CHECKPOINT_EVAL_ARTIFACT_CONTRACT_SCHEMA = "omnicoder.checkpoint_eval_artifact_contract_2026.v1"

TOKEN_FAMILY_TO_MODALITY = {
    "text": "text",
    "control": "control",
    "vision_semantic": "vision",
    "vision_residual": "vision",
    "speech_tts": "tts",
    "audio_music": "audio_music",
    "music_control": "music",
    "time_space": "time_space",
    "tool_agent": "tool_agent",
    "flow": "media_flow",
    "unknown": "unknown",
}


def _monotonic() -> float:
    return time.perf_counter()


def _should_log_interval(interval: int, step: int) -> bool:
    interval = int(interval or 0)
    if interval <= 0:
        return False
    return int(step) <= 1 or (int(step) % interval) == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(str(name), str(int(default))) or int(default))
    except Exception:
        return int(default)


def _cuda_synchronize_if_requested(device: torch.device, enabled: bool) -> None:
    if not bool(enabled) or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass


class PhaseTimer:
    def __init__(self, *, device: torch.device, cuda_sync: bool = False):
        self.device = device
        self.cuda_sync = bool(cuda_sync)
        self.started = _monotonic()
        self.spans: dict[str, float] = {}

    @contextlib.contextmanager
    def span(self, name: str):
        _cuda_synchronize_if_requested(self.device, self.cuda_sync)
        started = _monotonic()
        try:
            yield
        finally:
            _cuda_synchronize_if_requested(self.device, self.cuda_sync)
            self.spans[str(name)] = float(self.spans.get(str(name), 0.0) + (_monotonic() - started))

    def elapsed(self) -> float:
        _cuda_synchronize_if_requested(self.device, self.cuda_sync)
        return float(_monotonic() - self.started)

    def record(self) -> dict[str, float]:
        return {str(key): float(value) for key, value in self.spans.items()}


def _file_size_or_zero(path: str | Path) -> int:
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return 0


@dataclass(frozen=True)
class PipelineShardSpec:
    stage_index: int
    num_stages: int
    layer_start: int
    layer_end: int
    has_embed: bool
    has_head: bool


@dataclass(frozen=True)
class PipelineStepTensors:
    input_ids: torch.Tensor | None
    labels: torch.Tensor | None
    sample_weights: torch.Tensor | None


@dataclass(frozen=True)
class PipelineTargetSummary:
    valid_target_tokens: int
    optimized_target_tokens: int
    sample_weight_mean: float | None


def _pipeline_final_rank(world_size: int) -> int:
    return max(0, int(world_size) - 1)


def _loss_scalar_tensor(loss_tensor: torch.Tensor) -> torch.Tensor:
    return loss_tensor.detach().float().reshape(-1).mean()


def _tensor_scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _route_pipeline_step_tensors(
    *,
    rank: int,
    world_size: int,
    batch: torch.Tensor | None,
    labels: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dist_module: Any = dist,
) -> PipelineStepTensors:
    """Move only the tensors each pipeline role needs for this step."""

    rank = int(rank)
    final_rank = _pipeline_final_rank(int(world_size))
    input_ids = batch if rank == 0 else None
    target_labels: torch.Tensor | None = labels if rank == final_rank else None
    target_weights: torch.Tensor | None = sample_weights if rank == final_rank else None
    if final_rank == 0:
        if input_ids is None or target_labels is None or target_weights is None:
            raise RuntimeError("single-stage pipeline requires rank 0 to own inputs, labels, and sample weights")
        return PipelineStepTensors(input_ids=input_ids, labels=target_labels, sample_weights=target_weights)
    if rank == 0:
        if input_ids is None or labels is None or sample_weights is None:
            raise RuntimeError("rank 0 must fetch input ids, labels, and sample weights before target routing")
        dist_module.send(labels.contiguous(), dst=final_rank)
        dist_module.send(sample_weights.contiguous(), dst=final_rank)
        return PipelineStepTensors(input_ids=input_ids, labels=None, sample_weights=None)
    if rank == final_rank:
        target_labels = torch.empty((int(batch_size), int(seq_len)), dtype=torch.long, device=device)
        target_weights = torch.empty((int(batch_size),), dtype=torch.float32, device=device)
        dist_module.recv(target_labels, src=0)
        dist_module.recv(target_weights, src=0)
        return PipelineStepTensors(input_ids=None, labels=target_labels, sample_weights=target_weights)
    return PipelineStepTensors(input_ids=None, labels=None, sample_weights=None)


def _sync_pipeline_loss_to_rank0(
    *,
    rank: int,
    world_size: int,
    loss_tensor: torch.Tensor | None,
    device: torch.device,
    dist_module: Any = dist,
) -> torch.Tensor | None:
    final_rank = _pipeline_final_rank(int(world_size))
    rank = int(rank)
    if rank == final_rank:
        if loss_tensor is None:
            raise RuntimeError("final pipeline rank must have a local loss tensor")
        scalar = _loss_scalar_tensor(loss_tensor).to(device=device)
        if rank != 0:
            dist_module.send(scalar, dst=0)
        return scalar
    if rank == 0:
        scalar = torch.empty((), dtype=torch.float32, device=device)
        dist_module.recv(scalar, src=final_rank)
        return scalar
    return None


def _sync_pipeline_loss_for_checkpoint(
    *,
    rank: int,
    world_size: int,
    loss_tensor: torch.Tensor | None,
    device: torch.device,
    dist_module: Any = dist,
) -> float:
    final_rank = _pipeline_final_rank(int(world_size))
    if int(rank) == final_rank:
        if loss_tensor is None:
            raise RuntimeError("final pipeline rank must have a local loss tensor for checkpoint loss sync")
        scalar = _loss_scalar_tensor(loss_tensor).to(device=device)
    else:
        scalar = torch.empty((), dtype=torch.float32, device=device)
    dist_module.broadcast(scalar, src=final_rank)
    return _tensor_scalar_float(scalar)


def _target_summary_from_final_rank(
    loss_diagnostics: dict[str, Any] | None,
    sample_weights: torch.Tensor | None,
) -> PipelineTargetSummary:
    diagnostics = loss_diagnostics if isinstance(loss_diagnostics, dict) else {}
    sample_weight_mean = _tensor_scalar_float(sample_weights.detach().float().mean()) if sample_weights is not None else None
    return PipelineTargetSummary(
        valid_target_tokens=int(diagnostics.get("valid_target_tokens") or 0),
        optimized_target_tokens=int(diagnostics.get("optimized_target_tokens") or 0),
        sample_weight_mean=sample_weight_mean,
    )


def _sync_pipeline_target_summary_to_rank0(
    *,
    rank: int,
    world_size: int,
    loss_diagnostics: dict[str, Any] | None,
    sample_weights: torch.Tensor | None,
    device: torch.device,
    dist_module: Any = dist,
) -> PipelineTargetSummary | None:
    final_rank = _pipeline_final_rank(int(world_size))
    rank = int(rank)
    if rank == final_rank:
        summary = _target_summary_from_final_rank(loss_diagnostics, sample_weights)
        if rank != 0:
            payload = torch.tensor(
                [
                    float(summary.valid_target_tokens),
                    float(summary.optimized_target_tokens),
                    float("nan") if summary.sample_weight_mean is None else float(summary.sample_weight_mean),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist_module.send(payload, dst=0)
        return summary
    if rank == 0:
        payload = torch.empty((3,), dtype=torch.float64, device=device)
        dist_module.recv(payload, src=final_rank)
        values = payload.detach().cpu().tolist()
        sample_weight_mean = None if math.isnan(float(values[2])) else float(values[2])
        return PipelineTargetSummary(
            valid_target_tokens=int(values[0]),
            optimized_target_tokens=int(values[1]),
            sample_weight_mean=sample_weight_mean,
        )
    return None


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

    def __init__(
        self,
        cfg: OmniCoder2026Config,
        spec: PipelineShardSpec,
        *,
        checkpoint_blocks: bool = False,
        checkpoint_segment_size: int = 1,
    ):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.checkpoint_segment_size = max(1, int(checkpoint_segment_size or 1))
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
        self.latent_reasoner = AdaptiveLatentReasoner(cfg) if spec.has_head else nn.Identity()
        self.lm_head = (
            QuantAwareLinear(cfg.d_model, cfg.vocab_size, bias=False, fake_quant=False, group_size=cfg.fake_quant_group_size)
            if spec.has_head
            else nn.Identity()
        )
        self.last_lm_loss_diagnostics: dict[str, Any] = {}
        self.last_lm_loss_timing: dict[str, Any] = {}
        self.last_reasoning_diagnostics: dict[str, Any] = {}
        self.profile_block_timing = False
        self.profile_block_cuda_sync = False
        self.profile_record_functions = False
        self.sanitize_input_ids = False
        self.pipeline_reasoning_effort: int | str | None = None
        self.block_timing_records: list[dict[str, Any]] = []
        self._block_timing_call_index = 0
        reset_omnicoder2026_parameters(self, cfg)

    def _checkpoint_enabled(self) -> bool:
        return bool(self.checkpoint_blocks and self.training and torch.is_grad_enabled())

    def _forward_block_range(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(int(start), int(end)):
            x = self.blocks[index](x)
        return x

    def _configured_reasoning_effort(self) -> int | str | None:
        effort = getattr(self, "pipeline_reasoning_effort", None)
        if effort is not None:
            return effort
        raw_effort = os.getenv("OMNICODER2026_PIPELINE_REASONING_EFFORT", str(int(self.cfg.reasoning_default_steps)))
        try:
            return int(raw_effort or 0)
        except ValueError:
            return str(raw_effort or "")

    def _embed_input_ids(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.to(dtype=torch.long)
        if bool(getattr(self, "sanitize_input_ids", False)):
            x = x.remainder(int(self.cfg.vocab_size))
        return self.embed(x)

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.has_embed:
            x = self._embed_input_ids(x)
        checkpoint_enabled = self._checkpoint_enabled()
        segment_size = max(1, int(self.checkpoint_segment_size))
        index = int(self.spec.layer_start)
        layer_end = int(self.spec.layer_end)
        while index < layer_end:
            if checkpoint_enabled:
                segment_end = min(layer_end, index + segment_size)
                if segment_end == index + 1:
                    x = activation_checkpoint(self.blocks[index], x, use_reentrant=False)
                else:
                    x = activation_checkpoint(
                        lambda inp, start=index, end=segment_end: self._forward_block_range(inp, start, end),
                        x,
                        use_reentrant=False,
                    )
                index = segment_end
            else:
                x = self.blocks[index](x)
                index += 1
        if self.spec.has_head:
            if isinstance(self.latent_reasoner, AdaptiveLatentReasoner):
                x, _controls = self.latent_reasoner(x, effort=self._configured_reasoning_effort(), return_controls=False)
                self.last_reasoning_diagnostics = dict(self.latent_reasoner.last_diagnostics)
            x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        profile_blocks = bool(getattr(self, "profile_block_timing", False))
        block_cuda_sync = bool(getattr(self, "profile_block_cuda_sync", False))
        profile_records = bool(getattr(self, "profile_record_functions", False))
        if not profile_blocks and not profile_records:
            return self._forward_fast(x)
        block_spans: list[dict[str, Any]] = []

        @contextlib.contextmanager
        def block_span(name: str, **metadata: Any):
            rf_name = f"omnicoder.pipeline.{name}"
            if "layer_index" in metadata:
                rf_name = f"{rf_name}.{int(metadata['layer_index']):03d}"
            record_context = record_function(rf_name) if profile_records else contextlib.nullcontext()
            with record_context:
                if not profile_blocks:
                    yield
                    return
                device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
                _cuda_synchronize_if_requested(device, block_cuda_sync)
                started = _monotonic()
                try:
                    yield
                finally:
                    _cuda_synchronize_if_requested(device, block_cuda_sync)
                    record = {"name": str(name), "sec": float(_monotonic() - started)}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            record[str(key)] = value
                    block_spans.append(record)

        self._block_timing_call_index += 1
        call_index = int(self._block_timing_call_index)
        call_started = _monotonic()
        if self.spec.has_embed:
            with block_span("embed", layer_start=int(self.spec.layer_start), layer_end=int(self.spec.layer_end)):
                x = self._embed_input_ids(x)
        checkpoint_enabled = self._checkpoint_enabled()
        segment_size = max(1, int(self.checkpoint_segment_size))
        index = int(self.spec.layer_start)
        layer_end = int(self.spec.layer_end)
        while index < layer_end:
            if checkpoint_enabled and segment_size > 1:
                segment_end = min(layer_end, index + segment_size)
                with block_span("block_segment", layer_start=int(index), layer_end=int(segment_end), checkpointed=True):
                    x = activation_checkpoint(
                        lambda inp, start=index, end=segment_end: self._forward_block_range(inp, start, end),
                        x,
                        use_reentrant=False,
                    )
                index = segment_end
                continue
            block = self.blocks[index]
            with block_span("block", layer_index=int(index), checkpointed=bool(checkpoint_enabled)):
                if checkpoint_enabled:
                    x = activation_checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            index += 1
        if self.spec.has_head:
            if isinstance(self.latent_reasoner, AdaptiveLatentReasoner):
                with block_span("latent_reasoner"):
                    x, _controls = self.latent_reasoner(x, effort=self._configured_reasoning_effort(), return_controls=False)
                    self.last_reasoning_diagnostics = dict(self.latent_reasoner.last_diagnostics)
            with block_span("norm"):
                x = self.norm(x)
        if profile_blocks:
            self.block_timing_records.append(
                {
                    "call_index": call_index,
                    "stage_index": int(self.spec.stage_index),
                    "layer_start": int(self.spec.layer_start),
                    "layer_end": int(self.spec.layer_end),
                    "has_embed": bool(self.spec.has_embed),
                    "has_head": bool(self.spec.has_head),
                    "total_sec": float(_monotonic() - call_started),
                    "spans": block_spans,
                }
            )
        return x

    def chunked_lm_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        chunk_tokens: int = 128,
        sample_weights: torch.Tensor | None = None,
        loss_token_stride: int = 1,
        max_loss_tokens_per_sample: int = 0,
        loss_mask: torch.Tensor | None = None,
        target_boundary_weight: float = 1.0,
        target_prefix_weight: float = 1.0,
        target_prefix_tokens: int = 0,
        collect_diagnostics: bool = True,
        labels_are_sparse: bool | None = None,
    ) -> torch.Tensor:
        if not bool(getattr(self, "profile_record_functions", False)):
            return self._chunked_lm_loss_impl(
                hidden,
                labels,
                chunk_tokens,
                sample_weights,
                loss_token_stride,
                max_loss_tokens_per_sample,
                loss_mask,
                target_boundary_weight,
                target_prefix_weight,
                target_prefix_tokens,
                collect_diagnostics,
                labels_are_sparse,
            )
        with record_function("omnicoder.pipeline.chunked_lm_loss"):
            return self._chunked_lm_loss_impl(
                hidden,
                labels,
                chunk_tokens,
                sample_weights,
                loss_token_stride,
                max_loss_tokens_per_sample,
                loss_mask,
                target_boundary_weight,
                target_prefix_weight,
                target_prefix_tokens,
                collect_diagnostics,
                labels_are_sparse,
            )

    def _chunked_lm_loss_impl(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        chunk_tokens: int = 128,
        sample_weights: torch.Tensor | None = None,
        loss_token_stride: int = 1,
        max_loss_tokens_per_sample: int = 0,
        loss_mask: torch.Tensor | None = None,
        target_boundary_weight: float = 1.0,
        target_prefix_weight: float = 1.0,
        target_prefix_tokens: int = 0,
        collect_diagnostics: bool = True,
        labels_are_sparse: bool | None = None,
    ) -> torch.Tensor:
        if not self.spec.has_head:
            raise RuntimeError("LM loss can only be computed on the final pipeline stage")
        loss_started = _monotonic()
        loss_spans: dict[str, float] = {}

        @contextlib.contextmanager
        def loss_span(name: str):
            started = _monotonic()
            try:
                yield
            finally:
                loss_spans[str(name)] = float(loss_spans.get(str(name), 0.0) + (_monotonic() - started))

        def set_loss_diagnostics(diagnostics: dict[str, Any], *, sparse_path: bool) -> None:
            timing = {
                "schema": "omnicoder.lm_loss_timing_2026.v1",
                "total_sec": float(_monotonic() - loss_started),
                "spans": {str(key): float(value) for key, value in loss_spans.items()},
                "chunk_tokens": int(chunk_tokens),
                "collect_diagnostics": bool(collect_diagnostics),
                "sparse_path": bool(sparse_path),
                "labels_are_sparse": bool(sparse_target_labels),
                "loss_token_stride": int(loss_token_stride),
                "max_loss_tokens_per_sample": int(max_loss_tokens_per_sample),
            }
            self.last_lm_loss_timing = timing
            diagnostics = dict(diagnostics)
            diagnostics["timing"] = timing
            self.last_lm_loss_diagnostics = diagnostics

        with loss_span("label_to_device_sec"):
            if labels.device != hidden.device:
                labels = labels.to(hidden.device, non_blocking=True)
            if loss_mask is not None and loss_mask.device != hidden.device:
                loss_mask = loss_mask.to(hidden.device, non_blocking=True)
        with loss_span("mask_build_sec"):
            shifted_hidden = hidden[:, :-1, :]
            shifted_labels = labels[:, 1:]
            sparse_target_labels = bool(labels_are_sparse) if labels_are_sparse is not None else bool(labels.eq(-100).any())
            target_mask = shifted_labels.ge(0)
            if loss_mask is not None:
                if tuple(loss_mask.shape) != tuple(labels.shape):
                    raise ValueError(f"loss_mask shape mismatch: mask={tuple(loss_mask.shape)} labels={tuple(labels.shape)}")
                target_mask = target_mask & loss_mask[:, 1:].to(dtype=torch.bool)
        collect_diagnostics = bool(collect_diagnostics)
        ce_accumulator = _new_ce_accumulator() if collect_diagnostics else {}
        available_target_tokens = 0
        optimized_target_tokens = 0
        with loss_span("boundary_prefix_mask_sec"):
            boundary_weight = max(1.0, float(target_boundary_weight or 1.0))
            prev_target_mask = F.pad(target_mask[:, :-1], (1, 0), value=False)
            boundary_mask = target_mask & ~prev_target_mask
            prefix_weight = max(1.0, float(target_prefix_weight or 1.0))
            prefix_tokens = max(0, int(target_prefix_tokens or 0))
            prefix_mask = torch.zeros_like(target_mask, dtype=torch.bool)
            if prefix_tokens > 0:
                target_rank = target_mask.long().cumsum(dim=1)
                boundary_rank = target_rank * boundary_mask.long()
                last_boundary_rank = torch.cummax(boundary_rank, dim=1).values
                prefix_index = target_rank - last_boundary_rank
                prefix_mask = target_mask & last_boundary_rank.gt(0) & prefix_index.lt(prefix_tokens)

        def _token_weight(mask: torch.Tensor, boundary: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
            weights = torch.ones_like(mask, dtype=torch.float32)
            if boundary_weight > 1.0:
                weights = torch.where(boundary, torch.full_like(weights, boundary_weight), weights)
            if prefix_weight > 1.0:
                weights = torch.where(prefix, torch.maximum(weights, torch.full_like(weights, prefix_weight)), weights)
            return weights

        def _aligned_weights(batch_size: int) -> torch.Tensor | None:
            if sample_weights is None:
                return None
            weights = sample_weights.to(hidden.device, non_blocking=True).to(dtype=torch.float32).reshape(-1)
            if weights.numel() == batch_size:
                return weights
            if weights.numel() > 0:
                return weights.mean().expand(batch_size)
            return hidden.new_ones((batch_size,), dtype=torch.float32)
        loss_token_stride = max(1, int(loss_token_stride))
        max_loss_tokens_per_sample = max(0, int(max_loss_tokens_per_sample))
        if sparse_target_labels or loss_token_stride > 1 or max_loss_tokens_per_sample > 0:
            with loss_span("selected_position_scan_sec"):
                selected_hidden: list[torch.Tensor] = []
                selected_labels: list[torch.Tensor] = []
                selected_batches: list[torch.Tensor] = []
                selected_token_weights: list[torch.Tensor] = []
                for batch_index in range(int(shifted_hidden.shape[0])):
                    positions = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    available_target_tokens += int(positions.numel())
                    if positions.numel() == 1:
                        boundary_positions = positions
                    else:
                        span_starts = torch.empty_like(positions, dtype=torch.bool)
                        span_starts[0] = True
                        span_starts[1:] = positions[1:].ne(positions[:-1] + 1)
                        boundary_positions = positions[span_starts]
                    if not sparse_target_labels and loss_token_stride > 1:
                        positions = positions[::loss_token_stride]
                        if positions.numel() == 0:
                            positions = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()[-1:]
                    if boundary_positions.numel() > 0:
                        positions = torch.unique(torch.cat((positions, boundary_positions)), sorted=True)
                    prefix_positions = (
                        torch.nonzero(prefix_mask[batch_index], as_tuple=False).flatten()
                        if prefix_tokens > 0
                        else positions.new_empty((0,))
                    )
                    if prefix_positions.numel() > 0:
                        positions = torch.unique(torch.cat((positions, prefix_positions)), sorted=True)
                    if max_loss_tokens_per_sample > 0 and positions.numel() > max_loss_tokens_per_sample:
                        priority_positions = torch.unique(torch.cat((boundary_positions, prefix_positions)), sorted=True)
                        if sparse_target_labels and priority_positions.numel() > 0:
                            if priority_positions.numel() >= max_loss_tokens_per_sample:
                                pick = torch.linspace(
                                    0,
                                    priority_positions.numel() - 1,
                                    steps=max_loss_tokens_per_sample,
                                    device=priority_positions.device,
                                    dtype=torch.float32,
                                ).round().long().unique(sorted=True)
                                positions = priority_positions[pick]
                            else:
                                pick = torch.linspace(
                                    0,
                                    positions.numel() - 1,
                                    steps=max_loss_tokens_per_sample,
                                    device=positions.device,
                                    dtype=torch.float32,
                                ).round().long().unique(sorted=True)
                                positions = torch.unique(torch.cat((priority_positions, positions[pick])), sorted=True)
                                if positions.numel() > max_loss_tokens_per_sample:
                                    pick = torch.linspace(
                                        0,
                                        positions.numel() - 1,
                                        steps=max_loss_tokens_per_sample,
                                        device=positions.device,
                                        dtype=torch.float32,
                                    ).round().long().unique(sorted=True)
                                    positions = positions[pick]
                        else:
                            pick = torch.linspace(
                                0,
                                positions.numel() - 1,
                                steps=max_loss_tokens_per_sample,
                                device=positions.device,
                                dtype=torch.float32,
                            ).round().long().unique(sorted=True)
                            positions = positions[pick]
                    selected_hidden.append(shifted_hidden[batch_index, positions, :])
                    selected_labels.append(shifted_labels[batch_index, positions])
                    selected_batches.append(torch.full((positions.numel(),), batch_index, dtype=torch.long, device=hidden.device))
                    selected_token_weights.append(
                        _token_weight(
                            target_mask[batch_index, positions],
                            boundary_mask[batch_index, positions],
                            prefix_mask[batch_index, positions],
                        )
                    )
            if not selected_hidden:
                set_loss_diagnostics(
                    _lm_loss_diagnostics(labels, 0, ce_accumulator) if collect_diagnostics else _minimal_lm_loss_diagnostics(available_target_tokens, 0),
                    sparse_path=True,
                )
                return hidden.sum() * 0.0
            with loss_span("selected_concat_sec"):
                flat_hidden = torch.cat(selected_hidden, dim=0)
                flat_labels = torch.cat(selected_labels, dim=0)
                flat_batches = torch.cat(selected_batches, dim=0)
                flat_token_weights = torch.cat(selected_token_weights, dim=0) if selected_token_weights else None
                losses = flat_hidden.new_empty((flat_hidden.shape[0],), dtype=torch.float32)
            with loss_span("selected_lm_head_ce_sec"):
                for start in range(0, flat_hidden.shape[0], max(1, int(chunk_tokens))):
                    end = min(flat_hidden.shape[0], start + int(chunk_tokens))
                    logits = self.lm_head(flat_hidden[start:end, :])
                    losses[start:end] = F.cross_entropy(logits, flat_labels[start:end], reduction="none").float()
            optimized_target_tokens = int(flat_labels.numel())
            if collect_diagnostics:
                with loss_span("ce_diagnostics_sec"):
                    _accumulate_ce_by_token_family(ce_accumulator, flat_labels, losses, flat_token_weights)
            with loss_span("selected_reduce_sec"):
                if flat_token_weights is not None:
                    losses = losses * flat_token_weights
                if sample_weights is not None:
                    weights = _aligned_weights(int(shifted_hidden.shape[0]))
                    assert weights is not None
                    per_sample_sum = hidden.new_zeros((shifted_hidden.shape[0],), dtype=torch.float32)
                    per_sample_tokens = hidden.new_zeros((shifted_hidden.shape[0],), dtype=torch.float32)
                    per_sample_sum.index_add_(0, flat_batches, losses)
                    per_sample_tokens.index_add_(0, flat_batches, flat_token_weights if flat_token_weights is not None else torch.ones_like(losses))
                    per_sample = per_sample_sum / per_sample_tokens.clamp_min(1.0)
                    out = (per_sample * weights).mean().to(dtype=hidden.dtype)
                    set_loss_diagnostics(
                        _lm_loss_diagnostics(labels, optimized_target_tokens, ce_accumulator) if collect_diagnostics else _minimal_lm_loss_diagnostics(available_target_tokens, optimized_target_tokens),
                        sparse_path=True,
                    )
                    return out
                if flat_token_weights is not None:
                    out = (losses.sum() / flat_token_weights.sum().clamp_min(1.0)).to(dtype=hidden.dtype)
                    set_loss_diagnostics(
                        _lm_loss_diagnostics(labels, optimized_target_tokens, ce_accumulator) if collect_diagnostics else _minimal_lm_loss_diagnostics(available_target_tokens, optimized_target_tokens),
                        sparse_path=True,
                    )
                    return out
                out = losses.mean().to(dtype=hidden.dtype)
                set_loss_diagnostics(
                    _lm_loss_diagnostics(labels, optimized_target_tokens, ce_accumulator) if collect_diagnostics else _minimal_lm_loss_diagnostics(available_target_tokens, optimized_target_tokens),
                    sparse_path=True,
                )
                return out
        if sample_weights is not None:
            with loss_span("dense_weighted_lm_head_ce_sec"):
                weights = _aligned_weights(int(shifted_hidden.shape[0]))
                assert weights is not None
                per_sample_sum = hidden.new_zeros((shifted_hidden.shape[0],), dtype=torch.float32)
                per_sample_tokens = hidden.new_zeros((shifted_hidden.shape[0],), dtype=torch.float32)
                for start in range(0, shifted_hidden.shape[1], max(1, int(chunk_tokens))):
                    end = min(shifted_hidden.shape[1], start + int(chunk_tokens))
                    logits = self.lm_head(shifted_hidden[:, start:end, :])
                    token_losses = F.cross_entropy(logits.transpose(1, 2), shifted_labels[:, start:end], reduction="none").float()
                    mask = target_mask[:, start:end].float()
                    mask = mask * _token_weight(target_mask[:, start:end], boundary_mask[:, start:end], prefix_mask[:, start:end])
                    if collect_diagnostics:
                        with loss_span("ce_diagnostics_sec"):
                            _accumulate_ce_by_token_family(ce_accumulator, shifted_labels[:, start:end], token_losses, mask)
                    per_sample_sum = per_sample_sum + (token_losses * mask).sum(dim=1)
                    per_sample_tokens = per_sample_tokens + mask.sum(dim=1)
            with loss_span("dense_weighted_reduce_sec"):
                per_sample = per_sample_sum / per_sample_tokens.clamp_min(1.0)
                if collect_diagnostics:
                    with loss_span("diagnostic_scalar_sec"):
                        optimized_target_tokens = int(target_mask.sum().detach().cpu().item())
                else:
                    optimized_target_tokens = int(shifted_labels.numel())
                out = (per_sample * weights).mean().to(dtype=hidden.dtype)
                set_loss_diagnostics(
                    _lm_loss_diagnostics(labels, optimized_target_tokens, ce_accumulator) if collect_diagnostics else _minimal_lm_loss_diagnostics(optimized_target_tokens, optimized_target_tokens),
                    sparse_path=False,
                )
                return out
        with loss_span("dense_mask_sum_sec"):
            total_mask = target_mask.float() * _token_weight(target_mask, boundary_mask, prefix_mask)
            total_tokens = total_mask.sum().clamp_min(1.0)
        loss_sum = hidden.new_zeros(())
        with loss_span("dense_lm_head_ce_sec"):
            for start in range(0, shifted_hidden.shape[1], max(1, int(chunk_tokens))):
                end = min(shifted_hidden.shape[1], start + int(chunk_tokens))
                logits = self.lm_head(shifted_hidden[:, start:end, :])
                token_losses = F.cross_entropy(logits.transpose(1, 2), shifted_labels[:, start:end], reduction="none").float()
                if collect_diagnostics:
                    with loss_span("ce_diagnostics_sec"):
                        _accumulate_ce_by_token_family(ce_accumulator, shifted_labels[:, start:end], token_losses, total_mask[:, start:end])
                loss_sum = loss_sum + (token_losses * total_mask[:, start:end]).sum()
        if collect_diagnostics:
            with loss_span("diagnostic_scalar_sec"):
                optimized_target_tokens = int(target_mask.sum().detach().cpu().item())
        else:
            optimized_target_tokens = int(shifted_labels.numel())
        out = loss_sum / total_tokens.to(dtype=loss_sum.dtype)
        set_loss_diagnostics(
            _lm_loss_diagnostics(labels, optimized_target_tokens, ce_accumulator) if collect_diagnostics else _minimal_lm_loss_diagnostics(optimized_target_tokens, optimized_target_tokens),
            sparse_path=False,
        )
        return out

    def local_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if not key.endswith("._metadata")}


def load_full_checkpoint_shard(path: str | Path, shard: OmniCoder2026PipelineShard) -> tuple[int, float | None]:
    return load_checkpoint_shard(path, shard)


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _unlink_if_exists(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _rank_checkpoint_file(target: str | Path, rank: int) -> Path:
    return Path(target) / f"rank{int(rank):05d}.pt"


def _rank_complete_file(target: str | Path, rank: int) -> Path:
    return Path(str(_rank_checkpoint_file(target, rank)) + ".complete.json")


def _checkpoint_complete_marker(path: str | Path) -> Path:
    return Path(str(path) + ".complete.json")


def _read_json_dict(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _pid_exists(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError):
        return False
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _manifest_sha256(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("sha256", "data_sha256", "input_sha256", "source_sha256"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("data", "dataset", "source", "manifest"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            value = _manifest_sha256(nested)
            if value:
                return value
    return None


def _checkpoint_data_integrity(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_checkpoint_data_integrity", None)
    if isinstance(cached, dict):
        return cached
    data_path = str(getattr(args, "data", "") or "")
    manifest_path = str(getattr(args, "data_manifest", "") or "")
    explicit_sha = str(getattr(args, "data_sha256", "") or "").strip()
    policy = str(getattr(args, "checkpoint_data_hash_policy", "manifest") or "manifest").strip().lower()
    out: dict[str, Any] = {
        "path": data_path,
        "sha256": explicit_sha or None,
        "manifest": manifest_path or None,
        "hash_policy": policy,
        "hash_source": "explicit" if explicit_sha else None,
        "hash_duration_sec": 0.0,
    }
    if not out["sha256"] and manifest_path:
        started = _monotonic()
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            out["sha256"] = _manifest_sha256(manifest)
            out["hash_source"] = "manifest" if out["sha256"] else "manifest_missing_sha256"
        except Exception as exc:
            out["hash_source"] = "manifest_error"
            out["hash_error"] = str(exc)
        out["hash_duration_sec"] = float(_monotonic() - started)
    if not out["sha256"] and policy in {"once", "always"} and data_path and Path(data_path).exists():
        started = _monotonic()
        out["sha256"] = _sha256_file(data_path)
        out["hash_source"] = f"file_{policy}"
        out["hash_duration_sec"] = float(_monotonic() - started)
    if not out["sha256"] and not out["hash_source"]:
        out["hash_source"] = "skipped"
    setattr(args, "_checkpoint_data_integrity", out)
    return out


def _atomic_torch_save_with_metrics(payload: dict[str, Any], path: str | Path) -> dict[str, Any]:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    metrics: dict[str, Any] = {
        "path": str(final_path),
        "tmp_path": str(tmp_path),
        "tmp_bytes": 0,
        "final_bytes": 0,
        "torch_save_tmp_sec": 0.0,
        "rename_sec": 0.0,
        "marker_write_sec": 0.0,
        "bytes_per_sec": None,
    }
    try:
        started = _monotonic()
        torch.save(payload, tmp_path)
        metrics["torch_save_tmp_sec"] = float(_monotonic() - started)
        metrics["tmp_bytes"] = _file_size_or_zero(tmp_path)
        started = _monotonic()
        os.replace(tmp_path, final_path)
        metrics["rename_sec"] = float(_monotonic() - started)
        metrics["final_bytes"] = _file_size_or_zero(final_path)
        marker = {
            "status": "complete",
            "path": str(final_path),
            "bytes": metrics["final_bytes"],
            "format": payload.get("format"),
            "global_step": payload.get("global_step"),
            "last_loss": payload.get("last_loss"),
        }
        started = _monotonic()
        _checkpoint_complete_marker(final_path).write_text(json.dumps(marker, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        metrics["marker_write_sec"] = float(_monotonic() - started)
        save_time = float(metrics["torch_save_tmp_sec"] or 0.0)
        metrics["bytes_per_sec"] = (float(metrics["final_bytes"]) / save_time) if save_time > 0.0 else None
        return metrics
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _checkpoint_attempt_file(target: str | Path) -> Path:
    return Path(target) / CHECKPOINT_ATTEMPT_MARKER


def _rank_marker_matches(path: Path, *, attempt_id: str, rank: int, world_size: int, global_step: int) -> bool:
    payload = _read_json_dict(path)
    if not payload:
        return False
    return (
        payload.get("status") == "complete"
        and str(payload.get("checkpoint_attempt_id") or "") == str(attempt_id)
        and int(payload.get("rank", -1)) == int(rank)
        and int(payload.get("world_size", -1)) == int(world_size)
        and int(payload.get("global_step", -1)) == int(global_step)
        and _rank_checkpoint_file(path.parent, rank).exists()
    )


def _directory_marker_matches(path: Path, *, attempt_id: str, world_size: int, global_step: int) -> bool:
    payload = _read_json_dict(path)
    if not payload:
        return False
    return (
        payload.get("status") == "complete"
        and str(payload.get("checkpoint_attempt_id") or "") == str(attempt_id)
        and int(payload.get("world_size", -1)) == int(world_size)
        and int(payload.get("global_step", -1)) == int(global_step)
    )


def _wait_for_checkpoint_attempt(
    target: str | Path,
    *,
    world_size: int,
    global_step: int,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    attempt_path = _checkpoint_attempt_file(target)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        payload = _read_json_dict(attempt_path)
        rank0_host = str(payload.get("rank0_host") or "") if payload else ""
        rank0_alive = not rank0_host or rank0_host != socket.gethostname() or _pid_exists(payload.get("rank0_pid"))
        if (
            payload
            and payload.get("status") == "ready"
            and payload.get("checkpoint_attempt_id")
            and int(payload.get("world_size", -1)) == int(world_size)
            and int(payload.get("global_step", -1)) == int(global_step)
            and rank0_alive
        ):
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for checkpoint save attempt marker {attempt_path}")
        time.sleep(max(0.05, float(poll_seconds)))


def _wait_for_rank_checkpoint_markers(
    target: str | Path,
    *,
    world_size: int,
    global_step: int,
    attempt_id: str,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
) -> list[Path]:
    checkpoint_dir = Path(target)
    expected = [_rank_complete_file(checkpoint_dir, rank) for rank in range(int(world_size))]
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    missing = [
        path
        for rank, path in enumerate(expected)
        if not _rank_marker_matches(path, attempt_id=attempt_id, rank=rank, world_size=world_size, global_step=global_step)
    ]
    while missing:
        if time.monotonic() >= deadline:
            names = ", ".join(path.name for path in missing[:8])
            raise TimeoutError(f"timed out waiting for rank checkpoint markers in {checkpoint_dir}: {names}")
        time.sleep(max(0.05, float(poll_seconds)))
        missing = [
            path
            for rank, path in enumerate(expected)
            if not _rank_marker_matches(path, attempt_id=attempt_id, rank=rank, world_size=world_size, global_step=global_step)
        ]
    return expected


def _wait_for_directory_checkpoint_marker(
    target: str | Path,
    *,
    world_size: int,
    global_step: int,
    attempt_id: str,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
) -> None:
    complete_path = Path(target) / ".complete.json"
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        if _directory_marker_matches(complete_path, attempt_id=attempt_id, world_size=world_size, global_step=global_step):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for complete checkpoint marker {complete_path}")
        time.sleep(max(0.05, float(poll_seconds)))


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
        text = f"user: {prompt}\nassistant: {record.get('chosen', '')}"
        weight = 1.5
    else:
        text = _text_from_record(record)
        weight = 1.0 + max(0.0, reward) * 0.25
    return text.strip(), max(0.05, min(2.5, float(weight)))


MEDIA_TARGET_KEYS = {
    "artifact_path",
    "artifact_uri",
    "artifact_tokens",
    "output_modality",
    "codec",
    "codec_id",
    "media_tokens",
    "audio_tokens",
    "video_tokens",
    "image_tokens",
    "tts_tokens",
    "music_tokens",
    "ocr_result",
    "ocr_text",
    "artifact_refs",
    "artifacts",
    "artifact_paths",
    "media_refs",
    "media_paths",
}

MEDIA_INPUT_KEYS = {
    "input_modality",
    "source_modality",
    "media_tokens",
    "input_media_tokens",
    "artifact_tokens",
    "image_tokens",
    "video_tokens",
    "audio_tokens",
    "speech_tokens",
    "tts_tokens",
    "music_tokens",
    "ocr_image_tokens",
    "image_path",
    "video_path",
    "audio_path",
    "reference_image",
    "reference_video",
    "reference_audio",
}

MEDIA_ROUTE_NAMES = {"image", "video", "music", "tts", "speech", "audio", "ocr"}


def _ordered_json_value(value: object) -> object:
    if isinstance(value, dict):
        priority = (
            "input_modality",
            "source_modality",
            "output_modality",
            "task",
            "ocr_text",
            "input_media_tokens",
            "artifact_tokens",
            "media_tokens",
            "image_tokens",
            "video_tokens",
            "audio_tokens",
            "speech_tokens",
            "tts_tokens",
            "music_tokens",
            "ocr_image_tokens",
            "image_path",
            "video_path",
            "audio_path",
            "reference_image",
            "reference_video",
            "reference_audio",
            "artifact_path",
            "artifact_uri",
            "codec",
            "codec_id",
            "ocr_result",
        )
        ordered: dict[str, object] = {}
        for key in priority:
            if key in value:
                ordered[key] = _ordered_json_value(value[key])
        for key in sorted(str(k) for k in value.keys() if str(k) not in ordered):
            ordered[key] = _ordered_json_value(value[key])
        return ordered
    if isinstance(value, list):
        return [_ordered_json_value(item) for item in value]
    return value


def _content_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(_ordered_json_value(value), ensure_ascii=True, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _canonical_output_route(value: object) -> str:
    route = str(value or "").strip().lower()
    if route == "speech":
        return "tts"
    if route == "audio":
        return "music"
    return route


def _target_json_route(target_json: dict[str, Any]) -> str:
    task = _canonical_output_route(target_json.get("task"))
    if task == "ocr" or "ocr_text" in target_json or "ocr_result" in target_json:
        return "ocr"
    modality = _canonical_output_route(target_json.get("output_modality"))
    if modality in MEDIA_ROUTE_NAMES:
        return modality
    artifact = str(target_json.get("artifact_tokens") or target_json.get("media_tokens") or "").lower()
    if "<image_" in artifact:
        return "image"
    if "<video_" in artifact:
        return "video"
    if "<music_" in artifact:
        return "music"
    if "<speech_" in artifact or "<tts_" in artifact:
        return "tts"
    return ""


def _has_nonempty_media_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_nonempty_media_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_nonempty_media_value(item) for item in value)
    return True


def _target_json_has_media_payload(target_json: dict[str, Any]) -> bool:
    for key in MEDIA_TARGET_KEYS:
        if key == "output_modality":
            continue
        if key in target_json and _has_nonempty_media_value(target_json.get(key)):
            return True
    return False


def _record_output_route(record: dict[str, Any], target_json: dict[str, Any] | None = None) -> str:
    top_level = _canonical_output_route(record.get("modality") or record.get("target_modality"))
    if top_level in MEDIA_ROUTE_NAMES:
        return top_level
    if isinstance(target_json, dict):
        route = _target_json_route(target_json)
        if route:
            return route
    return ""


def _is_media_target_content(value: object) -> bool:
    if isinstance(value, dict):
        return _target_json_has_media_payload(value) or _target_json_route(value) in MEDIA_ROUTE_NAMES
    if isinstance(value, str):
        text = value.lower()
        return (
            '"output_modality"' in text
            or '"artifact_tokens"' in text
            or "<image_" in text
            or "<video_" in text
            or "<music_" in text
            or "<speech_" in text
            or '"ocr_text"' in text
            or '"task":"ocr"' in text
        )
    return False


def _input_context_payload(input_json: dict[str, Any], *, emitted_prompt: bool) -> object:
    consumed = {"messages", "content", "prompt", "text", "instruction"}
    payload = {str(key): value for key, value in input_json.items() if str(key) not in consumed and value is not None}
    if not payload:
        return {}
    has_media = any(key in payload for key in MEDIA_INPUT_KEYS)
    if has_media or not emitted_prompt:
        return payload
    return {}


def _content_output_route(value: object) -> str:
    if isinstance(value, dict):
        return _target_json_route(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                route = _target_json_route(payload)
                if route:
                    return route
        text = stripped.lower()
        if "<image_" in text:
            return "image"
        if "<video_" in text:
            return "video"
        if "<music_" in text:
            return "music"
        if "<speech_" in text or "<tts_" in text:
            return "tts"
        if '"task":"ocr"' in text or '"ocr_text"' in text:
            return "ocr"
    return ""


def _assistant_route_prefix(output_route: str) -> str:
    route = _canonical_output_route(output_route)
    return f"{route} | " if route in MEDIA_ROUTE_NAMES else ""


def _with_assistant_route_prefix(body: str, output_route: str) -> str:
    prefix = _assistant_route_prefix(output_route)
    if not prefix:
        return body
    stripped = body.lstrip()
    if stripped.lower().startswith(prefix.lower()):
        return body
    return f"{prefix}{stripped}"


def _token_id_list(value: object) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    ids: list[int] = []
    for item in value:
        try:
            ids.append(int(item))
        except Exception:
            return None
    return ids


def _mask_list(value: object, length: int) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    mask: list[float] = []
    for item in value[:length]:
        try:
            mask.append(1.0 if float(item) > 0.0 else 0.0)
        except Exception:
            mask.append(0.0)
    if len(mask) < length:
        mask.extend([0.0] * (length - len(mask)))
    return mask


def _labels_to_mask(value: object, length: int) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    mask: list[float] = []
    for item in value[:length]:
        try:
            label = int(item)
            mask.append(0.0 if label == -100 else 1.0)
        except Exception:
            mask.append(0.0)
    if len(mask) < length:
        mask.extend([0.0] * (length - len(mask)))
    return mask


def _contains_token_subsequence(haystack: list[int], needle: list[int]) -> bool:
    if not needle:
        return True
    if len(needle) > len(haystack):
        return False
    if len(needle) == len(haystack):
        return haystack == needle
    limit = len(haystack) - len(needle) + 1
    for start in range(limit):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


def _explicit_token_ids_and_mask(record: dict[str, Any]) -> tuple[list[int], list[float]] | None:
    prompt_ids: list[int] = []
    for key in ("prompt_token_ids", "prompt_ids", "input_token_ids"):
        value = _token_id_list(record.get(key))
        if value:
            prompt_ids = value
            break
    target_ids: list[int] = []
    for key in ("target_token_ids", "completion_token_ids", "assistant_token_ids", "media_token_ids", "artifact_token_ids"):
        value = _token_id_list(record.get(key))
        if value:
            # Some curated media rows carry artifact_token_ids as metadata for
            # indexing while target_token_ids already contains the exact same
            # output sequence. Do not silently double the supervised target.
            if not _contains_token_subsequence(target_ids, value):
                target_ids.extend(value)
    if target_ids:
        if not prompt_ids:
            # Explicit media/assistant target rows can be target-only. Prepend a
            # masked context token so even a one-token target has next-token CE.
            prompt_ids = [0]
        return prompt_ids + target_ids, ([0.0] * len(prompt_ids)) + ([1.0] * len(target_ids))
    ids = _ids_from_record(record)
    if not ids:
        return None
    ids = [int(x) for x in ids]
    for key in ("loss_mask", "target_mask", "assistant_loss_mask", "labels_mask"):
        mask = _mask_list(record.get(key), len(ids))
        if mask is not None:
            return ids, mask
    labels_mask = _labels_to_mask(record.get("labels"), len(ids))
    if labels_mask is not None:
        return ids, labels_mask
    return ids, [1.0] * len(ids)


def _tokenizer_encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    raw = getattr(tokenizer, "_tok", None)
    if raw is not None:
        try:
            encoding = raw.encode(text)
            ids = [int(x) for x in encoding.ids]
            offsets = [(int(start), int(end)) for start, end in encoding.offsets]
            if len(ids) == len(offsets) and all(0 <= start <= end <= len(text) for start, end in offsets):
                return ids, offsets
        except Exception:
            pass
        try:
            encoded = raw(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = [int(x) for x in getattr(encoded, "input_ids", encoded["input_ids"])]
            offsets_raw = getattr(encoded, "offset_mapping", encoded["offset_mapping"])
            offsets = [(int(start), int(end)) for start, end in offsets_raw]
            if len(ids) == len(offsets) and all(0 <= start <= end <= len(text) for start, end in offsets):
                return ids, offsets
        except Exception:
            pass
    ids = [int(x) for x in tokenizer.encode(text)]
    if len(ids) == len(text):
        return ids, [(index, index + 1) for index in range(len(text))]
    return ids, None


def _encode_segments_with_mask(tokenizer: Any, segments: list[tuple[str, bool]]) -> tuple[list[int], list[float]]:
    full_text = "".join(text for text, _is_target in segments if text)
    if full_text:
        char_mask: list[bool] = []
        for text, is_target in segments:
            if not text:
                continue
            char_mask.extend([bool(is_target)] * len(text))
        ids, offsets = _tokenizer_encode_with_offsets(tokenizer, full_text)
        if offsets is not None and len(offsets) == len(ids):
            mask: list[float] = []
            for start, end in offsets:
                if end <= start:
                    mask.append(0.0)
                    continue
                lo = max(0, int(start))
                hi = min(len(char_mask), int(end))
                mask.append(1.0 if any(char_mask[lo:hi]) else 0.0)
            return ids, mask
        if ids:
            mask = [0.0] * len(ids)
            cursor = 0
            while cursor < len(char_mask):
                if not char_mask[cursor]:
                    cursor += 1
                    continue
                start = cursor
                while cursor < len(char_mask) and char_mask[cursor]:
                    cursor += 1
                end = cursor
                # Offset-less tokenizers still need full-string tokenization so
                # merges across prompt/assistant boundaries stay intact. Map
                # target character spans to token spans by re-encoding bounded
                # prefixes instead of falling back to per-segment tokenization.
                try:
                    token_start = len([int(x) for x in tokenizer.encode(full_text[:start])])
                    token_end = len([int(x) for x in tokenizer.encode(full_text[:end])])
                except Exception:
                    token_start = token_end = 0
                if token_end <= token_start:
                    token_end = min(len(ids), token_start + 1)
                for index in range(max(0, token_start), min(len(ids), token_end)):
                    mask[index] = 1.0
            if any(value > 0.0 for value in mask) or not any(bool(is_target) for _text, is_target in segments):
                return ids, mask
    ids: list[int] = []
    mask: list[float] = []
    for text, is_target in segments:
        if not text:
            continue
        part = [int(x) for x in tokenizer.encode(text)]
        ids.extend(part)
        mask.extend([1.0 if is_target else 0.0] * len(part))
    return ids, mask


def _append_role_line_segments(
    segments: list[tuple[str, bool]],
    role: str,
    content: object,
    *,
    output_route: str = "",
) -> None:
    body = _content_to_text(content)
    if not body:
        return
    if segments:
        segments.append(("\n", False))
    normalized = str(role or "message").lower()
    if normalized == "assistant":
        # Keep the leading space in the target piece so BPE tokenizers learn the
        # first answer/media token after an inference prompt ending in
        # "assistant:".
        segments.append((f"{role}:", False))
        route = output_route or _content_output_route(content)
        if route and _is_media_target_content(content):
            body = _with_assistant_route_prefix(body, route)
        segments.append((f" {body}", True))
        return
    segments.append((f"{role}: {body}", False))


def _message_segments(messages: object, *, output_route: str = "") -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    if not isinstance(messages, list):
        return segments
    for message in messages:
        if not isinstance(message, dict):
            continue
        _append_role_line_segments(
            segments,
            str(message.get("role") or "message"),
            message.get("content"),
            output_route=output_route,
        )
        tool_calls = message.get("tool_calls")
        if str(message.get("role") or "").lower() == "assistant" and tool_calls:
            segments.append((" " + _content_to_text({"tool_calls": tool_calls}), True))
    return segments


def _encode_text_with_assistant_targets(tokenizer: Any, text: str, *, default_target: bool) -> tuple[list[int], list[float]]:
    segments: list[tuple[str, bool]] = []
    role_prefixes = ("user:", "system:", "developer:", "tool:", "observation:")
    for index, line in enumerate(str(text).splitlines()):
        if index > 0:
            previous_target = bool(segments[-1][1]) if segments else bool(default_target)
            segments.append(("\n", previous_target))
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]
        lower = stripped.lower()
        if lower.startswith("assistant:"):
            colon = line.lower().find("assistant:") + len("assistant:")
            segments.append((line[:colon], False))
            if len(line) > colon:
                segments.append((line[colon:], True))
        elif any(lower.startswith(prefix) for prefix in role_prefixes):
            segments.append((line, False))
        else:
            segments.append((leading + stripped, default_target))
    return _encode_segments_with_mask(tokenizer, segments)


def _target_json_segments(record: dict[str, Any]) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    target_json = record.get("target_json")
    if isinstance(target_json, dict):
        output_route = _record_output_route(record, target_json)
        emitted_text = False
        for key in ("content", "completion", "answer", "caption", "transcript", "ocr_text"):
            value = target_json.get(key)
            if value:
                _append_role_line_segments(segments, "assistant", value, output_route=output_route)
                emitted_text = True
        has_media_payload = _target_json_has_media_payload(target_json)
        if has_media_payload or not emitted_text:
            _append_role_line_segments(segments, "assistant", target_json, output_route=output_route)
    return segments


def _input_target_json_segments(record: dict[str, Any]) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    input_json = record.get("input_json")
    if isinstance(input_json, dict):
        segments.extend(_message_segments(input_json.get("messages")))
        emitted_prompt = bool(segments)
        for key in ("content", "prompt", "text", "instruction"):
            value = input_json.get(key)
            if value:
                _append_role_line_segments(segments, "user", value)
                emitted_prompt = True
        context_payload = _input_context_payload(input_json, emitted_prompt=emitted_prompt)
        if context_payload:
            _append_role_line_segments(segments, "user", context_payload)
    target_segments = _target_json_segments(record)
    if segments and target_segments:
        segments.append(("\n", False))
    segments.extend(target_segments)
    return segments


def _record_ids_weight_mask(record: dict[str, Any], tokenizer: Any) -> tuple[list[int], float, list[float]]:
    _, weight = _pipeline_record_to_text_and_weight(record)
    explicit = _explicit_token_ids_and_mask(record)
    if explicit is not None:
        ids, mask = explicit
        return ids, weight, mask
    if "native_media_features" in record or "native_media_targets" in record:
        raise ValueError(
            "native continuous media rows require the full OmniCoder2026 training path; "
            "the distributed pipeline trainer only accepts ledger-token media rows"
        )
    segments = _message_segments(record.get("messages"), output_route=_record_output_route(record))
    if segments and not any(bool(is_target) for _text, is_target in segments):
        target_segments = _target_json_segments(record)
        if target_segments:
            segments.append(("\n", False))
            segments.extend(target_segments)
    if not segments:
        segments = _input_target_json_segments(record)
    if segments:
        ids, mask = _encode_segments_with_mask(tokenizer, segments)
        return ids, weight, mask
    text, weight = _pipeline_record_to_text_and_weight(record)
    kind = str(record.get("training_kind") or "").lower()
    dialogue_hint = bool(kind or {"prompt", "chosen", "rejected"} <= set(record) or "assistant:" in text.lower())
    ids, mask = _encode_text_with_assistant_targets(tokenizer, text, default_target=not dialogue_hint)
    if not ids:
        ids = [int(x) for x in tokenizer.encode(text)]
        mask = [1.0] * len(ids)
    return ids, weight, mask


def _labels_from_ids_mask(ids: list[int], mask: list[float]) -> list[int]:
    labels: list[int] = []
    for token, keep in zip(ids, mask):
        token_id = int(token)
        labels.append(token_id if float(keep) > 0.0 else -100)
    if len(labels) < len(ids):
        labels.extend([-100] * (len(ids) - len(labels)))
    return labels


def record_ids_labels_weight(record: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int], float]:
    ids, weight, mask = _record_ids_weight_mask(record, tokenizer)
    if len(mask) != len(ids):
        mask = [1.0] * len(ids)
    return ids, _labels_from_ids_mask(ids, mask), weight


def _zero_token_family_counts() -> dict[str, int]:
    counts = {token_range.name: 0 for token_range in DEFAULT_LEDGER.ranges}
    counts["unknown"] = 0
    return counts


def _zero_modality_counts() -> dict[str, int]:
    modalities = sorted(set(TOKEN_FAMILY_TO_MODALITY.values()))
    return {name: 0 for name in modalities}


def _token_family_counts(labels: torch.Tensor | None) -> dict[str, int]:
    counts = _zero_token_family_counts()
    if labels is None:
        return counts
    with torch.no_grad():
        target_labels = labels.detach()
        if target_labels.ndim >= 2:
            target_labels = target_labels[:, 1:]
        elif target_labels.ndim == 1:
            target_labels = target_labels[1:]
        if target_labels.numel() == 0:
            return counts
        flat = target_labels.reshape(-1)
        valid = flat.ge(0)
        assigned = torch.zeros_like(valid, dtype=torch.bool)
        count_tensors: list[torch.Tensor] = []
        names: list[str] = []
        for token_range in DEFAULT_LEDGER.ranges:
            mask = valid & flat.ge(int(token_range.begin)) & flat.lt(int(token_range.end))
            count_tensors.append(mask.sum().to(dtype=torch.long))
            names.append(token_range.name)
            assigned = assigned | mask
        count_tensors.append((valid & ~assigned).sum().to(dtype=torch.long))
        names.append("unknown")
        packed_counts = torch.stack(count_tensors).to(device="cpu")
        for name, value in zip(names, packed_counts.tolist(), strict=False):
            counts[name] = int(value)
    return counts


def _modality_counts_from_token_families(family_counts: dict[str, int | float]) -> dict[str, int]:
    counts = _zero_modality_counts()
    for family, count in family_counts.items():
        modality = TOKEN_FAMILY_TO_MODALITY.get(str(family), "unknown")
        counts[modality] = int(counts.get(modality, 0) + int(count or 0))
    return counts


def _new_ce_accumulator() -> dict[str, dict[str, Any]]:
    acc = {token_range.name: {"loss_sum": 0.0, "weight_sum": 0.0, "tokens": 0.0} for token_range in DEFAULT_LEDGER.ranges}
    acc["unknown"] = {"loss_sum": 0.0, "weight_sum": 0.0, "tokens": 0.0}
    return acc


def _accumulator_add(bucket: dict[str, Any], key: str, value: torch.Tensor | float) -> None:
    current = bucket.get(key, 0.0)
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if isinstance(current, torch.Tensor):
            bucket[key] = current + detached.to(device=current.device)
        else:
            bucket[key] = detached + detached.new_tensor(float(current or 0.0))
        return
    if isinstance(current, torch.Tensor):
        bucket[key] = current + current.new_tensor(float(value or 0.0))
    else:
        bucket[key] = float(current or 0.0) + float(value or 0.0)


def _accumulator_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value or 0.0)


def _accumulate_ce_by_token_family(
    accumulator: dict[str, dict[str, Any]],
    target_labels: torch.Tensor,
    token_losses: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> None:
    with torch.no_grad():
        labels = target_labels.detach().reshape(-1)
        losses = token_losses.detach().float().reshape(-1)
        if labels.numel() == 0 or losses.numel() == 0:
            return
        if labels.numel() != losses.numel():
            size = min(int(labels.numel()), int(losses.numel()))
            labels = labels[:size]
            losses = losses[:size]
        valid = labels.ge(0) & torch.isfinite(losses)
        if token_weights is None:
            weights = torch.ones_like(losses, dtype=torch.float32)
        else:
            weights = token_weights.detach().float().reshape(-1)
            if weights.numel() != losses.numel():
                size = min(int(weights.numel()), int(losses.numel()))
                labels = labels[:size]
                losses = losses[:size]
                valid = valid[:size]
                weights = weights[:size]
        valid = valid & weights.gt(0)
        assigned = torch.zeros_like(valid, dtype=torch.bool)
        for token_range in DEFAULT_LEDGER.ranges:
            mask = valid & labels.ge(int(token_range.begin)) & labels.lt(int(token_range.end))
            weight_sum = weights[mask].sum()
            loss_sum = (losses[mask] * weights[mask]).sum()
            bucket = accumulator[token_range.name]
            _accumulator_add(bucket, "loss_sum", loss_sum)
            _accumulator_add(bucket, "weight_sum", weight_sum)
            _accumulator_add(bucket, "tokens", mask.sum().to(dtype=torch.float32))
            assigned = assigned | mask
        unknown_mask = valid & ~assigned
        weight_sum = weights[unknown_mask].sum()
        loss_sum = (losses[unknown_mask] * weights[unknown_mask]).sum()
        bucket = accumulator["unknown"]
        _accumulator_add(bucket, "loss_sum", loss_sum)
        _accumulator_add(bucket, "weight_sum", weight_sum)
        _accumulator_add(bucket, "tokens", unknown_mask.sum().to(dtype=torch.float32))


def _finalize_ce_accumulator(accumulator: dict[str, dict[str, Any]]) -> tuple[dict[str, float | None], dict[str, float | None], dict[str, int]]:
    ce_by_family: dict[str, float | None] = {}
    family_counts: dict[str, int] = {}
    modality_loss_sum: dict[str, float] = {}
    modality_weight_sum: dict[str, float] = {}
    for family, stats in accumulator.items():
        weight_sum = _accumulator_float(stats.get("weight_sum", 0.0))
        loss_sum = _accumulator_float(stats.get("loss_sum", 0.0))
        token_count = int(_accumulator_float(stats.get("tokens", 0.0)))
        family_counts[family] = token_count
        ce_by_family[family] = (loss_sum / weight_sum) if weight_sum > 0.0 else None
        modality = TOKEN_FAMILY_TO_MODALITY.get(str(family), "unknown")
        modality_loss_sum[modality] = modality_loss_sum.get(modality, 0.0) + loss_sum
        modality_weight_sum[modality] = modality_weight_sum.get(modality, 0.0) + weight_sum
    ce_by_modality: dict[str, float | None] = {}
    for modality in _zero_modality_counts():
        weight_sum = float(modality_weight_sum.get(modality, 0.0) or 0.0)
        ce_by_modality[modality] = (float(modality_loss_sum.get(modality, 0.0) or 0.0) / weight_sum) if weight_sum > 0.0 else None
    return ce_by_family, ce_by_modality, family_counts


def _lm_loss_diagnostics(labels: torch.Tensor, optimized_target_tokens: int, ce_accumulator: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target_counts = _token_family_counts(labels)
    ce_by_family, ce_by_modality, optimized_counts = _finalize_ce_accumulator(ce_accumulator)
    return {
        "schema": "omnicoder.lm_loss_diagnostics_2026.v1",
        "valid_target_tokens": int(sum(target_counts.values())),
        "optimized_target_tokens": int(optimized_target_tokens),
        "target_counts_by_token_family": target_counts,
        "target_counts_by_modality": _modality_counts_from_token_families(target_counts),
        "optimized_target_counts_by_token_family": optimized_counts,
        "optimized_target_counts_by_modality": _modality_counts_from_token_families(optimized_counts),
        "ce_by_token_family": ce_by_family,
        "ce_by_modality": ce_by_modality,
    }


def _minimal_lm_loss_diagnostics(valid_target_tokens: int, optimized_target_tokens: int) -> dict[str, Any]:
    target_counts = _zero_token_family_counts()
    optimized_counts = _zero_token_family_counts()
    target_counts["unknown"] = int(max(0, valid_target_tokens))
    optimized_counts["unknown"] = int(max(0, optimized_target_tokens))
    return {
        "schema": "omnicoder.lm_loss_diagnostics_2026.v1",
        "diagnostics_skipped": True,
        "valid_target_tokens": int(max(0, valid_target_tokens)),
        "optimized_target_tokens": int(max(0, optimized_target_tokens)),
        "target_counts_by_token_family": target_counts,
        "target_counts_by_modality": _modality_counts_from_token_families(target_counts),
        "optimized_target_counts_by_token_family": optimized_counts,
        "optimized_target_counts_by_modality": _modality_counts_from_token_families(optimized_counts),
        "ce_by_token_family": {},
        "ce_by_modality": {},
    }


class WeightedTextJsonlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Any,
        seq_len: int,
        max_records: int = 0,
        vocab_size: int = 0,
        *,
        max_source_rows: int = 0,
        max_indexed_windows: int = 0,
    ):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.records: list[tuple[Path, int, int, str]] = []
        self._overflow_records: list[tuple[Path, int, int, str]] = []
        self.source_row_keys: set[tuple[Path, int, str]] = set()
        self.row_metadata: dict[tuple[Path, int, str], dict[str, Any]] = {}
        self.source_rows_seen = 0
        self.max_source_rows = int(max_source_rows) if int(max_source_rows) > 0 else 0
        self.max_indexed_windows = int(max_indexed_windows) if int(max_indexed_windows) > 0 else int(max_records)
        self.fallback: tuple[list[int], list[int], float] = ([1] * self.seq_len, [1] * self.seq_len, 0.05)
        self.record_cache_max_bytes = max(0, _env_int("OMNICODER2026_DATASET_RECORD_CACHE_MAX_BYTES", 512 * 1024 * 1024))
        self._record_cache: dict[tuple[Path, int, str], tuple[list[int], list[int], float, int, int]] = {}
        self._record_cache_bytes = 0
        self._record_cache_hits = 0
        self._record_cache_misses = 0
        window_limit = int(self.max_indexed_windows) if int(self.max_indexed_windows) > 0 else None
        source_limit = int(self.max_source_rows) if int(self.max_source_rows) > 0 else None
        p = Path(path)
        paths = sorted(p.rglob("*.jsonl")) + sorted(p.rglob("*.txt")) if p.is_dir() else [p]
        for src in paths:
            if window_limit is not None and len(self.records) >= window_limit:
                break
            if source_limit is not None and self.source_rows_seen >= source_limit:
                break
            self._index_path(src, window_limit=window_limit, source_limit=source_limit)
        if window_limit is not None and len(self.records) < window_limit:
            for entry in self._overflow_records:
                if len(self.records) >= window_limit:
                    break
                self.records.append(entry)

    def _sanitize_id(self, value: int) -> int:
        token = int(value)
        if token < 0:
            return 0
        if self.vocab_size > 0 and token >= self.vocab_size:
            return 1
        return token

    def _estimate_chunks(self, raw_len: int) -> int:
        # Use bytes as a conservative upper bound for text-token count so late
        # assistant/media targets in long JSONL rows are indexed, not silently
        # skipped by an optimistic average-bytes-per-token estimate.
        approx_tokens = max(2, int(max(1, raw_len)))
        stride = max(1, self.seq_len - 1)
        return max(1, int(math.ceil(float(approx_tokens) / float(stride))))

    @staticmethod
    def _messages_have_target(messages: object) -> bool:
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() != "assistant":
                continue
            if _content_to_text(message.get("content")).strip() or message.get("tool_calls"):
                return True
        return False

    @classmethod
    def _jsonl_line_has_possible_target(cls, raw: bytes) -> bool:
        try:
            obj = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return True
        return cls._jsonl_obj_has_possible_target(obj)

    @classmethod
    def _jsonl_obj_has_possible_target(cls, obj: object) -> bool:
        if not isinstance(obj, dict):
            return True
        if any(_token_id_list(obj.get(key)) for key in ("target_token_ids", "completion_token_ids", "assistant_token_ids", "media_token_ids", "artifact_token_ids")):
            return True
        target_json = obj.get("target_json")
        if isinstance(target_json, dict) and bool(target_json):
            return True
        if cls._messages_have_target(obj.get("messages")):
            return True
        input_json = obj.get("input_json")
        if isinstance(input_json, dict):
            if cls._messages_have_target(input_json.get("messages")):
                return True
            # input_json-only rows are context unless paired with target_json.
            return False
        kind = str(obj.get("training_kind") or "").lower()
        if kind.endswith("_rlvr") or kind in {"tool_preference", "tool_reward"}:
            return True
        if kind == "tool_safety_negative":
            return bool(_content_to_text(obj.get("chosen")).strip())
        if {"prompt", "chosen", "rejected"} <= set(obj):
            return bool(_content_to_text(obj.get("chosen")).strip())
        if any(_content_to_text(obj.get(key)).strip() for key in ("text", "content", "completion", "answer", "caption", "transcript", "ocr_text")):
            return True
        if "messages" in obj:
            return False
        return bool(_content_to_text(obj.get("prompt")).strip())

    @staticmethod
    def _row_modality(obj: dict[str, Any], path: Path) -> str:
        for key in ("modality", "target_modality", "source_modality", "media_family", "kind", "training_kind"):
            value = obj.get(key)
            if value:
                text = str(value).strip().lower()
                for candidate in ("image", "video", "music", "tts", "speech", "audio", "ocr", "code", "tool", "math", "text"):
                    if candidate in text:
                        return "tts" if candidate == "speech" else candidate
                return text[:64]
        target_json = obj.get("target_json")
        if isinstance(target_json, dict):
            route = _target_json_route(target_json)
            if route:
                return route
        stem = str(path.stem).lower()
        for candidate in ("image", "video", "music", "tts", "audio", "ocr", "code", "tool", "math", "text"):
            if candidate in stem:
                return candidate
        return "unknown"

    @staticmethod
    def _row_origin_group(obj: dict[str, Any], path: Path) -> str:
        for key in ("origin_group", "proof_group", "group", "source_family", "dataset_name", "source_id"):
            value = obj.get(key)
            if value:
                return str(value).strip()[:128]
        return path.stem

    def _register_source_row(
        self,
        path: Path,
        offset: int,
        kind: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, int, str]:
        key = (path, int(offset), str(kind))
        if key not in self.source_row_keys:
            self.source_row_keys.add(key)
            self.source_rows_seen += 1
        if metadata is not None:
            self.row_metadata[key] = metadata
        return key

    def _append_row_first_chunks(self, path: Path, offset: int, kind: str, chunks: int, *, window_limit: int | None) -> None:
        total = max(1, int(chunks))
        if window_limit is None or len(self.records) < window_limit:
            self.records.append((path, int(offset), 0, kind))
        if total <= 1:
            return
        overflow = [(path, int(offset), chunk_index, kind) for chunk_index in range(1, total)]
        if window_limit is None:
            self.records.extend(overflow)
        else:
            self._overflow_records.extend(overflow)

    def _index_path(self, path: Path, *, window_limit: int | None, source_limit: int | None) -> None:
        if not path.exists():
            return
        if path.suffix.lower() == ".txt":
            if source_limit is not None and self.source_rows_seen >= source_limit:
                return
            raw_len = path.stat().st_size
            chunks = self._estimate_chunks(raw_len)
            self._register_source_row(
                path,
                0,
                "txt",
                metadata={
                    "source_id": path.name,
                    "origin_group": path.stem,
                    "modality": "text",
                    "kind": "txt",
                    "estimated_chunks": int(chunks),
                },
            )
            self._append_row_first_chunks(path, 0, "txt", chunks, window_limit=window_limit)
            return
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if window_limit is not None and len(self.records) >= window_limit:
                    break
                if source_limit is not None and self.source_rows_seen >= source_limit:
                    break
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    obj = {}
                if not self._jsonl_obj_has_possible_target(obj):
                    continue
                if not isinstance(obj, dict):
                    obj = {}
                if _row_trainability_rejection_reason(obj) is not None:
                    continue
                chunks = self._estimate_chunks(len(raw))
                self._register_source_row(
                    path,
                    offset,
                    "jsonl",
                    metadata={
                        "source_id": str(obj.get("source_id") or obj.get("id") or f"{path.name}:{offset}"),
                        "origin_group": self._row_origin_group(obj, path),
                        "modality": self._row_modality(obj, path),
                        "kind": "jsonl",
                        "estimated_chunks": int(chunks),
                    },
                )
                self._append_row_first_chunks(path, offset, "jsonl", chunks, window_limit=window_limit)

    def __len__(self) -> int:
        return max(1, len(self.records))

    def _cached_record_copy(self, key: tuple[Path, int, str]) -> tuple[list[int], list[int], float, int] | None:
        cached = self._record_cache.get(key)
        if cached is None:
            self._record_cache_misses += 1
            return None
        self._record_cache_hits += 1
        ids, labels, weight, raw_len, _estimated_bytes = cached
        return list(ids), list(labels), float(weight), int(raw_len)

    def _cache_record(
        self,
        key: tuple[Path, int, str],
        ids: list[int],
        labels: list[int],
        weight: float,
        raw_len: int,
    ) -> tuple[list[int], list[int], float, int]:
        max_bytes = int(self.record_cache_max_bytes or 0)
        if max_bytes <= 0:
            return ids, labels, float(weight), int(raw_len)
        estimated_bytes = 64 + (len(ids) + len(labels)) * 8
        if estimated_bytes <= max_bytes and self._record_cache_bytes + estimated_bytes <= max_bytes:
            self._record_cache[key] = (list(ids), list(labels), float(weight), int(raw_len), int(estimated_bytes))
            self._record_cache_bytes += int(estimated_bytes)
        return ids, labels, float(weight), int(raw_len)

    def _read_record(self, path: Path, offset: int, kind: str) -> tuple[list[int], list[int], float, int]:
        key = (path, int(offset), str(kind))
        cached = self._cached_record_copy(key)
        if cached is not None:
            return cached
        if kind == "txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            ids = [int(x) for x in self.tokenizer.encode(text)]
            labels = _labels_from_ids_mask(ids, [1.0] * len(ids))
            return self._cache_record(key, ids, labels, 1.0, len(ids))
        with path.open("rb") as handle:
            handle.seek(int(offset))
            line = handle.readline().decode("utf-8", errors="ignore")
        if not line.strip():
            return self._cache_record(key, list(self.fallback[0]), list(self.fallback[1]), self.fallback[2], len(self.fallback[0]))
        try:
            obj = json.loads(line)
        except Exception:
            obj = {"text": line}
        if not isinstance(obj, dict):
            obj = {"text": str(obj)}
        if _row_trainability_rejection_reason(obj) is not None:
            return self._cache_record(key, list(self.fallback[0]), list(self.fallback[1]), self.fallback[2], len(self.fallback[0]))
        ids, labels, weight = record_ids_labels_weight(obj, self.tokenizer)
        return self._cache_record(key, ids, labels, weight, len(ids))

    def _window_from_entry(self, entry: tuple[Path, int, int, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        path, offset, chunk_index, kind = entry
        raw_ids, raw_labels, weight, _ = self._read_record(path, offset, kind)
        cleaned = [self._sanitize_id(x) for x in raw_ids]
        labels = []
        sparse_source = any(int(label) < 0 for label in raw_labels[: len(cleaned)])
        for token, label in zip(cleaned, raw_labels[: len(cleaned)]):
            try:
                label_id = int(label)
            except Exception:
                label_id = -100
            labels.append(token if label_id >= 0 else -100)
        if len(labels) < len(cleaned):
            labels.extend([-100] * (len(cleaned) - len(labels)))
        if len(cleaned) < 2:
            cleaned, labels, weight = self.fallback
        # Keep true one-token overlap on every chunk. Starts are 0, seq-1,
        # 2*(seq-1), ... so any boundary target except raw position 0 appears
        # in at least one chunk with previous-token context.
        start = int(chunk_index) * max(1, self.seq_len - 1)
        if start >= max(1, len(cleaned) - 1):
            start = max(0, len(cleaned) - self.seq_len)
        target_positions = [index for index, label in enumerate(labels) if int(label) >= 0]
        shifted_target_positions = [index for index in target_positions if int(index) > 0]
        if shifted_target_positions and not any(int(label) >= 0 for label in labels[start + 1:start + self.seq_len]):
            target_pos = shifted_target_positions[int(chunk_index) % len(shifted_target_positions)]
            max_start = max(0, len(cleaned) - self.seq_len)
            # Put sparse assistant/media targets as far right as possible so
            # the shared trunk still sees maximum cross-modal prompt context.
            start = max(0, min(max_start, int(target_pos) - self.seq_len + 1))
        ids = cleaned[start:start + self.seq_len]
        target_labels = labels[start:start + self.seq_len]
        if sparse_source and target_labels:
            target_labels[0] = -100
        # The LM objective predicts labels[:, 1:] from hidden[:, :-1].
        # A target label at position 0 is visible in the tensor but contributes
        # no CE term, so target-contract windows must have a shifted target.
        has_targets = any(int(label) >= 0 for label in target_labels[1:])
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
            target_labels = target_labels + [-100] * (self.seq_len - len(target_labels))
        return (
            torch.tensor(ids[: self.seq_len], dtype=torch.long),
            torch.tensor(target_labels[: self.seq_len], dtype=torch.long),
            torch.tensor(float(weight), dtype=torch.float32),
            bool(has_targets),
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.records:
            ids, labels, weight = self.fallback
            return (
                torch.tensor(ids[: self.seq_len], dtype=torch.long),
                torch.tensor(labels[: self.seq_len], dtype=torch.long),
                torch.tensor(float(weight), dtype=torch.float32),
            )
        base_index = int(idx) % len(self.records)
        blocked_records: set[tuple[Path, int, str]] = set()
        scan_budget = min(len(self.records), max(64, self.seq_len * 4))
        last_window: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        cursor = base_index
        attempts = 0
        while attempts < scan_budget:
            path, offset, chunk_index, kind = self.records[cursor]
            key = (path, int(offset), kind)
            attempts += 1
            if key in blocked_records:
                cursor = self._next_record_index(cursor, key)
                continue
            ids, target_labels, weight, has_targets = self._window_from_entry((path, offset, chunk_index, kind))
            last_window = (ids, target_labels, weight)
            if has_targets:
                return ids, target_labels, weight
            blocked_records.add(key)
            cursor = self._next_record_index(cursor, key)
        raise RuntimeError(
            f"no shifted assistant/media target labels found after scanning {scan_budget} indexed windows"
        )

    def _next_record_index(self, cursor: int, blocked_key: tuple[Path, int, str]) -> int:
        if not self.records:
            return 0
        start = int(cursor) % len(self.records)
        cursor = (start + 1) % len(self.records)
        while cursor != start:
            path, offset, _chunk_index, kind = self.records[cursor]
            if (path, int(offset), kind) != blocked_key:
                return cursor
            cursor = (cursor + 1) % len(self.records)
        return cursor


def _validate_resume_payload(
    checkpoint: dict[str, Any],
    *,
    preset: object | None,
    args: argparse.Namespace | None,
    sharded: bool,
) -> bool:
    placement_changed = False
    saved_preset = checkpoint.get("preset") if isinstance(checkpoint.get("preset"), dict) else {}
    saved_name = str(saved_preset.get("name") or "")
    current_name = str(getattr(preset, "name", "")) if preset is not None else ""
    if saved_name and current_name and saved_name != current_name:
        raise ValueError(f"resume checkpoint preset mismatch: checkpoint={saved_name!r} current={current_name!r}")
    if args is None:
        return False
    if bool(getattr(args, "require_target_contract", False)) and saved_name and saved_name != TARGET_PRESET:
        raise ValueError(f"target contract resume requires {TARGET_PRESET!r}, got checkpoint preset {saved_name!r}")
    train_args = checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {}
    current_placement_counts = str(getattr(args, "placement_layer_counts", "") or "").strip()
    saved_placement_counts = str(train_args.get("placement_layer_counts") or "").strip()
    if not saved_placement_counts:
        saved_ranges = str(train_args.get("pipeline_stage_ranges") or "").strip()
        try:
            parsed_ranges = [
                tuple(int(part.strip()) for part in segment.split(":", 1))
                for segment in saved_ranges.split(",")
                if segment.strip()
            ]
            if parsed_ranges:
                saved_placement_counts = ",".join(str(end - start) for start, end in parsed_ranges)
        except Exception:
            saved_placement_counts = ""
    if sharded and current_placement_counts and saved_placement_counts and current_placement_counts != saved_placement_counts:
        placement_changed = True
    for key in ("pipeline_stage_ranges", "placement_layer_counts", "pipeline_microbatches", "pipeline_schedule", "fake_quant"):
        saved = train_args.get(key)
        if saved is None or saved == "":
            continue
        current = getattr(args, key, None)
        if key == "pipeline_stage_ranges" and str(getattr(args, "placement_layer_counts", "") or "").strip():
            continue
        if str(saved) != str(current):
            if sharded and key in {"pipeline_stage_ranges", "placement_layer_counts"}:
                placement_changed = True
                continue
            raise ValueError(f"resume checkpoint {key} mismatch: checkpoint={saved!r} current={current!r}")
    if sharded:
        saved_world = checkpoint.get("world_size")
        if saved_world is not None and int(saved_world) != int(dist.get_world_size()):
            raise ValueError(f"resume world_size mismatch: checkpoint={saved_world} current={dist.get_world_size()}")
    return placement_changed


def _fill_missing_from_checkpoint_state(
    filtered: dict[str, torch.Tensor],
    missing: list[str],
    checkpoint_state: dict[str, Any],
    current: dict[str, torch.Tensor],
) -> list[str]:
    remaining: list[str] = []
    for key in missing:
        candidate = checkpoint_state.get(key)
        if isinstance(candidate, torch.Tensor) and tuple(candidate.shape) == tuple(current[key].shape):
            filtered[key] = candidate
        elif key == "lm_head.weight":
            embed = checkpoint_state.get("embed.weight")
            if isinstance(embed, torch.Tensor) and tuple(embed.shape) == tuple(current[key].shape):
                filtered[key] = embed
            else:
                remaining.append(key)
        elif key.endswith(".o_a_proj.weight") and current[key].ndim == 3:
            groups = int(current[key].shape[0])
            legacy_prefix = key[: -len("o_a_proj.weight")]
            legacy_keys = [f"{legacy_prefix}o_a_groups.{idx}.weight" for idx in range(groups)]
            legacy_tensors = [checkpoint_state.get(legacy_key) for legacy_key in legacy_keys]
            if all(isinstance(tensor, torch.Tensor) for tensor in legacy_tensors):
                stacked = torch.stack([tensor for tensor in legacy_tensors if isinstance(tensor, torch.Tensor)], dim=0)
                if tuple(stacked.shape) == tuple(current[key].shape):
                    filtered[key] = stacked
                else:
                    remaining.append(key)
            else:
                remaining.append(key)
        else:
            remaining.append(key)
    return remaining


def _fill_missing_from_other_shards(
    source: Path,
    filtered: dict[str, torch.Tensor],
    missing: list[str],
    current: dict[str, torch.Tensor],
    *,
    local_rank: int,
) -> list[str]:
    remaining = list(missing)
    world_size = int(dist.get_world_size())
    for rank_index in range(world_size):
        if rank_index == local_rank or not remaining:
            continue
        checkpoint_path = source / f"rank{rank_index:05d}.pt"
        if not checkpoint_path.exists():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            continue
        remaining = _fill_missing_from_checkpoint_state(filtered, remaining, state, current)
        del checkpoint, state
    return remaining


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
    placement_changed = _validate_resume_payload(checkpoint, preset=preset, args=args, sharded=sharded)
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
        missing = _fill_missing_from_checkpoint_state(filtered, missing, state, current)
    if missing and sharded and placement_changed:
        missing = _fill_missing_from_other_shards(source, filtered, missing, current, local_rank=int(dist.get_rank()))
    if missing:
        raise ValueError(f"checkpoint {checkpoint_path} is missing local shard tensors: {missing[:8]}")
    shard.load_state_dict(filtered, strict=False)
    if optimizer is not None and not placement_changed and checkpoint.get("optimizer_state_dict"):
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
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    sync_backend = str(getattr(args, "checkpoint_sync_backend", "filesystem") or "filesystem").strip().lower()
    marker_timeout = float(getattr(args, "checkpoint_marker_timeout_seconds", 7200.0) or 7200.0)
    marker_poll = float(getattr(args, "checkpoint_marker_poll_seconds", 2.0) or 2.0)
    if sync_backend not in {"filesystem", "barrier"}:
        raise ValueError(f"unsupported checkpoint_sync_backend={sync_backend!r}")
    total_started = _monotonic()
    io_metrics: dict[str, Any] = {
        "schema": CHECKPOINT_IO_SCHEMA,
        "event": "checkpoint_save",
        "status": "started",
        "timestamp": time.time(),
        "rank": rank,
        "world_size": world_size,
        "checkpoint_dir": str(target),
        "global_step": int(global_step),
        "last_loss": last_loss,
        "sync_backend": sync_backend,
        "spans": {},
        "bytes": {},
        "data": {},
    }
    checkpoint_attempt_id = ""
    if sync_backend == "barrier":
        if rank == 0:
            for stale in (target / ".complete.json", target / "manifest.json"):
                _unlink_if_exists(stale)
        dist.barrier()
    else:
        if rank == 0:
            checkpoint_attempt_id = uuid.uuid4().hex
            for stale in (target / ".complete.json", target / "manifest.json", _checkpoint_attempt_file(target)):
                _unlink_if_exists(stale)
            for rank_index in range(world_size):
                _unlink_if_exists(_rank_complete_file(target, rank_index))
            _atomic_write_json(
                _checkpoint_attempt_file(target),
                {
                    "status": "ready",
                    "checkpoint_attempt_id": checkpoint_attempt_id,
                    "checkpoint_dir": str(target),
                    "world_size": world_size,
                    "global_step": int(global_step),
                    "rank0_pid": os.getpid(),
                    "rank0_host": socket.gethostname(),
                    "created_unix_seconds": time.time(),
                },
            )
        else:
            attempt = _wait_for_checkpoint_attempt(
                target,
                world_size=world_size,
                global_step=int(global_step),
                timeout_seconds=marker_timeout,
                poll_seconds=marker_poll,
            )
            checkpoint_attempt_id = str(attempt["checkpoint_attempt_id"])
    rank_path = _rank_checkpoint_file(target, rank)
    _unlink_if_exists(str(rank_path) + ".complete.json")
    started = _monotonic()
    model_state = shard.local_state_dict()
    io_metrics["spans"]["state_dict_collect_sec"] = float(_monotonic() - started)
    io_metrics["bytes"]["state_dict_cpu_bytes_estimate"] = int(
        sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in model_state.values() if isinstance(tensor, torch.Tensor))
    )
    started = _monotonic()
    optimizer_state = optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None
    io_metrics["spans"]["optimizer_state_dict_sec"] = float(_monotonic() - started)
    started = _monotonic()
    rng_state = _rng_state()
    io_metrics["spans"]["rng_state_sec"] = float(_monotonic() - started)
    data_integrity = _checkpoint_data_integrity(args)
    io_metrics["data"] = dict(data_integrity)
    payload = {
        "format": "omnicoder2026_pipeline_stage_checkpoint_v2",
        "rank": rank,
        "world_size": world_size,
        "checkpoint_attempt_id": checkpoint_attempt_id or None,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "rng_state": rng_state,
        "global_step": int(global_step),
        "last_loss": last_loss,
        "preset": preset.__dict__,
        "data": {
            "path": data_integrity.get("path") or args.data,
            "sha256": data_integrity.get("sha256"),
            "manifest": data_integrity.get("manifest") or getattr(args, "data_manifest", None),
            "hash_policy": data_integrity.get("hash_policy"),
            "hash_source": data_integrity.get("hash_source"),
            "hash_duration_sec": data_integrity.get("hash_duration_sec"),
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
            "fake_quant_chunk_rows": int(getattr(args, "fake_quant_chunk_rows", 0) or 0),
            "fake_quant_max_full_elements": int(getattr(args, "fake_quant_max_full_elements", 0) or 0),
            "lm_loss_chunk_tokens": int(getattr(args, "lm_loss_chunk_tokens", 0) or 0),
            "loss_token_stride": int(getattr(args, "loss_token_stride", 1) or 1),
            "max_loss_tokens_per_sample": int(getattr(args, "max_loss_tokens_per_sample", 0) or 0),
            "max_records": int(getattr(args, "max_records", 0) or 0),
            "max_source_rows": int(getattr(args, "max_source_rows", 0) or 0),
            "max_indexed_windows": int(getattr(args, "max_indexed_windows", 0) or 0),
            "target_boundary_weight": float(getattr(args, "target_boundary_weight", 1.0) or 1.0),
            "target_prefix_weight": float(getattr(args, "target_prefix_weight", 1.0) or 1.0),
            "target_prefix_tokens": int(getattr(args, "target_prefix_tokens", 0) or 0),
            "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1) or 1),
            "skip_final_optimizer_update": bool(getattr(args, "skip_final_optimizer_update", False)),
            "shuffle": bool(getattr(args, "shuffle", True)),
            "optimizer": str(args.optimizer),
            "optimizer_in_backward": bool(getattr(args, "optimizer_in_backward", False)),
            "optimizer_in_backward_update": str(getattr(args, "optimizer_in_backward_update", "")),
            "optimizer_in_backward_grad_clip": float(getattr(args, "optimizer_in_backward_grad_clip", 0.0) or 0.0),
            "optimizer_in_backward_adafactor_chunk_rows": int(getattr(args, "optimizer_in_backward_adafactor_chunk_rows", 0) or 0),
            "checkpoint_sync_backend": sync_backend,
            "checkpoint_marker_timeout_seconds": marker_timeout,
            "checkpoint_data_hash_policy": str(getattr(args, "checkpoint_data_hash_policy", "manifest") or "manifest"),
        },
        "spec": shard.spec.__dict__,
        "checkpoint_eval_artifact_contract": _checkpoint_eval_artifact_contract(target),
        "notes": {"pipeline_low_memory_optimizer": PIPELINE_LOW_MEMORY_OPTIMIZER_NOTE},
    }
    save_metrics = _atomic_torch_save_with_metrics(payload, rank_path)
    io_metrics["spans"].update(
        {
            "torch_save_tmp_sec": float(save_metrics.get("torch_save_tmp_sec") or 0.0),
            "rename_sec": float(save_metrics.get("rename_sec") or 0.0),
            "rank_file_marker_write_sec": float(save_metrics.get("marker_write_sec") or 0.0),
        }
    )
    io_metrics["bytes"].update(
        {
            "tmp_bytes": int(save_metrics.get("tmp_bytes") or 0),
            "rank_file_bytes": int(save_metrics.get("final_bytes") or 0),
            "rank_file_bytes_per_sec": save_metrics.get("bytes_per_sec"),
        }
    )
    if sync_backend == "barrier":
        started = _monotonic()
        dist.barrier()
        io_metrics["spans"]["barrier_after_rank_save_sec"] = float(_monotonic() - started)
    else:
        started = _monotonic()
        _atomic_write_json(
            _rank_complete_file(target, rank),
            {
                "status": "complete",
                "path": str(rank_path),
                "bytes": rank_path.stat().st_size,
                "format": payload.get("format"),
                "rank": rank,
                "world_size": world_size,
                "global_step": int(global_step),
                "last_loss": last_loss,
                "checkpoint_attempt_id": checkpoint_attempt_id,
            },
        )
        io_metrics["spans"]["rank_marker_write_sec"] = float(_monotonic() - started)
        if rank == 0:
            started = _monotonic()
            _wait_for_rank_checkpoint_markers(
                target,
                world_size=world_size,
                global_step=int(global_step),
                attempt_id=checkpoint_attempt_id,
                timeout_seconds=marker_timeout,
                poll_seconds=marker_poll,
            )
            io_metrics["spans"]["rank_wait_for_peers_sec"] = float(_monotonic() - started)
        else:
            started = _monotonic()
            _wait_for_directory_checkpoint_marker(
                target,
                world_size=world_size,
                global_step=int(global_step),
                attempt_id=checkpoint_attempt_id,
                timeout_seconds=marker_timeout,
                poll_seconds=marker_poll,
            )
            io_metrics["spans"]["directory_wait_sec"] = float(_monotonic() - started)
            io_metrics["status"] = "complete"
            io_metrics["total_sec"] = float(_monotonic() - total_started)
            _append_pipeline_telemetry(target / f"checkpoint_io.rank{rank:05d}.jsonl", io_metrics)
            return
    if rank == 0:
        manifest = {
            "format": "omnicoder2026_pipeline_stage_checkpoint_v2",
            "checkpoint_dir": str(target),
            "world_size": world_size,
            "rank_files": [f"rank{rank_index:05d}.pt" for rank_index in range(world_size)],
            "global_step": int(global_step),
            "last_loss": last_loss,
            "checkpoint_attempt_id": checkpoint_attempt_id or None,
            "sync_backend": sync_backend,
            "checkpoint_io_schema": CHECKPOINT_IO_SCHEMA,
            "preset": preset.__dict__,
            "train_args": payload["train_args"],
            "data": payload["data"],
            "checkpoint_eval_artifact_contract": _checkpoint_eval_artifact_contract(target),
            "note": "Per-stage pipeline checkpoint. Use merge tooling before GGUF/export.",
        }
        started = _monotonic()
        _atomic_write_json(target / "manifest.json", manifest)
        _atomic_write_json(target / ".complete.json", {"status": "complete", **manifest})
        io_metrics["spans"]["manifest_write_sec"] = float(_monotonic() - started)
    if sync_backend == "barrier":
        started = _monotonic()
        dist.barrier()
        io_metrics["spans"]["final_barrier_sec"] = float(_monotonic() - started)
    io_metrics["status"] = "complete"
    io_metrics["total_sec"] = float(_monotonic() - total_started)
    _append_pipeline_telemetry(target / f"checkpoint_io.rank{rank:05d}.jsonl", io_metrics)


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


def _rank_telemetry_path(args: argparse.Namespace, *, rank: int, world_size: int) -> Path:
    raw_path = str(getattr(args, "telemetry_file", "") or "").strip()
    if raw_path:
        if "{rank}" in raw_path:
            return Path(raw_path.format(rank=f"{int(rank):05d}", rank_int=int(rank), world_size=int(world_size)))
        path = Path(raw_path)
        if int(world_size) > 1:
            return path.with_name(f"{path.stem}.rank{int(rank):05d}{path.suffix or '.jsonl'}")
        return path
    return Path(args.out) / f"telemetry.rank{int(rank):05d}.jsonl"


def _rank_train_diagnostics_path(args: argparse.Namespace, *, rank: int, world_size: int) -> Path:
    raw_path = str(getattr(args, "train_diagnostics_file", "") or "").strip()
    if raw_path:
        if "{rank}" in raw_path:
            return Path(raw_path.format(rank=f"{int(rank):05d}", rank_int=int(rank), world_size=int(world_size)))
        path = Path(raw_path)
        if int(world_size) > 1:
            return path.with_name(f"{path.stem}.rank{int(rank):05d}{path.suffix or '.jsonl'}")
        return path
    return Path(args.out) / f"train_diagnostics.rank{int(rank):05d}.jsonl"


def _rank_step_timing_path(args: argparse.Namespace, *, rank: int, world_size: int) -> Path:
    raw_path = str(getattr(args, "step_timing_file", "") or "").strip()
    if raw_path:
        if "{rank}" in raw_path:
            return Path(raw_path.format(rank=f"{int(rank):05d}", rank_int=int(rank), world_size=int(world_size)))
        path = Path(raw_path)
        if int(world_size) > 1:
            return path.with_name(f"{path.stem}.rank{int(rank):05d}{path.suffix or '.jsonl'}")
        return path
    return Path(args.out) / f"step_timing.rank{int(rank):05d}.jsonl"


def _rank_block_timing_path(args: argparse.Namespace, *, rank: int, world_size: int) -> Path:
    raw_path = str(getattr(args, "block_timing_file", "") or "").strip()
    if raw_path:
        if "{rank}" in raw_path:
            return Path(raw_path.format(rank=f"{int(rank):05d}", rank_int=int(rank), world_size=int(world_size)))
        path = Path(raw_path)
        if int(world_size) > 1:
            return path.with_name(f"{path.stem}.rank{int(rank):05d}{path.suffix or '.jsonl'}")
        return path
    return Path(args.out) / f"block_timing.rank{int(rank):05d}.jsonl"


def _pipeline_telemetry_record(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    ranges: list[tuple[int, int]],
    spec: PipelineShardSpec,
    seq_len: int,
    step: int,
    local_step: int,
) -> dict[str, Any]:
    cuda_active = bool(device.type == "cuda" and torch.cuda.is_available())
    device_index = int(device.index or 0) if device.type == "cuda" else None
    memory = {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "max_allocated_bytes": 0,
        "max_reserved_bytes": 0,
        "free_bytes": 0,
        "total_bytes": 0,
    }
    device_name = str(device)
    device_capability = None
    if cuda_active:
        memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": 0,
            "total_bytes": 0,
        }
        try:
            device_name = str(torch.cuda.get_device_name(device))
        except Exception:
            device_name = str(device)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            memory["free_bytes"] = int(free_bytes)
            memory["total_bytes"] = int(total_bytes)
        except Exception:
            pass
        try:
            capability = torch.cuda.get_device_capability(device)
            device_capability = [int(capability[0]), int(capability[1])]
        except Exception:
            device_capability = None
    stage_infos = [
        {
            "stage_index": index,
            "layer_start": int(start),
            "layer_end": int(end),
            "layer_count": int(end - start),
        }
        for index, (start, end) in enumerate(ranges)
    ]
    return {
        "event": "pipeline_rank_memory_telemetry",
        "timestamp": time.time(),
        "rank": int(rank),
        "world_size": int(world_size),
        "device": str(device),
        "device_type": str(device.type),
        "device_index": device_index,
        "device_name": device_name,
        "device_capability": device_capability,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_active": cuda_active,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "local_rank": int(os.environ.get("LOCAL_RANK", rank) or rank),
        **memory,
        "seq_len": int(seq_len),
        "step": int(step),
        "local_step": int(local_step),
        "stage_index": int(spec.stage_index),
        "num_stages": int(spec.num_stages),
        "layer_start": int(spec.layer_start),
        "layer_end": int(spec.layer_end),
        "layer_count": int(spec.layer_end - spec.layer_start),
        "has_embed": bool(spec.has_embed),
        "has_head": bool(spec.has_head),
        "placement_layer_counts": [int(end - start) for start, end in ranges],
        "pipeline_stage_ranges": stage_infos,
        "pipeline_schedule": str(getattr(args, "pipeline_schedule", "")),
        "pipeline_microbatches": int(getattr(args, "pipeline_microbatches", 0) or 0),
    }


def _append_pipeline_telemetry(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")


def _dataset_source_summary(dataset: WeightedTextJsonlDataset | None) -> dict[str, Any]:
    if dataset is None:
        return {"available": False, "records": 0, "sources": {}, "kinds": {}}
    sources: dict[str, int] = {}
    kinds: dict[str, int] = {}
    row_sources: dict[str, int] = {}
    origin_groups: dict[str, int] = {}
    modalities: dict[str, int] = {}
    for path, _offset, _chunk_index, kind in getattr(dataset, "records", []):
        source = str(path)
        sources[source] = int(sources.get(source, 0) + 1)
        kinds[str(kind)] = int(kinds.get(str(kind), 0) + 1)
    row_metadata = getattr(dataset, "row_metadata", {}) or {}
    for key in getattr(dataset, "source_row_keys", set()):
        path, _offset, kind = key
        row_sources[str(path)] = int(row_sources.get(str(path), 0) + 1)
        metadata = row_metadata.get(key, {}) if isinstance(row_metadata, dict) else {}
        origin_group = str(metadata.get("origin_group") or Path(path).stem)
        modality = str(metadata.get("modality") or "unknown")
        origin_groups[origin_group] = int(origin_groups.get(origin_group, 0) + 1)
        modalities[modality] = int(modalities.get(modality, 0) + 1)
    return {
        "available": True,
        "records": int(len(getattr(dataset, "records", []))),
        "indexed_samples": int(len(dataset)),
        "source_rows": int(len(getattr(dataset, "source_row_keys", set()))),
        "max_source_rows": int(getattr(dataset, "max_source_rows", 0) or 0),
        "max_indexed_windows": int(getattr(dataset, "max_indexed_windows", 0) or 0),
        "fallback_active": bool(not getattr(dataset, "records", [])),
        "record_cache": _dataset_record_cache_summary(dataset),
        "sources": dict(sorted(sources.items(), key=lambda item: (-item[1], item[0]))[:32]),
        "row_sources": dict(sorted(row_sources.items(), key=lambda item: (-item[1], item[0]))[:32]),
        "source_count": int(len(sources)),
        "row_source_count": int(len(row_sources)),
        "kinds": dict(sorted(kinds.items())),
        "origin_groups": dict(sorted(origin_groups.items())),
        "modalities": dict(sorted(modalities.items())),
    }


def _dataset_record_cache_summary(dataset: WeightedTextJsonlDataset | None) -> dict[str, int]:
    return {
        "max_bytes": int(getattr(dataset, "record_cache_max_bytes", 0) or 0),
        "bytes": int(getattr(dataset, "_record_cache_bytes", 0) or 0),
        "entries": int(len(getattr(dataset, "_record_cache", {}) or {})),
        "hits": int(getattr(dataset, "_record_cache_hits", 0) or 0),
        "misses": int(getattr(dataset, "_record_cache_misses", 0) or 0),
    }


def _compact_source_summary(source_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source_summary, dict):
        return {"available": False}
    keep = (
        "available",
        "records",
        "indexed_samples",
        "source_rows",
        "max_source_rows",
        "max_indexed_windows",
        "fallback_active",
        "source_count",
        "row_source_count",
        "kinds",
        "modalities",
        "record_cache",
    )
    compact = {key: source_summary[key] for key in keep if key in source_summary}
    return compact or {"available": False}


def _sample_weight_summary(sample_weights: torch.Tensor | None) -> dict[str, Any]:
    if sample_weights is None or sample_weights.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    values = sample_weights.detach().float().reshape(-1).cpu()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _optimizer_lr_groups(optimizer: Any, fallback_lr: float) -> dict[str, float]:
    groups = getattr(optimizer, "param_groups", None)
    if isinstance(groups, list) and groups:
        out: dict[str, float] = {}
        for index, group in enumerate(groups):
            if isinstance(group, dict) and "lr" in group:
                out[f"group{index}"] = float(group.get("lr") or 0.0)
        if out:
            return out
    return {"group0": float(fallback_lr)}


def _module_grad_norm(model: nn.Module, *, enabled: bool) -> float | None:
    if not bool(enabled):
        return None
    try:
        chunk_elems = max(1, int(os.getenv("OMNICODER2026_GRAD_NORM_CHUNK_ELEMS", "262144")))
    except ValueError:
        chunk_elems = 262_144
    total_sq: torch.Tensor | None = None
    found = False
    with torch.no_grad():
        for parameter in model.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            found = True
            flat = grad.detach().reshape(-1)
            param_sq = torch.zeros((), device=flat.device, dtype=torch.float32)
            for start in range(0, flat.numel(), chunk_elems):
                chunk = flat[start : start + chunk_elems].float()
                param_sq = param_sq + torch.sum(chunk * chunk)
            total_sq = param_sq if total_sq is None else total_sq + param_sq.to(device=total_sq.device)
    if not found:
        return None
    total_sq_value = float(total_sq.detach().cpu().item()) if isinstance(total_sq, torch.Tensor) else 0.0
    return float(math.sqrt(max(0.0, total_sq_value)))


def _runtime_rank_memory_from_telemetry(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "rank",
        "device",
        "device_type",
        "device_index",
        "device_name",
        "cuda_active",
        "allocated_bytes",
        "reserved_bytes",
        "max_allocated_bytes",
        "max_reserved_bytes",
        "free_bytes",
        "total_bytes",
    )
    return {key: record.get(key) for key in keys if key in record}


def _checkpoint_eval_artifact_contract(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    return {
        "schema": CHECKPOINT_EVAL_ARTIFACT_CONTRACT_SCHEMA,
        "status": "required_after_checkpoint_save",
        "training_invoked": False,
        "checkpoint_dir": str(root),
        "artifacts": [
            {
                "name": "heldout_sample_loss_by_modality",
                "required": True,
                "path": str(root / "evals" / "heldout_pipeline_sample_loss.json"),
                "schema": "omnicoder.pipeline_sample_loss_2026.v1",
                "must_include": ["overall", "modalities"],
            },
            {
                "name": "target_token_rank_diagnostics",
                "required": True,
                "path": str(root / "evals" / "target_token_diagnostics.json"),
                "schema": "omnicoder.pipeline_target_token_diagnostics_2026.v1",
                "must_include": ["target_tokens", "rank_metrics"],
            },
            {
                "name": "text_code_tool_decode_probes",
                "required": True,
                "path": str(root / "evals" / "decode_probes_text_code_tool.jsonl"),
                "schema": "omnicoder.decode_probe_2026.v1",
                "must_include": ["text", "code", "tool_agent"],
            },
            {
                "name": "media_route_probe_attempts",
                "required": True,
                "path": str(root / "evals" / "media_route_probe_attempts.json"),
                "schema": "omnicoder.media_route_probe_2026.v1",
                "must_include": ["image", "video", "audio", "music", "tts", "ocr"],
            },
        ],
    }


def _train_diagnostics_record(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    spec: PipelineShardSpec,
    global_step: int,
    local_step: int,
    seq_len: int,
    batch_size: int,
    microbatch_size: int,
    loss: float,
    labels: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    optimizer: Any,
    optimizer_update: bool,
    grad_norm_pre_clip: float | None,
    grad_norm_post_clip: float | None,
    step_elapsed_sec: float,
    memory_record: dict[str, Any],
    loss_diagnostics: dict[str, Any] | None,
    source_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    loss_diagnostics = loss_diagnostics if isinstance(loss_diagnostics, dict) else {}
    loss_target_counts = loss_diagnostics.get("target_counts_by_token_family")
    if isinstance(loss_target_counts, dict) and any(int(value or 0) for value in loss_target_counts.values()):
        target_counts = {str(key): int(value or 0) for key, value in loss_target_counts.items()}
    elif labels is None and int(loss_diagnostics.get("valid_target_tokens") or 0) > 0:
        target_counts = _zero_token_family_counts()
        target_counts["unknown"] = int(loss_diagnostics.get("valid_target_tokens") or 0)
    else:
        target_counts = _token_family_counts(labels)
    optimized_counts = loss_diagnostics.get("optimized_target_counts_by_token_family")
    if not isinstance(optimized_counts, dict):
        optimized_counts = {key: 0 for key in target_counts}
        if "unknown" in optimized_counts:
            optimized_counts["unknown"] = int(loss_diagnostics.get("optimized_target_tokens") or 0)
    valid_target_tokens = int(sum(int(value or 0) for value in target_counts.values()))
    optimized_target_tokens = int(loss_diagnostics.get("optimized_target_tokens") or sum(int(value or 0) for value in optimized_counts.values()))
    elapsed = max(0.0, float(step_elapsed_sec or 0.0))
    total_tokens = int(batch_size) * int(seq_len)
    tokens_per_sec = (float(total_tokens) / elapsed) if elapsed > 0.0 else None
    ce_by_token_family = loss_diagnostics.get("ce_by_token_family")
    ce_by_modality = loss_diagnostics.get("ce_by_modality")
    return {
        "schema": TRAIN_DIAGNOSTICS_SCHEMA,
        "event": "train_step",
        "timestamp": time.time(),
        "rank": int(rank),
        "world_size": int(world_size),
        "stage_index": int(spec.stage_index),
        "has_head": bool(spec.has_head),
        "global_step": int(global_step),
        "local_step": int(local_step),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "microbatch_size": int(microbatch_size),
        "lr": _optimizer_lr_groups(optimizer, float(getattr(args, "lr", 0.0) or 0.0)),
        "optimizer": {
            "name": str(getattr(args, "optimizer", "")),
            "update": bool(optimizer_update),
            "gradient_accumulation_steps": int(getattr(args, "gradient_accumulation_steps", 1) or 1),
            "in_backward": bool(getattr(args, "optimizer_in_backward", False)),
            "in_backward_update": str(getattr(args, "optimizer_in_backward_update", "")),
            "clipping": {
                "configured": bool(float(getattr(args, "optimizer_in_backward_grad_clip", 0.0) or 0.0) > 0.0),
                "mode": str(getattr(args, "optimizer_in_backward_clip_mode", "")),
                "max_norm_or_rms": float(getattr(args, "optimizer_in_backward_grad_clip", 0.0) or 0.0),
                "adafactor_clip_threshold": float(getattr(args, "optimizer_in_backward_adafactor_clip_threshold", 0.0) or 0.0),
            },
            "grad_norm_pre_clip": grad_norm_pre_clip,
            "grad_norm_post_clip": grad_norm_post_clip,
        },
        "loss": {
            "total_ce": float(loss),
            "valid_target_tokens": valid_target_tokens,
            "optimized_target_tokens": optimized_target_tokens,
            "ce_by_modality": ce_by_modality if isinstance(ce_by_modality, dict) else {},
            "ce_by_token_family": ce_by_token_family if isinstance(ce_by_token_family, dict) else {},
        },
        "targets": {
            "by_modality": _modality_counts_from_token_families(target_counts),
            "by_token_family": target_counts,
            "optimized_by_modality": _modality_counts_from_token_families({str(k): int(v or 0) for k, v in optimized_counts.items()}),
            "optimized_by_token_family": {str(k): int(v or 0) for k, v in optimized_counts.items()},
        },
        "data": {
            "data_path": str(getattr(args, "data", "")),
            "data_manifest": str(getattr(args, "data_manifest", "") or ""),
            "sample_weights": _sample_weight_summary(sample_weights),
            "source_summary": _compact_source_summary(source_summary),
            "source_summary_ref": "dataset_index_done",
            "shuffle": bool(getattr(args, "shuffle", True)),
        },
        "runtime": {
            "elapsed_sec": elapsed,
            "tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "rank_memory": _runtime_rank_memory_from_telemetry(memory_record),
        },
    }


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


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
        "has_embed": bool(spec.has_embed),
        "has_head": bool(spec.has_head),
    }
    reports: list[dict[str, Any] | None] = [None for _ in range(int(dist.get_world_size()))]
    dist.all_gather_object(reports, local)
    complete = [item for item in reports if isinstance(item, dict)]
    if len(complete) != int(dist.get_world_size()):
        raise ValueError(f"incomplete device placement report: {reports!r}")
    p40_ranks = [item for item in complete if "P40" in str(item.get("name", "")).upper()]
    if p40_ranks and not bool(getattr(args, "allow_p40_target_contract_eval", False)):
        raise ValueError(f"target-contract pipeline may not include P40 devices: {p40_ranks!r}")
    head_layer_equivalent = max(0, int(os.getenv("OMNICODER2026_HEAD_LAYER_EQUIVALENT", "4") or 4))
    embed_layer_equivalent = max(0, int(os.getenv("OMNICODER2026_EMBED_LAYER_EQUIVALENT", "1") or 1))
    for item in complete:
        item["placement_load"] = (
            int(item["layers"])
            + (head_layer_equivalent if bool(item.get("has_head")) else 0)
            + (embed_layer_equivalent if bool(item.get("has_embed")) else 0)
        )
    max_layers = max(int(item["placement_load"]) for item in complete)
    max_memory = max(int(item["total_memory"]) for item in complete)
    heavy_ranks = [item for item in complete if int(item["placement_load"]) == max_layers]
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
        self.step_stats: dict[str, float | int] = {}
        if self.post_accumulate:
            sample = self.params[0] if self.params else None
            if sample is None or not hasattr(sample, "register_post_accumulate_grad_hook"):
                raise RuntimeError("optimizer-in-backward requires Tensor.register_post_accumulate_grad_hook")
            for param in self.params:
                self.handles.append(param.register_post_accumulate_grad_hook(self._hook_step))

    def reset_step_stats(self) -> None:
        self.step_stats = {
            "hook_calls": 0,
            "step_calls": 0,
            "params_2d": 0,
            "fallback_params": 0,
            "rows": 0,
            "elements": 0,
            "hook_step_sec": 0.0,
            "explicit_step_sec": 0.0,
        }

    def diagnostics(self) -> dict[str, float | int | bool]:
        stats = dict(self.step_stats)
        stats["post_accumulate"] = bool(self.post_accumulate)
        stats["chunk_rows"] = int(self.chunk_rows)
        return stats

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
            self.step_stats["fallback_params"] = int(self.step_stats.get("fallback_params", 0)) + 1
            self.step_stats["elements"] = int(self.step_stats.get("elements", 0)) + int(grad.numel())
            self._fallback_step(param, grad)
            param.grad = None
            return
        rows, cols = int(grad.shape[0]), int(grad.shape[1])
        self.step_stats["params_2d"] = int(self.step_stats.get("params_2d", 0)) + 1
        self.step_stats["rows"] = int(self.step_stats.get("rows", 0)) + rows
        self.step_stats["elements"] = int(self.step_stats.get("elements", 0)) + int(grad.numel())
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
            started = _monotonic()
            try:
                self._step_param(param)
            finally:
                self.step_stats["hook_calls"] = int(self.step_stats.get("hook_calls", 0)) + 1
                self.step_stats["hook_step_sec"] = float(self.step_stats.get("hook_step_sec", 0.0)) + float(_monotonic() - started)

    def step(self) -> None:
        if self.post_accumulate:
            return
        with torch.no_grad():
            started = _monotonic()
            for param in self.params:
                self._step_param(param)
            self.step_stats["step_calls"] = int(self.step_stats.get("step_calls", 0)) + 1
            self.step_stats["explicit_step_sec"] = float(self.step_stats.get("explicit_step_sec", 0.0)) + float(_monotonic() - started)

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
    parser.add_argument("--telemetry_file", default=os.getenv("OMNICODER2026_PIPELINE_TELEMETRY_FILE", ""))
    parser.add_argument("--train_diagnostics_file", default=os.getenv("OMNICODER2026_TRAIN_DIAGNOSTICS_FILE", ""))
    parser.add_argument("--step_timing_file", default=os.getenv("OMNICODER2026_STEP_TIMING_FILE", ""))
    parser.add_argument("--block_timing_file", default=os.getenv("OMNICODER2026_BLOCK_TIMING_FILE", ""))
    parser.add_argument("--step_timing_interval", type=int, default=int(os.getenv("OMNICODER2026_STEP_TIMING_INTERVAL", "0") or 0))
    parser.add_argument("--telemetry_interval", type=int, default=int(os.getenv("OMNICODER2026_TELEMETRY_INTERVAL", "8") or 8))
    parser.add_argument("--train_log_interval", type=int, default=int(os.getenv("OMNICODER2026_TRAIN_LOG_INTERVAL", "8") or 8))
    parser.add_argument("--detailed_event_log_interval", type=int, default=int(os.getenv("OMNICODER2026_DETAILED_EVENT_LOG_INTERVAL", "0") or 0))
    parser.add_argument("--timing_cuda_sync", action="store_true", default=(os.getenv("OMNICODER2026_TIMING_CUDA_SYNC", "0") == "1"))
    parser.add_argument("--block_timing", action="store_true", default=(os.getenv("OMNICODER2026_BLOCK_TIMING", "0") == "1"))
    parser.add_argument("--block_timing_cuda_sync", action="store_true", default=(os.getenv("OMNICODER2026_BLOCK_TIMING_CUDA_SYNC", "0") == "1"))
    parser.add_argument("--record_functions", action="store_true", default=(os.getenv("OMNICODER2026_RECORD_FUNCTIONS", "0") == "1"))
    parser.add_argument("--rank_skew_interval", type=int, default=int(os.getenv("OMNICODER2026_RANK_SKEW_INTERVAL", "0") or 0))
    parser.add_argument("--loss_diagnostics_interval", type=int, default=int(os.getenv("OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL", "8") or 8))
    parser.add_argument("--diagnostics_grad_norm", action="store_true", default=(os.getenv("OMNICODER2026_DIAGNOSTICS_GRAD_NORM", "0") == "1"))
    parser.add_argument("--data_manifest", default="")
    parser.add_argument("--data_sha256", default=os.getenv("OMNICODER2026_DATA_SHA256", ""))
    parser.add_argument("--checkpoint_data_hash_policy", default=os.getenv("OMNICODER2026_CHECKPOINT_DATA_HASH_POLICY", "manifest"), choices=["manifest", "never", "once", "always"])
    parser.add_argument("--preset", default="ledger_probe")
    parser.add_argument("--rank_device_map", default="")
    parser.add_argument("--pipeline_stage_ranges", default="")
    parser.add_argument("--placement_layer_counts", default="")
    parser.add_argument("--pipeline_schedule", default="1f1b", choices=["1f1b", "gpipe"])
    parser.add_argument("--schedule", dest="pipeline_schedule", choices=["1f1b", "gpipe"])
    parser.add_argument("--pipeline_microbatches", type=int, default=2)
    parser.add_argument("--n_microbatches", dest="pipeline_microbatches", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=int(os.getenv("OMNICODER2026_GRADIENT_ACCUMULATION_STEPS", "1") or 1))
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=True)
    parser.add_argument("--no_shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--max_source_rows", type=int, default=0)
    parser.add_argument("--max_indexed_windows", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=int(os.getenv("OMNICODER2026_DATALOADER_NUM_WORKERS", "0") or 0))
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=int(os.getenv("OMNICODER2026_DATALOADER_PREFETCH_FACTOR", "2") or 2))
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=(os.getenv("OMNICODER2026_DATALOADER_PIN_MEMORY", "auto").lower() in {"1", "true", "yes", "auto"}))
    parser.add_argument("--no_dataloader_pin_memory", dest="dataloader_pin_memory", action="store_false")
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=(os.getenv("OMNICODER2026_DATALOADER_PERSISTENT_WORKERS", "0") == "1"))
    parser.add_argument("--skip_final_optimizer_update", action="store_true")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--init_dtype", default="auto")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--optimizer_in_backward", action="store_true")
    parser.add_argument("--optimizer_in_backward_update", default=os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_UPDATE", ""), choices=["", "lowmem_adafactor", "chunked_adafactor"])
    parser.add_argument("--optimizer_in_backward_grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer_in_backward_clip_mode", default="rms", choices=["rms", "clamp"])
    parser.add_argument("--optimizer_in_backward_adafactor_chunk_rows", type=int, default=256)
    parser.add_argument("--optimizer_in_backward_adafactor_clip_threshold", type=float, default=1.0)
    parser.add_argument("--optimizer_in_backward_adafactor_decay_rate", type=float, default=-0.8)
    parser.add_argument("--optimizer_in_backward_adafactor_eps1", type=float, default=1.0e-30)
    parser.add_argument("--activation_checkpointing", "--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--activation_checkpoint_segment_size",
        "--activation-checkpoint-segment-size",
        type=int,
        default=int(os.getenv("OMNICODER2026_ACTIVATION_CHECKPOINT_SEGMENT_SIZE", "1") or 1),
    )
    parser.add_argument("--sanitize_input_ids", action="store_true", default=(os.getenv("OMNICODER2026_SANITIZE_INPUT_IDS", "0") == "1"))
    parser.add_argument("--pipeline_reasoning_effort", default=os.getenv("OMNICODER2026_PIPELINE_REASONING_EFFORT", ""))
    parser.add_argument("--fake_quant", action="store_true")
    parser.add_argument("--fake_quant_chunk_rows", type=int, default=0)
    parser.add_argument("--fake_quant_max_full_elements", type=int, default=0)
    parser.add_argument("--lm_loss_chunk_tokens", type=int, default=int(os.getenv("OMNICODER2026_LM_LOSS_CHUNK_TOKENS", "128") or 128))
    parser.add_argument("--loss_token_stride", type=int, default=int(os.getenv("OMNICODER2026_LOSS_TOKEN_STRIDE", "1") or 1))
    parser.add_argument("--max_loss_tokens_per_sample", type=int, default=int(os.getenv("OMNICODER2026_MAX_LOSS_TOKENS_PER_SAMPLE", "0") or 0))
    parser.add_argument("--target_boundary_weight", type=float, default=float(os.getenv("OMNICODER2026_TARGET_BOUNDARY_WEIGHT", "1.0") or 1.0))
    parser.add_argument("--target_prefix_weight", type=float, default=float(os.getenv("OMNICODER2026_TARGET_PREFIX_WEIGHT", "1.0") or 1.0))
    parser.add_argument("--target_prefix_tokens", type=int, default=int(os.getenv("OMNICODER2026_TARGET_PREFIX_TOKENS", "0") or 0))
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--skip_final_save", action="store_true")
    parser.add_argument("--require_target_contract", action="store_true")
    parser.add_argument("--allow_p40_target_contract_eval", action="store_true")
    parser.add_argument("--allow_probe", action="store_true")
    parser.add_argument("--debug_events", action="store_true")
    parser.add_argument("--checkpoint_sync_backend", default=os.getenv("OMNICODER2026_CHECKPOINT_SYNC_BACKEND", "filesystem"), choices=["filesystem", "barrier"])
    parser.add_argument("--checkpoint_marker_timeout_seconds", type=float, default=float(os.getenv("OMNICODER2026_CHECKPOINT_MARKER_TIMEOUT_SECONDS", "7200") or 7200))
    parser.add_argument("--checkpoint_marker_poll_seconds", type=float, default=float(os.getenv("OMNICODER2026_CHECKPOINT_MARKER_POLL_SECONDS", "2") or 2))
    parser.add_argument("--dist_timeout_seconds", type=float, default=float(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "3600") or 3600))
    args = parser.parse_args(argv)

    if int(args.fake_quant_chunk_rows or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_CHUNK_ROWS"] = str(int(args.fake_quant_chunk_rows))
    if int(args.fake_quant_max_full_elements or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS"] = str(int(args.fake_quant_max_full_elements))
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=float(args.dist_timeout_seconds)))
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    device = rank_device(rank, args.rank_device_map)

    preset = get_omnicoder2026_preset(args.preset)
    if bool(args.require_target_contract) and _is_probe_name(preset.name) and not bool(args.allow_probe):
        raise ValueError(
            f"Refusing to train verifier preset {preset.name!r} for a target-contract run. "
            f"Pass --allow_probe only for explicit validation runs, or use --preset {TARGET_PRESET}."
        )
    validate_target_contract_preset(
        preset,
        require_target_contract=bool(args.require_target_contract),
        allow_probe=bool(args.allow_probe),
        fake_quant_enabled=bool(args.fake_quant),
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
    gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps or 1))
    if gradient_accumulation_steps > 1 and bool(args.optimizer_in_backward):
        raise ValueError("--gradient_accumulation_steps > 1 is incompatible with --optimizer_in_backward hooks; omit --optimizer_in_backward for delayed low-memory Adafactor accumulation")
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
    telemetry_path = _rank_telemetry_path(args, rank=rank, world_size=world_size)
    train_diagnostics_path = _rank_train_diagnostics_path(args, rank=rank, world_size=world_size)
    final_train_diagnostics_path = _rank_train_diagnostics_path(args, rank=world_size - 1, world_size=world_size)
    step_timing_path = _rank_step_timing_path(args, rank=rank, world_size=world_size)
    block_timing_path = _rank_block_timing_path(args, rank=rank, world_size=world_size)
    init_dtype_name = str(args.init_dtype or "auto").lower()
    if init_dtype_name == "auto":
        init_dtype_name = str(args.precision or "fp32").lower()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(_dtype_from_name(init_dtype_name))
    try:
        print(json.dumps({"event": "model_build_start", "rank": int(rank), "layer_start": int(spec.layer_start), "layer_end": int(spec.layer_end), "seq_len": int(seq_len)}), flush=True)
        with torch.device(device):
            shard = OmniCoder2026PipelineShard(
                cfg,
                spec,
                checkpoint_blocks=bool(args.activation_checkpointing),
                checkpoint_segment_size=int(args.activation_checkpoint_segment_size),
            ).to(device)
            shard.profile_block_timing = bool(args.block_timing)
            shard.profile_block_cuda_sync = bool(args.block_timing_cuda_sync)
            shard.profile_record_functions = bool(args.record_functions)
            shard.sanitize_input_ids = bool(args.sanitize_input_ids)
            raw_effort = str(args.pipeline_reasoning_effort or "").strip()
            if raw_effort:
                try:
                    shard.pipeline_reasoning_effort = int(raw_effort)
                except ValueError:
                    shard.pipeline_reasoning_effort = raw_effort
            else:
                shard.pipeline_reasoning_effort = int(cfg.reasoning_default_steps)
        print(json.dumps({"event": "model_build_done", "rank": int(rank), "layer_start": int(spec.layer_start), "layer_end": int(spec.layer_end)}), flush=True)
    finally:
        torch.set_default_dtype(old_dtype)

    start_step, last_loss = 0, None
    shard.train()
    microbatch_size = batch_size // pipeline_microbatches
    example_input = torch.zeros((microbatch_size, seq_len), dtype=torch.long, device=device)
    if not spec.has_embed:
        example_input = torch.zeros((example_input.shape[0], seq_len, cfg.d_model), dtype=_dtype_from_name(init_dtype_name), device=device)

    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

    print(json.dumps({"event": "pipeline_stage_build_start", "rank": int(rank), "microbatch_size": int(microbatch_size), "seq_len": int(seq_len)}), flush=True)
    example_output_dtype = torch.float32 if spec.has_head else _dtype_from_name(init_dtype_name)
    example_output = torch.empty((microbatch_size, seq_len, cfg.d_model), dtype=example_output_dtype, device=device)
    stage = PipelineStage(
        shard,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=(example_input,),
        output_args=example_output,
    )
    print(json.dumps({"event": "pipeline_stage_build_done", "rank": int(rank)}), flush=True)
    optimizer = build_optimizer(args, shard)
    if args.resume:
        print(json.dumps({"event": "resume_load_start", "rank": int(rank), "resume": str(args.resume)}), flush=True)
        start_step, last_loss = load_checkpoint_shard(args.resume, shard, optimizer, preset=preset, args=args)
        print(json.dumps({"event": "resume_load_done", "rank": int(rank), "start_step": int(start_step), "last_loss": last_loss}), flush=True)

    def _unused_nonfinal_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Enable pipeline backward plumbing on non-final ranks.

        PyTorch's single-stage schedules only build backward send/recv handling
        when a loss function is present. The schedule calls the loss function
        only on the final stage, but non-final stages still need a non-None
        loss_fn so gradients flow instead of rank 0 finishing early while later
        ranks wait forever.
        """

        return output.sum() * 0.0

    current_sample_weights: dict[str, Any] = {"loss_scale": 1.0}
    loss_fn = (
        lambda hidden, labels: shard.chunked_lm_loss(
            hidden,
            labels,
            int(args.lm_loss_chunk_tokens),
            current_sample_weights.get("weights"),
            int(args.loss_token_stride),
            int(args.max_loss_tokens_per_sample),
            target_boundary_weight=float(args.target_boundary_weight),
            target_prefix_weight=float(args.target_prefix_weight),
            target_prefix_tokens=int(args.target_prefix_tokens),
            collect_diagnostics=bool(current_sample_weights.get("collect_loss_diagnostics", True)),
            labels_are_sparse=bool(current_sample_weights.get("labels_are_sparse", True)),
        )
        * float(current_sample_weights.get("loss_scale") or 1.0)
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
    if rank == 0:
        _write_log(
            args.log_file,
            {
                "event": "dataset_index_start",
                "data": str(args.data),
                "seq_len": int(seq_len),
                "max_records": int(args.max_records),
                "max_source_rows": int(args.max_source_rows),
                "max_indexed_windows": int(args.max_indexed_windows),
            },
        )
    data = (
        WeightedTextJsonlDataset(
            args.data,
            tokenizer,
            seq_len=seq_len,
            max_records=args.max_records,
            max_source_rows=args.max_source_rows,
            max_indexed_windows=args.max_indexed_windows,
            vocab_size=int(getattr(preset, "vocab_size", 0) or 0),
        )
        if rank == 0
        else None
    )
    dataset_source_summary = _dataset_source_summary(data) if rank == 0 else None
    if rank == 0:
        _write_log(
            args.log_file,
            {
                "event": "dataset_index_done",
                "samples": int(len(data) if data is not None else 0),
                "data": str(args.data),
                "seq_len": int(seq_len),
                "source_summary": dataset_source_summary,
            },
        )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": bool(args.shuffle),
        "drop_last": True,
    }
    if rank == 0:
        num_workers = max(0, int(args.dataloader_num_workers or 0))
        pin_memory = bool(args.dataloader_pin_memory) and device.type == "cuda"
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["pin_memory"] = pin_memory
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(args.dataloader_persistent_workers)
            loader_kwargs["prefetch_factor"] = max(1, int(args.dataloader_prefetch_factor or 1))
    loader = DataLoader(data, **loader_kwargs) if rank == 0 else None
    it = iter(loader) if loader is not None else None
    source_summary_box: list[Any] = [dataset_source_summary if rank == 0 else None]
    dist.broadcast_object_list(source_summary_box, src=0)
    source_summary = source_summary_box[0] if isinstance(source_summary_box[0], dict) else {"available": False}
    data_integrity_box: list[Any] = [_checkpoint_data_integrity(args) if rank == 0 else None]
    dist.broadcast_object_list(data_integrity_box, src=0)
    if isinstance(data_integrity_box[0], dict):
        setattr(args, "_checkpoint_data_integrity", data_integrity_box[0])
    def debug_event(message: str) -> None:
        if bool(args.debug_events):
            print(json.dumps({"event": "pipeline_debug", "rank": rank, "message": message}, ensure_ascii=True), flush=True)

    for local_step in range(int(args.steps)):
        step_started_at = time.time()
        global_step = start_step + local_step + 1
        step_timer = PhaseTimer(device=device, cuda_sync=bool(args.timing_cuda_sync))
        final_step = (local_step + 1) == int(args.steps)
        checkpoint_due = int(args.save_interval) > 0 and (start_step + local_step + 1) % int(args.save_interval) == 0
        log_step_timing = _should_log_interval(int(args.step_timing_interval), global_step)
        train_log_due = _should_log_interval(int(args.train_log_interval), global_step) or final_step or checkpoint_due
        collect_telemetry = _should_log_interval(int(args.telemetry_interval), global_step) or final_step
        collect_loss_diagnostics = _should_log_interval(int(args.loss_diagnostics_interval), global_step) or final_step
        collect_rank_skew = _should_log_interval(int(args.rank_skew_interval), global_step)
        log_detail_events = _should_log_interval(int(args.detailed_event_log_interval), global_step)
        if collect_telemetry:
            _reset_cuda_peak_memory(device)
        if (local_step % gradient_accumulation_steps) == 0:
            optimizer.zero_grad(set_to_none=True)
            if hasattr(optimizer, "reset_step_stats"):
                optimizer.reset_step_stats()
        losses: list[torch.Tensor] = []
        sample_weight_mean: float | None = None
        batch: torch.Tensor | None = None
        batch_labels: torch.Tensor | None = None
        batch_weights: torch.Tensor | None = None
        if rank == 0:
            if log_detail_events:
                _write_log(args.log_file, {"event": "batch_fetch_start", "local_step": int(local_step + 1), "global_step": int(global_step)})
            debug_event("rank0_fetch_batch_start")
            with step_timer.span("batch_fetch_sec"):
                try:
                    batch_item = next(it)  # type: ignore[arg-type]
                except StopIteration:
                    it = iter(loader)  # type: ignore[arg-type]
                    batch_item = next(it)
            batch, batch_labels, batch_weights = batch_item
            with step_timer.span("host_to_device_sec"):
                batch = batch.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_weights = batch_weights.to(device, non_blocking=True).float()
            debug_event("rank0_fetch_batch_done")
            if log_detail_events:
                _write_log(args.log_file, {"event": "batch_fetch_done", "local_step": int(local_step + 1), "global_step": int(global_step), "sample_weight_mean": sample_weight_mean})
        debug_event("route_step_tensors_start")
        with step_timer.span("route_step_tensors_sec"):
            routed_tensors = _route_pipeline_step_tensors(
                rank=rank,
                world_size=world_size,
                batch=batch,
                labels=batch_labels,
                sample_weights=batch_weights,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
        batch = routed_tensors.input_ids
        batch_labels = routed_tensors.labels
        batch_weights = routed_tensors.sample_weights
        current_sample_weights["weights"] = batch_weights if spec.has_head else None
        current_sample_weights["loss_scale"] = 1.0 / float(gradient_accumulation_steps)
        current_sample_weights["collect_loss_diagnostics"] = collect_loss_diagnostics
        current_sample_weights["labels_are_sparse"] = True
        debug_event("route_step_tensors_done")
        with step_timer.span("schedule_step_sec"):
            with autocast_context(device, str(args.precision)):
                if rank == 0:
                    if log_detail_events:
                        _write_log(args.log_file, {"event": "schedule_step_start", "local_step": int(local_step + 1), "global_step": int(global_step), "rank": int(rank)})
                    debug_event("schedule_step_rank0_start")
                    if batch is None:
                        raise RuntimeError("rank 0 must own input ids before schedule.step")
                    if spec.has_head:
                        if batch_labels is None:
                            raise RuntimeError("single-stage final rank must own labels before schedule.step")
                        schedule.step(batch, target=batch_labels, losses=losses)
                    else:
                        schedule.step(batch, losses=losses)
                    debug_event("schedule_step_rank0_done")
                elif spec.has_head:
                    if batch_labels is None:
                        raise RuntimeError("final pipeline rank must own labels before schedule.step")
                    debug_event("schedule_step_final_start")
                    schedule.step(target=batch_labels, losses=losses)
                    debug_event("schedule_step_final_done")
                else:
                    debug_event("schedule_step_nonzero_start")
                    schedule.step(losses=losses)
                    debug_event("schedule_step_nonzero_done")
        if rank == 0 and log_detail_events:
            _write_log(args.log_file, {"event": "schedule_step_done", "local_step": int(local_step + 1), "global_step": int(global_step), "rank": int(rank)})
        if spec.has_head:
            if losses:
                loss_tensor = torch.stack([loss.detach().float() for loss in losses]).mean().to(device=device)
            else:
                loss_tensor = torch.tensor(float(last_loss) if last_loss is not None else -1.0, device=device)
        else:
            loss_tensor = None
        rank0_loss_tensor: torch.Tensor | None = None
        if train_log_due:
            with step_timer.span("loss_rank0_sync_sec"):
                rank0_loss_tensor = _sync_pipeline_loss_to_rank0(
                    rank=rank,
                    world_size=world_size,
                    loss_tensor=loss_tensor,
                    device=device,
                )
        rank0_loss_value = _tensor_scalar_float(rank0_loss_tensor) if rank == 0 and rank0_loss_tensor is not None else None
        local_loss_value: float | None = rank0_loss_value if spec.has_head and rank == 0 else None
        if rank == 0 and rank0_loss_value is not None:
            last_loss = rank0_loss_value
        if spec.has_head and bool(args.require_target_contract):
            if local_loss_value is None and loss_tensor is not None:
                local_loss_value = _tensor_scalar_float(loss_tensor)
            if local_loss_value is not None and local_loss_value <= 0.0:
                raise RuntimeError(
                    "target contract produced a non-positive training loss; "
                    "check assistant/media target coverage before continuing"
                )
        if bool(args.require_target_contract) and rank == 0 and rank0_loss_value is not None and rank0_loss_value <= 0.0:
            raise RuntimeError(
                "target contract produced a non-positive training loss; "
                "check assistant/media target coverage before continuing"
            )
        should_update = ((local_step + 1) % gradient_accumulation_steps) == 0 or final_step
        if bool(getattr(args, "skip_final_optimizer_update", False)) and final_step:
            should_update = False
        with step_timer.span("grad_norm_pre_sec"):
            grad_norm_pre_clip = _module_grad_norm(shard, enabled=bool(args.diagnostics_grad_norm) and should_update and collect_loss_diagnostics)
        with step_timer.span("optimizer_step_sec"):
            if should_update:
                optimizer.step()
        with step_timer.span("grad_norm_post_sec"):
            grad_norm_post_clip = _module_grad_norm(shard, enabled=bool(args.diagnostics_grad_norm) and should_update and collect_loss_diagnostics)
        if rank == 0 and log_detail_events:
            _write_log(
                args.log_file,
                {
                    "event": "optimizer_step_done" if should_update else "optimizer_step_deferred",
                    "local_step": int(local_step + 1),
                    "global_step": int(start_step + local_step + 1),
                    "rank": int(rank),
                    "gradient_accumulation_steps": int(gradient_accumulation_steps),
                },
            )
        debug_event("optimizer_step_done" if should_update else "optimizer_step_deferred")
        with step_timer.span("telemetry_sec"):
            if collect_telemetry:
                memory_record = _pipeline_telemetry_record(
                    args=args,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    ranges=ranges,
                    spec=spec,
                    seq_len=seq_len,
                    step=global_step,
                    local_step=local_step + 1,
                )
                _append_pipeline_telemetry(telemetry_path, memory_record)
            else:
                memory_record = {
                    "event": "pipeline_rank_memory_telemetry_skipped",
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "device": str(device),
                    "device_type": str(device.type),
                    "device_index": int(device.index or 0) if device.type == "cuda" else None,
                    "cuda_active": bool(device.type == "cuda" and torch.cuda.is_available()),
                    "step": int(global_step),
                    "local_step": int(local_step + 1),
                    "telemetry_interval": int(args.telemetry_interval),
                }
        loss_diagnostics = getattr(shard, "last_lm_loss_diagnostics", {}) if spec.has_head else {}
        target_summary: PipelineTargetSummary | None = None
        with step_timer.span("target_summary_sync_sec"):
            if collect_loss_diagnostics:
                target_summary = _sync_pipeline_target_summary_to_rank0(
                    rank=rank,
                    world_size=world_size,
                    loss_diagnostics=loss_diagnostics if spec.has_head else None,
                    sample_weights=batch_weights if spec.has_head else None,
                    device=device,
                )
        if rank == 0 and target_summary is not None:
            sample_weight_mean = target_summary.sample_weight_mean
        step_elapsed_sec = step_timer.elapsed()
        with step_timer.span("diagnostics_write_sec"):
            if spec.has_head and collect_loss_diagnostics:
                if local_loss_value is None and loss_tensor is not None:
                    local_loss_value = _tensor_scalar_float(loss_tensor)
                _append_pipeline_telemetry(
                    train_diagnostics_path,
                    _train_diagnostics_record(
                        args=args,
                        rank=rank,
                        world_size=world_size,
                        spec=spec,
                        global_step=global_step,
                        local_step=local_step + 1,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        microbatch_size=microbatch_size,
                        loss=float(local_loss_value if local_loss_value is not None else -1.0),
                        labels=batch_labels,
                        sample_weights=batch_weights,
                        optimizer=optimizer,
                        optimizer_update=bool(should_update),
                        grad_norm_pre_clip=grad_norm_pre_clip,
                        grad_norm_post_clip=grad_norm_post_clip,
                        step_elapsed_sec=step_elapsed_sec,
                        memory_record=memory_record,
                        loss_diagnostics=loss_diagnostics,
                        source_summary=source_summary,
                    ),
                )
        if rank == 0 and train_log_due:
            with step_timer.span("log_write_sec"):
                _write_log(args.log_file, {"step": global_step, "local_step": local_step + 1, "loss": rank0_loss_value, "preset": preset.name, "seq_len": seq_len, "distributed": "pipeline", "world_size": world_size, "pipeline_schedule": args.pipeline_schedule, "pipeline_microbatches": pipeline_microbatches, "microbatch_size": microbatch_size, "batch_size": int(batch_size), "sample_weight_mean": sample_weight_mean, "optimizer": str(args.optimizer), "optimizer_in_backward": bool(args.optimizer_in_backward), "optimizer_in_backward_update": str(args.optimizer_in_backward_update), "loss_token_stride": int(args.loss_token_stride), "max_loss_tokens_per_sample": int(args.max_loss_tokens_per_sample), "target_boundary_weight": float(args.target_boundary_weight), "target_prefix_weight": float(args.target_prefix_weight), "target_prefix_tokens": int(args.target_prefix_tokens), "gradient_accumulation_steps": int(gradient_accumulation_steps), "optimizer_update": bool(should_update), "shuffle": bool(args.shuffle), "train_diagnostics_file": str(final_train_diagnostics_path), "step_timing_file": str(step_timing_path), "loss_diagnostics_collected": bool(collect_loss_diagnostics), "valid_target_tokens": int(target_summary.valid_target_tokens if target_summary is not None else 0), "optimized_target_tokens": int(target_summary.optimized_target_tokens if target_summary is not None else 0)})
        if log_step_timing:
            rank_elapsed: list[float] = []
            if collect_rank_skew:
                elapsed_tensor = torch.tensor([float(step_elapsed_sec)], device=device)
                gathered = [torch.zeros_like(elapsed_tensor) for _ in range(world_size)]
                with step_timer.span("rank_skew_gather_sec"):
                    dist.all_gather(gathered, elapsed_tensor)
                rank_elapsed = [float(item.detach().cpu()[0].item()) for item in gathered]
            timing_record = {
                "schema": STEP_TIMING_SCHEMA,
                "event": "pipeline_step_timing",
                "timestamp": time.time(),
                "rank": int(rank),
                "world_size": int(world_size),
                "global_step": int(global_step),
                "local_step": int(local_step + 1),
                "total_sec": float(step_timer.elapsed()),
                "spans": step_timer.record(),
                "cuda_sync_timing": bool(args.timing_cuda_sync),
                "loss_diagnostics_collected": bool(collect_loss_diagnostics),
                "rank_elapsed_sec": rank_elapsed,
                "rank_skew_sec": (max(rank_elapsed) - min(rank_elapsed)) if rank_elapsed else None,
                "optimizer_diagnostics": optimizer.diagnostics() if hasattr(optimizer, "diagnostics") else {},
                "lm_loss_timing": getattr(shard, "last_lm_loss_timing", {}) if spec.has_head else {},
            }
            _append_pipeline_telemetry(step_timing_path, timing_record)
        if bool(args.block_timing) and getattr(shard, "block_timing_records", None):
            records = list(getattr(shard, "block_timing_records", []))
            shard.block_timing_records.clear()
            _append_pipeline_telemetry(
                block_timing_path,
                {
                    "schema": "omnicoder.pipeline_block_timing_2026.v1",
                    "event": "pipeline_block_timing",
                    "timestamp": time.time(),
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "global_step": int(global_step),
                    "local_step": int(local_step + 1),
                    "cuda_sync_timing": bool(args.block_timing_cuda_sync),
                    "records": records,
                },
            )
        if checkpoint_due:
            checkpoint_loss = _sync_pipeline_loss_for_checkpoint(
                rank=rank,
                world_size=world_size,
                loss_tensor=loss_tensor,
                device=device,
            )
            last_loss = checkpoint_loss
            save_sharded_checkpoint(Path(args.out).with_name(f"{Path(args.out).stem}.step{global_step}"), shard, preset=preset, args=args, optimizer=optimizer, global_step=global_step, last_loss=checkpoint_loss)

    if not bool(args.skip_final_save):
        final_loss = _sync_pipeline_loss_for_checkpoint(
            rank=rank,
            world_size=world_size,
            loss_tensor=loss_tensor,
            device=device,
        )
        last_loss = final_loss
        save_sharded_checkpoint(args.out, shard, preset=preset, args=args, optimizer=optimizer, global_step=start_step + int(args.steps), last_loss=final_loss)
    if rank == 0:
        _write_log(args.log_file, {"status": "ok", "out": args.out, "last_loss": last_loss, "global_step": start_step + int(args.steps), "distributed": "pipeline", "world_size": world_size, "final_save_skipped": bool(args.skip_final_save)})
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

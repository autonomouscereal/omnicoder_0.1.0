from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
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


REQUEST_SCHEMA = "omnicoder.reportable_prediction_request_2026.v1"
OUTPUT_KEYS = ("prediction", "model_patch", "tool_call", "model_actions", "artifact_path")
CONFIG_FIELDS = {field.name for field in fields(OmniCoder2026Config)}


class RunnerError(ValueError):
    """Raised for input, checkpoint, or decode failures that must fail closed."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunnerError(f"{path} must contain one JSON object")
    return payload


def _read_stdin_request() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise RunnerError("checkpoint runner expected one JSON request on stdin")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"invalid JSON request on stdin: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerError("checkpoint runner request must be a JSON object")
    return payload


def _checkpoint_dir(request: dict[str, Any]) -> Path:
    raw = str(request.get("checkpoint_path") or "").strip()
    if not raw:
        raise RunnerError("request is missing checkpoint_path")
    path = Path(raw)
    if not path.exists() or not path.is_dir():
        raise RunnerError(f"pipeline checkpoint runner requires a sharded checkpoint directory, got: {path}")
    return path


def _rank_files(checkpoint: Path) -> list[Path]:
    return sorted(path for path in checkpoint.glob("rank*.pt") if path.is_file())


def _load_manifest(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / ".complete.json"
    if not manifest_path.exists():
        raise RunnerError(f"pipeline checkpoint is missing manifest.json: {checkpoint}")
    if not complete_path.exists():
        raise RunnerError(f"pipeline checkpoint is missing .complete.json: {checkpoint}")
    manifest = _read_json(manifest_path)
    rank_files = _rank_files(checkpoint)
    if not rank_files:
        raise RunnerError(f"pipeline checkpoint has no rank*.pt files: {checkpoint}")
    world_size = int(manifest.get("world_size") or len(rank_files))
    expected = [checkpoint / f"rank{rank:05d}.pt" for rank in range(world_size)]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise RunnerError(f"pipeline checkpoint rank files are not contiguous; missing: {missing}")
    marker_missing = [f"rank{rank:05d}.pt.complete.json" for rank in range(world_size) if not (checkpoint / f"rank{rank:05d}.pt.complete.json").exists()]
    if marker_missing:
        raise RunnerError(f"pipeline checkpoint is missing rank completion markers: {marker_missing}")
    return manifest


def _checkpoint_train_args(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return {}
    train_args = manifest.get("train_args")
    if isinstance(train_args, dict):
        return train_args
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
        raise RunnerError(f"{checkpoint / 'rank00000.pt'} is not a checkpoint payload")
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
    kwargs["tie_embeddings"] = False
    return kwargs, saved_name or preset.name


def _build_shard(args: argparse.Namespace) -> tuple[OmniCoder2026PipelineShard, torch.device, int, int]:
    if not dist.is_initialized():
        backend = str(args.dist_backend or "auto").lower()
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_seconds = int(getattr(args, "dist_timeout_seconds", 0) or os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200"))
        dist.init_process_group(backend=backend, timeout=_dt.timedelta(seconds=max(1, timeout_seconds)))
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    device = rank_device(rank, args.rank_device_map)
    checkpoint = Path(args.checkpoint)
    train_args = _checkpoint_train_args(checkpoint)
    kwargs, saved_preset_name = _checkpoint_kwargs(checkpoint, args.preset)
    if bool(args.fake_quant) or bool(train_args.get("fake_quant")):
        kwargs["fake_quant"] = True
    cfg = OmniCoder2026Config(**kwargs)
    placement_counts = str(args.placement_layer_counts or "").strip() or _placement_counts_from_train_args(train_args)
    ranges = stage_ranges(int(cfg.n_layers), placement_counts)
    if len(ranges) != world_size:
        raise RunnerError(f"world_size={world_size} must match pipeline stages {ranges}")
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
    shard.eval()
    return shard, device, int(cfg.d_model), int(cfg.vocab_size)


def _output_field(request: dict[str, Any]) -> str:
    task = request.get("task") if isinstance(request.get("task"), dict) else {}
    text = " ".join(
        str(task.get(key) or request.get(key) or "")
        for key in ("benchmark_id", "adapter_id", "adapter_kind", "axis", "task_format", "source")
    ).lower()
    if "swe" in text or "patch" in text or "git" in text or task.get("repo"):
        return "model_patch"
    if "tool" in text or "bfcl" in text or "mcp" in text:
        return "tool_call"
    if "arc_agi3" in text or "interactive" in text:
        return "model_actions"
    if "image_generation" in text or "video_generation" in text or "audio_generation" in text or "music_generation" in text:
        return "artifact_path"
    return "prediction"


def _validate_request(request: dict[str, Any], args: argparse.Namespace) -> tuple[str, int]:
    schema = str(request.get("schema") or "")
    if schema and schema != REQUEST_SCHEMA:
        raise RunnerError(f"unsupported request schema: {schema}")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RunnerError("request is missing non-empty prompt")
    try:
        max_new = int(request.get("max_output_tokens") or args.max_new_tokens_limit)
    except Exception as exc:
        raise RunnerError("request max_output_tokens must be an integer") from exc
    if max_new <= 0:
        raise RunnerError("request max_output_tokens must be positive")
    max_new = min(max_new, int(args.max_new_tokens_limit))
    temperature = float(request.get("temperature") or 0.0)
    if abs(temperature) > 1.0e-8:
        raise RunnerError("pipeline checkpoint runner supports only greedy decode with temperature=0")
    return prompt, max_new


def _broadcast_ids(batch: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    rank = int(dist.get_rank())
    meta = torch.tensor([int(batch.shape[1]) if batch is not None else 0], dtype=torch.long, device=device)
    dist.broadcast(meta, src=0)
    length = int(meta[0].item())
    if length <= 0:
        raise RunnerError("cannot decode an empty prompt")
    if rank != 0:
        batch = torch.empty((1, length), dtype=torch.long, device=device)
    else:
        batch = batch.to(device, non_blocking=True)
    dist.broadcast(batch, src=0)
    return batch


def _pipeline_next_token(
    shard: OmniCoder2026PipelineShard,
    batch: torch.Tensor,
    *,
    device: torch.device,
    hidden_dtype: torch.dtype,
    d_model: int,
    precision: str,
) -> int:
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    length = int(batch.shape[1])
    with torch.no_grad(), autocast_context(device, precision):
        if world_size == 1:
            hidden = shard(batch)
            logits = shard.lm_head(hidden[:, -1:, :]).float()
            token = torch.argmax(logits[:, -1, :], dim=-1).to(dtype=torch.long)
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
        token = torch.argmax(logits[:, -1, :], dim=-1).to(dtype=torch.long)
        dist.broadcast(token, src=rank)
        return int(token.detach().cpu().item())


def _decode_worker(args: argparse.Namespace) -> dict[str, Any] | None:
    request = _read_json(Path(args.request))
    prompt, max_new = _validate_request(request, args)
    shard, device, d_model, vocab_size = _build_shard(args)
    rank = int(dist.get_rank())
    tokenizer = get_text_tokenizer(prefer_hf=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []
    ids = [int(item) for item in tokenizer.encode(prompt)]
    if not ids:
        raise RunnerError("tokenizer produced no prompt tokens")
    if len(ids) > int(args.max_prompt_tokens):
        raise RunnerError(
            f"prompt token length {len(ids)} exceeds --max-prompt-tokens={int(args.max_prompt_tokens)}"
        )
    if rank == 0:
        ids = [int(item) % max(2, vocab_size) for item in ids]
        generated = list(ids)
    hidden_dtype_name = str(args.init_dtype if str(args.init_dtype or "auto").lower() != "auto" else args.precision)
    hidden_dtype = _dtype_from_name(hidden_dtype_name)
    started = time.perf_counter()
    for _ in range(max_new):
        batch = None
        if rank == 0:
            batch = torch.tensor([generated], dtype=torch.long, device=device)
        batch = _broadcast_ids(batch, device)
        next_token = _pipeline_next_token(
            shard,
            batch,
            device=device,
            hidden_dtype=hidden_dtype,
            d_model=d_model,
            precision=str(args.precision),
        )
        if rank == 0:
            generated.append(int(next_token) % max(2, vocab_size))
        if isinstance(eos_id, int) and int(next_token) == int(eos_id):
            break
    if rank != 0:
        return None
    new_tokens = generated[-max_new:]
    text = tokenizer.decode(new_tokens).strip()  # type: ignore[union-attr]
    if not text:
        raise RunnerError("greedy decode produced empty text")
    field = _output_field(request)
    if field == "artifact_path":
        raise RunnerError("text-only pipeline checkpoint decode cannot produce a real artifact_path")
    return {
        field: text,
        "metadata": {
            "backend": "pipeline_checkpoint_predict_2026",
            "checkpoint_path": str(request.get("checkpoint_path") or ""),
            "benchmark_id": str(request.get("benchmark_id") or ""),
            "task_id": str(request.get("task_id") or ""),
            "generated_tokens": len(new_tokens),
            "latency_seconds": round(time.perf_counter() - started, 6),
        },
    }


def _torchrun_world_size(checkpoint: Path, manifest: dict[str, Any], explicit: int) -> int:
    if explicit > 0:
        return explicit
    raw = manifest.get("world_size")
    if raw is not None:
        return int(raw)
    return len(_rank_files(checkpoint))


def _json_line_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RunnerError("worker did not emit a JSON object on stdout")


def _parent_main(args: argparse.Namespace) -> int:
    request = _read_stdin_request()
    checkpoint = _checkpoint_dir(request)
    manifest = _load_manifest(checkpoint)
    nproc = _torchrun_world_size(checkpoint, manifest, int(args.nproc_per_node or 0))
    if nproc <= 0:
        raise RunnerError(f"invalid torchrun world size: {nproc}")
    request["checkpoint_path"] = str(checkpoint)
    with tempfile.TemporaryDirectory(prefix="omnicoder_pipeline_predict_") as tmp:
        request_path = Path(tmp) / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=True, sort_keys=True), encoding="utf-8")
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
            "omnicoder.eval.pipeline_checkpoint_predict_2026",
            "--worker",
            "--request",
            str(request_path),
            "--checkpoint",
            str(checkpoint),
            "--preset",
            str(args.preset),
            "--precision",
            str(args.precision),
            "--init-dtype",
            str(args.init_dtype),
            "--max-new-tokens-limit",
            str(args.max_new_tokens_limit),
            "--max-prompt-tokens",
            str(args.max_prompt_tokens),
            "--dist-backend",
            str(args.dist_backend),
            "--dist-timeout-seconds",
            str(args.dist_timeout_seconds),
        ]
        if str(args.rank_device_map or "").strip():
            cmd.extend(["--rank-device-map", str(args.rank_device_map)])
        if str(args.placement_layer_counts or "").strip():
            cmd.extend(["--placement-layer-counts", str(args.placement_layer_counts)])
        if bool(args.fake_quant):
            cmd.append("--fake-quant")
        if bool(args.require_target_contract):
            cmd.append("--require-target-contract")
        if bool(args.allow_p40_target_contract_eval):
            cmd.append("--allow-p40-target-contract-eval")
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            timeout=float(args.torchrun_timeout_seconds),
            check=False,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise RunnerError(f"torchrun worker failed with exit {proc.returncode}: {detail}")
    payload = _json_line_from_stdout(proc.stdout)
    if not any(payload.get(key) not in (None, "", [], {}) for key in OUTPUT_KEYS):
        raise RunnerError(f"worker JSON did not contain a model output field: {payload!r}")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint-runner helper for sharded Omnicoder2026 pipeline checkpoints")
    parser.add_argument("--worker", action="store_true", help="Internal torchrun worker mode")
    parser.add_argument("--request", default="", help="Worker-mode request JSON path")
    parser.add_argument("--checkpoint", default="", help="Worker-mode checkpoint directory")
    parser.add_argument("--preset", default=os.getenv("OMNICODER2026_PIPELINE_PREDICT_PRESET", "omnicoder2026_20b_1m"))
    parser.add_argument("--rank-device-map", "--rank_device_map", dest="rank_device_map", default=os.getenv("OMNICODER2026_PIPELINE_PREDICT_RANK_DEVICE_MAP", ""))
    parser.add_argument("--placement-layer-counts", "--placement_layer_counts", dest="placement_layer_counts", default=os.getenv("OMNICODER2026_PIPELINE_PREDICT_PLACEMENT_LAYER_COUNTS", ""))
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_PREDICT_PRECISION", "fp16"))
    parser.add_argument("--init-dtype", "--init_dtype", dest="init_dtype", choices=["auto", "fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PIPELINE_PREDICT_INIT_DTYPE", "auto"))
    parser.add_argument("--dist-backend", default=os.getenv("OMNICODER2026_PIPELINE_PREDICT_DIST_BACKEND", "auto"))
    parser.add_argument("--dist-timeout-seconds", "--dist_timeout_seconds", dest="dist_timeout_seconds", type=int, default=int(os.getenv("OMNICODER2026_DIST_TIMEOUT_SECONDS", "7200") or 7200))
    parser.add_argument("--nproc-per-node", "--nproc_per_node", dest="nproc_per_node", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_PREDICT_NPROC_PER_NODE", "0") or 0))
    parser.add_argument("--max-new-tokens-limit", "--max_new_tokens_limit", dest="max_new_tokens_limit", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_PREDICT_MAX_NEW_TOKENS", "256") or 256))
    parser.add_argument("--max-prompt-tokens", "--max_prompt_tokens", dest="max_prompt_tokens", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_PREDICT_MAX_PROMPT_TOKENS", "4096") or 4096))
    parser.add_argument("--torchrun-timeout-seconds", "--torchrun_timeout_seconds", dest="torchrun_timeout_seconds", type=float, default=float(os.getenv("OMNICODER2026_PIPELINE_PREDICT_TIMEOUT_SECONDS", "1800") or 1800))
    parser.add_argument("--fake-quant", "--fake_quant", dest="fake_quant", action="store_true")
    parser.add_argument("--require-target-contract", "--require_target_contract", dest="require_target_contract", action="store_true")
    parser.add_argument("--allow-p40-target-contract-eval", "--allow_p40_target_contract_eval", dest="allow_p40_target_contract_eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker:
            if not args.request or not args.checkpoint:
                raise RunnerError("--worker requires --request and --checkpoint")
            result = _decode_worker(args)
            if result is not None:
                print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
            return 0
        return _parent_main(args)
    except (RunnerError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        payload = {"status": "error", "error": str(exc), "runner": "pipeline_checkpoint_predict_2026"}
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

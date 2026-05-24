from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Block, OmniCoder2026Config

try:  # Optional on CPU-only or stripped-down PyTorch installs.
    from torch.distributed.fsdp import (
        CPUOffload,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except Exception:  # pragma: no cover - depends on the installed torch build.
    CPUOffload = None  # type: ignore[assignment]
    FullStateDictConfig = None  # type: ignore[assignment]
    FSDP = None  # type: ignore[assignment]
    MixedPrecision = None  # type: ignore[assignment]
    ShardingStrategy = None  # type: ignore[assignment]
    StateDictType = None  # type: ignore[assignment]
    transformer_auto_wrap_policy = None  # type: ignore[assignment]


FSDP_LOCAL_FORMAT = "omnicoder2026_native_train_checkpoint_v3_fsdp_local"
CONSOLIDATED_FORMAT = "omnicoder2026_native_train_checkpoint_v2_from_fsdp_local"
CONFIG_FIELDS = {field.name for field in fields(OmniCoder2026Config)}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _dtype_from_name(name: str) -> torch.dtype:
    key = str(name or "fp32").lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.float32


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_fsdp_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("format") != FSDP_LOCAL_FORMAT:
            raise ValueError(f"{manifest_path} is not an Omnicoder2026 rank-local FSDP manifest")
        return manifest
    rank_paths = rank_checkpoint_files(path)
    if not rank_paths:
        raise FileNotFoundError(f"no rank-local FSDP files found in {path}")
    return {
        "format": FSDP_LOCAL_FORMAT,
        "checkpoint_dir": str(path),
        "rank_files": [item.name for item in rank_paths],
        "world_size": len(rank_paths),
    }


def rank_checkpoint_files(checkpoint_dir: str | Path) -> list[Path]:
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        return []
    return sorted(item for item in path.glob("rank*.pt") if item.is_file())


def is_fsdp_rank_local_checkpoint_dir(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    manifest_path = candidate / "manifest.json"
    if manifest_path.exists():
        try:
            return load_fsdp_manifest(candidate).get("format") == FSDP_LOCAL_FORMAT
        except Exception:
            return False
    return bool(rank_checkpoint_files(candidate))


def fsdp_world_size(checkpoint_dir: str | Path) -> int:
    manifest = load_fsdp_manifest(checkpoint_dir)
    raw = manifest.get("world_size")
    if raw is None:
        rank_files = manifest.get("rank_files")
        if isinstance(rank_files, list):
            raw = len(rank_files)
    return int(raw or len(rank_checkpoint_files(checkpoint_dir)) or 1)


def fsdp_rank_file(checkpoint_dir: str | Path, rank: int) -> Path:
    path = Path(checkpoint_dir)
    candidate = path / f"rank{int(rank):05d}.pt"
    if candidate.exists():
        return candidate
    rank_files = rank_checkpoint_files(path)
    if 0 <= int(rank) < len(rank_files):
        return rank_files[int(rank)]
    raise FileNotFoundError(f"missing rank-local checkpoint for rank {rank}: {candidate}")


def load_rank_payload(checkpoint_dir: str | Path, rank: int, map_location: Any = "cpu") -> dict[str, Any]:
    rank_path = fsdp_rank_file(checkpoint_dir, rank)
    payload = torch.load(rank_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FSDP_LOCAL_FORMAT:
        raise ValueError(f"{rank_path} is not an Omnicoder2026 rank-local FSDP checkpoint")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(f"{rank_path} does not contain model_state_dict")
    return payload


def checkpoint_model_kwargs(checkpoint: dict[str, Any], profile: str) -> dict[str, Any]:
    preset = get_omnicoder2026_preset(profile)
    kwargs = preset_to_model_kwargs(preset)

    for key in ("config", "model_config"):
        value = checkpoint.get(key)
        if value is not None and not isinstance(value, dict) and hasattr(value, "__dict__"):
            value = value.__dict__
        if isinstance(value, dict):
            kwargs.update({k: v for k, v in value.items() if k in CONFIG_FIELDS})

    preset_payload = checkpoint.get("preset")
    if isinstance(preset_payload, dict):
        kwargs.update({k: v for k, v in preset_payload.items() if k in CONFIG_FIELDS})

    train_args = checkpoint.get("train_args")
    if isinstance(train_args, dict) and "fake_quant" in train_args:
        kwargs["fake_quant"] = bool(train_args.get("fake_quant"))

    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict):
        embed = state.get("embed.weight")
        if isinstance(embed, torch.Tensor) and embed.ndim == 2:
            kwargs["vocab_size"] = int(embed.shape[0])
            kwargs["d_model"] = int(embed.shape[1])
        layer_ids: set[int] = set()
        for name in state:
            if str(name).startswith("blocks."):
                parts = str(name).split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_ids.add(int(parts[1]))
        if layer_ids:
            kwargs["n_layers"] = max(layer_ids) + 1
        gate = state.get("blocks.0.ffn.gate.weight")
        if isinstance(gate, torch.Tensor) and gate.ndim == 2:
            kwargs["mlp_dim"] = int(gate.shape[0])

    if isinstance(kwargs.get("layer_pattern"), list):
        kwargs["layer_pattern"] = tuple(kwargs["layer_pattern"])
    return kwargs


def checkpoint_fingerprint(path: str | Path) -> str:
    candidate = Path(path)
    digest = hashlib.sha256()
    if candidate.is_file():
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not candidate.is_dir():
        digest.update(str(candidate).encode("utf-8"))
        return digest.hexdigest()
    manifest = candidate / "manifest.json"
    if manifest.exists():
        digest.update(manifest.read_bytes())
    for rank_path in rank_checkpoint_files(candidate):
        stat = rank_path.stat()
        digest.update(rank_path.name.encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def torchrun_hint(checkpoint_dir: str | Path, module: str, extra_args: list[str] | None = None) -> str:
    args = " ".join(extra_args or [])
    suffix = f" {args}" if args else ""
    return (
        f"{sys.executable} -m torch.distributed.run --nproc_per_node {fsdp_world_size(checkpoint_dir)} "
        f"-m {module}{suffix}"
    )


def distributed_context(device_arg: str | torch.device, backend: str = "nccl", *, require_initialized: bool = False) -> dict[str, Any]:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    device_text = str(device_arg)
    use_cuda = torch.cuda.is_available() and device_text != "cpu"
    chosen_backend = str(backend or "nccl")
    if not use_cuda and chosen_backend == "nccl":
        chosen_backend = "gloo"
    should_init = world_size > 1 or (require_initialized and os.environ.get("RANK") is not None)
    if should_init and not dist.is_initialized():
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable; cannot load rank-local FSDP checkpoint")
        dist.init_process_group(backend=chosen_backend)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = _env_int("LOCAL_RANK", rank)
    elif dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = _env_int("LOCAL_RANK", rank)
    elif require_initialized:
        raise RuntimeError("rank-local FSDP checkpoint requires torchrun with a matching WORLD_SIZE")

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "device": device,
        "backend": chosen_backend,
        "is_main": int(rank) == 0,
        "distributed": int(world_size) > 1,
    }


def _require_fsdp() -> None:
    if (
        FSDP is None
        or StateDictType is None
        or FullStateDictConfig is None
        or CPUOffload is None
        or MixedPrecision is None
        or ShardingStrategy is None
        or transformer_auto_wrap_policy is None
    ):
        raise RuntimeError("torch.distributed.fsdp is unavailable in this environment")


def wrap_fsdp_for_eval(
    model: OmniCoder2026,
    *,
    device: torch.device,
    precision: str = "fp32",
    cpu_offload: bool = False,
) -> torch.nn.Module:
    _require_fsdp()
    import functools

    mp_dtype = _dtype_from_name(precision)
    mixed_precision = None
    if mp_dtype in (torch.float16, torch.bfloat16):
        mixed_precision = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={OmniCoder2026Block})
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=bool(cpu_offload)),
        device_id=device if device.type == "cuda" else None,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def load_fsdp_rank_local_model(
    checkpoint_dir: str | Path,
    profile: str,
    device_arg: str | torch.device,
    *,
    dist_backend: str = "nccl",
    precision: str = "fp32",
    cpu_offload: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if not is_fsdp_rank_local_checkpoint_dir(checkpoint_dir):
        raise ValueError(f"{checkpoint_dir} is not a rank-local FSDP checkpoint directory")
    expected_world = fsdp_world_size(checkpoint_dir)
    ctx = distributed_context(device_arg, dist_backend, require_initialized=True)
    if int(ctx["world_size"]) != int(expected_world):
        raise RuntimeError(
            f"FSDP checkpoint has world_size={expected_world}, but current torchrun world_size={ctx['world_size']}. "
            f"Use: {torchrun_hint(checkpoint_dir, 'omnicoder.eval.sample_loss_2026')}"
        )
    payload = load_rank_payload(checkpoint_dir, int(ctx["rank"]), map_location=ctx["device"])
    cfg = OmniCoder2026Config(**checkpoint_model_kwargs(payload, profile))
    raw_model = OmniCoder2026(cfg)
    model = wrap_fsdp_for_eval(
        raw_model,
        device=ctx["device"],
        precision=precision,
        cpu_offload=cpu_offload,
    )
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, ctx, payload


def consolidate_fsdp_rank_local_checkpoint(
    checkpoint_dir: str | Path,
    out: str | Path,
    profile: str,
    device_arg: str,
    *,
    dist_backend: str = "nccl",
    precision: str = "fp32",
    cpu_offload: bool = False,
) -> dict[str, Any] | None:
    model, ctx, payload = load_fsdp_rank_local_model(
        checkpoint_dir,
        profile,
        device_arg,
        dist_backend=dist_backend,
        precision=precision,
        cpu_offload=cpu_offload,
    )
    _require_fsdp()
    full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
        full_state = model.state_dict()
    result: dict[str, Any] | None = None
    if ctx["is_main"]:
        manifest = load_fsdp_manifest(checkpoint_dir)
        native_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"model_state_dict", "optimizer_state_dict", "rng_state"}
        }
        native_payload.update(
            {
                "format": CONSOLIDATED_FORMAT,
                "model_state_dict": full_state,
                "optimizer_state_dict": None,
                "fsdp_source": {
                    "checkpoint_dir": str(Path(checkpoint_dir)),
                    "manifest": manifest,
                    "source_format": FSDP_LOCAL_FORMAT,
                    "world_size": int(ctx["world_size"]),
                },
            }
        )
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(native_payload, out_path)
        result = {
            "status": "ok",
            "out": str(out_path),
            "checkpoint_dir": str(Path(checkpoint_dir)),
            "world_size": int(ctx["world_size"]),
            "format": CONSOLIDATED_FORMAT,
            "model_keys": len(full_state),
        }
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return result


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.checkpoint_dir)
    payload = {
        "checkpoint_dir": str(path),
        "is_fsdp_rank_local": is_fsdp_rank_local_checkpoint_dir(path),
        "fingerprint": checkpoint_fingerprint(path),
    }
    if payload["is_fsdp_rank_local"]:
        manifest = load_fsdp_manifest(path)
        payload.update(
            {
                "format": manifest.get("format"),
                "world_size": fsdp_world_size(path),
                "rank_files": [item.name for item in rank_checkpoint_files(path)],
                "torchrun_sample_loss_hint": torchrun_hint(path, "omnicoder.eval.sample_loss_2026"),
                "torchrun_consolidate_hint": torchrun_hint(path, "omnicoder.eval.fsdp_checkpoint_2026"),
            }
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    result = consolidate_fsdp_rank_local_checkpoint(
        args.checkpoint_dir,
        args.out,
        args.profile,
        args.device,
        dist_backend=args.dist_backend,
        precision=args.precision,
        cpu_offload=bool(args.cpu_offload),
    )
    if result is not None:
        print(json.dumps(result, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or consolidate Omnicoder2026 rank-local FSDP checkpoints")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect a rank-local FSDP checkpoint directory")
    inspect_p.add_argument("--checkpoint-dir", required=True)
    inspect_p.set_defaults(func=cmd_inspect)

    consolidate_p = sub.add_parser("consolidate", help="Consolidate rank-local shards into a native .pt checkpoint")
    consolidate_p.add_argument("--checkpoint-dir", required=True)
    consolidate_p.add_argument("--out", required=True)
    consolidate_p.add_argument("--profile", default="ledger_probe")
    consolidate_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    consolidate_p.add_argument("--dist-backend", default=os.getenv("OMNICODER2026_DIST_BACKEND", "nccl"))
    consolidate_p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=os.getenv("OMNICODER2026_PRECISION", "fp32"))
    consolidate_p.add_argument("--cpu-offload", action="store_true")
    consolidate_p.set_defaults(func=cmd_consolidate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

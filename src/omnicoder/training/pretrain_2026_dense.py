from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from omnicoder.config_2026 import get_omnicoder2026_preset, preset_to_model_kwargs
from omnicoder.model_contract_2026 import validate_target_contract_preset
from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Block, OmniCoder2026Config
from omnicoder.training.simple_tokenizer import get_text_tokenizer

try:  # Optional on CPU-only development hosts.
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
except Exception:  # pragma: no cover - exercised on the AI-server CUDA image.
    CheckpointImpl = None  # type: ignore[assignment]
    apply_activation_checkpointing = None  # type: ignore[assignment]
    checkpoint_wrapper = None  # type: ignore[assignment]
    CPUOffload = None  # type: ignore[assignment]
    FSDP = None  # type: ignore[assignment]
    MixedPrecision = None  # type: ignore[assignment]
    ShardingStrategy = None  # type: ignore[assignment]
    StateDictType = None  # type: ignore[assignment]
    transformer_auto_wrap_policy = None  # type: ignore[assignment]
    ShardedGradScaler = None  # type: ignore[assignment]


TARGET_PRESET = "omnicoder2026_20b_1m"
PROBE_PRESET_NAMES = {"probe", "native1m_probe", "ledger_probe", "full_ledger_probe", "omnicoder2026_native1m_probe", "omnicoder2026_full_ledger_probe"}


def _message_text(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    role = str(message.get("role") or "message")
    content = message.get("content")
    if isinstance(content, str):
        return f"{role}: {content}"
    if isinstance(content, (dict, list)):
        return f"{role}: {json.dumps(content, ensure_ascii=True, sort_keys=True)}"
    return ""


def _text_from_record(obj: dict) -> str:
    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        return "\n".join(part for part in (_message_text(m) for m in messages) if part)
    parts: list[str] = []
    input_json = obj.get("input_json")
    target_json = obj.get("target_json")
    if isinstance(input_json, dict):
        imessages = input_json.get("messages")
        if isinstance(imessages, list):
            parts.extend(part for part in (_message_text(m) for m in imessages) if part)
        for key in ("content", "prompt", "text"):
            value = input_json.get(key)
            if isinstance(value, str) and value:
                parts.append(f"user: {value}")
    if isinstance(target_json, dict):
        for key in ("content", "completion", "answer", "caption"):
            value = target_json.get(key)
            if isinstance(value, str) and value:
                parts.append(f"assistant: {value}")
        if target_json.get("artifact_path"):
            parts.append(f"assistant: {json.dumps(target_json, ensure_ascii=True, sort_keys=True)}")
    if parts:
        return "\n".join(parts)
    text = obj.get("text") or obj.get("prompt") or obj.get("completion") or obj.get("content") or ""
    if isinstance(text, list):
        return "\n".join(str(x) for x in text)
    return str(text)


def _ids_from_record(obj: dict) -> list[int] | None:
    for key in ("token_ids", "input_ids", "ids"):
        value = obj.get(key)
        if isinstance(value, list) and value:
            ids: list[int] = []
            for item in value:
                try:
                    ids.append(int(item))
                except Exception:
                    return None
            return ids
    return None


class TextJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int, max_records: int = 0, vocab_size: int = 0):
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.samples: list[list[int]] = []
        limit = int(max_records) if int(max_records) > 0 else None
        p = Path(path)
        if p.is_dir():
            paths = sorted(p.rglob("*.jsonl")) + sorted(p.rglob("*.txt"))
        else:
            paths = [p]
        for src in paths:
            if limit is not None and len(self.samples) >= limit:
                break
            self._load_path(src, limit)
        if not self.samples:
            self.samples.append([1] * self.seq_len)

    def _load_path(self, path: Path, limit: int | None) -> None:
        if not path.exists():
            return
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            self._append_text(text, limit)
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
            ids = _ids_from_record(obj) if isinstance(obj, dict) else None
            if ids:
                self._append_ids(ids, limit)
                continue
            self._append_text(_text_from_record(obj), limit)

    def _append_text(self, text: str, limit: int | None) -> None:
        if limit is not None and len(self.samples) >= limit:
            return
        ids = [int(x) for x in self.tokenizer.encode(text)]
        if not ids:
            return
        step = max(1, self.seq_len)
        for start in range(0, len(ids), step):
            chunk = ids[start:start + self.seq_len]
            if len(chunk) < 2:
                continue
            self.samples.append(chunk)
            if limit is not None and len(self.samples) >= limit:
                break

    def _append_ids(self, ids: list[int], limit: int | None) -> None:
        if limit is not None and len(self.samples) >= limit:
            return
        cleaned = [self._sanitize_id(x) for x in ids]
        if len(cleaned) < 2:
            return
        step = max(1, self.seq_len)
        for start in range(0, len(cleaned), step):
            chunk = cleaned[start:start + self.seq_len]
            if len(chunk) < 2:
                continue
            self.samples.append(chunk)
            if limit is not None and len(self.samples) >= limit:
                break

    def _sanitize_id(self, value: int) -> int:
        token = int(value)
        if token < 0:
            return 0
        if self.vocab_size > 0 and token >= self.vocab_size:
            return 1
        return token

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.samples[idx]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        return torch.tensor(ids[: self.seq_len], dtype=torch.long)


@contextlib.contextmanager
def _default_torch_dtype(dtype: torch.dtype):
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original)


def _dtype_from_name(name: str) -> torch.dtype:
    key = str(name or "fp32").lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.float32


def _is_probe_name(name: str) -> bool:
    return str(name or "").strip().lower().replace("-", "_") in PROBE_PRESET_NAMES


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _rank_device_index(local_rank: int, device_count: int, raw_map: str | None) -> int:
    mapping = [part.strip() for part in str(raw_map or "").split(",") if part.strip()]
    if mapping:
        if local_rank >= len(mapping):
            raise ValueError(f"LOCAL_RANK={local_rank} is outside --rank_device_map={raw_map!r}")
        device_index = int(mapping[local_rank])
    else:
        device_index = int(local_rank)
    if device_count <= 0:
        raise RuntimeError("FSDP requested but no CUDA devices are visible")
    if device_index < 0 or device_index >= device_count:
        raise ValueError(f"rank device index {device_index} is outside visible CUDA device count {device_count}")
    return device_index


def _distributed_context(args: argparse.Namespace) -> dict[str, Any]:
    world_size = _env_int("WORLD_SIZE", 1)
    requested = str(args.distributed or "none").lower()
    if requested == "auto":
        requested = "fsdp" if world_size > 1 else "none"
    enabled = requested == "fsdp" or world_size > 1
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if enabled:
        if FSDP is None:
            raise RuntimeError("FSDP requested but torch.distributed.fsdp is unavailable in this environment")
        if not dist.is_available():
            raise RuntimeError("FSDP requested but torch.distributed is unavailable")
        if not dist.is_initialized():
            dist.init_process_group(backend=str(args.dist_backend))
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = _env_int("LOCAL_RANK", rank)
        if torch.cuda.is_available():
            device_index = _rank_device_index(local_rank, torch.cuda.device_count(), str(args.rank_device_map or os.getenv("OMNICODER2026_RANK_DEVICE_MAP", "")))
            torch.cuda.set_device(device_index)
            device = torch.device("cuda", device_index)
        else:
            device = torch.device("cpu")
        return {
            "enabled": True,
            "mode": "fsdp",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device": device,
            "rank_device_map": str(args.rank_device_map or os.getenv("OMNICODER2026_RANK_DEVICE_MAP", "")),
            "is_main": rank == 0,
        }
    return {"enabled": False, "mode": "none", "rank": 0, "local_rank": 0, "world_size": 1, "device": torch.device(args.device), "is_main": True}


def _build_model(args: argparse.Namespace) -> tuple[OmniCoder2026, object]:
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
    if args.seq_len:
        preset.train_seq_len = int(args.seq_len)
    cfg = OmniCoder2026Config(**kwargs)
    init_dtype_name = str(args.init_dtype or "auto").lower()
    if init_dtype_name == "auto":
        init_dtype_name = str(args.precision or "fp32").lower()
    init_dtype = _dtype_from_name(init_dtype_name)
    init_layer_devices: list[torch.device] | None = None
    init_embed_device: torch.device | None = None
    init_head_device: torch.device | None = None
    if str(getattr(args, "placement", "single") or "single") == "weighted_layers" and torch.cuda.is_available():
        devices = _parse_cuda_devices(str(getattr(args, "placement_devices", "") or ""))
        counts = _parse_layer_counts(str(getattr(args, "placement_layer_counts", "") or ""), int(cfg.n_layers), len(devices), devices)
        head_index = int(getattr(args, "placement_head_device", -1))
        if head_index < 0:
            head_index = max(range(len(devices)), key=lambda i: torch.cuda.get_device_properties(i).total_memory if devices[i].type == "cuda" else 0)
        if head_index >= len(devices):
            raise ValueError(f"--placement_head_device={head_index} outside placement devices {devices}")
        init_layer_devices = []
        for device, count in zip(devices, counts, strict=True):
            init_layer_devices.extend([device] * int(count))
        init_embed_device = devices[head_index]
        init_head_device = devices[head_index]
    _validate_target_training_devices(args, preset)
    with _default_torch_dtype(init_dtype):
        model = OmniCoder2026(
            cfg,
            init_layer_devices=init_layer_devices,
            init_embed_device=init_embed_device,
            init_head_device=init_head_device,
            checkpoint_blocks=bool(getattr(args, "activation_checkpointing", False)),
        )
    return model, preset


def _parse_cuda_devices(raw: str) -> list[torch.device]:
    items = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not items:
        if torch.cuda.is_available():
            items = [str(index) for index in range(torch.cuda.device_count())]
        else:
            items = ["cpu"]
    devices: list[torch.device] = []
    for item in items:
        if item == "cpu" or item.startswith("cpu:"):
            devices.append(torch.device("cpu"))
        elif item.startswith("cuda"):
            devices.append(torch.device(item))
        else:
            devices.append(torch.device("cuda", int(item)))
    return devices


def _device_total_memory(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device} is outside visible CUDA device count {torch.cuda.device_count()}")
    return float(torch.cuda.get_device_properties(index).total_memory)


def _device_name(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device} is outside visible CUDA device count {torch.cuda.device_count()}")
    return str(torch.cuda.get_device_name(index))


def _validate_target_training_devices(args: argparse.Namespace, preset: object) -> dict[str, Any]:
    if not bool(args.require_target_contract) or str(getattr(preset, "name", "")) != TARGET_PRESET:
        return {"status": "skipped", "reason": "not_target_contract"}
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "cuda_unavailable"}
    if str(args.placement or "single") == "weighted_layers":
        devices = _parse_cuda_devices(str(args.placement_devices or ""))
        if len(devices) < 3:
            raise ValueError("target 20B weighted training requires at least three visible fast CUDA devices")
    else:
        device = torch.device(args.device)
        devices = [device] if device.type == "cuda" else []
    selected = [{"device": str(device), "name": _device_name(device)} for device in devices if device.type == "cuda"]
    p40 = [item for item in selected if "p40" in item["name"].lower()]
    if p40:
        names = ", ".join(f'{item["device"]}:{item["name"]}' for item in p40)
        raise ValueError(f"target synchronous training selected P40 device(s): {names}")
    return {"status": "passed", "selected_cuda_devices": selected}


def _placement_head_reserve_fraction() -> float:
    raw = os.getenv("OMNICODER2026_PLACEMENT_HEAD_RESERVE_FRACTION", "0.10")
    try:
        value = float(raw)
    except Exception:
        value = 0.10
    return min(0.50, max(0.0, value))


def _parse_layer_counts(raw: str, n_layers: int, n_devices: int, devices: list[torch.device]) -> list[int]:
    if raw.strip():
        counts = [int(item.strip()) for item in raw.split(",") if item.strip()]
        if len(counts) != n_devices:
            raise ValueError(f"--placement_layer_counts must provide {n_devices} counts, got {counts}")
        if sum(counts) != n_layers:
            raise ValueError(f"--placement_layer_counts sum must equal {n_layers}, got {counts}")
        if any(count < 0 for count in counts):
            raise ValueError(f"--placement_layer_counts cannot contain negatives: {counts}")
        return counts
    if n_devices == 1:
        return [n_layers]
    if n_layers < n_devices:
        raise ValueError(f"weighted placement needs at least one layer per device; got {n_layers} layers over {n_devices} devices")
    weights: list[float] = []
    for device in devices:
        weights.append(_device_total_memory(device) or 1.0)
    # Leave room for the tied 330k-token embedding/output head on the largest card,
    # but do not flatten the largest shard by default; the RTX 8000 is meant to
    # carry materially more layers than the 24GB cards.
    largest = max(range(n_devices), key=lambda i: weights[i])
    adjusted = list(weights)
    adjusted[largest] = max(1.0, adjusted[largest] * (1.0 - _placement_head_reserve_fraction()))
    total = sum(adjusted)
    counts = [max(1, int(n_layers * weight / total)) for weight in adjusted]
    while sum(counts) < n_layers:
        target = max(range(n_devices), key=lambda i: adjusted[i] / max(1, counts[i]))
        counts[target] += 1
    while sum(counts) > n_layers:
        target = max(range(n_devices), key=lambda i: counts[i])
        if counts[target] <= 1:
            break
        counts[target] -= 1
    smallest_weight = max(1.0, min(weights))
    if n_devices >= 3 and weights[largest] / smallest_weight >= 1.75:
        largest_floor = min(n_layers - (n_devices - 1), max(counts[largest], int(round(n_layers * 0.50))))
        if largest_floor > counts[largest]:
            remaining = n_layers - largest_floor
            other_indices = [index for index in range(n_devices) if index != largest]
            other_total = sum(weights[index] for index in other_indices) or float(len(other_indices))
            other_counts = {
                index: max(1, int(remaining * weights[index] / other_total))
                for index in other_indices
            }
            while sum(other_counts.values()) < remaining:
                target = max(other_indices, key=lambda i: weights[i] / max(1, other_counts[i]))
                other_counts[target] += 1
            while sum(other_counts.values()) > remaining:
                target = max(other_indices, key=lambda i: other_counts[i])
                if other_counts[target] <= 1:
                    break
                other_counts[target] -= 1
            for index in other_indices:
                counts[index] = other_counts[index]
            counts[largest] = largest_floor
    if sum(counts) != n_layers:
        raise ValueError(f"could not derive weighted layer counts for {n_layers} layers over {n_devices} devices")
    return counts


def _checkpoint_complete_marker(path: str | Path) -> Path:
    return Path(str(path) + ".complete.json")


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)
        marker = {
            "status": "complete",
            "path": str(final_path),
            "bytes": final_path.stat().st_size,
            "format": payload.get("format"),
            "global_step": payload.get("global_step"),
            "last_loss": payload.get("last_loss"),
        }
        _checkpoint_complete_marker(final_path).write_text(json.dumps(marker, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _validate_resume_checkpoint_metadata(checkpoint: dict[str, Any], preset: object, args: argparse.Namespace) -> None:
    saved_preset = checkpoint.get("preset") if isinstance(checkpoint.get("preset"), dict) else {}
    saved_name = str(saved_preset.get("name") or "")
    current_name = str(getattr(preset, "name", ""))
    if saved_name and current_name and saved_name != current_name:
        raise ValueError(f"resume checkpoint preset mismatch: checkpoint={saved_name!r} current={current_name!r}")
    train_args = checkpoint.get("train_args") if isinstance(checkpoint.get("train_args"), dict) else {}
    saved_contract = saved_name == TARGET_PRESET
    if bool(args.require_target_contract) and saved_name and not saved_contract and not bool(getattr(args, "allow_probe", False)):
        raise ValueError(f"target contract resume requires {TARGET_PRESET!r}, got checkpoint preset {saved_name!r}")
    saved_fake_quant = train_args.get("fake_quant")
    if saved_fake_quant is not None and bool(saved_fake_quant) != bool(args.fake_quant):
        raise ValueError("resume checkpoint fake_quant setting does not match current target run")
    for key in ("placement", "placement_devices", "placement_layer_counts", "placement_head_device"):
        saved = train_args.get(key)
        current = getattr(args, key, None)
        if saved is None or saved == "":
            continue
        if str(saved) != str(current):
            raise ValueError(f"resume checkpoint {key} mismatch: checkpoint={saved!r} current={current!r}")


def _apply_weighted_placement(model: OmniCoder2026, args: argparse.Namespace) -> dict[str, object]:
    devices = _parse_cuda_devices(str(args.placement_devices or ""))
    if not devices:
        raise ValueError("weighted placement needs at least one device")
    n_layers = len(model.blocks)
    counts = _parse_layer_counts(str(args.placement_layer_counts or ""), n_layers, len(devices), devices)
    head_index = int(args.placement_head_device)
    if head_index < 0:
        head_index = max(range(len(devices)), key=lambda i: torch.cuda.get_device_properties(i).total_memory if devices[i].type == "cuda" and torch.cuda.is_available() else 0)
    if head_index >= len(devices):
        raise ValueError(f"--placement_head_device={head_index} outside placement devices {devices}")
    layer_devices: list[torch.device] = []
    for device, count in zip(devices, counts, strict=True):
        layer_devices.extend([device] * int(count))
    summary = model.apply_weighted_device_map(
        layer_devices,
        embed_device=devices[head_index],
        head_device=devices[head_index],
        checkpoint_blocks=bool(args.activation_checkpointing),
    )
    summary["requested_counts"] = counts
    return summary


def _architecture_manifest(model: torch.nn.Module) -> dict[str, Any]:
    raw = model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod  # type: ignore[attr-defined]
    if FSDP is not None and isinstance(raw, FSDP):
        raw = raw.module
    if hasattr(raw, "architecture_manifest"):
        return raw.architecture_manifest()  # type: ignore[no-any-return, attr-defined]
    return {}


def _apply_activation_checkpointing(model: torch.nn.Module) -> None:
    if apply_activation_checkpointing is None or checkpoint_wrapper is None or CheckpointImpl is None:
        raise RuntimeError("Activation checkpointing requested but checkpoint_wrapper is unavailable")
    wrapper = lambda module: checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, OmniCoder2026Block),
    )


def _wrap_fsdp(model: OmniCoder2026, args: argparse.Namespace, ctx: dict[str, Any]) -> torch.nn.Module:
    if not ctx["enabled"]:
        return model
    if FSDP is None or MixedPrecision is None or CPUOffload is None or ShardingStrategy is None or transformer_auto_wrap_policy is None:
        raise RuntimeError("FSDP requested but required FSDP helpers are unavailable")
    import functools

    if bool(args.activation_checkpointing):
        _apply_activation_checkpointing(model)
    mp_dtype = _dtype_from_name(str(args.precision))
    mixed_precision = None
    if mp_dtype in (torch.float16, torch.bfloat16):
        mixed_precision = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={OmniCoder2026Block})
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=bool(args.cpu_offload)),
        device_id=ctx["device"] if ctx["device"].type == "cuda" else None,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def _build_optimizer_for_params(args: argparse.Namespace, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    optimizer_name = str(args.optimizer or "adamw").lower()
    if optimizer_name == "adafactor":
        try:
            from transformers.optimization import Adafactor

            return Adafactor(
                params,
                lr=float(args.lr),
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=0.0,
            )
        except Exception as exc:
            if bool(args.require_adafactor):
                raise RuntimeError("Adafactor optimizer requested but unavailable") from exc
    return torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.1)


def _build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    return _build_optimizer_for_params(args, model.parameters())


def _install_optimizer_in_backward(args: argparse.Namespace, model: torch.nn.Module) -> list[Any]:
    handles: list[Any] = []
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        return handles
    sample = params[0]
    if not hasattr(sample, "register_post_accumulate_grad_hook"):
        raise RuntimeError("optimizer-in-backward requires Tensor.register_post_accumulate_grad_hook")
    update_mode = str(getattr(args, "optimizer_in_backward_update", "lowmem_sgd") or "lowmem_sgd").lower()
    if update_mode in {"lowmem_adafactor", "chunked_adafactor"}:
        lr = float(args.lr)
        chunk_rows = max(1, int(getattr(args, "optimizer_in_backward_adafactor_chunk_rows", 256) or 256))
        clip_threshold = float(getattr(args, "optimizer_in_backward_adafactor_clip_threshold", 1.0) or 1.0)
        decay_rate = float(getattr(args, "optimizer_in_backward_adafactor_decay_rate", -0.8) or -0.8)
        eps1 = float(getattr(args, "optimizer_in_backward_adafactor_eps1", 1.0e-30) or 1.0e-30)
        fallback_clip = float(getattr(args, "optimizer_in_backward_grad_clip", 1.0) or 0.0)
        states: dict[int, dict[str, Any]] = {}

        def fallback_step(param: torch.nn.Parameter, grad: torch.Tensor) -> None:
            if grad.is_sparse:
                param.add_(grad, alpha=-lr)
                return
            if grad.is_floating_point():
                limit = fallback_clip if fallback_clip > 0 else 1.0
                torch.nan_to_num_(grad, nan=0.0, posinf=limit, neginf=-limit)
                if fallback_clip > 0:
                    norm = torch.linalg.vector_norm(grad.detach())
                    target = fallback_clip * math.sqrt(max(1, int(grad.numel())))
                    scale = torch.clamp(torch.as_tensor(target, device=grad.device, dtype=norm.dtype) / norm.clamp_min(1.0e-12), max=1.0)
                    grad.mul_(scale.to(dtype=grad.dtype))
            param.add_(grad, alpha=-lr)

        def step_and_clear(param: torch.nn.Parameter) -> None:
            grad = param.grad
            if grad is None:
                return
            with torch.no_grad():
                if grad.is_sparse or grad.ndim != 2:
                    fallback_step(param, grad)
                    param.grad = None
                    return
                rows, cols = int(grad.shape[0]), int(grad.shape[1])
                state = states.setdefault(
                    id(param),
                    {
                        "step": 0,
                        "row": torch.zeros(rows, device=param.device, dtype=torch.float32),
                        "col": torch.zeros(cols, device=param.device, dtype=torch.float32),
                    },
                )
                state["step"] = int(state["step"]) + 1
                beta2 = 1.0 - (float(state["step"]) ** decay_rate)
                one_minus_beta2 = 1.0 - beta2
                row_state = state["row"]
                col_state = state["col"]
                assert isinstance(row_state, torch.Tensor) and isinstance(col_state, torch.Tensor)
                torch.nan_to_num_(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                col_sum = torch.zeros_like(col_state)
                for start in range(0, rows, chunk_rows):
                    end = min(rows, start + chunk_rows)
                    g2 = grad[start:end].detach().to(torch.float32, copy=True)
                    g2.square_().add_(eps1)
                    row_state[start:end].mul_(beta2).add_(g2.mean(dim=1), alpha=one_minus_beta2)
                    col_sum.add_(g2.sum(dim=0))
                col_state.mul_(beta2).add_(col_sum.div_(max(1, rows)), alpha=one_minus_beta2)
                row_mean = row_state.mean().clamp_min(eps1)
                col_factor = col_state.clamp_min(eps1).rsqrt()
                update_sq = torch.zeros((), device=param.device, dtype=torch.float32)
                for start in range(0, rows, chunk_rows):
                    end = min(rows, start + chunk_rows)
                    update = grad[start:end].detach().to(torch.float32, copy=True)
                    row_factor = (row_state[start:end].clamp_min(eps1) / row_mean).rsqrt()
                    update.mul_(row_factor[:, None]).mul_(col_factor[None, :])
                    update_sq.add_(update.square_().sum())
                update_rms = torch.sqrt(update_sq / max(1, grad.numel()))
                denom = torch.clamp(update_rms / max(1.0e-12, clip_threshold), min=1.0)
                for start in range(0, rows, chunk_rows):
                    end = min(rows, start + chunk_rows)
                    update = grad[start:end].detach().to(torch.float32, copy=True)
                    row_factor = (row_state[start:end].clamp_min(eps1) / row_mean).rsqrt()
                    update.mul_(row_factor[:, None]).mul_(col_factor[None, :])
                    update.mul_(lr).div_(denom)
                    param.data[start:end].add_(update.to(dtype=param.dtype), alpha=-1.0)
                param.grad = None

        for parameter in params:
            handles.append(parameter.register_post_accumulate_grad_hook(step_and_clear))
        return handles

    if update_mode in {"lowmem_sgd", "sgd", "inplace_sgd"}:
        lr = float(args.lr)
        clip = float(getattr(args, "optimizer_in_backward_grad_clip", 1.0) or 0.0)
        clip_mode = str(getattr(args, "optimizer_in_backward_clip_mode", "rms") or "rms").lower()

        def step_and_clear(param: torch.nn.Parameter) -> None:
            grad = param.grad
            if grad is None:
                return
            with torch.no_grad():
                if grad.is_sparse:
                    param.add_(grad, alpha=-lr)
                    param.grad = None
                    return
                if grad.is_floating_point():
                    limit = clip if clip > 0 else 1.0
                    torch.nan_to_num_(grad, nan=0.0, posinf=limit, neginf=-limit)
                    if clip > 0 and clip_mode == "rms":
                        norm = torch.linalg.vector_norm(grad.detach())
                        target = float(clip) * math.sqrt(max(1, int(grad.numel())))
                        scale = torch.clamp(torch.as_tensor(target, device=grad.device, dtype=norm.dtype) / norm.clamp_min(1.0e-12), max=1.0)
                        grad.mul_(scale.to(dtype=grad.dtype))
                    elif clip > 0:
                        grad.clamp_(min=-clip, max=clip)
                param.add_(grad, alpha=-lr)
                param.grad = None

        for parameter in params:
            handles.append(parameter.register_post_accumulate_grad_hook(step_and_clear))
        return handles

    for parameter in params:
        optimizer = _build_optimizer_for_params(args, [parameter])

        def step_and_clear(param: torch.nn.Parameter, opt: torch.optim.Optimizer = optimizer) -> None:
            if param.grad is None:
                return
            opt.step()
            opt.zero_grad(set_to_none=True)

        handles.append(parameter.register_post_accumulate_grad_hook(step_and_clear))
    return handles


def _autocast_context(device: torch.device, precision: str):
    key = str(precision or "fp32").lower()
    if device.type == "cuda" and key in {"fp16", "bf16"}:
        return torch.autocast(device_type="cuda", dtype=_dtype_from_name(key))
    return contextlib.nullcontext()


def _uses_fp16_parameter_storage(args: argparse.Namespace) -> bool:
    init_dtype_name = str(args.init_dtype or "auto").lower()
    if init_dtype_name == "auto":
        init_dtype_name = str(args.precision or "fp32").lower()
    return init_dtype_name == "fp16"


def _clip_grad_norm(model: torch.nn.Module, max_norm: float) -> None:
    if hasattr(model, "clip_grad_norm_"):
        model.clip_grad_norm_(max_norm)  # type: ignore[attr-defined]
    else:
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return
        total_sq = 0.0
        for parameter in parameters:
            grad = parameter.grad.detach()
            total_sq += float(grad.float().norm(2).cpu().item() ** 2)
        total_norm = math.sqrt(total_sq)
        clip_coef = float(max_norm) / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for parameter in parameters:
                parameter.grad.detach().mul_(clip_coef)


def _sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    if state.get("python_random") is not None:
        try:
            random.setstate(state["python_random"])
        except Exception:
            pass
    torch_cpu = state.get("torch_cpu")
    if isinstance(torch_cpu, torch.Tensor):
        torch.set_rng_state(torch_cpu.cpu())
    # CUDA RNG formats vary across PyTorch builds and visible-device mappings.
    # Checkpoint resume should not fail just because the RNG payload cannot be restored.


def _write_log(log_file: str | None, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    print(line)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _save_checkpoint(
    path: str,
    raw_model: OmniCoder2026,
    opt: torch.optim.Optimizer | None,
    preset: object,
    args: argparse.Namespace,
    step: int,
    last_loss: float | None,
) -> None:
    payload = {
        "format": "omnicoder2026_native_train_checkpoint_v2",
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "rng_state": _rng_state(),
        "global_step": int(step),
        "preset": preset.__dict__,
        "architecture_manifest": raw_model.architecture_manifest(),
        "last_loss": last_loss,
        "data": {
            "path": args.data,
            "sha256": _sha256_file(args.data),
            "manifest": args.data_manifest,
        },
        "train_args": {
            "seq_len": int(args.seq_len or getattr(preset, "train_seq_len", 0)),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "max_records": int(args.max_records),
            "device": args.device,
            "placement": str(args.placement),
            "placement_devices": str(args.placement_devices or ""),
            "placement_layer_counts": str(args.placement_layer_counts or ""),
            "placement_head_device": int(args.placement_head_device),
            "placement_schedule": str(args.placement_schedule),
            "pipeline_microbatches": int(args.pipeline_microbatches),
            "pipeline_async_streams": bool(args.pipeline_async_streams),
            "optimizer_in_backward": bool(args.optimizer_in_backward),
            "optimizer_in_backward_update": str(args.optimizer_in_backward_update),
            "optimizer_in_backward_grad_clip": float(args.optimizer_in_backward_grad_clip),
            "optimizer_in_backward_clip_mode": str(args.optimizer_in_backward_clip_mode),
            "optimizer_in_backward_adafactor_chunk_rows": int(args.optimizer_in_backward_adafactor_chunk_rows),
            "optimizer_in_backward_adafactor_clip_threshold": float(args.optimizer_in_backward_adafactor_clip_threshold),
            "optimizer_in_backward_adafactor_decay_rate": float(args.optimizer_in_backward_adafactor_decay_rate),
            "optimizer_in_backward_adafactor_eps1": float(args.optimizer_in_backward_adafactor_eps1),
            "fake_quant": bool(args.fake_quant),
            "fake_quant_chunk_rows": int(args.fake_quant_chunk_rows or 0),
            "fake_quant_max_full_elements": int(args.fake_quant_max_full_elements or 0),
            "aux_probe": bool(args.aux_probe),
        },
        "provenance": {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    _atomic_torch_save(payload, path)


def _save_distributed_checkpoint(
    path: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    preset: object,
    args: argparse.Namespace,
    step: int,
    last_loss: float | None,
    ctx: dict[str, Any],
) -> None:
    if FSDP is None or StateDictType is None:
        raise RuntimeError("Distributed checkpoint requested but FSDP state-dict support is unavailable")
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    rank = int(ctx["rank"])
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        payload = {
            "format": "omnicoder2026_native_train_checkpoint_v3_fsdp_local",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "rng_state": _rng_state(),
            "global_step": int(step),
            "preset": preset.__dict__,
            "last_loss": last_loss,
            "rank": rank,
            "world_size": int(ctx["world_size"]),
            "train_args": {
                "seq_len": int(args.seq_len or getattr(preset, "train_seq_len", 0)),
                "batch_size": int(args.batch_size),
                "steps": int(args.steps),
                "lr": float(args.lr),
                "optimizer": str(args.optimizer),
                "precision": str(args.precision),
                "distributed": str(args.distributed),
                "rank_device_map": str(args.rank_device_map or ""),
                "fake_quant": bool(args.fake_quant),
                "fake_quant_chunk_rows": int(args.fake_quant_chunk_rows or 0),
                "fake_quant_max_full_elements": int(args.fake_quant_max_full_elements or 0),
                "aux_probe": bool(args.aux_probe),
            },
            "provenance": {
                "pid": os.getpid(),
                "cwd": os.getcwd(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(ctx["device"]) if torch.cuda.is_available() and ctx["device"].type == "cuda" else None,
            },
        }
        rank_path = target / f"rank{rank:05d}.pt"
        _atomic_torch_save(payload, rank_path)
    if ctx["is_main"]:
        manifest = {
            "format": "omnicoder2026_native_train_checkpoint_v3_fsdp_local",
            "checkpoint_dir": str(target),
            "rank_files": [f"rank{idx:05d}.pt" for idx in range(int(ctx["world_size"]))],
            "global_step": int(step),
            "last_loss": last_loss,
            "preset": preset.__dict__,
            "architecture_manifest": _architecture_manifest(model),
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        _checkpoint_complete_marker(target).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "path": str(target),
                    "format": manifest.get("format"),
                    "global_step": manifest.get("global_step"),
                    "last_loss": manifest.get("last_loss"),
                    "rank_files": manifest.get("rank_files"),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    dist.barrier()


def _load_distributed_checkpoint(
    path: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    ctx: dict[str, Any],
) -> tuple[int, float | None]:
    if FSDP is None or StateDictType is None:
        raise RuntimeError("Distributed checkpoint requested but FSDP state-dict support is unavailable")
    source = Path(path)
    rank_file = source / f"rank{int(ctx['rank']):05d}.pt"
    if not rank_file.exists():
        raise FileNotFoundError(f"missing rank-local checkpoint: {rank_file}")
    checkpoint = torch.load(rank_file, map_location=ctx["device"], weights_only=False)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint.get("optimizer_state_dict"):
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in opt.param_groups:
            group["lr"] = float(group.get("lr", 0.0) or 0.0) or group["lr"]
    _restore_rng_state(checkpoint.get("rng_state") or {})
    return int(checkpoint.get("global_step") or 0), checkpoint.get("last_loss")


def main() -> None:
    ap = argparse.ArgumentParser(description="Omnicoder 2026 dense native-1M pretraining entrypoint")
    ap.add_argument("--preset", default=os.getenv("OMNICODER2026_PRESET", TARGET_PRESET))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="weights/omnicoder2026_probe.pt")
    ap.add_argument("--seq_len", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_records", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fake_quant", action="store_true")
    ap.add_argument("--fake_quant_chunk_rows", type=int, default=int(os.getenv("OMNICODER2026_FAKE_QUANT_CHUNK_ROWS", "0") or 0))
    ap.add_argument("--fake_quant_max_full_elements", type=int, default=int(os.getenv("OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS", "0") or 0))
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--aux_probe", action="store_true", help="also exercise flow/grounding/sync heads")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save_interval", type=int, default=0)
    ap.add_argument("--log_file", default=None)
    ap.add_argument("--data_manifest", default=None)
    ap.add_argument("--distributed", default=os.getenv("OMNICODER2026_DISTRIBUTED", "none"), choices=["none", "auto", "fsdp"])
    ap.add_argument("--dist_backend", default=os.getenv("OMNICODER2026_DIST_BACKEND", "nccl"))
    ap.add_argument("--rank_device_map", default=os.getenv("OMNICODER2026_RANK_DEVICE_MAP", ""))
    ap.add_argument("--placement", default=os.getenv("OMNICODER2026_PLACEMENT", "single"), choices=["single", "weighted_layers"])
    ap.add_argument("--placement_devices", default=os.getenv("OMNICODER2026_PLACEMENT_DEVICES", ""))
    ap.add_argument("--placement_layer_counts", default=os.getenv("OMNICODER2026_PLACEMENT_LAYER_COUNTS", ""))
    ap.add_argument("--placement_head_device", type=int, default=int(os.getenv("OMNICODER2026_PLACEMENT_HEAD_DEVICE", "-1") or -1))
    ap.add_argument("--placement_schedule", default=os.getenv("OMNICODER2026_PLACEMENT_SCHEDULE", "sequential"), choices=["sequential", "microbatch_pipeline"])
    ap.add_argument("--pipeline_microbatches", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_MICROBATCHES", "1") or 1))
    ap.add_argument("--pipeline_async_streams", dest="pipeline_async_streams", action="store_true", default=os.getenv("OMNICODER2026_PIPELINE_ASYNC_STREAMS", "0") in {"1", "true", "True", "yes", "YES"})
    ap.add_argument("--no_pipeline_async_streams", dest="pipeline_async_streams", action="store_false")
    ap.add_argument("--precision", default=os.getenv("OMNICODER2026_PRECISION", "fp32"), choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--init_dtype", default=os.getenv("OMNICODER2026_INIT_DTYPE", "auto"), choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--optimizer", default=os.getenv("OMNICODER2026_OPTIMIZER", "adamw"), choices=["adamw", "adafactor"])
    ap.add_argument("--optimizer_in_backward", action="store_true")
    ap.add_argument("--optimizer_in_backward_update", default=os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_UPDATE", "lowmem_sgd"), choices=["lowmem_sgd", "sgd", "inplace_sgd", "optimizer", "lowmem_adafactor", "chunked_adafactor"])
    ap.add_argument("--optimizer_in_backward_grad_clip", type=float, default=float(os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_GRAD_CLIP", "1.0") or 1.0))
    ap.add_argument("--optimizer_in_backward_clip_mode", default=os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_CLIP_MODE", "rms"), choices=["rms", "clamp"])
    ap.add_argument("--optimizer_in_backward_adafactor_chunk_rows", type=int, default=int(os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_ADAFACTOR_CHUNK_ROWS", "256") or 256))
    ap.add_argument("--optimizer_in_backward_adafactor_clip_threshold", type=float, default=float(os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_ADAFACTOR_CLIP_THRESHOLD", "1.0") or 1.0))
    ap.add_argument("--optimizer_in_backward_adafactor_decay_rate", type=float, default=float(os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_ADAFACTOR_DECAY_RATE", "-0.8") or -0.8))
    ap.add_argument("--optimizer_in_backward_adafactor_eps1", type=float, default=float(os.getenv("OMNICODER2026_OPTIMIZER_IN_BACKWARD_ADAFACTOR_EPS1", "1e-30") or 1.0e-30))
    ap.add_argument("--require_adafactor", action="store_true")
    ap.add_argument("--activation_checkpointing", action="store_true")
    ap.add_argument("--cpu_offload", action="store_true")
    ap.add_argument("--require_target_contract", action="store_true")
    ap.add_argument("--allow_probe", action="store_true")
    args = ap.parse_args()

    if int(args.fake_quant_chunk_rows or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_CHUNK_ROWS"] = str(int(args.fake_quant_chunk_rows))
    if int(args.fake_quant_max_full_elements or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS"] = str(int(args.fake_quant_max_full_elements))

    ctx = _distributed_context(args)
    model, preset = _build_model(args)
    seq_len = int(args.seq_len or preset.train_seq_len)
    tokenizer = get_text_tokenizer(prefer_hf=True)
    data = TextJsonlDataset(args.data, tokenizer, seq_len=seq_len, max_records=args.max_records, vocab_size=int(getattr(preset, "vocab_size", 0) or 0))
    sampler = DistributedSampler(data, num_replicas=int(ctx["world_size"]), rank=int(ctx["rank"]), shuffle=True, drop_last=False) if ctx["enabled"] else None
    loader = DataLoader(data, batch_size=int(args.batch_size), shuffle=sampler is None, sampler=sampler, drop_last=False)
    weighted_placement = str(args.placement or "single") == "weighted_layers"
    if weighted_placement and ctx["enabled"]:
        raise ValueError("weighted_layers placement is a single-process model-parallel path; use --distributed none")
    placement_schedule = str(args.placement_schedule or "sequential")
    pipeline_microbatches = max(1, int(args.pipeline_microbatches or 1))
    if placement_schedule == "microbatch_pipeline" and not weighted_placement:
        raise ValueError("--placement_schedule microbatch_pipeline requires --placement weighted_layers")
    if placement_schedule == "microbatch_pipeline" and bool(args.compile):
        raise ValueError("--compile is not supported with microbatch_pipeline placement scheduling")
    placement_summary: dict[str, object] | None = None
    if weighted_placement:
        placement_summary = _apply_weighted_placement(model, args)
        head_device_name = str(placement_summary.get("head_device") or "cuda:0")
        device = torch.device(head_device_name)
    else:
        device = ctx["device"]
    if ctx["enabled"]:
        model = _wrap_fsdp(model, args, ctx)
    elif not weighted_placement:
        model.to(device)
    if bool(args.compile) and not ctx["enabled"]:
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer_in_backward = bool(args.optimizer_in_backward)
    opt = None if optimizer_in_backward else _build_optimizer(args, model)
    backward_optimizer_handles: list[Any] = []
    if optimizer_in_backward:
        backward_optimizer_handles = _install_optimizer_in_backward(args, model)
    scaler = None
    if device.type == "cuda" and str(args.precision).lower() == "fp16" and not _uses_fp16_parameter_storage(args) and not optimizer_in_backward:
        if ctx["enabled"] and ShardedGradScaler is not None:
            scaler = ShardedGradScaler()
        else:
            scaler = torch.amp.GradScaler("cuda")
    start_step = 0
    last_loss = None
    if args.resume:
        if ctx["enabled"] and Path(args.resume).is_dir():
            start_step, last_loss = _load_distributed_checkpoint(args.resume, model, opt, ctx)
            for group in opt.param_groups:
                group["lr"] = float(args.lr)
        else:
            checkpoint_map = "cpu" if weighted_placement else device
            checkpoint = torch.load(args.resume, map_location=checkpoint_map, weights_only=False)
            _validate_resume_checkpoint_metadata(checkpoint, preset, args)
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if opt is not None and checkpoint.get("optimizer_state_dict"):
                opt.load_state_dict(checkpoint["optimizer_state_dict"])
                for group in opt.param_groups:
                    group["lr"] = float(args.lr)
            _restore_rng_state(checkpoint.get("rng_state") or {})
            start_step = int(checkpoint.get("global_step") or 0)
            last_loss = checkpoint.get("last_loss")
        if ctx["is_main"]:
            _write_log(args.log_file, {"event": "resumed", "resume": args.resume, "global_step": start_step, "last_loss": last_loss})
    if ctx["is_main"] and placement_summary:
        _write_log(args.log_file, {"event": "weighted_placement", **placement_summary})

    it = iter(loader)
    for local_step in range(int(args.steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if not weighted_placement:
            batch = batch.to(device, non_blocking=True)
        with _autocast_context(device, str(args.precision)):
            if (
                weighted_placement
                and placement_schedule == "microbatch_pipeline"
                and pipeline_microbatches > 1
                and not bool(args.aux_probe)
            ):
                loss = model.forward_weighted_pipeline_loss(  # type: ignore[attr-defined]
                    batch,
                    batch,
                    microbatches=pipeline_microbatches,
                    async_streams=bool(args.pipeline_async_streams),
                )
            else:
                out = model(
                    batch,
                    labels=batch,
                    return_aux=bool(args.aux_probe),
                    return_logits=False,
                    return_hidden=False,
                )
                loss = out["loss"]
                if loss is None:
                    logits = out["logits"]
                    target = batch.to(logits.device, non_blocking=True) if batch.device != logits.device else batch
                    loss = F.cross_entropy(logits[:, :-1, :].transpose(1, 2), target[:, 1:])
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            _clip_grad_norm(model, 1.0)
            scaler.step(opt)
            scaler.update()
        elif optimizer_in_backward:
            loss.backward()
        else:
            loss.backward()
            _clip_grad_norm(model, 1.0)
            opt.step()
        last_loss = float(loss.detach().cpu())
        global_step = start_step + local_step + 1
        if ctx["is_main"]:
            _write_log(args.log_file, {"step": global_step, "local_step": local_step + 1, "loss": last_loss, "preset": preset.name, "seq_len": seq_len, "distributed": ctx["mode"], "world_size": ctx["world_size"], "placement": str(args.placement), "placement_schedule": placement_schedule, "pipeline_microbatches": pipeline_microbatches, "pipeline_async_streams": bool(args.pipeline_async_streams), "optimizer_in_backward": optimizer_in_backward, "optimizer_in_backward_update": str(args.optimizer_in_backward_update)})
        if int(args.save_interval) > 0 and global_step % int(args.save_interval) == 0:
            interval_path = str(Path(args.out).with_name(f"{Path(args.out).stem}.step{global_step}.pt"))
            if ctx["enabled"]:
                _save_distributed_checkpoint(interval_path, model, opt, preset, args, global_step, last_loss, ctx)
            else:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                _save_checkpoint(interval_path, raw_model, opt, preset, args, global_step, last_loss)

    if ctx["enabled"]:
        _save_distributed_checkpoint(args.out, model, opt, preset, args, start_step + int(args.steps), last_loss, ctx)
    else:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        _save_checkpoint(args.out, raw_model, opt, preset, args, start_step + int(args.steps), last_loss)
    if ctx["is_main"]:
        _write_log(args.log_file, {"status": "ok", "out": args.out, "last_loss": last_loss, "global_step": start_step + int(args.steps), "distributed": ctx["mode"], "world_size": ctx["world_size"]})
    for handle in backward_optimizer_handles:
        try:
            handle.remove()
        except Exception:
            pass
    if ctx["enabled"]:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

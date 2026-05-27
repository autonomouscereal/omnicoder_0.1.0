from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Config
from omnicoder.eval.fsdp_checkpoint_2026 import (
    checkpoint_model_kwargs as _checkpoint_model_kwargs,
    distributed_context,
    is_fsdp_rank_local_checkpoint_dir,
    load_fsdp_rank_local_model,
    torchrun_hint,
)
from omnicoder.training.pretrain_2026_dense import _ids_from_record, _text_from_record
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _read_jsonl(path: Path, max_records: int) -> Iterable[dict[str, Any]]:
    seen = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if max_records > 0 and seen >= max_records:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                obj = {"text": line.rstrip("\n")}
            if isinstance(obj, dict):
                seen += 1
                yield obj


AGGREGATE_JSONL_NAMES = {"curated_records.jsonl", "train_all_modalities.jsonl"}


def _candidate_data_files(
    data_paths: list[str],
    data_dir: str | None,
    *,
    exclude_aggregate_jsonl: bool = False,
) -> list[Path]:
    files: list[Path] = []
    if data_dir:
        root = Path(data_dir)
        files.extend(
            sorted(
                path
                for path in root.rglob("*.jsonl")
                if path.is_file() and not (exclude_aggregate_jsonl and path.name in AGGREGATE_JSONL_NAMES)
            )
        )
    for item in data_paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(src for src in path.rglob("*.jsonl") if src.is_file()))
        else:
            files.append(path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _record_modality(record: dict[str, Any], fallback: str) -> str:
    modality = record.get("modality")
    if isinstance(modality, str) and modality:
        return modality
    modalities = record.get("modalities")
    if isinstance(modalities, list):
        for item in modalities:
            if isinstance(item, str) and item and item != "text":
                return item
        for item in modalities:
            if isinstance(item, str) and item:
                return item
    return fallback


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    raw = model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod  # type: ignore[attr-defined]
    if hasattr(raw, "module"):
        raw = raw.module  # type: ignore[attr-defined]
    return raw


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


def _autocast_context(device: torch.device, precision: str):
    key = str(precision or "fp32").lower()
    if device.type == "cuda" and key in {"fp16", "bf16"}:
        return torch.autocast(device_type="cuda", dtype=_dtype_from_name(key))
    return contextlib.nullcontext()


def _model_vocab_size(model: torch.nn.Module) -> int:
    return int(getattr(_unwrap_model(model), "vocab_size"))


def _model_max_seq_len(model: torch.nn.Module) -> int:
    return int(getattr(_unwrap_model(model), "max_seq_len"))


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def _model_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    raw = _unwrap_model(model)
    embed = getattr(raw, "embed", None)
    if isinstance(embed, torch.nn.Module):
        return _module_device(embed)
    return fallback


def _parse_devices(raw: str) -> list[torch.device]:
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
            raise ValueError(f"--placement-layer-counts must provide {n_devices} counts, got {counts}")
        if sum(counts) != n_layers:
            raise ValueError(f"--placement-layer-counts sum must equal {n_layers}, got {counts}")
        if any(count < 0 for count in counts):
            raise ValueError(f"--placement-layer-counts cannot contain negatives: {counts}")
        return counts
    if n_devices == 1:
        return [n_layers]
    if n_layers < n_devices:
        raise ValueError(f"weighted placement needs at least one layer per device; got {n_layers} layers over {n_devices} devices")
    weights = [_device_total_memory(device) or 1.0 for device in devices]
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


def _weighted_placement(
    cfg: OmniCoder2026Config,
    *,
    placement_devices: str,
    placement_layer_counts: str,
    placement_head_device: int,
) -> tuple[list[torch.device], torch.device, list[int]]:
    devices = _parse_devices(placement_devices)
    if not devices:
        raise ValueError("weighted placement needs at least one device")
    counts = _parse_layer_counts(str(placement_layer_counts or ""), int(cfg.n_layers), len(devices), devices)
    head_index = int(placement_head_device)
    if head_index < 0:
        head_index = max(range(len(devices)), key=lambda i: _device_total_memory(devices[i]))
    if head_index >= len(devices):
        raise ValueError(f"--placement-head-device={head_index} outside placement devices {devices}")
    layer_devices: list[torch.device] = []
    for device, count in zip(devices, counts, strict=True):
        layer_devices.extend([device] * int(count))
    return layer_devices, devices[head_index], counts


def _set_fake_quant_env(fake_quant_chunk_rows: int, fake_quant_max_full_elements: int) -> None:
    if int(fake_quant_chunk_rows or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_CHUNK_ROWS"] = str(int(fake_quant_chunk_rows))
    if int(fake_quant_max_full_elements or 0) > 0:
        os.environ["OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS"] = str(int(fake_quant_max_full_elements))


def _load_checkpoint_file(checkpoint_path: str | Path, map_location: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": map_location, "weights_only": False}
    try:
        checkpoint = torch.load(checkpoint_path, mmap=Path(checkpoint_path).is_file(), **kwargs)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, **kwargs)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{checkpoint_path} is not an Omnicoder2026 native training checkpoint")
    return checkpoint


def load_native_checkpoint(
    checkpoint_path: str | Path,
    profile: str,
    device: torch.device,
    *,
    dist_backend: str = "nccl",
    precision: str = "fp32",
    init_dtype: str = "auto",
    cpu_offload: bool = False,
    placement: str = "single",
    placement_devices: str = "",
    placement_layer_counts: str = "",
    placement_head_device: int = -1,
    activation_checkpointing: bool = False,
    fake_quant_chunk_rows: int = 0,
    fake_quant_max_full_elements: int = 0,
) -> torch.nn.Module:
    weighted_placement = str(placement or "single") == "weighted_layers"
    _set_fake_quant_env(fake_quant_chunk_rows, fake_quant_max_full_elements)
    if Path(checkpoint_path).is_dir():
        if weighted_placement:
            raise ValueError("weighted_layers placement expects a native .pt checkpoint, not a rank-local FSDP directory")
        if not is_fsdp_rank_local_checkpoint_dir(checkpoint_path):
            raise ValueError(f"{checkpoint_path} is a directory, but not an Omnicoder2026 rank-local FSDP checkpoint")
        try:
            model, _, _ = load_fsdp_rank_local_model(
                checkpoint_path,
                profile,
                device,
                dist_backend=dist_backend,
                precision=precision,
                cpu_offload=cpu_offload,
            )
            return model
        except RuntimeError as exc:
            hint = torchrun_hint(checkpoint_path, "omnicoder.eval.sample_loss_2026")
            raise RuntimeError(f"{exc}. Run sample loss with torchrun, for example: {hint}") from exc

    checkpoint = _load_checkpoint_file(checkpoint_path, map_location="cpu")
    cfg = OmniCoder2026Config(**_checkpoint_model_kwargs(checkpoint, profile))
    init_dtype_name = str(init_dtype or "auto").lower()
    if init_dtype_name == "auto":
        init_dtype_name = str(precision or "fp32").lower()
    init_kwargs: dict[str, Any] = {}
    placement_summary: dict[str, object] | None = None
    if weighted_placement:
        layer_devices, head_device, requested_counts = _weighted_placement(
            cfg,
            placement_devices=placement_devices,
            placement_layer_counts=placement_layer_counts,
            placement_head_device=int(placement_head_device),
        )
        init_kwargs.update(
            {
                "init_layer_devices": layer_devices,
                "init_embed_device": head_device,
                "init_head_device": head_device,
                "checkpoint_blocks": bool(activation_checkpointing),
            }
        )
    with _default_torch_dtype(_dtype_from_name(init_dtype_name)):
        model = OmniCoder2026(cfg, **init_kwargs)
    if weighted_placement:
        placement_summary = model.apply_weighted_device_map(
            layer_devices,
            embed_device=head_device,
            head_device=head_device,
            checkpoint_blocks=bool(activation_checkpointing),
        )
        placement_summary["requested_counts"] = requested_counts
    state_dict = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    for name, value in list(state_dict.items()):
        target = model_state.get(name)
        if target is None or not hasattr(value, "shape"):
            continue
        if tuple(value.shape) != tuple(target.shape) and value.numel() == target.numel():
            state_dict[name] = value.reshape_as(target)
    model.load_state_dict(state_dict, strict=True)
    if not weighted_placement:
        model.to(device)
    if placement_summary is not None:
        setattr(model, "_eval_placement_summary", placement_summary)
    model.eval()
    return model


def _sanitize_ids(ids: list[int], vocab_size: int) -> list[int]:
    cleaned: list[int] = []
    for item in ids:
        token = int(item)
        cleaned.append(token if 0 <= token < vocab_size else 1)
    return cleaned


def _record_ids(record: dict[str, Any], tokenizer: Any, vocab_size: int) -> list[int]:
    ids = _ids_from_record(record)
    if ids is None:
        ids = [int(x) for x in tokenizer.encode(_text_from_record(record))]
    return _sanitize_ids(ids, vocab_size)


def _chunks(ids: list[int], seq_len: int) -> Iterable[list[int]]:
    width = max(2, int(seq_len))
    for start in range(0, len(ids), width):
        chunk = ids[start:start + width]
        if len(chunk) >= 2:
            yield chunk


def _new_bucket(path: str | None = None) -> dict[str, Any]:
    bucket: dict[str, Any] = {
        "records": 0,
        "samples": 0,
        "tokens": 0,
        "loss_sum": 0.0,
        "avg_loss": None,
        "loss": None,
    }
    if path is not None:
        bucket["path"] = path
        bucket["modalities"] = {}
    return bucket


def _add_loss(bucket: dict[str, Any], loss_sum: float, tokens: int) -> None:
    bucket["samples"] += 1
    bucket["tokens"] += int(tokens)
    bucket["loss_sum"] += float(loss_sum)


def _bucket_avg_loss(bucket: dict[str, Any] | None) -> float | None:
    if not isinstance(bucket, dict):
        return None
    for key in ("avg_loss", "loss"):
        value = bucket.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    tokens = int(bucket.get("tokens") or 0)
    if tokens > 0 and bucket.get("loss_sum") not in (None, ""):
        return float(bucket.get("loss_sum") or 0.0) / tokens
    return None


def _bucket_loss_sum(bucket: dict[str, Any]) -> float:
    value = bucket.get("loss_sum")
    if value not in (None, ""):
        return float(value)
    avg_loss = _bucket_avg_loss(bucket)
    tokens = int(bucket.get("tokens") or 0)
    return float(avg_loss) * tokens if avg_loss is not None and tokens > 0 else 0.0


def _finalize(bucket: dict[str, Any]) -> None:
    tokens = int(bucket.get("tokens") or 0)
    avg_loss = (float(bucket["loss_sum"]) / tokens) if tokens else None
    bucket["avg_loss"] = avg_loss
    bucket["loss"] = avg_loss
    bucket["perplexity"] = math.exp(min(50.0, float(avg_loss))) if avg_loss is not None else None
    if "modalities" in bucket:
        for child in bucket["modalities"].values():
            _finalize(child)


def _merge_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["records"] += int(source.get("records") or 0)
    target["samples"] += int(source.get("samples") or 0)
    target["tokens"] += int(source.get("tokens") or 0)
    target["loss_sum"] += _bucket_loss_sum(source)


def _merge_eval_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged_files: dict[str, dict[str, Any]] = {}
    merged_modalities: dict[str, dict[str, Any]] = {}
    overall = _new_bucket()
    for result in results:
        _merge_bucket(overall, result.get("overall") or {})
        for name, bucket in (result.get("modalities") or {}).items():
            _merge_bucket(merged_modalities.setdefault(str(name), _new_bucket()), bucket)
        for file_bucket in result.get("files") or []:
            path = str(file_bucket.get("path") or "")
            target_file = merged_files.setdefault(path, _new_bucket(path))
            _merge_bucket(target_file, file_bucket)
            for name, bucket in (file_bucket.get("modalities") or {}).items():
                child = target_file["modalities"].setdefault(str(name), _new_bucket())
                _merge_bucket(child, bucket)
    _finalize(overall)
    for bucket in merged_modalities.values():
        _finalize(bucket)
    for bucket in merged_files.values():
        _finalize(bucket)
    return {
        "files": [merged_files[key] for key in sorted(merged_files)],
        "modalities": {key: merged_modalities[key] for key in sorted(merged_modalities)},
        "overall": overall,
    }


def _gather_eval_results(result: dict[str, Any], rank: int, world_size: int) -> dict[str, Any] | None:
    if world_size <= 1:
        return result
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed sample-loss merge requires an initialized process group")
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result)
    if rank != 0:
        return None
    return _merge_eval_results([item for item in gathered if isinstance(item, dict)])


def _loss_delta(current: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, Any]:
    current_loss = _bucket_avg_loss(current)
    baseline_loss = _bucket_avg_loss(baseline)
    delta = None
    if current_loss is not None and baseline_loss is not None:
        delta = float(current_loss) - float(baseline_loss)
    return {
        "current_avg_loss": current_loss,
        "baseline_avg_loss": baseline_loss,
        "delta_avg_loss": delta,
    }


def compare_baseline(result: dict[str, Any], baseline_path: str | Path) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"{baseline_path} must contain a sample_loss_2026 JSON object")

    current_modalities = result.get("modalities") if isinstance(result.get("modalities"), dict) else {}
    baseline_modalities = baseline.get("modalities") if isinstance(baseline.get("modalities"), dict) else {}
    modality_names = sorted(set(current_modalities) | set(baseline_modalities))
    return {
        "baseline_path": str(baseline_path),
        "overall": _loss_delta(result.get("overall"), baseline.get("overall")),
        "modalities": {
            name: _loss_delta(current_modalities.get(name), baseline_modalities.get(name))
            for name in modality_names
        },
    }


@torch.no_grad()
def evaluate_files(
    model: torch.nn.Module,
    files: list[Path],
    *,
    seq_len: int,
    max_records_per_file: int,
    device: torch.device,
    precision: str = "fp32",
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    tokenizer = get_text_tokenizer(prefer_hf=True)
    overall = _new_bucket()
    by_modality: dict[str, dict[str, Any]] = {}
    file_results: list[dict[str, Any]] = []
    vocab_size = _model_vocab_size(model)

    for path in files:
        file_bucket = _new_bucket(str(path))
        for record_index, record in enumerate(_read_jsonl(path, max_records_per_file)):
            if world_size > 1 and (record_index % world_size) != rank:
                continue
            modality = _record_modality(record, path.stem)
            ids = _record_ids(record, tokenizer, vocab_size)
            if len(ids) < 2:
                continue
            file_bucket["records"] += 1
            overall["records"] += 1
            modality_bucket = by_modality.setdefault(modality, _new_bucket())
            modality_bucket["records"] += 1
            file_modality = file_bucket["modalities"].setdefault(modality, _new_bucket())
            file_modality["records"] += 1

            for chunk in _chunks(ids, seq_len):
                batch = torch.tensor([chunk], dtype=torch.long, device=device)
                with _autocast_context(device, precision):
                    logits = model(batch, return_hidden=False)["logits"]
                labels = batch if batch.device == logits.device else batch.to(logits.device, non_blocking=True)
                loss_logits = logits[:, :-1, :].float()
                loss = F.cross_entropy(
                    loss_logits.transpose(1, 2),
                    labels[:, 1:],
                    reduction="sum",
                )
                token_count = max(0, len(chunk) - 1)
                loss_value = float(loss.detach().cpu())
                _add_loss(file_bucket, loss_value, token_count)
                _add_loss(file_modality, loss_value, token_count)
                _add_loss(modality_bucket, loss_value, token_count)
                _add_loss(overall, loss_value, token_count)
        _finalize(file_bucket)
        file_results.append(file_bucket)

    _finalize(overall)
    for bucket in by_modality.values():
        _finalize(bucket)
    return {"files": file_results, "modalities": by_modality, "overall": overall}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate sample loss for Omnicoder2026 native checkpoints over curated JSONL files")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--exclude-aggregate-jsonl",
        action="store_true",
        help="With --data-dir, skip aggregate JSONL files curated_records.jsonl and train_all_modalities.jsonl",
    )
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--profile", default="ledger_probe")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dist-backend", default="nccl", help="Process-group backend for rank-local FSDP checkpoints")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Eval autocast precision; also used for FSDP mixed precision")
    ap.add_argument("--init-dtype", "--init_dtype", dest="init_dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    ap.add_argument("--placement", choices=["single", "weighted_layers"], default=os.getenv("OMNICODER2026_PLACEMENT", "single"))
    ap.add_argument("--placement-devices", "--placement_devices", dest="placement_devices", default=os.getenv("OMNICODER2026_PLACEMENT_DEVICES", ""))
    ap.add_argument("--placement-layer-counts", "--placement_layer_counts", dest="placement_layer_counts", default=os.getenv("OMNICODER2026_PLACEMENT_LAYER_COUNTS", ""))
    ap.add_argument("--placement-head-device", "--placement_head_device", dest="placement_head_device", type=int, default=int(os.getenv("OMNICODER2026_PLACEMENT_HEAD_DEVICE", "-1") or -1))
    ap.add_argument("--activation-checkpointing", "--activation_checkpointing", dest="activation_checkpointing", action="store_true")
    ap.add_argument("--fake-quant-chunk-rows", "--fake_quant_chunk_rows", dest="fake_quant_chunk_rows", type=int, default=int(os.getenv("OMNICODER2026_FAKE_QUANT_CHUNK_ROWS", "0") or 0))
    ap.add_argument("--fake-quant-max-full-elements", "--fake_quant_max_full_elements", dest="fake_quant_max_full_elements", type=int, default=int(os.getenv("OMNICODER2026_FAKE_QUANT_MAX_FULL_ELEMENTS", "0") or 0))
    ap.add_argument("--cpu-offload", action="store_true", help="Enable FSDP CPU offload when loading rank-local checkpoints")
    ap.add_argument("--seq-len", type=int, default=0)
    ap.add_argument("--max-records-per-file", type=int, default=0)
    ap.add_argument("--compare-baseline", default=None, help="Optional prior sample_loss_2026 JSON result to diff against")
    ap.add_argument("--out", required=True)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if str(args.placement or "single") == "weighted_layers" and int(os.environ.get("WORLD_SIZE", "1") or 1) > 1:
        raise ValueError("weighted_layers evaluation is a single-process model-parallel path; run without torchrun/FSDP")
    files = _candidate_data_files(args.data, args.data_dir, exclude_aggregate_jsonl=bool(args.exclude_aggregate_jsonl))
    if not files:
        raise SystemExit("no JSONL files found; pass --data-dir or repeated --data")
    device = torch.device(args.device)
    model = load_native_checkpoint(
        args.checkpoint,
        args.profile,
        device,
        dist_backend=str(args.dist_backend),
        precision=str(args.precision),
        init_dtype=str(args.init_dtype),
        cpu_offload=bool(args.cpu_offload),
        placement=str(args.placement),
        placement_devices=str(args.placement_devices or ""),
        placement_layer_counts=str(args.placement_layer_counts or ""),
        placement_head_device=int(args.placement_head_device),
        activation_checkpointing=bool(args.activation_checkpointing),
        fake_quant_chunk_rows=int(args.fake_quant_chunk_rows or 0),
        fake_quant_max_full_elements=int(args.fake_quant_max_full_elements or 0),
    )
    ctx = distributed_context(device, str(args.dist_backend))
    seq_len = int(args.seq_len or min(1024, _model_max_seq_len(model)))
    eval_device = _model_input_device(model, ctx["device"])
    local_result = evaluate_files(
        model,
        files,
        seq_len=seq_len,
        max_records_per_file=int(args.max_records_per_file),
        device=eval_device,
        precision=str(args.precision),
        rank=int(ctx["rank"]),
        world_size=int(ctx["world_size"]),
    )
    result = _gather_eval_results(local_result, int(ctx["rank"]), int(ctx["world_size"]))
    if result is None:
        return
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "profile": str(args.profile),
            "device": str(eval_device),
            "placement": getattr(model, "_eval_placement_summary", {"mode": "single"}),
            "distributed": {
                "world_size": int(ctx["world_size"]),
                "rank_local_fsdp": Path(args.checkpoint).is_dir() and is_fsdp_rank_local_checkpoint_dir(args.checkpoint),
            },
            "seq_len": int(seq_len),
            "max_records_per_file": int(args.max_records_per_file),
        }
    )
    if args.compare_baseline:
        result["comparison"] = compare_baseline(result, args.compare_baseline)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()

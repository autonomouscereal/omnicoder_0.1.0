from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


SCHEMA_HELDOUT = "omnicoder.decode_diagnostics_2026.heldout_sample_loss.v1"
SCHEMA_DECODE = "omnicoder.decode_diagnostics_2026.decode_sanity.v1"
SCHEMA_OVERFIT = "omnicoder.decode_diagnostics_2026.tiny_text_overfit.v1"
DEFAULT_REQUIRED_MODALITIES = ("text", "code", "tool", "math", "media")
DEFAULT_TINY_TEXT_SAMPLES = (
    "Omnicoder tiny overfit check: readable text generation is working and the decoder writes normal words.",
    "Diagnostic answer: the model should emit words, a number like 42, and a short useful sentence.",
    "Before long training, this tiny sample proves the language head can learn a simple text pattern.",
)
DEFAULT_DECODE_PROMPTS = (
    {"id": "plain_text", "modality": "text", "prompt": "Write one clear sentence about decode diagnostics."},
    {"id": "code", "modality": "code", "prompt": "Write a Python function name and one short comment for adding two numbers."},
    {"id": "math", "modality": "math", "prompt": "Answer in words: what is two plus three?"},
    {
        "id": "tool_call",
        "modality": "tool",
        "prompt": "Return one compact JSON tool call with a tool name and arguments for checking disk usage.",
    },
    {
        "id": "image_route",
        "modality": "image",
        "prompt": "Return a compact image generation route with an artifact token marker and output artifact field.",
    },
    {
        "id": "video_route",
        "modality": "video",
        "prompt": "Return a compact video generation route with an artifact token marker and output artifact field.",
    },
    {
        "id": "music_route",
        "modality": "music",
        "prompt": "Return a compact music generation route with an audio artifact token marker and output artifact field.",
    },
    {
        "id": "tts_route",
        "modality": "tts",
        "prompt": "Return a compact TTS generation route with a speech artifact token marker and output artifact field.",
    },
    {
        "id": "ocr_route",
        "modality": "ocr",
        "prompt": "Return a compact OCR route that reads an image artifact and outputs extracted text.",
    },
)
REFUSAL_DECODE_RE = re.compile(
    r"\b(?:as an ai(?: language)? model|cannot assist|can't assist|unable to assist|violates? (?:the )?policy|must refuse)\b",
    re.IGNORECASE,
)
TOOL_DECODE_RE = re.compile(r"(\{[^{}]{2,}\}|\"(?:tool|name|arguments)\"|\b(?:tool|function|arguments?|parameters?)\b)", re.IGNORECASE)
MEDIA_ROUTE_RE = re.compile(
    r"(<(?:image|video|audio|music|tts|speech|media)[^>]{0,80}>|\b(?:artifact|artifact_path|output_path|media_token|route|decoder?|output_route)\b)",
    re.IGNORECASE,
)
OCR_ROUTE_RE = re.compile(r"\b(?:ocr|extracted text|recognized text|read text|transcription|image artifact)\b", re.IGNORECASE)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _read_jsonl(path: str | Path, max_records: int = 0) -> Iterable[dict[str, Any]]:
    seen = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if max_records > 0 and seen >= max_records:
                break
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = {"text": text}
            if isinstance(obj, dict):
                seen += 1
                yield obj


def candidate_jsonl_files(data: list[str] | None = None, data_dir: str | None = None) -> list[Path]:
    files: list[Path] = []
    if data_dir:
        root = Path(data_dir)
        files.extend(sorted(path for path in root.rglob("*.jsonl") if path.is_file()))
    for item in data or []:
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


def _stringify_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or item.get("speaker")
                content = item.get("content")
                if content is None:
                    content = item.get("value") or item.get("text")
                rendered = _stringify_text_value(content)
                if rendered:
                    parts.append(f"{role}: {rendered}" if role else rendered)
            else:
                rendered = _stringify_text_value(item)
                if rendered:
                    parts.append(rendered)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        nested = record_text(value)
        if nested:
            return nested
        try:
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def record_text(record: dict[str, Any]) -> str:
    message_text = _stringify_text_value(record.get("messages") or record.get("conversations"))
    if message_text:
        return message_text

    paired_parts: list[str] = []
    for key in ("prompt", "instruction", "question", "input", "completion", "response", "answer", "output"):
        rendered = _stringify_text_value(record.get(key))
        if rendered:
            paired_parts.append(rendered)
    if paired_parts:
        return "\n".join(paired_parts).strip()

    for key in (
        "text",
        "content",
        "caption",
        "description",
        "transcript",
        "code",
        "patch",
        "trace",
        "tool_call",
        "tool_calls",
        "function_call",
        "input_json",
        "target_json",
        "task",
    ):
        rendered = _stringify_text_value(record.get(key))
        if rendered:
            return rendered
    return ""


def normalize_modality(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if any(token in raw for token in ("image", "video", "audio", "music", "media", "tts", "asr", "speech", "ltx", "qwen_image", "comfy")):
        return "media"
    if any(token in raw for token in ("tool", "function", "agent", "action", "api_call", "json_tool")):
        return "tool"
    if any(token in raw for token in ("math", "algebra", "geometry", "calculus", "proof", "arithmetic")):
        return "math"
    if any(token in raw for token in ("code", "python", "javascript", "typescript", "repo", "patch", "program", "software")):
        return "code"
    if any(token in raw for token in ("text", "chat", "conversation", "plain", "language")):
        return "text"
    return None


def infer_modality(record: dict[str, Any], fallback: str = "text") -> str:
    for key in ("modality", "modalities", "domain", "task_domain", "category", "kind", "teacher_name", "source", "media_type"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                normalized = normalize_modality(item)
                if normalized:
                    return normalized
        else:
            normalized = normalize_modality(value)
            if normalized:
                return normalized

    if any(key in record for key in ("tool_call", "tool_calls", "function_call", "actions")):
        return "tool"
    if any(key in record for key in ("image_path", "video_path", "audio_path", "media_path", "artifact_path", "generated_artifact")):
        return "media"
    text = record_text(record).lower()
    normalized = normalize_modality(fallback)
    if normalized and normalized != "text":
        return normalized
    if re.search(r"\b(def|class|import|return|pytest|function|patch|diff --git)\b", text):
        return "code"
    if re.search(r"\b(equation|solve|proof|theorem|algebra|derivative|integral)\b|[0-9]\s*[+\-*/=]\s*[0-9]", text):
        return "math"
    return normalized or "text"


def _new_bucket() -> dict[str, Any]:
    return {
        "records_seen": 0,
        "records_evaluated": 0,
        "records_skipped": 0,
        "tokens": 0,
        "loss_sum": 0.0,
        "loss": None,
        "perplexity": None,
    }


def _add_loss(bucket: dict[str, Any], loss_sum: float, tokens: int) -> None:
    bucket["records_evaluated"] += 1
    bucket["tokens"] += int(tokens)
    bucket["loss_sum"] += float(loss_sum)


def _skip(bucket: dict[str, Any]) -> None:
    bucket["records_skipped"] += 1


def _safe_perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    if not math.isfinite(float(loss)):
        return None
    if float(loss) > 80.0:
        return float("inf")
    return float(math.exp(float(loss)))


def _finalize_bucket(bucket: dict[str, Any]) -> None:
    tokens = int(bucket.get("tokens") or 0)
    if tokens > 0:
        loss = float(bucket.get("loss_sum") or 0.0) / tokens
        bucket["loss"] = loss
        bucket["perplexity"] = _safe_perplexity(loss)
    else:
        bucket["loss"] = None
        bucket["perplexity"] = None


def _dtype_from_name(name: str) -> torch.dtype | str:
    key = str(name or "auto").lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    if key == "fp32":
        return torch.float32
    return "auto"


def _autocast_context(device: torch.device, precision: str):
    key = str(precision or "fp32").lower()
    if device.type == "cuda" and key in {"fp16", "bf16"}:
        dtype = torch.float16 if key == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _extract_loss(output: Any) -> torch.Tensor | None:
    if hasattr(output, "loss"):
        return output.loss
    if isinstance(output, dict):
        loss = output.get("loss")
        return loss if isinstance(loss, torch.Tensor) else None
    return None


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError("model output did not include logits")


def _model_device(model: torch.nn.Module, fallback: str | torch.device = "cpu") -> torch.device:
    if hasattr(model, "device"):
        try:
            return torch.device(getattr(model, "device"))
        except Exception:
            pass
    for parameter in model.parameters(recurse=True):
        return parameter.device
    for buffer in model.buffers(recurse=True):
        return buffer.device
    return torch.device(fallback)


def _safe_model_to(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if hasattr(model, "hf_device_map"):
        return model
    return model.to(device)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _tokenize(tokenizer: Any, text: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=int(max_length))
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return _move_batch(encoded, device)


def sample_loss_for_text(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, Any] | None:
    batch = _tokenize(tokenizer, text, max_length, device)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    token_count = int(attention_mask.sum().item()) - 1 if isinstance(attention_mask, torch.Tensor) else int(input_ids.numel()) - 1
    if input_ids.shape[-1] < 2 or token_count <= 0:
        return None
    labels = input_ids.clone()
    with torch.no_grad(), _autocast_context(device, precision):
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = _extract_loss(output)
    if loss is None:
        return None
    loss_value = float(loss.detach().float().cpu())
    if not math.isfinite(loss_value):
        return None
    return {"loss": loss_value, "loss_sum": loss_value * token_count, "tokens": token_count}


def evaluate_heldout_sample_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    files: list[Path],
    *,
    checkpoint: str,
    required_modalities: tuple[str, ...] = DEFAULT_REQUIRED_MODALITIES,
    max_length: int = 1024,
    max_records_per_file: int = 0,
    max_samples_per_modality: int = 8,
    device: torch.device | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    started = time.time()
    eval_device = device or _model_device(model)
    required = tuple(dict.fromkeys(required_modalities))
    overall = _new_bucket()
    by_modality: dict[str, dict[str, Any]] = {name: _new_bucket() for name in required}
    file_reports: list[dict[str, Any]] = []

    for path in files:
        file_bucket = _new_bucket()
        file_bucket["path"] = str(path)
        file_bucket["modalities"] = {name: _new_bucket() for name in required}
        for record in _read_jsonl(path, max_records=max_records_per_file):
            modality = infer_modality(record, path.stem)
            if modality not in by_modality:
                by_modality[modality] = _new_bucket()
            if modality not in file_bucket["modalities"]:
                file_bucket["modalities"][modality] = _new_bucket()
            modality_bucket = by_modality[modality]
            file_modality_bucket = file_bucket["modalities"][modality]
            for bucket in (overall, file_bucket, modality_bucket, file_modality_bucket):
                bucket["records_seen"] += 1

            if max_samples_per_modality > 0 and int(modality_bucket["records_evaluated"]) >= int(max_samples_per_modality):
                for bucket in (overall, file_bucket, modality_bucket, file_modality_bucket):
                    _skip(bucket)
                continue
            text = record_text(record)
            if len(text.strip()) < 2:
                for bucket in (overall, file_bucket, modality_bucket, file_modality_bucket):
                    _skip(bucket)
                continue
            loss = sample_loss_for_text(
                model,
                tokenizer,
                text,
                max_length=max_length,
                device=eval_device,
                precision=precision,
            )
            if loss is None:
                for bucket in (overall, file_bucket, modality_bucket, file_modality_bucket):
                    _skip(bucket)
                continue
            for bucket in (overall, file_bucket, modality_bucket, file_modality_bucket):
                _add_loss(bucket, float(loss["loss_sum"]), int(loss["tokens"]))

        _finalize_bucket(file_bucket)
        for child in file_bucket["modalities"].values():
            _finalize_bucket(child)
        file_reports.append(file_bucket)

    _finalize_bucket(overall)
    for bucket in by_modality.values():
        _finalize_bucket(bucket)
    missing = [name for name in required if by_modality.get(name, {}).get("loss") is None]
    reasons: list[str] = []
    if not files:
        reasons.append("no_input_files")
    if overall.get("loss") is None:
        reasons.append("overall_loss_null")
    if missing:
        reasons.append("required_modality_loss_null:" + ",".join(missing))
    passed = not reasons
    return {
        "schema": SCHEMA_HELDOUT,
        "status": "passed" if passed else "failed",
        "checkpoint": checkpoint,
        "device": str(eval_device),
        "precision": precision,
        "max_length": int(max_length),
        "max_records_per_file": int(max_records_per_file),
        "max_samples_per_modality": int(max_samples_per_modality),
        "required_modalities": list(required),
        "files": file_reports,
        "modalities": {key: by_modality[key] for key in sorted(by_modality)},
        "overall": overall,
        "gate": {
            "passed": passed,
            "reasons": reasons,
            "missing_non_null_loss_modalities": missing,
            "non_null_loss_modalities": sorted(name for name, bucket in by_modality.items() if bucket.get("loss") is not None),
        },
        "elapsed_sec": round(time.time() - started, 6),
    }


def analyze_decode_text(
    text: str,
    *,
    min_chars: int = 8,
    min_words: int = 2,
    min_alnum_fraction: float = 0.20,
    max_punctuation_fraction: float = 0.75,
    max_char_run: int = 12,
    min_unique_chars: int = 4,
    max_top_token_fraction: float = 0.80,
) -> dict[str, Any]:
    stripped = str(text or "").strip()
    chars = len(stripped)
    alnum = sum(1 for ch in stripped if ch.isalnum())
    alpha = sum(1 for ch in stripped if ch.isalpha())
    punct = sum(1 for ch in stripped if (not ch.isalnum() and not ch.isspace()))
    words = re.findall(r"[A-Za-z0-9_]+", stripped)
    unique_chars = len(set(stripped))
    max_run = 0
    current_run = 0
    previous = None
    for ch in stripped:
        current_run = current_run + 1 if ch == previous else 1
        max_run = max(max_run, current_run)
        previous = ch
    top_token_fraction = 0.0
    if words:
        counts: dict[str, int] = {}
        for word in words:
            lowered = word.lower()
            counts[lowered] = counts.get(lowered, 0) + 1
        top_token_fraction = max(counts.values()) / max(1, len(words))

    reasons: list[str] = []
    if not stripped:
        reasons.append("empty")
    if chars < int(min_chars):
        reasons.append("too_short")
    if chars and alnum == 0:
        reasons.append("punctuation_only")
    if chars and (alnum / chars) < float(min_alnum_fraction):
        reasons.append("low_alnum_fraction")
    if chars and (punct / chars) > float(max_punctuation_fraction):
        reasons.append("high_punctuation_fraction")
    if len(words) < int(min_words):
        reasons.append("too_few_words")
    if chars >= int(min_chars) and unique_chars < int(min_unique_chars):
        reasons.append("low_unique_chars")
    if max_run > int(max_char_run):
        reasons.append("long_repeated_char_run")
    if len(words) >= 4 and top_token_fraction > float(max_top_token_fraction):
        reasons.append("single_token_repetition")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "chars": chars,
            "alnum_chars": alnum,
            "alpha_chars": alpha,
            "punctuation_chars": punct,
            "words": len(words),
            "unique_chars": unique_chars,
            "max_char_run": max_run,
            "alnum_fraction": (alnum / chars) if chars else 0.0,
            "punctuation_fraction": (punct / chars) if chars else 0.0,
            "top_token_fraction": top_token_fraction,
        },
    }


def modality_decode_reasons(modality: str, text: str) -> list[str]:
    normalized = str(modality or "text").strip().lower().replace("-", "_")
    stripped = str(text or "").strip()
    reasons: list[str] = []
    if REFUSAL_DECODE_RE.search(stripped):
        reasons.append("refusal_decode")
    if normalized in {"tool", "agent", "function_call"} and not TOOL_DECODE_RE.search(stripped):
        reasons.append("tool_decode_missing_structured_call")
    if normalized in {"image", "video", "audio", "music", "tts", "speech"} and not MEDIA_ROUTE_RE.search(stripped):
        reasons.append(f"{normalized}_decode_missing_media_route")
    if normalized == "ocr" and not (OCR_ROUTE_RE.search(stripped) or MEDIA_ROUTE_RE.search(stripped)):
        reasons.append("ocr_decode_missing_route_or_text_extraction")
    return reasons


def _decode_tokenizer(tokenizer: Any, ids: torch.Tensor | list[int]) -> str:
    if isinstance(ids, torch.Tensor):
        raw_ids = ids.detach().cpu().tolist()
    else:
        raw_ids = ids
    if raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
    return str(tokenizer.decode(raw_ids, skip_special_tokens=True))


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
) -> tuple[str, int]:
    batch = _tokenize(tokenizer, prompt, max_input_tokens, device)
    input_ids = batch["input_ids"]
    input_len = int(input_ids.shape[-1])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if hasattr(model, "generate"):
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": batch.get("attention_mask"),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0.0,
            "temperature": float(temperature) if float(temperature) > 0.0 else None,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        output = model.generate(**kwargs)
        new_ids = output[0, input_len:]
        return _decode_tokenizer(tokenizer, new_ids), int(new_ids.numel())

    generated = input_ids
    for _ in range(int(max_new_tokens)):
        output = model(input_ids=generated)
        logits = _extract_logits(output)[:, -1, :]
        if float(temperature) > 0.0:
            probs = torch.softmax(logits / float(temperature), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=-1)
    new_ids = generated[0, input_len:]
    return _decode_tokenizer(tokenizer, new_ids), int(new_ids.numel())


def load_prompt_records(paths: list[str] | None, prompts: list[str] | None) -> list[dict[str, str]]:
    loaded: list[dict[str, str]] = []
    for item in prompts or []:
        loaded.append({"id": f"prompt_{len(loaded) + 1}", "modality": "text", "prompt": str(item)})
    for path in paths or []:
        for record in _read_jsonl(path):
            prompt = record.get("prompt") or record_text(record)
            if isinstance(prompt, str) and prompt.strip():
                loaded.append(
                    {
                        "id": str(record.get("id") or record.get("task_id") or f"prompt_{len(loaded) + 1}"),
                        "modality": infer_modality(record, Path(path).stem),
                        "prompt": prompt,
                    }
                )
    return loaded or [dict(item) for item in DEFAULT_DECODE_PROMPTS]


def run_decode_sanity(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[dict[str, str]],
    *,
    checkpoint: str,
    max_input_tokens: int = 512,
    max_new_tokens: int = 48,
    device: torch.device | None = None,
    temperature: float = 0.0,
    min_chars: int = 8,
    min_words: int = 2,
) -> dict[str, Any]:
    started = time.time()
    eval_device = device or _model_device(model)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(prompts):
        prompt = str(item.get("prompt") or "")
        generated, generated_tokens = generate_completion(
            model,
            tokenizer,
            prompt,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            device=eval_device,
            temperature=temperature,
        )
        analysis = analyze_decode_text(generated, min_chars=min_chars, min_words=min_words)
        modality_reasons = modality_decode_reasons(str(item.get("modality") or "text"), generated)
        if modality_reasons:
            analysis = {
                **analysis,
                "passed": False,
                "reasons": [*analysis.get("reasons", []), *modality_reasons],
            }
        rows.append(
            {
                "id": str(item.get("id") or f"prompt_{index + 1}"),
                "modality": str(item.get("modality") or "text"),
                "prompt": prompt,
                "generated_text": generated,
                "generated_tokens": generated_tokens,
                "sanity": analysis,
            }
        )
    failed = [row["id"] for row in rows if not row["sanity"]["passed"]]
    reasons = ["decode_sanity_failed:" + ",".join(failed)] if failed else []
    return {
        "schema": SCHEMA_DECODE,
        "status": "passed" if not failed and rows else "failed",
        "checkpoint": checkpoint,
        "device": str(eval_device),
        "max_input_tokens": int(max_input_tokens),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "prompts": rows,
        "gate": {"passed": not failed and bool(rows), "reasons": reasons, "failed_prompt_ids": failed},
        "elapsed_sec": round(time.time() - started, 6),
    }


def _average_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[str],
    *,
    max_length: int,
    device: torch.device,
    precision: str,
) -> float | None:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for text in samples:
        loss = sample_loss_for_text(model, tokenizer, text, max_length=max_length, device=device, precision=precision)
        if loss is None:
            continue
        total_loss += float(loss["loss_sum"])
        total_tokens += int(loss["tokens"])
    if total_tokens <= 0:
        return None
    return total_loss / total_tokens


def _parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def _target_modules_arg(value: str) -> str | list[str]:
    raw = str(value or "all-linear").strip()
    if raw == "all-linear":
        return raw
    return [item.strip() for item in raw.split(",") if item.strip()]


def configure_tiny_overfit_trainables(
    model: torch.nn.Module,
    *,
    mode: str,
    max_trainable_params: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    total_params = _parameter_count(model.parameters())
    selected_mode = str(mode or "auto").lower()
    if selected_mode == "auto":
        selected_mode = "all" if total_params <= int(max_trainable_params) else "lora"

    if selected_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError("tiny overfit mode lora requires peft; choose --train-mode all for tiny local models") from exc
        config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=_target_modules_arg(target_modules),
        )
        model = get_peft_model(model, config)
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if selected_mode == "all":
            if total_params > int(max_trainable_params):
                raise RuntimeError(
                    f"refusing to train all {total_params} parameters; increase --max-trainable-params or use --train-mode lora"
                )
            for parameter in model.parameters():
                parameter.requires_grad = True
        elif selected_mode == "lm_head":
            matched = 0
            for name, parameter in model.named_parameters():
                if any(part in name for part in ("lm_head", "embed_out", "output_projection")):
                    parameter.requires_grad = True
                    matched += parameter.numel()
            if matched <= 0:
                raise RuntimeError("train-mode lm_head did not find lm_head/embed_out/output_projection parameters")
        else:
            raise RuntimeError(f"unknown tiny overfit train mode: {mode}")

    trainable_params = _parameter_count(parameter for parameter in model.parameters() if parameter.requires_grad)
    if trainable_params <= 0:
        raise RuntimeError("tiny overfit has no trainable parameters")
    return model, {"mode": selected_mode, "total_params": total_params, "trainable_params": trainable_params}


def load_text_samples(paths: list[str] | None, samples: list[str] | None, limit: int = 0) -> list[str]:
    texts = [str(item) for item in samples or [] if str(item).strip()]
    for path in paths or []:
        for record in _read_jsonl(path, max_records=limit):
            text = record_text(record)
            if text.strip():
                texts.append(text)
    return texts or list(DEFAULT_TINY_TEXT_SAMPLES)


def run_tiny_text_overfit(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[str],
    *,
    checkpoint: str,
    steps: int = 20,
    learning_rate: float = 2e-4,
    max_length: int = 256,
    device: torch.device | None = None,
    precision: str = "fp32",
    train_mode: str = "auto",
    max_trainable_params: int = 5_000_000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: str = "all-linear",
    min_loss_drop_ratio: float = 0.01,
    decode_prompt: str = "Omnicoder tiny overfit check:",
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    started = time.time()
    eval_device = device or _model_device(model)
    model = _safe_model_to(model, eval_device)
    model, trainable = configure_tiny_overfit_trainables(
        model,
        mode=train_mode,
        max_trainable_params=max_trainable_params,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = _safe_model_to(model, eval_device)
    clean_samples = [sample for sample in samples if sample.strip()]
    if not clean_samples:
        raise RuntimeError("tiny overfit needs at least one non-empty text sample")

    initial_loss = _average_loss(model, tokenizer, clean_samples, max_length=max_length, device=eval_device, precision=precision)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(learning_rate))
    step_losses: list[float] = []
    model.train()
    for step in range(int(steps)):
        text = clean_samples[step % len(clean_samples)]
        batch = _tokenize(tokenizer, text, max_length, eval_device)
        if batch["input_ids"].shape[-1] < 2:
            continue
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(eval_device, precision):
            output = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch["input_ids"])
            loss = _extract_loss(output)
        if loss is None:
            raise RuntimeError("model did not return loss during tiny overfit")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()
        step_losses.append(float(loss.detach().float().cpu()))

    final_loss = _average_loss(model, tokenizer, clean_samples, max_length=max_length, device=eval_device, precision=precision)
    generated_text, generated_tokens = generate_completion(
        model.eval(),
        tokenizer,
        decode_prompt,
        max_input_tokens=max_length,
        max_new_tokens=max_new_tokens,
        device=eval_device,
        temperature=0.0,
    )
    decode = analyze_decode_text(generated_text)
    loss_drop = None
    loss_drop_ratio = None
    if initial_loss is not None and final_loss is not None:
        loss_drop = float(initial_loss) - float(final_loss)
        loss_drop_ratio = loss_drop / max(1e-12, float(initial_loss))

    reasons: list[str] = []
    if initial_loss is None:
        reasons.append("initial_loss_null")
    if final_loss is None:
        reasons.append("final_loss_null")
    if loss_drop is None or loss_drop <= 0.0:
        reasons.append("loss_did_not_drop")
    if loss_drop_ratio is None or loss_drop_ratio < float(min_loss_drop_ratio):
        reasons.append("loss_drop_below_threshold")
    if not decode["passed"]:
        reasons.append("decode_sanity_failed")
    passed = not reasons
    return {
        "schema": SCHEMA_OVERFIT,
        "status": "passed" if passed else "failed",
        "checkpoint": checkpoint,
        "device": str(eval_device),
        "precision": precision,
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "max_length": int(max_length),
        "samples": {"count": len(clean_samples), "chars": sum(len(item) for item in clean_samples)},
        "trainable": trainable,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": loss_drop,
        "loss_drop_ratio": loss_drop_ratio,
        "step_losses": step_losses,
        "decode": {
            "prompt": decode_prompt,
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "sanity": decode,
        },
        "gate": {"passed": passed, "reasons": reasons, "min_loss_drop_ratio": float(min_loss_drop_ratio)},
        "elapsed_sec": round(time.time() - started, 6),
    }


def _adapter_base_model(adapter_path: Path) -> str:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return ""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    value = config.get("base_model_name_or_path")
    return str(value) if value else ""


def load_hf_checkpoint(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("HF diagnostics require transformers to load checkpoint paths") from exc

    checkpoint = Path(args.checkpoint)
    adapter = Path(args.adapter) if str(getattr(args, "adapter", "") or "").strip() else None
    checkpoint_is_adapter = checkpoint.is_dir() and (checkpoint / "adapter_config.json").exists()
    if adapter is None and checkpoint_is_adapter:
        adapter = checkpoint

    base_model = str(getattr(args, "base_model", "") or "").strip()
    if adapter is not None and not base_model:
        base_model = _adapter_base_model(adapter)
    if adapter is not None:
        model_source = base_model or ("" if checkpoint_is_adapter else str(checkpoint))
    else:
        model_source = str(checkpoint)
    if not model_source:
        raise RuntimeError("adapter checkpoints need --base-model or adapter_config.json base_model_name_or_path")

    tokenizer_source = str(getattr(args, "tokenizer", "") or "").strip()
    if not tokenizer_source:
        if adapter is None and checkpoint.is_dir() and (checkpoint / "tokenizer_config.json").exists():
            tokenizer_source = str(checkpoint)
        else:
            tokenizer_source = model_source

    dtype = _dtype_from_name(str(getattr(args, "dtype", "auto")))
    trust_remote_code = not bool(getattr(args, "no_trust_remote_code", False))
    local_files_only = bool(getattr(args, "local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    device_map = str(getattr(args, "device_map", "") or "").strip()
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if adapter is not None:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("adapter checkpoints require peft") from exc
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=local_files_only)
    if not device_map:
        model.to(torch.device(getattr(args, "device", "cpu")))
    model.eval()
    meta = {
        "checkpoint": str(checkpoint),
        "model_source": model_source,
        "adapter": str(adapter) if adapter else "",
        "tokenizer_source": tokenizer_source,
        "device_map": device_map,
        "dtype": str(getattr(args, "dtype", "auto")),
        "local_files_only": local_files_only,
    }
    return model, tokenizer, meta


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="Local HF model directory or PEFT adapter directory")
    parser.add_argument("--base-model", "--base_model", dest="base_model", default="", help="Base model for adapter-only checkpoints")
    parser.add_argument("--adapter", default="", help="Optional PEFT adapter path when --checkpoint is the base model")
    parser.add_argument("--tokenizer", default="", help="Optional tokenizer path/name; defaults to checkpoint or base model")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, etc. Defaults to CPU to avoid active training GPUs")
    parser.add_argument("--device-map", "--device_map", dest="device_map", default="", help="Optional transformers device_map, for example auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Autocast precision for forward passes")
    parser.add_argument("--local-files-only", "--local_files_only", dest="local_files_only", action="store_true")
    parser.add_argument("--no-trust-remote-code", "--no_trust_remote_code", dest="no_trust_remote_code", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode correctness diagnostics for local Omnicoder/HF checkpoints")
    sub = parser.add_subparsers(dest="command", required=True)

    heldout = sub.add_parser("heldout-loss", help="Compute heldout sample loss/perplexity by modality")
    _add_model_args(heldout)
    heldout.add_argument("--data", action="append", default=[], help="JSONL file or directory; repeatable")
    heldout.add_argument("--data-dir", "--data_dir", dest="data_dir", default="")
    heldout.add_argument("--required-modalities", "--required_modalities", dest="required_modalities", default="text,code,tool,math,media")
    heldout.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=1024)
    heldout.add_argument("--max-records-per-file", "--max_records_per_file", dest="max_records_per_file", type=int, default=0)
    heldout.add_argument("--max-samples-per-modality", "--max_samples_per_modality", dest="max_samples_per_modality", type=int, default=8)
    heldout.add_argument("--out", required=True)

    decode = sub.add_parser("decode-sanity", help="Generate short completions and reject punctuation-only/junk output")
    _add_model_args(decode)
    decode.add_argument("--prompt", action="append", default=[])
    decode.add_argument("--prompts-jsonl", "--prompts_jsonl", dest="prompts_jsonl", action="append", default=[])
    decode.add_argument("--max-input-tokens", "--max_input_tokens", dest="max_input_tokens", type=int, default=512)
    decode.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=48)
    decode.add_argument("--temperature", type=float, default=0.0)
    decode.add_argument("--min-chars", "--min_chars", dest="min_chars", type=int, default=8)
    decode.add_argument("--min-words", "--min_words", dest="min_words", type=int, default=2)
    decode.add_argument("--out", required=True)

    overfit = sub.add_parser("tiny-overfit", help="In-memory tiny text overfit check with a decode sanity gate")
    _add_model_args(overfit)
    overfit.add_argument("--sample", action="append", default=[])
    overfit.add_argument("--samples-jsonl", "--samples_jsonl", dest="samples_jsonl", action="append", default=[])
    overfit.add_argument("--sample-limit", "--sample_limit", dest="sample_limit", type=int, default=0)
    overfit.add_argument("--steps", type=int, default=20)
    overfit.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=2e-4)
    overfit.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=256)
    overfit.add_argument("--train-mode", "--train_mode", dest="train_mode", choices=["auto", "all", "lm_head", "lora"], default="auto")
    overfit.add_argument("--max-trainable-params", "--max_trainable_params", dest="max_trainable_params", type=int, default=5_000_000)
    overfit.add_argument("--lora-r", "--lora_r", dest="lora_r", type=int, default=8)
    overfit.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=int, default=16)
    overfit.add_argument("--lora-dropout", "--lora_dropout", dest="lora_dropout", type=float, default=0.0)
    overfit.add_argument("--target-modules", "--target_modules", dest="target_modules", default="all-linear")
    overfit.add_argument("--min-loss-drop-ratio", "--min_loss_drop_ratio", dest="min_loss_drop_ratio", type=float, default=0.01)
    overfit.add_argument("--decode-prompt", "--decode_prompt", dest="decode_prompt", default="Omnicoder tiny overfit check:")
    overfit.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=48)
    overfit.add_argument("--out", required=True)
    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, meta = load_hf_checkpoint(args)
    checkpoint_label = str(meta["checkpoint"])
    device = _model_device(model, getattr(args, "device", "cpu"))
    if args.command == "heldout-loss":
        files = candidate_jsonl_files(args.data, args.data_dir)
        if not files:
            raise RuntimeError("no JSONL files found; pass --data or --data-dir")
        report = evaluate_heldout_sample_loss(
            model,
            tokenizer,
            files,
            checkpoint=checkpoint_label,
            required_modalities=_csv_tuple(args.required_modalities),
            max_length=int(args.max_length),
            max_records_per_file=int(args.max_records_per_file),
            max_samples_per_modality=int(args.max_samples_per_modality),
            device=device,
            precision=str(args.precision),
        )
    elif args.command == "decode-sanity":
        prompts = load_prompt_records(args.prompts_jsonl, args.prompt)
        report = run_decode_sanity(
            model,
            tokenizer,
            prompts,
            checkpoint=checkpoint_label,
            max_input_tokens=int(args.max_input_tokens),
            max_new_tokens=int(args.max_new_tokens),
            device=device,
            temperature=float(args.temperature),
            min_chars=int(args.min_chars),
            min_words=int(args.min_words),
        )
    elif args.command == "tiny-overfit":
        samples = load_text_samples(args.samples_jsonl, args.sample, limit=int(args.sample_limit))
        report = run_tiny_text_overfit(
            model,
            tokenizer,
            samples,
            checkpoint=checkpoint_label,
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            max_length=int(args.max_length),
            device=device,
            precision=str(args.precision),
            train_mode=str(args.train_mode),
            max_trainable_params=int(args.max_trainable_params),
            lora_r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=str(args.target_modules),
            min_loss_drop_ratio=float(args.min_loss_drop_ratio),
            decode_prompt=str(args.decode_prompt),
            max_new_tokens=int(args.max_new_tokens),
        )
    else:
        raise RuntimeError(f"unknown command: {args.command}")
    report["model_load"] = meta
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_command(args)
        write_json(args.out, report)
        print(json.dumps(report, ensure_ascii=True, sort_keys=True, default=_json_default))
        return 0 if report.get("status") == "passed" else 2
    except Exception as exc:
        payload = {
            "schema": "omnicoder.decode_diagnostics_2026.error.v1",
            "status": "error",
            "command": getattr(args, "command", ""),
            "error": str(exc),
        }
        out = getattr(args, "out", "")
        if out:
            write_json(out, payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

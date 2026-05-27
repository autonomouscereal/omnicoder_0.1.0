from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "omnicoder.checkpoint_readiness_2026.v1"
MEDIA_MODALITIES = {"image", "video", "audio", "music", "speech"}
JUNK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"__OMNICODER_(?:EMPTY_DECODE|SKIPPED)__",
        r"(?:_ph){3,}",
        r"^(.)\1{15,}$",
    )
)


@dataclass(frozen=True)
class ReadinessThresholds:
    max_avg_loss: float = 20.0
    max_perplexity: float = 1_000_000.0
    min_tokens: int = 1
    min_weight_std: float = 1.0e-5
    max_weight_std: float = 0.20


def checkpoint_fingerprint(path: str | Path) -> str:
    candidate = Path(str(path))
    digest = hashlib.sha256()
    if candidate.is_file():
        stat = candidate.stat()
        digest.update(candidate.name.encode("utf-8", errors="ignore"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
        return digest.hexdigest()
    if not candidate.is_dir():
        digest.update(str(candidate).encode("utf-8", errors="ignore"))
        return digest.hexdigest()
    manifest = candidate / "manifest.json"
    if manifest.exists():
        digest.update(manifest.read_bytes())
    complete = candidate / ".complete.json"
    if complete.exists():
        digest.update(complete.read_bytes())
    rank_files = sorted(
        [
            *candidate.glob("rank*.pt"),
            *candidate.glob("shard*.pt"),
            *candidate.glob("*.pt"),
            *candidate.glob("*.safetensors"),
        ],
        key=lambda item: item.name,
    )
    for rank_path in rank_files:
        stat = rank_path.stat()
        digest.update(rank_path.name.encode("utf-8", errors="ignore"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def _checkpoint_aliases(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    raw = str(path).strip()
    if not raw:
        return set()
    aliases = {raw.replace("\\", "/").rstrip("/")}
    normalized = raw.replace("\\", "/").rstrip("/")
    for prefix in ("/workspace/weights/", "/home/cereal/omnicoder_2026_work/weights/"):
        if normalized.startswith(prefix):
            aliases.add(normalized[len(prefix) :])
    return {alias for alias in aliases if alias}


def _checkpoint_path_matches(observed: Any, expected: str | Path | None) -> bool:
    observed_aliases = _checkpoint_aliases(str(observed) if observed is not None else None)
    expected_aliases = _checkpoint_aliases(expected)
    if not observed_aliases or not expected_aliases:
        return False
    if observed_aliases & expected_aliases:
        return True
    return any(
        observed.endswith(expected) or expected.endswith(observed)
        for observed in observed_aliases
        for expected in expected_aliases
    )


def checkpoint_binding_payload(
    checkpoint: str | Path,
    *,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    path = Path(str(checkpoint))
    complete = path / ".complete.json" if path.is_dir() else None
    complete_payload: dict[str, Any] = {}
    if complete is not None and complete.exists():
        try:
            parsed = json.loads(complete.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                complete_payload = parsed
        except Exception:
            complete_payload = {}
    return {
        "checkpoint": str(path),
        "aliases": sorted(_checkpoint_aliases(path)),
        "fingerprint": checkpoint_fingerprint(path),
        "expected_world_size": expected_world_size,
        "completion_marker": str(complete) if complete is not None else "",
        "completion_marker_status": complete_payload.get("status"),
        "completion_marker_step": complete_payload.get("step"),
    }


def validate_checkpoint_binding(
    topk_probe: dict[str, Any],
    sample_loss: dict[str, Any],
    readiness_report: dict[str, Any] | None = None,
    *,
    expected_checkpoint: str | Path | None = None,
    expected_fingerprint: str | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if expected_checkpoint is None and expected_fingerprint is None and expected_world_size is None:
        return {"status": "passed", "reasons": [], "diagnostics": diagnostics}

    for label, payload in (("topk_probe", topk_probe), ("sample_loss", sample_loss)):
        observed = payload.get("checkpoint") or payload.get("checkpoint_path") or payload.get("resume_checkpoint")
        if readiness_report is not None and observed is None:
            continue
        diagnostics.append({"label": label, "checkpoint": observed})
        if expected_checkpoint is not None and not _checkpoint_path_matches(observed, expected_checkpoint):
            reasons.append(f"{label}_checkpoint_mismatch")

    if readiness_report is not None:
        binding = readiness_report.get("checkpoint_binding")
        if not isinstance(binding, dict):
            reasons.append("checkpoint_readiness_report_binding_missing")
        else:
            diagnostics.append({"label": "readiness_report", "checkpoint": binding.get("checkpoint"), "fingerprint": binding.get("fingerprint")})
            if expected_checkpoint is not None and not _checkpoint_path_matches(binding.get("checkpoint"), expected_checkpoint):
                reasons.append("checkpoint_readiness_report_checkpoint_mismatch")
            if expected_fingerprint and binding.get("fingerprint") != expected_fingerprint:
                reasons.append("checkpoint_readiness_report_fingerprint_mismatch")
            if expected_world_size is not None and binding.get("expected_world_size") != expected_world_size:
                reasons.append("checkpoint_readiness_report_world_size_mismatch")

    if reasons:
        reasons.append("checkpoint_binding_invalid")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": sorted(set(reasons)),
        "diagnostics": diagnostics,
        "expected": {
            "checkpoint": str(expected_checkpoint) if expected_checkpoint is not None else None,
            "fingerprint": expected_fingerprint,
            "expected_world_size": expected_world_size,
        },
    }


def _load_json(value: dict[str, Any] | str | Path, label: str) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, dict):
        return value, None
    path = Path(value)
    try:
        if path.suffix.lower() == ".jsonl":
            payload = None
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        payload = json.loads(line)
                        break
            if payload is None:
                return {}, f"{label}_jsonl_empty"
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"{label}_json_unreadable:{exc}"
    if not isinstance(payload, dict):
        return {}, f"{label}_json_not_object"
    return payload, None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _diagnostic_status_reasons(payload: dict[str, Any], label: str) -> list[str]:
    status = payload.get("status")
    if not isinstance(status, str) or not status.strip():
        return []
    normalized = status.strip().lower()
    if normalized in {"ok", "pass", "passed", "success"}:
        return []
    return [f"{label}_status_{normalized}"]


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


def _token_range(topk_probe: dict[str, Any]) -> tuple[int | None, int | None]:
    raw = topk_probe.get("text_range")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None, None
    return _int_or_none(raw[0]), _int_or_none(raw[1])


def validate_tokenizer_range(topk_probe: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    begin, end = _token_range(topk_probe)
    model_vocab_size = _int_or_none(topk_probe.get("model_vocab_size"))
    tokenizer = topk_probe.get("tokenizer") if isinstance(topk_probe.get("tokenizer"), dict) else {}
    tokenizer_vocab_size = _int_or_none(tokenizer.get("vocab_size"))

    if begin is None or end is None:
        reasons.append("tokenizer_text_range_missing")
    elif begin < 0 or end <= begin:
        reasons.append("tokenizer_text_range_invalid")

    if model_vocab_size is None or model_vocab_size <= 0:
        reasons.append("model_vocab_size_missing")
    elif begin is not None and end is not None and end > model_vocab_size:
        reasons.append("tokenizer_text_range_exceeds_model_vocab")

    if tokenizer_vocab_size is not None and tokenizer_vocab_size <= 0:
        reasons.append("tokenizer_vocab_size_invalid")
    if (
        tokenizer_vocab_size is not None
        and model_vocab_size is not None
        and tokenizer_vocab_size > model_vocab_size
    ):
        reasons.append("tokenizer_vocab_size_exceeds_model_vocab")

    if begin is not None and end is not None:
        raw_generated = topk_probe.get("generated_token_ids")
        if isinstance(raw_generated, list):
            for item in raw_generated:
                token_id = _int_or_none(item)
                if token_id is not None and model_vocab_size is not None and token_id >= model_vocab_size:
                    reasons.append("generated_token_id_outside_model_vocab")
                    break
                if token_id is not None and not (begin <= token_id < end):
                    reasons.append("generated_token_id_outside_text_range")
                    break
        raw_prompt = topk_probe.get("prompt_token_ids")
        if isinstance(raw_prompt, list):
            for item in raw_prompt:
                token_id = _int_or_none(item)
                if token_id is not None and token_id < 0:
                    reasons.append("prompt_token_ids_contains_negative_id")
                    break
                if token_id is not None and model_vocab_size is not None and token_id >= model_vocab_size:
                    reasons.append("prompt_token_id_outside_model_vocab")
                    break
        for step in topk_probe.get("steps") if isinstance(topk_probe.get("steps"), list) else []:
            if not isinstance(step, dict):
                continue
            for item in step.get("text_topk") if isinstance(step.get("text_topk"), list) else []:
                if not isinstance(item, dict):
                    continue
                token_id = _int_or_none(item.get("token_id"))
                if token_id is not None and not (begin <= token_id < end):
                    reasons.append("text_topk_token_outside_text_range")
                    break
            if "text_topk_token_outside_text_range" in reasons:
                break

    if any(
        reason.startswith("tokenizer_") or "outside_" in reason or reason == "text_topk_token_outside_text_range"
        for reason in reasons
    ):
        reasons.append("tokenizer_range_invalid")

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "text_range": [begin, end] if begin is not None and end is not None else None,
        "model_vocab_size": model_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
    }


def validate_topk_probe(topk_probe: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = _diagnostic_status_reasons(topk_probe, "topk_probe")
    generated_text = topk_probe.get("generated_text")
    if not isinstance(generated_text, str) or not generated_text.strip():
        reasons.append("topk_generated_text_missing")
        decode = {"passed": False, "reasons": ["empty"], "metrics": {}}
    else:
        decode = analyze_decode_text(
            generated_text,
            min_chars=4,
            min_words=1,
            min_alnum_fraction=0.10,
            max_punctuation_fraction=0.85,
        )
        junk_reasons = [
            reason
            for reason in decode.get("reasons", [])
            if reason
            in {
                "empty",
                "punctuation_only",
                "low_alnum_fraction",
                "high_punctuation_fraction",
                "low_unique_chars",
                "long_repeated_char_run",
                "single_token_repetition",
            }
        ]
        if any(pattern.search(generated_text.strip()) for pattern in JUNK_PATTERNS):
            junk_reasons.append("junk_pattern")
        if junk_reasons:
            reasons.append("topk_generated_text_junk")
            reasons.extend(f"topk_generated_text_{reason}" for reason in sorted(set(junk_reasons)))

    tokenizer_range = validate_tokenizer_range(topk_probe)
    reasons.extend(str(reason) for reason in tokenizer_range["reasons"])
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "generated_text": generated_text if isinstance(generated_text, str) else "",
        "decode": decode,
        "tokenizer_range": tokenizer_range,
    }


def validate_weight_stats(
    topk_probe: dict[str, Any],
    thresholds: ReadinessThresholds,
    *,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    checked: list[dict[str, Any]] = []
    rank_reports = topk_probe.get("rank_reports")
    if not isinstance(rank_reports, list) or not rank_reports:
        return {"status": "failed", "reasons": ["rank_weight_stats_missing"], "checked": checked}
    rank_ids: list[int] = []
    block_tensor_counts: list[int] = []
    for rank in rank_reports:
        if not isinstance(rank, dict):
            continue
        rank_id = _int_or_none(rank.get("rank"))
        if rank_id is not None:
            rank_ids.append(rank_id)
        block_count = _int_or_none(rank.get("block_tensor_count"))
        if block_count is not None:
            block_tensor_counts.append(block_count)
        tensors = rank.get("tensors") if isinstance(rank.get("tensors"), dict) else {}
        for name in ("embed.weight", "lm_head.weight"):
            stats = tensors.get(name)
            if not isinstance(stats, dict):
                continue
            std = _finite_float(stats.get("std_sample"))
            finite = stats.get("finite_sample")
            row = {"rank": rank.get("rank"), "tensor": name, "std_sample": std, "finite_sample": finite}
            checked.append(row)
            if finite is not True:
                reasons.append(f"{name}:nonfinite_weight_sample")
            if std is None:
                reasons.append(f"{name}:std_missing")
            elif std < float(thresholds.min_weight_std):
                reasons.append(f"{name}:std_below_threshold")
            elif std > float(thresholds.max_weight_std):
                reasons.append(f"{name}:std_over_threshold")
    names = {row.get("tensor") for row in checked}
    if "embed.weight" not in names:
        reasons.append("embed_weight_stats_missing")
    if "lm_head.weight" not in names:
        reasons.append("lm_head_weight_stats_missing")
    if expected_world_size is not None:
        if len(rank_reports) != int(expected_world_size):
            reasons.append("rank_weight_stats_world_size_mismatch")
        if sorted(set(rank_ids)) != list(range(int(expected_world_size))):
            reasons.append("rank_weight_stats_rank_ids_incomplete")
        if len(block_tensor_counts) != int(expected_world_size):
            reasons.append("rank_weight_stats_block_counts_missing")
        elif any(count <= 0 for count in block_tensor_counts):
            reasons.append("rank_weight_stats_empty_block_shard")
    elif block_tensor_counts and all(count <= 0 for count in block_tensor_counts):
        reasons.append("rank_weight_stats_no_block_tensors")
    if reasons:
        reasons.append("checkpoint_weight_stats_invalid")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": sorted(set(reasons)),
        "checked": checked,
        "rank_ids": sorted(set(rank_ids)),
        "block_tensor_counts": block_tensor_counts,
        "thresholds": {
            "min_weight_std": float(thresholds.min_weight_std),
            "max_weight_std": float(thresholds.max_weight_std),
        },
    }


def validate_sample_loss(sample_loss: dict[str, Any], thresholds: ReadinessThresholds) -> dict[str, Any]:
    overall = sample_loss.get("overall") if isinstance(sample_loss.get("overall"), dict) else {}
    avg_loss = _finite_float(overall.get("avg_loss") if overall.get("avg_loss") is not None else overall.get("loss"))
    perplexity = _finite_float(overall.get("perplexity"))
    tokens = _int_or_none(overall.get("tokens"))
    reasons: list[str] = _diagnostic_status_reasons(sample_loss, "sample_loss")
    returncode = _int_or_none(sample_loss.get("returncode"))
    if returncode not in (None, 0):
        reasons.append("sample_loss_returncode_nonzero")

    if avg_loss is None:
        reasons.append("heldout_avg_loss_missing")
    elif avg_loss > float(thresholds.max_avg_loss):
        reasons.append("heldout_avg_loss_over_threshold")

    if perplexity is None:
        reasons.append("heldout_perplexity_missing")
    elif perplexity > float(thresholds.max_perplexity):
        reasons.append("heldout_perplexity_over_threshold")

    if tokens is None or tokens <= 0:
        reasons.append("heldout_tokens_missing")
    elif tokens < int(thresholds.min_tokens):
        reasons.append("heldout_tokens_below_threshold")

    if any(reason.startswith("heldout_") for reason in reasons):
        reasons.append("heldout_metrics_missing_or_over_threshold")

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "tokens": tokens,
        "returncode": returncode,
        "thresholds": {
            "max_avg_loss": float(thresholds.max_avg_loss),
            "max_perplexity": float(thresholds.max_perplexity),
            "min_tokens": int(thresholds.min_tokens),
        },
    }


def _route_from_container(container: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("output_route", "media_route", "route"):
        value = container.get(key)
        if isinstance(value, dict):
            return value
    metadata = container.get("generation_metadata") if isinstance(container.get("generation_metadata"), dict) else {}
    value = metadata.get("output_route")
    return value if isinstance(value, dict) else None


def _collect_routes(media_route_probe: dict[str, Any]) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_route(route: dict[str, Any] | None) -> None:
        if route is None:
            return
        key = json.dumps(route, sort_keys=True, default=str)
        if key in seen:
            return
        seen.add(key)
        routes.append(route)

    route = _route_from_container(media_route_probe)
    add_route(route)
    for key in ("routes", "rows", "predictions"):
        raw_items = media_route_probe.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            route = _route_from_container(item)
            add_route(route)
    return routes


def _route_is_media(route: dict[str, Any]) -> bool:
    modality = str(route.get("output_modality") or "").strip().lower()
    artifact_kind = str(route.get("artifact_kind") or "").strip().lower()
    return bool(route.get("requires_artifact_decoder")) or modality in MEDIA_MODALITIES or artifact_kind in MEDIA_MODALITIES


def validate_media_route_probe(media_route_probe: dict[str, Any]) -> dict[str, Any]:
    routes = _collect_routes(media_route_probe)
    media_routes = [route for route in routes if _route_is_media(route)]
    reasons: list[str] = _diagnostic_status_reasons(media_route_probe, "media_route_probe")
    if not media_routes:
        reasons.append("media_router_metadata_missing")
    for route in media_routes:
        for key in ("name", "output_field", "output_modality", "token_ranges", "requires_artifact_decoder", "artifact_kind"):
            if key not in route:
                reasons.append(f"media_route_missing_{key}")
        token_ranges = route.get("token_ranges")
        if not isinstance(token_ranges, list) or not token_ranges:
            reasons.append("media_route_token_ranges_missing")
        else:
            for item in token_ranges:
                if not isinstance(item, dict):
                    reasons.append("media_route_token_range_invalid")
                    break
                begin = _int_or_none(item.get("begin"))
                end = _int_or_none(item.get("end"))
                if begin is None or end is None or begin < 0 or end <= begin:
                    reasons.append("media_route_token_range_invalid")
                    break
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": sorted(set(reasons)),
        "routes": media_routes,
        "media_route_count": len(media_routes),
    }


def checkpoint_readiness(
    topk_probe: dict[str, Any] | str | Path,
    sample_loss: dict[str, Any] | str | Path,
    media_route_probe: dict[str, Any] | str | Path,
    *,
    thresholds: ReadinessThresholds | None = None,
    max_avg_loss: float | None = None,
    max_perplexity: float | None = None,
    min_tokens: int | None = None,
    expected_checkpoint: str | Path | None = None,
    expected_fingerprint: str | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or ReadinessThresholds()
    if max_avg_loss is not None or max_perplexity is not None or min_tokens is not None:
        thresholds = ReadinessThresholds(
            max_avg_loss=float(max_avg_loss if max_avg_loss is not None else thresholds.max_avg_loss),
            max_perplexity=float(max_perplexity if max_perplexity is not None else thresholds.max_perplexity),
            min_tokens=int(min_tokens if min_tokens is not None else thresholds.min_tokens),
            min_weight_std=float(thresholds.min_weight_std),
            max_weight_std=float(thresholds.max_weight_std),
        )
    reasons: list[str] = []
    topk_payload, topk_error = _load_json(topk_probe, "topk_probe")
    sample_payload, sample_error = _load_json(sample_loss, "sample_loss")
    media_payload, media_error = _load_json(media_route_probe, "media_route_probe")

    load_errors = [error for error in (topk_error, sample_error, media_error) if error]
    reasons.extend(load_errors)
    topk = validate_topk_probe(topk_payload) if not topk_error else {"status": "failed", "reasons": [topk_error]}
    binding = validate_checkpoint_binding(
        topk_payload,
        sample_payload,
        expected_checkpoint=expected_checkpoint,
        expected_fingerprint=expected_fingerprint,
        expected_world_size=expected_world_size,
    ) if not (topk_error or sample_error) else {"status": "failed", "reasons": [error for error in (topk_error, sample_error) if error]}
    weights = validate_weight_stats(topk_payload, thresholds, expected_world_size=expected_world_size) if not topk_error else {"status": "failed", "reasons": [topk_error]}
    sample = validate_sample_loss(sample_payload, thresholds) if not sample_error else {"status": "failed", "reasons": [sample_error]}
    media = validate_media_route_probe(media_payload) if not media_error else {"status": "failed", "reasons": [media_error]}
    reasons.extend(str(reason) for check in (binding, topk, weights, sample, media) for reason in check.get("reasons", []))
    reasons = sorted(set(reasons))
    status = "passed" if not reasons else "failed"
    checkpoint_binding = (
        checkpoint_binding_payload(expected_checkpoint, expected_world_size=expected_world_size)
        if expected_checkpoint is not None
        else None
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "passed": status == "passed",
        "reasons": reasons,
        "reason": "checkpoint_diagnostics_ready" if status == "passed" else ",".join(reasons),
        "checkpoint_binding": checkpoint_binding,
        "checks": {
            "checkpoint_binding": binding,
            "topk_probe": topk,
            "weight_stats": weights,
            "sample_loss": sample,
            "media_route_probe": media,
        },
    }


evaluate_checkpoint_readiness = checkpoint_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed readiness gate for checkpoint diagnostic JSONs.")
    parser.add_argument("--topk-probe", required=True)
    parser.add_argument("--sample-loss", required=True)
    parser.add_argument("--media-route-probe", required=True)
    parser.add_argument("--max-avg-loss", type=float, default=ReadinessThresholds.max_avg_loss)
    parser.add_argument("--max-perplexity", type=float, default=ReadinessThresholds.max_perplexity)
    parser.add_argument("--min-tokens", type=int, default=ReadinessThresholds.min_tokens)
    parser.add_argument("--min-weight-std", type=float, default=ReadinessThresholds.min_weight_std)
    parser.add_argument("--max-weight-std", type=float, default=ReadinessThresholds.max_weight_std)
    parser.add_argument("--expected-checkpoint", default="")
    parser.add_argument("--expected-fingerprint", default="")
    parser.add_argument("--expected-world-size", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = checkpoint_readiness(
        args.topk_probe,
        args.sample_loss,
        args.media_route_probe,
        thresholds=ReadinessThresholds(
            max_avg_loss=float(args.max_avg_loss),
            max_perplexity=float(args.max_perplexity),
            min_tokens=int(args.min_tokens),
            min_weight_std=float(args.min_weight_std),
            max_weight_std=float(args.max_weight_std),
        ),
        expected_checkpoint=args.expected_checkpoint or None,
        expected_fingerprint=args.expected_fingerprint or None,
        expected_world_size=int(args.expected_world_size or 0) or None,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

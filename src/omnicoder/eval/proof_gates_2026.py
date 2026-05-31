from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


SCHEMA = "omnicoder.pre_full_training_proof_gate_2026.v1"
DEFAULT_REQUIRED_MODALITIES = (
    "text",
    "code",
    "tool",
    "math",
    "long_context",
    "image",
    "video",
    "audio",
    "music",
    "tts",
    "ocr",
)
DEFAULT_REQUIRED_MEDIA_MODALITIES = ("image", "video", "audio", "music", "tts")
DEFAULT_REQUIRED_REASONING_VARIANTS = (
    "fakequant_chunk2048_loss64",
    "reasoning_effort2_q4_chunk2048_loss64",
    "reasoning_efforthigh_q4_chunk2048_loss64",
)
DEFAULT_REQUIRED_Q4_PROFILES = (
    {
        "variant": "ffn_chunk1024_headroom_q4_chunk8192_loss64",
        "seq_len": 1024,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
        },
    },
    {
        "variant": "ffn_chunk1024_headroom_q4_chunk8192_loss64",
        "seq_len": 2048,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
        },
    },
)
DEFAULT_REQUIRED_CONTEXT_RUNGS = (8192, 32768, 131072, 262144, 524288, 1048576)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_modality(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "speech":
        return "tts"
    if text in {"tool_agent", "tools"}:
        return "tool"
    if text in {"vision", "image_edit", "image_generation"}:
        return "image"
    if "long" in text and "context" in text:
        return "long_context"
    return text


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}
    if not isinstance(payload, dict):
        return {"_read_error": "json_not_object", "_path": str(path)}
    payload.setdefault("_path", str(path))
    return payload


def _read_jsonl(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    row.setdefault("_path", str(path))
                    row.setdefault("_line_number", line_number)
                    rows.append(row)
                if max_rows and len(rows) >= int(max_rows):
                    break
    except Exception:
        return rows
    return rows


def _load_payloads(paths: list[str], *, max_jsonl_rows: int = 0) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            payloads.append({"_read_error": "path_missing", "_path": str(path)})
            continue
        if path.suffix.lower() == ".jsonl":
            payloads.extend(_read_jsonl(path, max_rows=max_jsonl_rows))
        else:
            payloads.append(_read_json(path))
    return payloads


def _status_ok(payload: dict[str, Any]) -> bool:
    if payload.get("_read_error"):
        return False
    status = str(payload.get("status") or payload.get("gate_decision") or "").strip().lower()
    if not status:
        return False
    return status in {"ok", "pass", "passed", "ready", "success", "completed", "accepted"}


def _positive_any(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = payload.get(key)
        if _safe_float(value) is not None and float(value) > 0.0:
            return True
        if isinstance(value, (list, tuple, set)) and len(value) > 0:
            return True
        if isinstance(value, dict) and len(value) > 0:
            if key == "counts":
                if any(
                    _safe_float(nested) is not None and float(nested) > 0.0
                    or isinstance(nested, (list, tuple, set)) and len(nested) > 0
                    or isinstance(nested, dict) and _positive_any(nested, tuple(str(inner_key) for inner_key in nested))
                    for nested in value.values()
                ):
                    return True
                continue
            return True
    return False


def _extract_modality_counts(payload: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    valid: dict[str, int] = {}
    optimized: dict[str, int] = {}

    def add(target: dict[str, int], key: Any, value: Any) -> None:
        name = _normalize_modality(key)
        amount = _safe_int(value)
        if name and amount is not None:
            target[name] = target.get(name, 0) + max(0, int(amount))

    targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
    for key, value in (targets.get("by_modality") if isinstance(targets.get("by_modality"), dict) else {}).items():
        add(valid, key, value)
    for key, value in (targets.get("optimized_by_modality") if isinstance(targets.get("optimized_by_modality"), dict) else {}).items():
        add(optimized, key, value)

    loss = payload.get("loss") if isinstance(payload.get("loss"), dict) else {}
    loss_diag = payload.get("loss_diagnostics") if isinstance(payload.get("loss_diagnostics"), dict) else {}
    for source in (payload, loss, loss_diag):
        for key, value in (source.get("target_counts_by_modality") if isinstance(source.get("target_counts_by_modality"), dict) else {}).items():
            add(valid, key, value)
        for key, value in (source.get("optimized_target_counts_by_modality") if isinstance(source.get("optimized_target_counts_by_modality"), dict) else {}).items():
            add(optimized, key, value)

    modalities = payload.get("modalities") if isinstance(payload.get("modalities"), dict) else {}
    for key, bucket in modalities.items():
        if not isinstance(bucket, dict):
            continue
        tokens = bucket.get("target_tokens", bucket.get("valid_target_tokens", bucket.get("tokens")))
        add(valid, key, tokens)
        # Target-token diagnostic probes score target positions rather than train
        # optimizer tokens, so count those as checked target positions.
        add(optimized, key, bucket.get("optimized_target_tokens", tokens))
    return valid, optimized


def target_coverage_check(payloads: list[dict[str, Any]], required_modalities: tuple[str, ...], min_tokens: int) -> dict[str, Any]:
    valid: dict[str, int] = {}
    optimized: dict[str, int] = {}
    read_errors: list[str] = []
    for payload in payloads:
        if payload.get("_read_error"):
            read_errors.append(f"{payload.get('_path')}: {payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            read_errors.append(f"{payload.get('_path', '<memory>')}: status_not_passed")
            continue
        payload_valid, payload_optimized = _extract_modality_counts(payload)
        for key, value in payload_valid.items():
            valid[key] = valid.get(key, 0) + int(value)
        for key, value in payload_optimized.items():
            optimized[key] = optimized.get(key, 0) + int(value)
    missing_valid = [item for item in required_modalities if valid.get(_normalize_modality(item), 0) < int(min_tokens)]
    missing_optimized = [item for item in required_modalities if optimized.get(_normalize_modality(item), 0) < int(min_tokens)]
    reasons = [f"target_valid_missing_{item}" for item in missing_valid]
    reasons.extend(f"target_optimized_missing_{item}" for item in missing_optimized)
    reasons.extend(f"target_payload_read_error:{item}" for item in read_errors)
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "valid_by_modality": valid,
        "optimized_by_modality": optimized,
        "required_modalities": list(required_modalities),
        "min_tokens_per_modality": int(min_tokens),
    }


def _extract_loss_bucket(bucket: dict[str, Any]) -> tuple[float | None, int]:
    loss = _safe_float(bucket.get("avg_loss", bucket.get("loss", bucket.get("ce"))))
    tokens = _safe_int(bucket.get("tokens", bucket.get("target_tokens", bucket.get("count"))))
    return loss, max(0, int(tokens or 0))


def heldout_loss_check(payloads: list[dict[str, Any]], required_modalities: tuple[str, ...], min_tokens: int, max_loss: float) -> dict[str, Any]:
    by_modality: dict[str, dict[str, float | int | None]] = {}
    read_errors: list[str] = []
    for payload in payloads:
        if payload.get("_read_error"):
            read_errors.append(f"{payload.get('_path')}: {payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            read_errors.append(f"{payload.get('_path', '<memory>')}: status_not_passed")
            continue
        modality_maps = []
        for key in ("modalities", "by_modality", "loss_by_modality", "per_modality"):
            value = payload.get(key)
            if isinstance(value, dict):
                modality_maps.append(value)
        for mapping in modality_maps:
            for raw_name, bucket in mapping.items():
                if not isinstance(bucket, dict):
                    continue
                name = _normalize_modality(raw_name)
                loss, tokens = _extract_loss_bucket(bucket)
                current = by_modality.setdefault(name, {"tokens": 0, "loss": None})
                current["tokens"] = int(current.get("tokens") or 0) + tokens
                if loss is not None:
                    current["loss"] = loss if current.get("loss") is None else max(float(current["loss"]), loss)
    reasons: list[str] = []
    for item in required_modalities:
        name = _normalize_modality(item)
        bucket = by_modality.get(name, {})
        tokens = int(bucket.get("tokens") or 0)
        loss = _safe_float(bucket.get("loss"))
        if tokens < int(min_tokens):
            reasons.append(f"heldout_tokens_missing_{name}")
        if loss is None:
            reasons.append(f"heldout_loss_missing_{name}")
        elif loss <= 0.0 or loss > float(max_loss):
            reasons.append(f"heldout_loss_invalid_{name}")
    reasons.extend(f"heldout_payload_read_error:{item}" for item in read_errors)
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "loss_by_modality": by_modality,
        "required_modalities": list(required_modalities),
        "min_tokens_per_modality": int(min_tokens),
        "max_loss": float(max_loss),
    }


def release_gate_check(payloads: list[dict[str, Any]], required_modalities: tuple[str, ...]) -> dict[str, Any]:
    accepted: set[str] = set()
    reasons: list[str] = []
    for payload in payloads:
        if payload.get("_read_error"):
            reasons.append(f"release_gate_read_error:{payload.get('_path')}:{payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            reasons.append(f"release_gate_status_{payload.get('status') or 'failed'}")
        for item in payload.get("accepted_modalities") or payload.get("covered_modalities") or []:
            accepted.add(_normalize_modality(item))
        counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        for key, value in counts.items():
            text = str(key)
            if text.startswith("modality_") and int(value or 0) > 0:
                accepted.add(_normalize_modality(text.removeprefix("modality_")))
    missing = [_normalize_modality(item) for item in required_modalities if _normalize_modality(item) not in accepted]
    reasons.extend(f"release_gate_missing_{item}" for item in missing)
    return {
        "status": "passed" if payloads and not reasons else "failed",
        "reasons": reasons if payloads else ["release_gate_missing"],
        "accepted_modalities": sorted(accepted),
        "required_modalities": list(required_modalities),
    }


def profile_matrix_check(
    payloads: list[dict[str, Any]],
    *,
    required_variants: tuple[str, ...] = (),
    required_profiles: tuple[dict[str, Any], ...] = (),
    min_coverage: float,
) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    reasons: list[str] = []
    for payload in payloads:
        if payload.get("_read_error"):
            reasons.append(f"profile_read_error:{payload.get('_path')}:{payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            reasons.append(f"profile_status_failed:{payload.get('_path', '<memory>')}")
            continue
        for item in payload.get("variants") or []:
            if isinstance(item, dict) and item.get("variant"):
                variants.append(item)
    specs = list(required_profiles)
    specs.extend({"variant": name} for name in required_variants)
    required_profile_specs: list[dict[str, Any]] = []
    available_profile_specs: list[dict[str, Any]] = []
    for item in variants:
        requested_env = item.get("requested_env") if isinstance(item.get("requested_env"), dict) else {}
        available_profile_specs.append(
            {
                "variant": str(item.get("variant") or ""),
                "seq_len": _safe_int(item.get("last_seq_len", item.get("seq_len"))),
                "status": item.get("status"),
                "target_coverage": _safe_float(item.get("last_target_token_coverage")),
                "tps": _safe_float(item.get("sequence_tokens_per_sec", item.get("training_tokens_per_sec"))),
                "env": {str(key): str(value) for key, value in sorted(requested_env.items())},
            }
        )
    for spec in specs:
        name = str(spec.get("variant") or "")
        seq_len = _safe_int(spec.get("seq_len"))
        required_profile_specs.append(
            {
                "variant": name,
                "seq_len": seq_len,
                "env": {str(key): str(value) for key, value in sorted((spec.get("env") if isinstance(spec.get("env"), dict) else {}).items())},
            }
        )
        label = f"{name}_seq{seq_len}" if seq_len is not None else name
        matches = [item for item in variants if str(item.get("variant") or "") == name]
        if seq_len is not None:
            matches = [item for item in matches if _safe_int(item.get("last_seq_len", item.get("seq_len"))) == seq_len]
        item = matches[-1] if matches else None
        if not item:
            reasons.append(f"profile_variant_missing_{label}")
            continue
        if str(item.get("status") or "").lower() != "passed":
            reasons.append(f"profile_variant_failed_{label}")
        if item.get("oom_killed") or (isinstance(item.get("container_state"), dict) and item["container_state"].get("oom_killed")):
            reasons.append(f"profile_variant_oom_{label}")
        if isinstance(item.get("container_state"), dict):
            state = item["container_state"]
            if state.get("exit_code") not in {None, 0}:
                reasons.append(f"profile_variant_exit_code_{label}_{state.get('exit_code')}")
        coverage = _safe_float(item.get("last_target_token_coverage"))
        if coverage is None or coverage < float(min_coverage):
            reasons.append(f"profile_variant_target_coverage_low_{label}")
        tps = _safe_float(item.get("sequence_tokens_per_sec", item.get("training_tokens_per_sec")))
        if tps is None or tps <= 0.0:
            reasons.append(f"profile_variant_tps_missing_{label}")
        if not bool(item.get("no_checkpoint_written", False)):
            reasons.append(f"profile_variant_checkpoint_written_{label}")
        requested_env = item.get("requested_env") if isinstance(item.get("requested_env"), dict) else {}
        for key, expected in (spec.get("env") if isinstance(spec.get("env"), dict) else {}).items():
            if str(requested_env.get(key, "")) != str(expected):
                reasons.append(f"profile_variant_env_mismatch_{label}_{key}")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "required_variants": [str(spec.get("variant") or "") for spec in specs],
        "available_variants": sorted({str(item.get("variant") or "") for item in variants}),
        "required_profile_specs": required_profile_specs,
        "available_profile_specs": sorted(available_profile_specs, key=lambda item: (item["variant"], item.get("seq_len") or 0)),
        "min_target_coverage": float(min_coverage),
    }


def simple_status_check(payloads: list[dict[str, Any]], *, name: str, require_non_manifest_only: bool = False) -> dict[str, Any]:
    reasons: list[str] = []
    usable = 0
    for payload in payloads:
        if payload.get("_read_error"):
            reasons.append(f"{name}_read_error:{payload.get('_path')}:{payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            reasons.append(f"{name}_status_{payload.get('status') or payload.get('gate_decision') or 'failed'}")
            continue
        if require_non_manifest_only and bool(payload.get("manifest_only") or payload.get("dry_run") or payload.get("validation_only")):
            reasons.append(f"{name}_manifest_only")
            continue
        if name == "coverage" and not _positive_any(payload, ("modalities", "coverage_by_modality", "required_modalities", "datasets", "train_files", "records", "counts")):
            reasons.append(f"{name}_missing_coverage_evidence")
            continue
        if name == "reportable_scores" and not _positive_any(payload, ("official", "official_scores", "scored_tasks", "reportable_scores", "task_results", "scores")):
            reasons.append(f"{name}_missing_official_scores")
            continue
        if name == "gguf_runtime":
            artifact = payload.get("artifact") or payload.get("gguf_path") or payload.get("path")
            runtime_evidence = _positive_any(payload, ("tokens_per_second", "latency_ms", "prefill_tps", "decode_tps", "peak_vram_gib", "runtime_results"))
            if not artifact:
                reasons.append(f"{name}_missing_artifact")
                continue
            if not runtime_evidence:
                reasons.append(f"{name}_missing_runtime_evidence")
                continue
            peak_vram = _safe_float(payload.get("peak_vram_gib"))
            if peak_vram is not None and peak_vram > 24.0:
                reasons.append(f"{name}_vram_over_24gb")
                continue
        usable += 1
    if usable <= 0:
        reasons.append(f"{name}_missing_or_unusable")
    return {"status": "passed" if not reasons else "failed", "reasons": reasons, "usable_reports": usable}


def model_contract_check(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    usable = 0
    for payload in payloads:
        if payload.get("_read_error"):
            reasons.append(f"model_contract_read_error:{payload.get('_path')}:{payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            reasons.append(f"model_contract_status_{payload.get('status') or 'failed'}")
            continue
        contract = payload.get("release_training_contract") if isinstance(payload.get("release_training_contract"), dict) else payload
        checks = contract.get("checks") if isinstance(contract.get("checks"), dict) else {}
        required = {
            "n_layers": 64,
            "d_model": 4096,
            "mlp_dim": 15360,
            "mtp_heads": 2,
            "vocab_size": 330000,
        }
        for key, expected in required.items():
            bucket = checks.get(key) if isinstance(checks.get(key), dict) else {}
            actual = _safe_int(bucket.get("actual", contract.get(key)))
            if actual != expected:
                reasons.append(f"model_contract_{key}_mismatch")
        if _safe_int(contract.get("target_context_length")) != 1048576:
            reasons.append("model_contract_context_not_1m")
        if str(contract.get("residual_mode") or "").lower() != "block_attnres":
            reasons.append("model_contract_residual_mode_missing")
        if str(contract.get("status") or "").lower() not in {"passed", "ready", "ok", "success", "completed", "accepted"}:
            reasons.append("model_contract_inner_status_failed")
        usable += 1
    if usable <= 0:
        reasons.append("model_contract_missing_or_unusable")
    return {"status": "passed" if not reasons else "failed", "reasons": reasons, "usable_reports": usable}


def context_ladder_check(payloads: list[dict[str, Any]], required_rungs: tuple[int, ...] = DEFAULT_REQUIRED_CONTEXT_RUNGS) -> dict[str, Any]:
    rung_reports: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    for payload in payloads:
        if payload.get("_read_error"):
            reasons.append(f"context_ladder_read_error:{payload.get('_path')}:{payload.get('_read_error')}")
            continue
        if not _status_ok(payload):
            reasons.append(f"context_ladder_status_{payload.get('status') or 'failed'}")
            continue
        raw_rungs = payload.get("rungs", payload.get("context_rungs", payload.get("ladder")))
        if isinstance(raw_rungs, dict):
            iterable = raw_rungs.values()
        elif isinstance(raw_rungs, list):
            iterable = raw_rungs
        else:
            iterable = [payload]
        for item in iterable:
            if not isinstance(item, dict):
                continue
            rung = _safe_int(item.get("seq_len", item.get("context_length", item.get("rung"))))
            if rung is None:
                continue
            loss = _safe_float(item.get("loss", item.get("avg_loss", item.get("heldout_loss"))))
            recall_passed = bool(item.get("recall_passed", item.get("probe_passed", item.get("passed", False))))
            status_passed = _status_ok(item) if item.get("status") or item.get("gate_decision") else recall_passed
            if loss is not None and loss > 0.0 and recall_passed and status_passed:
                rung_reports[int(rung)] = {"loss": loss, "recall_passed": True, "status": "passed"}
    for rung in required_rungs:
        if int(rung) not in rung_reports:
            reasons.append(f"context_rung_missing_{int(rung)}")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "required_rungs": [int(item) for item in required_rungs],
        "passed_rungs": sorted(rung_reports),
    }


def build_proof_gate(
    *,
    target_payloads: list[dict[str, Any]],
    heldout_payloads: list[dict[str, Any]],
    release_gate_payloads: list[dict[str, Any]],
    q4_profile_payloads: list[dict[str, Any]],
    reasoning_profile_payloads: list[dict[str, Any]],
    coverage_payloads: list[dict[str, Any]],
    reportable_payloads: list[dict[str, Any]],
    gguf_payloads: list[dict[str, Any]],
    contract_payloads: list[dict[str, Any]] | None = None,
    context_ladder_payloads: list[dict[str, Any]] | None = None,
    required_modalities: tuple[str, ...] = DEFAULT_REQUIRED_MODALITIES,
    required_media_modalities: tuple[str, ...] = DEFAULT_REQUIRED_MEDIA_MODALITIES,
    min_tokens_per_modality: int = 8,
    max_heldout_loss: float = 40.0,
    min_profile_target_coverage: float = 1.0,
) -> dict[str, Any]:
    target = target_coverage_check(target_payloads, required_modalities, min_tokens_per_modality)
    heldout = heldout_loss_check(heldout_payloads, required_modalities, min_tokens_per_modality, max_heldout_loss)
    release = release_gate_check(release_gate_payloads, (*required_media_modalities, "text", "code", "tool", "ocr"))
    q4 = profile_matrix_check(q4_profile_payloads, required_profiles=DEFAULT_REQUIRED_Q4_PROFILES, min_coverage=min_profile_target_coverage)
    reasoning = profile_matrix_check(reasoning_profile_payloads, required_variants=DEFAULT_REQUIRED_REASONING_VARIANTS, min_coverage=min_profile_target_coverage)
    coverage = simple_status_check(coverage_payloads, name="coverage")
    reportable = simple_status_check(reportable_payloads, name="reportable_scores")
    gguf = simple_status_check(gguf_payloads, name="gguf_runtime", require_non_manifest_only=True)
    contract = model_contract_check(contract_payloads or [])
    context = context_ladder_check(context_ladder_payloads or [])
    checks = {
        "model_contract": contract,
        "target_token_coverage": target,
        "heldout_loss_by_modality": heldout,
        "decode_and_media_release_gate": release,
        "q4_profile": q4,
        "reasoning_profile": reasoning,
        "context_ladder": context,
        "data_coverage": coverage,
        "reportable_scores": reportable,
        "gguf_runtime": gguf,
    }
    blockers = [f"{name}:{reason}" for name, check in checks.items() if check.get("status") != "passed" for reason in check.get("reasons", [])]
    status = "ready" if not blockers else "blocked"
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "status": status,
        "ready_for_full_training": status == "ready",
        "blockers": sorted(set(blockers)),
        "checks": checks,
        "policy": {
            "quality_preservation": "fail_closed; do not trade away model contract for speed",
            "required_modalities": list(required_modalities),
            "required_media_modalities": list(required_media_modalities),
            "heldout_loss_required_by_modality": True,
            "decode_media_artifacts_required": True,
            "q4_runtime_profile_required": True,
            "context_ladder_required": list(DEFAULT_REQUIRED_CONTEXT_RUNGS),
            "gguf_manifest_only_blocks": True,
            "reportable_scores_required": True,
        },
    }


def _csv_tuple(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(_normalize_modality(item) for item in str(value or "").split(",") if item.strip())
    return items or default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed pre-full-training proof gate for Omnicoder 20B readiness artifacts.")
    parser.add_argument("--target-diagnostics", action="append", default=[])
    parser.add_argument("--train-diagnostics", action="append", default=[])
    parser.add_argument("--heldout-sample-loss", action="append", default=[])
    parser.add_argument("--decode-release-gate", action="append", default=[])
    parser.add_argument("--q4-profile-summary", action="append", default=[])
    parser.add_argument("--reasoning-profile-summary", action="append", default=[])
    parser.add_argument("--coverage-report", action="append", default=[])
    parser.add_argument("--reportable-summary", action="append", default=[])
    parser.add_argument("--gguf-runtime-proof", action="append", default=[])
    parser.add_argument("--contract-report", action="append", default=[])
    parser.add_argument("--context-ladder-proof", action="append", default=[])
    parser.add_argument("--required-modalities", default=",".join(DEFAULT_REQUIRED_MODALITIES))
    parser.add_argument("--required-media-modalities", default=",".join(DEFAULT_REQUIRED_MEDIA_MODALITIES))
    parser.add_argument("--min-tokens-per-modality", type=int, default=8)
    parser.add_argument("--max-heldout-loss", type=float, default=40.0)
    parser.add_argument("--min-profile-target-coverage", type=float, default=1.0)
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_payloads = _load_payloads([*args.target_diagnostics, *args.train_diagnostics], max_jsonl_rows=1000)
    report = build_proof_gate(
        target_payloads=target_payloads,
        heldout_payloads=_load_payloads(args.heldout_sample_loss),
        release_gate_payloads=_load_payloads(args.decode_release_gate),
        q4_profile_payloads=_load_payloads(args.q4_profile_summary),
        reasoning_profile_payloads=_load_payloads(args.reasoning_profile_summary),
        coverage_payloads=_load_payloads(args.coverage_report),
        reportable_payloads=_load_payloads(args.reportable_summary),
        gguf_payloads=_load_payloads(args.gguf_runtime_proof),
        contract_payloads=_load_payloads(args.contract_report),
        context_ladder_payloads=_load_payloads(args.context_ladder_proof),
        required_modalities=_csv_tuple(args.required_modalities, DEFAULT_REQUIRED_MODALITIES),
        required_media_modalities=_csv_tuple(args.required_media_modalities, DEFAULT_REQUIRED_MEDIA_MODALITIES),
        min_tokens_per_modality=max(1, int(args.min_tokens_per_modality)),
        max_heldout_loss=float(args.max_heldout_loss),
        min_profile_target_coverage=float(args.min_profile_target_coverage),
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0 if report["ready_for_full_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


JSON_RE = re.compile(r"\{.*\}")
TRAINABILITY_MODALITIES_2026 = (
    "text",
    "vision",
    "tts",
    "audio_music",
    "music",
    "time_space",
    "tool_agent",
    "media_flow",
)
TRAINABILITY_TOKEN_FAMILIES_2026 = (
    "text",
    "vision_semantic",
    "vision_residual",
    "speech_tts",
    "audio_music",
    "music_control",
    "time_space",
    "tool_agent",
    "flow",
)
CHECKPOINT_EVAL_CONTRACT_SCHEMA = "omnicoder.checkpoint_eval_artifact_contract_2026.v1"
EVAL_SUMMARY_SCHEMAS = {
    "omnicoder.pipeline_sample_loss_2026.v1",
    "omnicoder.pipeline_target_token_diagnostics_2026.v1",
    "omnicoder.checkpoint_readiness_2026.v1",
    "omnicoder.media_route_probe_2026.v1",
}
EVAL_SUMMARY_TYPES = {
    "benchmark_summary",
    "benchmark_reportable_summary",
    "prediction_summary",
}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _raw_map(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): raw for key, raw in value.items()}


def _new_loss_quality() -> dict[str, int]:
    return {
        "observations": 0,
        "finite_count": 0,
        "positive_count": 0,
        "non_positive_count": 0,
        "non_finite_count": 0,
        "non_numeric_count": 0,
        "missing_count": 0,
    }


def _record_loss_observation(quality: dict[str, int], value: Any) -> float | None:
    if value is None or value == "":
        quality["missing_count"] += 1
        return None
    quality["observations"] += 1
    try:
        number = float(value)
    except (TypeError, ValueError):
        quality["non_numeric_count"] += 1
        return None
    if not math.isfinite(number):
        quality["non_finite_count"] += 1
        return None
    quality["finite_count"] += 1
    if number <= 0.0:
        quality["non_positive_count"] += 1
    else:
        quality["positive_count"] += 1
    return number


def _series_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "first": None, "last": None, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def iter_json_events(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    raw_text = p.read_text(encoding="utf-8", errors="ignore")
    stripped = raw_text.strip()
    if not stripped:
        return
    try:
        payload = json.loads(stripped)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        candidates = [text]
        match = JSON_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload
                break


def _target_counts(payload: dict[str, Any], key: str) -> dict[str, int]:
    targets = payload.get("targets")
    if isinstance(targets, dict):
        counts = targets.get(key)
        if isinstance(counts, dict):
            return {str(name): _as_int(value) for name, value in counts.items()}
    return {}


def _raw_loss_map(payload: dict[str, Any], key: str) -> dict[str, Any]:
    loss = payload.get("loss")
    if isinstance(loss, dict):
        return _raw_map(loss.get(key))
    return {}


def _accumulate_bucket(
    buckets: dict[str, dict[str, Any]],
    counts: dict[str, int],
    optimized_counts: dict[str, int],
    raw_losses: dict[str, Any],
) -> None:
    names = set(counts) | set(optimized_counts) | set(raw_losses)
    for name in sorted(names):
        bucket = buckets.setdefault(
            name,
            {
                "target_tokens": 0,
                "optimized_target_tokens": 0,
                "ce": [],
                "loss_quality": _new_loss_quality(),
            },
        )
        bucket["target_tokens"] += int(counts.get(name, 0))
        bucket["optimized_target_tokens"] += int(optimized_counts.get(name, 0))
        if name in raw_losses:
            number = _record_loss_observation(bucket["loss_quality"], raw_losses[name])
            if number is not None:
                bucket["ce"].append(number)
        elif name in counts or name in optimized_counts:
            bucket["loss_quality"]["missing_count"] += 1


def _finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for name, bucket in sorted(buckets.items()):
        target_tokens = int(bucket["target_tokens"])
        optimized_target_tokens = int(bucket["optimized_target_tokens"])
        finalized[name] = {
            "target_tokens": target_tokens,
            "optimized_target_tokens": optimized_target_tokens,
            "optimized_target_coverage": (
                float(optimized_target_tokens) / float(target_tokens)
                if target_tokens > 0
                else None
            ),
            "has_target_tokens": target_tokens > 0,
            "has_optimized_target_tokens": optimized_target_tokens > 0,
            "optimized_exceeds_target_tokens": target_tokens >= 0 and optimized_target_tokens > target_tokens,
            "ce": _series_summary([float(value) for value in bucket["ce"]]),
            "loss_quality": {key: int(value) for key, value in bucket["loss_quality"].items()},
        }
    return finalized


def _coverage_summary(
    buckets: dict[str, dict[str, Any]],
    required_names: Iterable[str],
) -> dict[str, Any]:
    required = sorted({str(name) for name in required_names})
    all_names = sorted(set(required) | set(buckets))
    missing_target = [
        name
        for name in required
        if int(buckets.get(name, {}).get("target_tokens", 0) or 0) <= 0
    ]
    missing_optimized = [
        name
        for name in required
        if int(buckets.get(name, {}).get("optimized_target_tokens", 0) or 0) <= 0
    ]
    optimized_exceeds_target = [
        name
        for name in all_names
        if bool(buckets.get(name, {}).get("optimized_exceeds_target_tokens"))
    ]
    missing_loss = [
        name
        for name in all_names
        if int(buckets.get(name, {}).get("target_tokens", 0) or 0) > 0
        and int(buckets.get(name, {}).get("ce", {}).get("count", 0) or 0) <= 0
    ]
    non_positive_loss = [
        name
        for name in all_names
        if int(buckets.get(name, {}).get("loss_quality", {}).get("non_positive_count", 0) or 0) > 0
    ]
    non_finite_loss = [
        name
        for name in all_names
        if int(buckets.get(name, {}).get("loss_quality", {}).get("non_finite_count", 0) or 0) > 0
    ]
    non_numeric_loss = [
        name
        for name in all_names
        if int(buckets.get(name, {}).get("loss_quality", {}).get("non_numeric_count", 0) or 0) > 0
    ]
    return {
        "required": required,
        "covered_target": [
            name
            for name in all_names
            if int(buckets.get(name, {}).get("target_tokens", 0) or 0) > 0
        ],
        "covered_optimized": [
            name
            for name in all_names
            if int(buckets.get(name, {}).get("optimized_target_tokens", 0) or 0) > 0
        ],
        "missing_target": missing_target,
        "missing_optimized": missing_optimized,
        "optimized_exceeds_target": optimized_exceeds_target,
        "missing_loss": missing_loss,
        "non_positive_loss": non_positive_loss,
        "non_finite_loss": non_finite_loss,
        "non_numeric_loss": non_numeric_loss,
    }


def _trainability_summary(
    *,
    diagnostic_events: int,
    by_modality: dict[str, dict[str, Any]],
    by_token_family: dict[str, dict[str, Any]],
    total_loss_quality: dict[str, int],
) -> dict[str, Any]:
    modality_coverage = _coverage_summary(by_modality, TRAINABILITY_MODALITIES_2026)
    family_coverage = _coverage_summary(by_token_family, TRAINABILITY_TOKEN_FAMILIES_2026)
    reasons: list[str] = []
    if int(diagnostic_events) <= 0:
        reasons.append("no_train_diagnostic_events")
    for key in ("non_positive_count", "non_finite_count", "non_numeric_count", "missing_count"):
        if int(total_loss_quality.get(key, 0) or 0) > 0:
            label = key.replace("_count", "")
            reasons.append(f"total_ce_{label}")
    for scope, coverage in (("modality", modality_coverage), ("token_family", family_coverage)):
        for name in coverage["missing_target"]:
            reasons.append(f"{scope}:{name}:missing_target_tokens")
        for name in coverage["missing_optimized"]:
            reasons.append(f"{scope}:{name}:missing_optimized_target_tokens")
        for name in coverage["optimized_exceeds_target"]:
            reasons.append(f"{scope}:{name}:optimized_target_tokens_exceed_target_tokens")
        for name in coverage["missing_loss"]:
            reasons.append(f"{scope}:{name}:missing_ce")
        for name in coverage["non_positive_loss"]:
            reasons.append(f"{scope}:{name}:non_positive_ce")
        for name in coverage["non_finite_loss"]:
            reasons.append(f"{scope}:{name}:non_finite_ce")
        for name in coverage["non_numeric_loss"]:
            reasons.append(f"{scope}:{name}:non_numeric_ce")
    reasons = sorted(set(reasons))
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "loss_quality": {key: int(value) for key, value in total_loss_quality.items()},
        "coverage": {
            "modalities": modality_coverage,
            "token_families": family_coverage,
        },
    }


def _count_by_key(items: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _compact_overall(overall: Any) -> dict[str, Any]:
    if not isinstance(overall, dict):
        return {}
    keys = (
        "records",
        "samples",
        "tokens",
        "target_tokens",
        "avg_loss",
        "loss",
        "loss_sum",
        "perplexity",
    )
    compact: dict[str, Any] = {}
    for key in keys:
        if key not in overall:
            continue
        number = _as_float(overall.get(key))
        compact[key] = number if number is not None else overall.get(key)
    return compact


def _compact_eval_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    schema = str(payload.get("schema") or "")
    summary_type = str(payload.get("type") or "")
    if schema not in EVAL_SUMMARY_SCHEMAS and summary_type not in EVAL_SUMMARY_TYPES:
        return None
    modalities = payload.get("modalities") if isinstance(payload.get("modalities"), dict) else {}
    modality_names = sorted(str(name) for name in modalities)
    checkpoint = payload.get("checkpoint") or payload.get("checkpoint_path") or payload.get("resume_checkpoint")
    return {
        "schema": schema or None,
        "type": summary_type or None,
        "status": payload.get("status"),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "overall": _compact_overall(payload.get("overall")),
        "modality_count": len(modality_names),
        "modalities": modality_names,
        "result_count": _as_int(payload.get("total_results")),
        "reportable": _as_int(payload.get("reportable")),
        "failed": _as_int(payload.get("failed")),
        "skipped": _as_int(payload.get("skipped")),
        "local_only": _as_int(payload.get("local_only")),
        "gate_decision": payload.get("gate_decision"),
    }


def _compact_checkpoint_contract(payload: dict[str, Any]) -> dict[str, Any] | None:
    contract = payload.get("checkpoint_eval_artifact_contract")
    if not isinstance(contract, dict) and payload.get("schema") == CHECKPOINT_EVAL_CONTRACT_SCHEMA:
        contract = payload
    if not isinstance(contract, dict):
        return None
    artifacts = contract.get("artifacts") if isinstance(contract.get("artifacts"), list) else []
    compact_artifacts: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        compact_artifacts.append(
            {
                "name": artifact.get("name"),
                "required": bool(artifact.get("required")),
                "path": str(artifact.get("path") or ""),
                "schema": artifact.get("schema"),
                "must_include": artifact.get("must_include") if isinstance(artifact.get("must_include"), list) else [],
            }
        )
    return {
        "schema": str(contract.get("schema") or ""),
        "status": contract.get("status"),
        "training_invoked": bool(contract.get("training_invoked")),
        "checkpoint_dir": str(contract.get("checkpoint_dir") or ""),
        "artifact_count": len(compact_artifacts),
        "required_artifact_count": sum(1 for item in compact_artifacts if item["required"]),
        "artifacts": compact_artifacts,
    }


def summarize_train_diagnostics_log(path: str | Path) -> dict[str, Any]:
    modality_buckets: dict[str, dict[str, Any]] = {}
    family_buckets: dict[str, dict[str, Any]] = {}
    total_ce: list[float] = []
    ranks: set[int] = set()
    world_sizes: set[int] = set()
    global_steps: list[int] = []
    local_steps: list[int] = []
    runtime_tokens = 0
    runtime_elapsed = 0.0
    runtime_tps: list[float] = []
    total_ce_quality = _new_loss_quality()
    events = 0
    last_event: dict[str, Any] | None = None

    for payload in iter_json_events(path):
        targets = payload.get("targets")
        loss = payload.get("loss")
        is_diagnostic = (
            payload.get("event") == "train_step"
            and isinstance(targets, dict)
            and isinstance(loss, dict)
        )
        if not is_diagnostic:
            continue
        events += 1
        last_event = payload
        ranks.add(_as_int(payload.get("rank")))
        world_sizes.add(_as_int(payload.get("world_size")))
        if payload.get("global_step") is not None:
            global_steps.append(_as_int(payload.get("global_step")))
        if payload.get("local_step") is not None:
            local_steps.append(_as_int(payload.get("local_step")))
        ce = _record_loss_observation(total_ce_quality, loss.get("total_ce"))
        if ce is not None:
            total_ce.append(ce)
        runtime = payload.get("runtime")
        if isinstance(runtime, dict):
            runtime_tokens += _as_int(runtime.get("tokens"))
            elapsed = _as_float(runtime.get("elapsed_sec"))
            if elapsed is not None:
                runtime_elapsed += elapsed
            tps = _as_float(runtime.get("tokens_per_sec"))
            if tps is not None:
                runtime_tps.append(tps)
        _accumulate_bucket(
            modality_buckets,
            _target_counts(payload, "by_modality"),
            _target_counts(payload, "optimized_by_modality"),
            _raw_loss_map(payload, "ce_by_modality"),
        )
        _accumulate_bucket(
            family_buckets,
            _target_counts(payload, "by_token_family"),
            _target_counts(payload, "optimized_by_token_family"),
            _raw_loss_map(payload, "ce_by_token_family"),
        )

    by_modality = _finalize_buckets(modality_buckets)
    by_token_family = _finalize_buckets(family_buckets)
    total_target_tokens = sum(bucket["target_tokens"] for bucket in modality_buckets.values())
    total_optimized_target_tokens = sum(bucket["optimized_target_tokens"] for bucket in modality_buckets.values())
    return {
        "path": str(path),
        "diagnostic_events": events,
        "ranks": sorted(ranks),
        "world_sizes": sorted(world_sizes),
        "global_step_first": global_steps[0] if global_steps else None,
        "global_step_last": global_steps[-1] if global_steps else None,
        "global_step_max": max(global_steps) if global_steps else None,
        "local_step_last": local_steps[-1] if local_steps else None,
        "loss_total_ce": _series_summary(total_ce),
        "loss_quality_total_ce": total_ce_quality,
        "total_target_tokens": total_target_tokens,
        "total_optimized_target_tokens": total_optimized_target_tokens,
        "optimized_target_coverage": (
            float(total_optimized_target_tokens) / float(total_target_tokens)
            if total_target_tokens > 0
            else None
        ),
        "by_modality": by_modality,
        "by_token_family": by_token_family,
        "trainability": _trainability_summary(
            diagnostic_events=events,
            by_modality=by_modality,
            by_token_family=by_token_family,
            total_loss_quality=total_ce_quality,
        ),
        "runtime": {
            "tokens": runtime_tokens,
            "elapsed_sec": runtime_elapsed,
            "tokens_per_sec": _series_summary(runtime_tps),
        },
        "last_event": last_event,
    }


def summarize_training_log(path: str | Path) -> dict[str, Any]:
    losses: list[float] = []
    steps: list[int] = []
    eval_summaries: list[dict[str, Any]] = []
    checkpoint_contracts: list[dict[str, Any]] = []
    events = 0
    last: dict[str, Any] | None = None
    for payload in iter_json_events(path):
        events += 1
        last = payload
        eval_summary = _compact_eval_summary(payload)
        if eval_summary is not None:
            eval_summaries.append(eval_summary)
        checkpoint_contract = _compact_checkpoint_contract(payload)
        if checkpoint_contract is not None:
            checkpoint_contracts.append(checkpoint_contract)
        loss_value = _as_float(payload.get("loss"))
        if loss_value is not None:
            losses.append(loss_value)
        if payload.get("step") is not None:
            steps.append(_as_int(payload["step"]))
    summary = {
        "path": str(path),
        "json_events": events,
        "steps": max(steps) if steps else 0,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "last_event": last,
    }
    train_diagnostics = summarize_train_diagnostics_log(path)
    if train_diagnostics["diagnostic_events"] > 0:
        summary["train_diagnostics"] = train_diagnostics
    if eval_summaries:
        summary["eval_summaries"] = {
            "count": len(eval_summaries),
            "schemas": _count_by_key(eval_summaries, "schema"),
            "types": _count_by_key(eval_summaries, "type"),
            "checkpoints": sorted({item["checkpoint"] for item in eval_summaries if item.get("checkpoint")}),
            "items": eval_summaries,
        }
    if checkpoint_contracts:
        summary["checkpoint_reports"] = {
            "eval_artifact_contract_count": len(checkpoint_contracts),
            "checkpoints": sorted(
                {item["checkpoint_dir"] for item in checkpoint_contracts if item.get("checkpoint_dir")}
            ),
            "eval_artifact_contracts": checkpoint_contracts,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Omnicoder 2026 training JSON logs")
    ap.add_argument("--log", required=True)
    ap.add_argument("--train-diagnostics", action="store_true", help="Summarize train-diagnostics JSONL fields by modality and token family")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summary = summarize_train_diagnostics_log(args.log) if args.train_diagnostics else summarize_training_log(args.log)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()

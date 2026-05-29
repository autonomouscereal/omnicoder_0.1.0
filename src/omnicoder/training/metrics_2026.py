from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


JSON_RE = re.compile(r"\{.*\}")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _numeric_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        number = _as_float(raw)
        if number is not None:
            out[str(key)] = number
    return out


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
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
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


def _loss_map(payload: dict[str, Any], key: str) -> dict[str, float]:
    loss = payload.get("loss")
    if isinstance(loss, dict):
        return _numeric_map(loss.get(key))
    return {}


def _accumulate_bucket(
    buckets: dict[str, dict[str, Any]],
    counts: dict[str, int],
    optimized_counts: dict[str, int],
    losses: dict[str, float],
) -> None:
    names = set(counts) | set(optimized_counts) | set(losses)
    for name in sorted(names):
        bucket = buckets.setdefault(name, {"target_tokens": 0, "optimized_target_tokens": 0, "ce": []})
        bucket["target_tokens"] += int(counts.get(name, 0))
        bucket["optimized_target_tokens"] += int(optimized_counts.get(name, 0))
        if name in losses:
            bucket["ce"].append(float(losses[name]))


def _finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for name, bucket in sorted(buckets.items()):
        finalized[name] = {
            "target_tokens": int(bucket["target_tokens"]),
            "optimized_target_tokens": int(bucket["optimized_target_tokens"]),
            "ce": _series_summary([float(value) for value in bucket["ce"]]),
        }
    return finalized


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
        ce = _as_float(loss.get("total_ce"))
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
            _loss_map(payload, "ce_by_modality"),
        )
        _accumulate_bucket(
            family_buckets,
            _target_counts(payload, "by_token_family"),
            _target_counts(payload, "optimized_by_token_family"),
            _loss_map(payload, "ce_by_token_family"),
        )

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
        "total_target_tokens": sum(bucket["target_tokens"] for bucket in modality_buckets.values()),
        "total_optimized_target_tokens": sum(bucket["optimized_target_tokens"] for bucket in modality_buckets.values()),
        "by_modality": _finalize_buckets(modality_buckets),
        "by_token_family": _finalize_buckets(family_buckets),
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
    events = 0
    last: dict[str, Any] | None = None
    for payload in iter_json_events(path):
        events += 1
        last = payload
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

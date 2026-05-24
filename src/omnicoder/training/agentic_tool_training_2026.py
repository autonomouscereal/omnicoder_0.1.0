from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2026-05-23"
TOOL_HINTS = (
    "tool",
    "function",
    "terminal",
    "shell",
    "command",
    "mcp",
    "browser",
    "postgres",
    "api",
    "json",
    "trace",
    "approval",
)
RISK_HINTS = (
    "secret",
    "password",
    "credential",
    "token",
    "delete",
    "destructive",
    "exfiltrate",
    "ignore previous",
    "bypass",
    "hidden test",
    "answer key",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return
    for line_no, line in enumerate(source.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            yield {"parse_error": str(exc), "line_no": line_no, "text": line}
            continue
        if isinstance(payload, dict):
            yield payload


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def record_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        input_json = record.get("input_json") if isinstance(record.get("input_json"), dict) else {}
        target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
        messages = input_json.get("messages")
        if not isinstance(messages, list):
            messages = []
            prompt = input_json.get("prompt") or input_json.get("content") or record.get("text")
            if prompt:
                messages.append({"role": "user", "content": str(prompt)})
            answer = target_json.get("answer") or target_json.get("completion") or target_json.get("content")
            if answer:
                messages.append({"role": "assistant", "content": str(answer)})
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def record_text(record: dict[str, Any]) -> str:
    parts = [message["content"] for message in record_messages(record)]
    for key in ("content", "text", "prompt", "completion", "answer", "normalized_text"):
        value = record.get(key)
        if isinstance(value, str):
            parts.append(value)
    for container_key in ("input_json", "target_json", "lineage", "tool_calls", "tool_results"):
        value = record.get(container_key)
        if isinstance(value, (dict, list)):
            parts.append(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str))
    return "\n".join(parts)


def trace_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else metadata.get("lineage", {})
    if not isinstance(lineage, dict):
        lineage = {}
    for key in ("trace_id", "session_id", "conversation_id", "record_hash", "curated_id"):
        value = record.get(key) or metadata.get(key) or lineage.get(key)
        if value:
            return str(value)
    return stable_hash(record)[:24]


def quality_score(record: dict[str, Any]) -> float:
    for container in (record.get("quality"), record.get("metadata", {}).get("quality") if isinstance(record.get("metadata"), dict) else None):
        if isinstance(container, dict):
            for key in ("score", "overall", "avg", "quality"):
                if container.get(key) is not None:
                    try:
                        return float(container[key])
                    except Exception:
                        pass
    return 1.0


def has_hidden_material(record: dict[str, Any]) -> bool:
    contamination = record.get("contamination") if isinstance(record.get("contamination"), dict) else {}
    if contamination.get("status") == "contaminated":
        return True
    text = record_text(record).lower()
    return "hidden test" in text or "answer key" in text or "gold patch" in text


def has_tool_signal(record: dict[str, Any]) -> bool:
    if record.get("tool_calls") or record.get("tool_results"):
        return True
    messages = record_messages(record)
    if any(message["role"] == "tool" for message in messages):
        return True
    text = record_text(record).lower()
    return any(hint in text for hint in TOOL_HINTS)


def risk_labels(record: dict[str, Any]) -> list[str]:
    text = record_text(record).lower()
    labels = [hint.replace(" ", "_") for hint in RISK_HINTS if hint in text]
    if has_hidden_material(record):
        labels.append("protected_eval_material")
    return sorted(set(labels))


def extract_json_objects(text: str, limit: int = 8) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{[^{}]{2,2000}\}", text, flags=re.DOTALL):
        if len(objects) >= limit:
            break
        try:
            payload = json.loads(match.group(0))
        except Exception:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
    return objects


def tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    raw = record.get("tool_calls")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                calls.append(item)
    for message in record_messages(record):
        if message["role"] == "assistant":
            for payload in extract_json_objects(message["content"]):
                if any(key in payload for key in ("tool", "tool_name", "name", "arguments", "args")):
                    calls.append(payload)
    return calls


def tool_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    raw = record.get("tool_results")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                results.append(item)
            elif isinstance(item, str):
                results.append({"content": item})
    for message in record_messages(record):
        if message["role"] == "tool":
            results.append({"content": message["content"]})
    return results


def normalize_messages_for_tools(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record_messages(record)
    if messages:
        return messages
    text = record_text(record).strip()
    return [{"role": "user", "content": text}] if text else []


def tool_reward(record: dict[str, Any], risks: list[str]) -> float:
    reward = quality_score(record)
    if tool_calls(record) or tool_results(record):
        reward += 0.15
    if risks:
        reward -= min(0.8, 0.18 * len(risks))
    if has_hidden_material(record):
        reward = min(reward, 0.0)
    return max(-1.0, min(1.0, round(reward, 4)))


def source_date(record: dict[str, Any]) -> str | None:
    if record.get("source_date"):
        return str(record["source_date"])
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if metadata.get("source_date"):
        return str(metadata["source_date"])
    return None


def eligible(record: dict[str, Any], min_quality: float) -> bool:
    if has_hidden_material(record):
        return False
    if quality_score(record) < min_quality:
        return False
    return has_tool_signal(record) and bool(normalize_messages_for_tools(record))


def build_rows(records: Iterable[dict[str, Any]], min_quality: float, limit: int = 0) -> dict[str, list[dict[str, Any]]]:
    outputs = {"sft": [], "preference": [], "reward": [], "rlvr": [], "safety": []}
    for record in records:
        if limit and len(outputs["sft"]) >= limit:
            break
        risks = risk_labels(record)
        if risks and has_tool_signal(record):
            outputs["safety"].append(build_safety_row(record, risks))
        if not eligible(record, min_quality):
            continue
        calls = tool_calls(record)
        results = tool_results(record)
        reward = tool_reward(record, risks)
        base = {
            "schema": "omnicoder.agentic_tool_training_2026.v1",
            "trace_id": trace_id(record),
            "record_hash": stable_hash(record),
            "source_date": source_date(record),
            "tool_calls": calls,
            "tool_results": results,
            "risk_labels": risks,
            "quality_score": quality_score(record),
        }
        outputs["sft"].append(
            {
                **base,
                "training_kind": "tool_sft",
                "messages": normalize_messages_for_tools(record),
                "metadata": {
                    "assistant_only_loss": True,
                    "tool_schema_masking": True,
                    "state_tracking": bool(results),
                },
            }
        )
        outputs["reward"].append(
            {
                **base,
                "training_kind": "tool_reward",
                "prompt": record_text(record)[:20000],
                "reward": reward,
                "reward_components": {
                    "quality": quality_score(record),
                    "tool_call_count": len(calls),
                    "tool_result_count": len(results),
                    "risk_penalty": len(risks),
                },
            }
        )
        chosen = json.dumps({"tool_calls": calls, "final": normalize_messages_for_tools(record)[-1]}, ensure_ascii=True, sort_keys=True)
        rejected = json.dumps({"tool_calls": [], "final": "No valid tool plan."}, ensure_ascii=True, sort_keys=True)
        outputs["preference"].append(
            {
                **base,
                "training_kind": "tool_preference",
                "prompt": record_text(record)[:20000],
                "chosen": chosen,
                "rejected": rejected,
                "preference_reason": "Prefer valid tool calls with state-aware use and no protected material.",
            }
        )
        outputs["rlvr"].append(
            {
                **base,
                "training_kind": "tool_rlvr",
                "prompt": record_text(record)[:20000],
                "verifier": {
                    "checks": [
                        "tool_schema_valid",
                        "state_update_consistent",
                        "no_protected_material",
                        "no_credential_leak",
                        "task_outcome_passed",
                    ],
                    "reward": reward,
                },
                "environment": {
                    "families": ["bfcl", "tau", "mcpmark", "terminal_bench", "internal_traces"],
                    "rollout_policy": "group_relative_with_verifiable_rewards",
                },
            }
        )
    return outputs


def build_safety_row(record: dict[str, Any], risks: list[str]) -> dict[str, Any]:
    return {
        "schema": "omnicoder.agentic_tool_training_2026.v1",
        "training_kind": "tool_safety_negative",
        "trace_id": trace_id(record),
        "record_hash": stable_hash(record),
        "prompt": record_text(record)[:20000],
        "risk_labels": risks,
        "chosen": "Refuse unsafe tool use, protect credentials, and keep protected evaluation material out of training.",
        "rejected": record_text(record)[:2000],
        "reward": -1.0,
    }


def posttrain_manifest(algorithm: str, train_jsonl: Path, out_dir: Path, model: str, dry_run: bool) -> dict[str, Any]:
    return {
        "schema": "omnicoder.agentic_tool_posttrain_manifest_2026.v1",
        "algorithm": algorithm,
        "model": model,
        "train_jsonl": str(train_jsonl),
        "out_dir": str(out_dir / algorithm),
        "dry_run": dry_run,
        "tool_training_contract": {
            "assistant_only_loss": algorithm == "sft",
            "tool_schema_masking": True,
            "state_tracking_rewards": algorithm in {"reward", "grpo", "rloo"},
            "safety_negatives": True,
            "q4_recovery_ready": True,
        },
    }


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    cfg = profile.get("agentic_tool_training") if isinstance(profile.get("agentic_tool_training"), dict) else {}
    min_quality = float(args.min_quality if args.min_quality is not None else cfg.get("min_quality", 0.0))
    limit = int(args.limit or cfg.get("limit") or 0)
    out_dir = Path(args.out_dir or cfg.get("out_dir") or "weights/agentic_tool_training_2026")
    source = Path(args.input or cfg.get("input_jsonl") or "")
    if not source.exists():
        raise SystemExit(json.dumps({"status": "error", "error": "input_jsonl not found", "input": str(source)}))
    rows = build_rows(iter_jsonl(source), min_quality=min_quality, limit=limit)
    paths = {
        "sft": out_dir / "tool_sft.jsonl",
        "preference": out_dir / "tool_preference.jsonl",
        "reward": out_dir / "tool_reward.jsonl",
        "rlvr": out_dir / "tool_rlvr.jsonl",
        "safety": out_dir / "tool_safety_negatives.jsonl",
    }
    counts = {name: write_jsonl(paths[name], rows[name]) for name in paths}
    model = str(args.model or cfg.get("model") or profile.get("base_model") or "Qwen/Qwen3-4B")
    bridge_dir = out_dir / "posttrain_manifests"
    bridge_rows = {
        "sft": posttrain_manifest("sft", paths["sft"], bridge_dir, model, bool(args.dry_run)),
        "reward": posttrain_manifest("reward", paths["reward"], bridge_dir, model, bool(args.dry_run)),
        "dpo": posttrain_manifest("dpo", paths["preference"], bridge_dir, model, bool(args.dry_run)),
        "grpo": posttrain_manifest("grpo", paths["rlvr"], bridge_dir, model, bool(args.dry_run)),
        "kto": posttrain_manifest("kto", paths["safety"], bridge_dir, model, bool(args.dry_run)),
    }
    bridge_paths: dict[str, str] = {}
    for name, payload in bridge_rows.items():
        path = bridge_dir / f"{name}_tool_manifest.json"
        write_json(path, payload)
        bridge_paths[name] = str(path)
    manifest = {
        "schema": "omnicoder.agentic_tool_training_manifest_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "source": str(source),
        "out_dir": str(out_dir),
        "counts": counts,
        "paths": {name: str(path) for name, path in paths.items()},
        "posttrain_manifests": bridge_paths,
        "training_sequence": ["tool_sft", "tool_reward", "tool_preference", "tool_rlvr", "tool_safety_negative"],
        "release_gate_links": ["bfcl_v4", "tau3", "mcpmark", "terminal_bench", "safety_tool_security"],
    }
    manifest_path = out_dir / "agentic_tool_training_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def validate_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    cfg = profile.get("agentic_tool_training") if isinstance(profile.get("agentic_tool_training"), dict) else profile
    required = ["input_jsonl", "out_dir", "min_quality", "stages"]
    missing = [key for key in required if key not in cfg]
    return {
        "status": "ok" if not missing else "missing_config",
        "missing": missing,
        "stages": ensure_list(cfg.get("stages")),
        "reward_axes": ensure_list(cfg.get("reward_axes")),
        "safety_negatives": bool(cfg.get("safety_negatives", True)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build agentic tool-calling training artifacts")
    parser.add_argument("--profile", default="profiles/agentic_tool_training_2026.json")
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate")
    val.set_defaults(func=validate_profile)

    build = sub.add_parser("build")
    build.add_argument("--input", default=None)
    build.add_argument("--out-dir", default=None)
    build.add_argument("--model", default=None)
    build.add_argument("--min-quality", type=float, default=None)
    build.add_argument("--limit", type=int, default=0)
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=run_build)

    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

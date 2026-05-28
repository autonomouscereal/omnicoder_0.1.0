from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2026-05-24"
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
MATH_HINTS = ("solve", "equation", "proof", "theorem", "latex", "\\boxed", "olympiad", "aime", "gsm8k", "math")
CODE_HINTS = ("pytest", "unit test", "def ", "class ", "traceback", "patch", "diff --git", "compiler", "leetcode", "codeforces")
TERMINAL_HINTS = ("exit_code", "stdout", "stderr", "powershell", "bash", "cmd.exe", "shell", "terminal", "docker", "nvidia-smi")
BROWSER_HINTS = ("browser", "playwright", "click", "screenshot", "url", "citation", "websearch", "web_research", "http://", "https://")
DOMAIN_DEFAULTS = {
    "math": {
        "checks": ["final_answer_exact", "symbolic_equivalence", "numeric_tolerance", "no_answer_key_leak"],
        "reward_axes": ["math_answer_exact", "reasoning_step_validity", "format_compliance"],
    },
    "code": {
        "checks": ["unit_tests_pass", "no_hidden_tests_leak", "patch_applies", "no_forbidden_dependency"],
        "reward_axes": ["code_tests_passed", "minimal_patch", "runtime_safety", "style_consistency"],
    },
    "terminal": {
        "checks": ["exit_code_zero", "stdout_matches", "filesystem_state_matches", "no_destructive_unapproved_action"],
        "reward_axes": ["terminal_exit_success", "command_relevance", "state_change_correctness", "recovery_after_error"],
    },
    "browser": {
        "checks": ["answer_exact", "citation_supports_claim", "page_state_reached", "no_stale_source"],
        "reward_axes": ["browser_answer_exactness", "citation_support", "navigation_efficiency", "source_freshness_2026"],
    },
    "tool": {
        "checks": ["tool_schema_valid", "argument_exactness", "state_update_consistent", "task_outcome_passed"],
        "reward_axes": ["tool_schema_valid", "argument_exactness", "state_update_consistency", "task_outcome"],
    },
}
DEFAULT_REWARD_WEIGHTS = {
    "quality": 0.15,
    "schema_valid": 0.12,
    "argument_exactness": 0.12,
    "state_consistency": 0.12,
    "outcome_passed": 0.22,
    "domain_verifier": 0.20,
    "rollout_efficiency": 0.07,
    "risk_penalty": -0.35,
}
PURE_VERIFIABLE_RLVR_DOMAINS = {"math", "code"}


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
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                yield {"parse_error": str(exc), "line_no": line_no, "text": line.rstrip("\n")}
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


def first_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("content", "text", "answer", "response", "completion", "final", "reason"):
            text = first_string(value.get(key))
            if text:
                return text
    if isinstance(value, list):
        parts = [first_string(item) for item in value[:8]]
        return "\n".join(part for part in parts if part)
    return ""


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


def task_domains(record: dict[str, Any]) -> list[str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    explicit = record.get("domains") or metadata.get("domains") or record.get("task_family") or metadata.get("task_family")
    domains: set[str] = set()
    if isinstance(explicit, str):
        domains.add(explicit.lower().replace("tool_use", "tool"))
    elif isinstance(explicit, list):
        domains.update(str(item).lower().replace("tool_use", "tool") for item in explicit)
    text = record_text(record).lower()
    if any(hint in text for hint in MATH_HINTS):
        domains.add("math")
    if any(hint in text for hint in CODE_HINTS):
        domains.add("code")
    if any(hint in text for hint in TERMINAL_HINTS):
        domains.add("terminal")
    if any(hint in text for hint in BROWSER_HINTS):
        domains.add("browser")
    if has_tool_signal(record):
        domains.add("tool")
    return sorted(domain for domain in domains if domain in DOMAIN_DEFAULTS)


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


def parse_json_values(text: str, limit: int = 4) -> list[Any]:
    values: list[Any] = []
    decoder = json.JSONDecoder()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    for candidate in candidates:
        probe = candidate.strip()
        if not probe:
            continue
        starts = [idx for idx, char in enumerate(probe) if char in "[{"]
        if 0 not in starts:
            starts.insert(0, 0)
        for start in starts:
            if len(values) >= limit:
                return values
            try:
                value, _ = decoder.raw_decode(probe[start:])
            except Exception:
                continue
            values.append(value)
    return values


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def teacher_signal(record: dict[str, Any]) -> dict[str, Any]:
    target_json = record.get("target_json") if isinstance(record.get("target_json"), dict) else {}
    content = str(target_json.get("content") or target_json.get("completion") or target_json.get("answer") or "")
    if not content:
        return {}
    parsed: dict[str, Any] = target_json.get("teacher_signal") if isinstance(target_json.get("teacher_signal"), dict) else {}
    for value in parse_json_values(content):
        if isinstance(value, dict):
            parsed = {**parsed, **value}
            break
    corrected_response = first_string(
        parsed.get("corrected_response")
        or parsed.get("corrected_answer")
        or parsed.get("final")
        or parsed.get("answer")
        or parsed.get("response")
    )
    if not corrected_response:
        corrected_response = content.strip()
    corrected_tool_calls = (
        _coerce_list_of_dicts(parsed.get("corrected_tool_calls"))
        or _coerce_list_of_dicts(parsed.get("tool_calls"))
        or _coerce_list_of_dicts(parsed.get("actions"))
    )
    reward_components = parsed.get("reward_components") or parsed.get("reward_vector") or parsed.get("scores")
    if not isinstance(reward_components, dict):
        reward_components = {}
    reward = _coerce_float(parsed.get("reward") or parsed.get("score") or parsed.get("quality_score"), 0.75 if str(target_json.get("teacher_status") or "").lower() == "ok" else 0.0)
    verifier_labels = parsed.get("verifier_labels") or parsed.get("verifier") or parsed.get("checks") or parsed.get("process_labels")
    if isinstance(verifier_labels, dict):
        verifier_labels = [verifier_labels]
    elif not isinstance(verifier_labels, list):
        verifier_labels = []
    chosen = first_string(parsed.get("chosen")) or corrected_response
    rejected = first_string(parsed.get("rejected")) or first_string(parsed.get("negative")) or "No valid tool plan."
    return {
        "is_teacher_rollout": str(record.get("schema") or "").startswith("omnicoder.openai_teacher_rollout")
        or "teacher_status" in target_json,
        "corrected_response": corrected_response,
        "corrected_tool_calls": corrected_tool_calls,
        "chosen": chosen,
        "rejected": rejected,
        "reward": max(-1.0, min(1.0, reward)),
        "reward_components": {str(key): _coerce_float(value, 0.0) for key, value in reward_components.items()},
        "verifier_labels": verifier_labels,
        "raw_content_hash": stable_hash(content),
    }


def tool_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    signal = teacher_signal(record)
    if signal.get("corrected_tool_calls"):
        calls.extend(_coerce_list_of_dicts(signal.get("corrected_tool_calls")))
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


def _numeric_ratio(value: Any, total: Any) -> float | None:
    try:
        numerator = float(value)
        denominator = float(total)
    except (TypeError, ValueError):
        return None
    if denominator <= 0:
        return None
    return max(0.0, min(1.0, numerator / denominator))


def reward_components(
    record: dict[str, Any],
    calls: list[dict[str, Any]],
    results: list[dict[str, Any]],
    risks: list[str],
    domains: list[str],
) -> dict[str, float]:
    text = record_text(record).lower()
    components: dict[str, float] = {
        "quality": max(0.0, min(1.0, quality_score(record))),
        "schema_valid": 1.0 if calls or results else 0.0,
        "argument_exactness": 1.0 if all(isinstance(call, dict) for call in calls) else 0.0,
        "state_consistency": 1.0 if results else 0.5 if calls else 0.0,
        "outcome_passed": 1.0 if any(marker in text for marker in ("passed", "\"ok\"", "success", "exit_code\":0", "exit_code\": 0")) else 0.0,
        "risk_penalty": 1.0 if risks or has_hidden_material(record) else 0.0,
        "rollout_efficiency": max(0.0, 1.0 - min(1.0, len(calls) / 24.0)),
        "domain_verifier": 0.0,
    }
    if "math" in domains:
        exact = any(marker in text for marker in ("\\boxed", "final answer", "ground_truth", "answer"))
        components["math_answer_exact"] = 1.0 if exact else 0.0
        components["domain_verifier"] = max(components["domain_verifier"], components["math_answer_exact"])
    if "code" in domains:
        ratio = None
        for result in results:
            if isinstance(result, dict):
                ratio = _numeric_ratio(result.get("tests_passed"), result.get("tests_total"))
                if ratio is not None:
                    break
        if ratio is None:
            ratio = 1.0 if any(marker in text for marker in ("pytest passed", "tests passed", "pass_to_pass")) else 0.0
        components["code_tests_passed"] = ratio
        components["domain_verifier"] = max(components["domain_verifier"], ratio)
    if "terminal" in domains:
        exit_success = 1.0 if any(("exit_code" in json.dumps(result, ensure_ascii=True).lower() and ": 0" in json.dumps(result, ensure_ascii=True)) or result.get("exit_code") == 0 for result in results if isinstance(result, dict)) else 0.0
        components["terminal_exit_success"] = exit_success
        components["domain_verifier"] = max(components["domain_verifier"], exit_success)
    if "browser" in domains:
        support = 1.0 if any(marker in text for marker in ("citation", "source", "url", "screenshot")) else 0.0
        components["browser_evidence_support"] = support
        components["browser_answer_exactness"] = 1.0 if "answer" in text and support else 0.0
        components["domain_verifier"] = max(components["domain_verifier"], support)
    if "tool" in domains:
        components["tool_schema_valid"] = components["schema_valid"]
        components["task_outcome"] = components["outcome_passed"]
    return {key: round(float(value), 6) for key, value in components.items()}


def compose_reward(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    active_weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    reward = 0.0
    for axis, value in components.items():
        if axis == "risk_penalty":
            reward += active_weights.get("risk_penalty", -0.35) * value
        else:
            reward += active_weights.get(axis, 0.0) * value
    if not any(axis in active_weights for axis in components):
        reward = components.get("quality", 0.0)
    return max(-1.0, min(1.0, round(reward, 4)))


def tool_reward(
    record: dict[str, Any],
    risks: list[str],
    components: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    if components is None:
        calls = tool_calls(record)
        results = tool_results(record)
        components = reward_components(record, calls, results, risks, task_domains(record))
    if has_hidden_material(record):
        components = {**components, "risk_penalty": 1.0, "domain_verifier": min(components.get("domain_verifier", 0.0), 0.0)}
    return compose_reward(components, weights)


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
    if not normalize_messages_for_tools(record):
        return False
    return has_tool_signal(record) or bool(pure_verifiable_rlvr_domains(record))


def pure_verifiable_rlvr_domains(record: dict[str, Any]) -> list[str]:
    if has_tool_signal(record):
        return []
    domains = [domain for domain in task_domains(record) if domain in PURE_VERIFIABLE_RLVR_DOMAINS]
    if not domains:
        return []
    components = reward_components(record, [], [], risk_labels(record), domains)
    verified: list[str] = []
    for domain in domains:
        if domain == "math" and components.get("math_answer_exact", 0.0) > 0.0:
            verified.append(domain)
        if domain == "code" and components.get("code_tests_passed", 0.0) > 0.0:
            verified.append(domain)
    return verified


def domain_config(profile_cfg: dict[str, Any] | None, domain: str) -> dict[str, Any]:
    rlvr = profile_cfg.get("rlvr_domains") if isinstance(profile_cfg, dict) and isinstance(profile_cfg.get("rlvr_domains"), dict) else {}
    configured = rlvr.get(domain) if isinstance(rlvr.get(domain), dict) else {}
    defaults = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS["tool"])
    return {
        "enabled": bool(configured.get("enabled", True)),
        "checks": list(configured.get("checks") or defaults["checks"]),
        "reward_axes": list(configured.get("reward_axes") or defaults["reward_axes"]),
        "export": str(configured.get("export") or f"{domain}_rlvr.jsonl"),
    }


def build_rlvr_row(
    record: dict[str, Any],
    base: dict[str, Any],
    reward: float,
    components: dict[str, float],
    domains: list[str],
    profile_cfg: dict[str, Any] | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    active_domains = [domain] if domain else domains
    checks: list[str] = []
    axes: list[str] = []
    for item in active_domains:
        cfg = domain_config(profile_cfg, item)
        checks.extend(str(check) for check in cfg["checks"])
        axes.extend(str(axis) for axis in cfg["reward_axes"])
    return {
        **base,
        "training_kind": "tool_rlvr" if domain is None else f"{domain}_rlvr",
        "domains": active_domains,
        "prompt": record_text(record)[:20000],
        "reward": reward,
        "reward_components": components,
        "verifier": {
            "checks": sorted(set(checks or DOMAIN_DEFAULTS["tool"]["checks"])),
            "reward_axes": sorted(set(axes or DOMAIN_DEFAULTS["tool"]["reward_axes"])),
            "expected_artifacts": ["tool_trace", "state_delta", "test_or_observation_evidence"],
            "reward": reward,
        },
        "environment": {
            "family": domain or "mixed_agentic_tool",
            "families": ["bfcl", "tau", "mcpmark", "terminal_bench", "swe_gym", "internal_traces"],
            "sandbox": "container_or_mocked_tool_environment",
            "timeout_s": 600,
            "rollout_policy": "group_relative_with_verifiable_rewards",
        },
    }


def build_rows(
    records: Iterable[dict[str, Any]],
    min_quality: float,
    limit: int = 0,
    profile_cfg: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    outputs = {"sft": [], "preference": [], "reward": [], "rlvr": [], "safety": []}
    for domain in DOMAIN_DEFAULTS:
        outputs[f"{domain}_rlvr"] = []
    for record in records:
        if limit and len(outputs["sft"]) >= limit:
            break
        for name, row in rows_for_record(record, min_quality, profile_cfg).items():
            if isinstance(row, list):
                outputs.setdefault(name, []).extend(row)
    return outputs


def rows_for_record(
    record: dict[str, Any],
    min_quality: float,
    profile_cfg: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {"sft": [], "preference": [], "reward": [], "rlvr": [], "safety": []}
    for domain in DOMAIN_DEFAULTS:
        outputs[f"{domain}_rlvr"] = []
    risks = risk_labels(record)
    safety_negatives_enabled = bool(profile_cfg.get("safety_negatives", False)) if isinstance(profile_cfg, dict) else False
    if safety_negatives_enabled and risks and has_tool_signal(record):
        outputs["safety"].append(build_safety_row(record, risks))
    if not eligible(record, min_quality):
        return outputs
    is_tool_row = has_tool_signal(record)
    reward_weights = profile_cfg.get("reward_weights") if isinstance(profile_cfg, dict) and isinstance(profile_cfg.get("reward_weights"), dict) else {}
    signal = teacher_signal(record)
    calls = tool_calls(record)
    results = tool_results(record)
    domains = task_domains(record) or ["tool"]
    if not is_tool_row:
        domains = pure_verifiable_rlvr_domains(record)
    messages = normalize_messages_for_tools(record)
    if signal.get("is_teacher_rollout") and signal.get("corrected_response"):
        user_messages = [message for message in messages if message["role"] in {"system", "user"}]
        if user_messages:
            messages = user_messages + [{"role": "assistant", "content": str(signal["corrected_response"])}]
    if not messages:
        return outputs
    text = record_text(record)[:20000]
    components = reward_components(record, calls, results, risks, domains)
    if signal.get("is_teacher_rollout"):
        teacher_components = signal.get("reward_components") if isinstance(signal.get("reward_components"), dict) else {}
        components = {**components, **teacher_components, "teacher_reward": float(signal.get("reward") or 0.0)}
    reward = float(signal.get("reward")) if signal.get("is_teacher_rollout") else tool_reward(record, risks, components, reward_weights)
    base = {
        "schema": "omnicoder.agentic_tool_training_2026.v1",
        "trace_id": trace_id(record),
        "record_hash": stable_hash(record),
        "source_date": source_date(record),
        "tool_calls": calls,
        "tool_results": results,
        "risk_labels": risks,
        "domains": domains,
        "quality_score": quality_score(record),
    }
    if signal.get("is_teacher_rollout"):
        base["teacher_signal"] = signal
    if is_tool_row:
        outputs["sft"].append(
            {
                **base,
                "training_kind": "tool_sft",
                "messages": messages,
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
                "prompt": text,
                "reward": reward,
                "reward_components": components,
            }
        )
        if signal.get("is_teacher_rollout"):
            chosen = str(signal.get("chosen") or signal.get("corrected_response") or messages[-1]["content"])
            rejected = str(signal.get("rejected") or "No valid tool plan.")
        else:
            chosen = json.dumps({"tool_calls": calls, "final": messages[-1]}, ensure_ascii=True, sort_keys=True)
            rejected = json.dumps({"tool_calls": [], "final": "No valid tool plan."}, ensure_ascii=True, sort_keys=True)
        outputs["preference"].append(
            {
                **base,
                "training_kind": "tool_preference",
                "prompt": text,
                "chosen": chosen,
                "rejected": rejected,
                "preference_reason": "Prefer valid tool calls with state-aware use and no protected material.",
            }
        )
        outputs["rlvr"].append(build_rlvr_row(record, base, reward, components, domains, profile_cfg))
    for domain in domains:
        if domain_config(profile_cfg, domain)["enabled"]:
            outputs[f"{domain}_rlvr"].append(build_rlvr_row(record, base, reward, components, domains, profile_cfg, domain=domain))
    return outputs


def build_safety_row(record: dict[str, Any], risks: list[str]) -> dict[str, Any]:
    return {
        "schema": "omnicoder.agentic_tool_training_2026.v1",
        "training_kind": "tool_safety_negative",
        "trace_id": trace_id(record),
        "record_hash": stable_hash(record),
        "prompt": record_text(record)[:20000],
        "risk_labels": risks,
        "chosen": "",
        "rejected": record_text(record)[:2000],
        "reward": -1.0,
    }


def posttrain_manifest(
    algorithm: str,
    train_jsonl: Path,
    out_dir: Path,
    model: str,
    dry_run: bool,
    domain: str | None = None,
    reward_axes: list[str] | None = None,
    checks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "omnicoder.agentic_tool_posttrain_manifest_2026.v1",
        "algorithm": algorithm,
        "model": model,
        "train_jsonl": str(train_jsonl),
        "out_dir": str(out_dir / algorithm),
        "dry_run": dry_run,
        "domain": domain,
        "verifier_contract": {
            "checks": checks or [],
            "reward_axes": reward_axes or [],
            "eval_gates": ["heldout_sample_loss", "domain_rlvr_replay", "protected_benchmark_decontam"],
        },
        "tool_training_contract": {
            "assistant_only_loss": algorithm == "sft",
            "tool_schema_masking": True,
            "state_tracking_rewards": algorithm in {"reward", "grpo", "rloo"},
            "safety_negatives": False,
            "q4_recovery_ready": True,
        },
    }


def training_export_paths(out_dir: Path, profile_cfg: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "sft": out_dir / "tool_sft.jsonl",
        "preference": out_dir / "tool_preference.jsonl",
        "reward": out_dir / "tool_reward.jsonl",
        "rlvr": out_dir / "tool_rlvr.jsonl",
        "safety": out_dir / "tool_safety_negatives.jsonl",
    }
    for domain in DOMAIN_DEFAULTS:
        cfg = domain_config(profile_cfg, domain)
        paths[f"{domain}_rlvr"] = out_dir / cfg["export"]
    return paths


def build_training_exports(
    rows: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    profile_cfg: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, int]]:
    paths = training_export_paths(out_dir, profile_cfg)
    counts = {name: write_jsonl(paths[name], rows.get(name, [])) for name in paths}
    return paths, counts


def build_training_exports_streaming(
    records: Iterable[dict[str, Any]],
    out_dir: Path,
    min_quality: float,
    limit: int,
    profile_cfg: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, int]]:
    paths = training_export_paths(out_dir, profile_cfg)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in paths}
    with ExitStack() as stack:
        handles = {name: stack.enter_context(path.open("w", encoding="utf-8")) for name, path in paths.items()}
        for record in records:
            if limit and counts["sft"] >= limit:
                break
            emitted = rows_for_record(record, min_quality, profile_cfg)
            for name, rows in emitted.items():
                handle = handles.get(name)
                if handle is None:
                    continue
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")
                    counts[name] += 1
    return paths, counts


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    profile = read_json(args.profile)
    cfg = profile.get("agentic_tool_training") if isinstance(profile.get("agentic_tool_training"), dict) else {}
    min_quality = float(args.min_quality if args.min_quality is not None else cfg.get("min_quality", 0.0))
    limit = int(args.limit or cfg.get("limit") or 0)
    out_dir = Path(args.out_dir or cfg.get("out_dir") or "weights/agentic_tool_training_2026")
    source = Path(args.input or cfg.get("input_jsonl") or "")
    if not source.exists():
        raise SystemExit(json.dumps({"status": "error", "error": "input_jsonl not found", "input": str(source)}))
    paths, counts = build_training_exports_streaming(iter_jsonl(source), out_dir, min_quality=min_quality, limit=limit, profile_cfg=cfg)
    model = str(args.model or cfg.get("model") or profile.get("base_model") or "Qwen/Qwen3-4B")
    bridge_dir = out_dir / "posttrain_manifests"
    bridge_rows = {
        "sft": posttrain_manifest("sft", paths["sft"], bridge_dir, model, bool(args.dry_run)),
        "reward": posttrain_manifest("reward", paths["reward"], bridge_dir, model, bool(args.dry_run)),
        "dpo": posttrain_manifest("dpo", paths["preference"], bridge_dir, model, bool(args.dry_run)),
        "grpo": posttrain_manifest("grpo", paths["rlvr"], bridge_dir, model, bool(args.dry_run)),
        "kto": posttrain_manifest("kto", paths["safety"], bridge_dir, model, bool(args.dry_run)),
    }
    for domain in DOMAIN_DEFAULTS:
        domain_key = f"{domain}_rlvr"
        cfg_domain = domain_config(cfg, domain)
        bridge_rows[domain_key] = posttrain_manifest(
            "grpo",
            paths[domain_key],
            bridge_dir,
            model,
            bool(args.dry_run),
            domain=domain,
            reward_axes=cfg_domain["reward_axes"],
            checks=cfg_domain["checks"],
        )
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
        "training_sequence": [
            "tool_sft",
            "tool_reward",
            "tool_preference",
            "math_rlvr",
            "code_rlvr",
            "terminal_rlvr",
            "browser_rlvr",
            "tool_rlvr",
            "tool_safety_negative",
        ],
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
    rlvr_domains = cfg.get("rlvr_domains") if isinstance(cfg.get("rlvr_domains"), dict) else {}
    missing_domains = sorted(set(DOMAIN_DEFAULTS) - set(rlvr_domains))
    return {
        "status": "ok" if not missing else "missing_config",
        "missing": missing,
        "missing_rlvr_domains": missing_domains,
        "stages": ensure_list(cfg.get("stages")),
        "reward_axes": ensure_list(cfg.get("reward_axes")),
        "safety_negatives": bool(cfg.get("safety_negatives", False)),
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

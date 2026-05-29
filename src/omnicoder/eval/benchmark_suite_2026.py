from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2026-05-23"
DEFAULT_PROFILE = "profiles/benchmark_suite_2026.json"
DEFAULT_OUT_DIR = "weights/benchmarks_2026"
DEFAULT_TIMEOUT_SECONDS = 30
FSDP_LOCAL_FORMAT = "omnicoder2026_native_train_checkpoint_v3_fsdp_local"
JUNK_OUTPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"__OMNICODER_EMPTY_DECODE__",
        r"(?:_ph){3,}",
        r"^\W*$",
        r"^(.)\1{15,}$",
    )
)

INLINE_PROFILE: dict[str, Any] = {
    "profile_name": "benchmark_suite_2026_inline",
    "version": "2026-05-23.inline",
    "benchmarks": [
        {
            "benchmark_id": "agent_bfcl_v4_2026",
            "axis": "agent_tool",
            "adapter_kind": "tool_call_state_scorer",
            "source": "https://gorilla.cs.berkeley.edu/leaderboard",
            "splits": {"smoke": "single nested multi-turn fixture"},
            "metrics": ["tool_selection_accuracy", "argument_f1"],
            "holdout_policy": ["hide_expected_asts"],
            "release_gate": "agent_tool_release",
        },
        {
            "benchmark_id": "coding_swe_bench_live_2026",
            "axis": "coding",
            "adapter_kind": "fresh_git_container_patch",
            "source": "https://github.com/microsoft/SWE-bench-Live",
            "splits": {"smoke": "one post-cutoff issue fixture"},
            "metrics": ["patch_applies", "hidden_tests_pass_rate"],
            "holdout_policy": ["hide_hidden_tests", "hide_gold_patches"],
            "release_gate": "coding_release",
        },
    ],
    "release_gates": {
        "coding_release": ["coding_swe_bench_live_2026"],
        "agent_tool_release": ["agent_bfcl_v4_2026"],
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_output_quality_reason(value: Any) -> str:
    if value in (None, "", [], {}):
        return "missing_output"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "empty_text"
        for pattern in JUNK_OUTPUT_PATTERNS:
            if pattern.search(text):
                return f"junk_text:{pattern.pattern}"
        return ""
    if isinstance(value, (dict, list)):
        if not value:
            return "empty_structured_output"
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        for pattern in JUNK_OUTPUT_PATTERNS:
            if pattern.search(text):
                return f"junk_structured_output:{pattern.pattern}"
        return ""
    return ""


def is_fsdp_rank_local_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    manifest = path / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            return isinstance(payload, dict) and payload.get("format") == FSDP_LOCAL_FORMAT
        except Exception:
            return False
    return any(path.glob("rank*.pt"))


def checkpoint_dir_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    manifest = path / "manifest.json"
    if manifest.exists():
        digest.update(manifest.read_bytes())
    for rank_path in sorted(item for item in path.glob("rank*.pt") if item.is_file()):
        stat = rank_path.stat()
        digest.update(rank_path.name.encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def missing_reportable_policy(profile: dict[str, Any], override: str = "") -> str:
    raw = override
    if not raw:
        policy = profile.get("reportability_policy")
        if isinstance(policy, dict):
            raw = str(policy.get("missing_reportable_policy") or "")
    value = (raw or "fail").lower()
    return value if value in {"fail", "allow", "warn", "skip"} else "fail"


def reportability_decision(status: str, policy: str) -> tuple[str, str, bool]:
    if status == "ok":
        return "enforced", "passed", False
    if policy in {"allow", "warn", "skip"}:
        return "fail_open", "allowed_needs_data", False
    return "fail_closed", "blocked_needs_data", True


def file_sha256(path: Path) -> str | None:
    if is_fsdp_rank_local_checkpoint_dir(path):
        return checkpoint_dir_fingerprint(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def profile_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def load_profile(path: str) -> dict[str, Any]:
    candidate = profile_path(path)
    if candidate.exists():
        profile = read_json(candidate)
    else:
        profile = dict(INLINE_PROFILE)
    validate_profile(profile)
    return profile


def record_id(record: dict[str, Any]) -> str:
    value = record.get("benchmark_id") or record.get("id") or record.get("name") or record.get("benchmark")
    if not value:
        raise ValueError(f"benchmark record is missing an id: {record!r}")
    return str(value)


def adapter_id(record: dict[str, Any]) -> str:
    return record_id(record)


def records_from_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw_records = profile.get("benchmarks")
    if raw_records is None:
        raw_records = profile.get("adapters", [])
    if not isinstance(raw_records, list):
        raise ValueError("profile benchmarks/adapters must be a list")
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError(f"profile contains a non-object benchmark entry: {raw!r}")
        records.append(normalize_record(raw))
    return records


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    benchmark_id = record_id(record)
    splits = record.get("splits") if isinstance(record.get("splits"), dict) else {}
    smoke = record.get("smoke") or splits.get("smoke")
    commands = record.get("commands") if isinstance(record.get("commands"), dict) else {}
    normalized = {
        "benchmark_id": benchmark_id,
        "adapter_id": benchmark_id,
        "axis": record.get("axis") or infer_axis(benchmark_id, str(record.get("kind") or "")),
        "adapter_kind": record.get("adapter_kind") or record.get("kind") or "unknown",
        "source": record.get("source"),
        "task_format": record.get("task_format"),
        "modalities": list(record.get("modalities") or []),
        "splits": splits,
        "smoke": smoke,
        "metrics": list(record.get("metrics") or []),
        "holdout_policy": list(record.get("holdout_policy") or record.get("holdout") or []),
        "release_gate": record.get("release_gate"),
        "command": record.get("command"),
        "commands": commands,
        "raw": record,
    }
    return normalized


def infer_axis(benchmark_id: str, kind: str) -> str:
    text = f"{benchmark_id} {kind}".lower()
    if "swe" in text or "code" in text or "repo" in text:
        return "coding"
    if "tool" in text or "mcp" in text or "tau" in text or "terminal" in text:
        return "agent_tool"
    if "video" in text or "audio" in text or "mmmu" in text or "joint" in text:
        return "multimodal_understanding"
    if "image" in text:
        return "image_generation"
    if "context" in text or "long" in text:
        return "long_context"
    return "reasoning"


def validate_profile(profile: dict[str, Any]) -> None:
    records = records_from_profile(profile)
    if not records:
        raise ValueError("benchmark profile has no benchmarks")
    seen: set[str] = set()
    for record in records:
        rid = record["benchmark_id"]
        if rid in seen:
            raise ValueError(f"duplicate benchmark id: {rid}")
        seen.add(rid)
        if not record["adapter_kind"]:
            raise ValueError(f"{rid} is missing adapter_kind/kind")
        if not record.get("smoke"):
            raise ValueError(f"{rid} is missing a smoke split description")

    gates = profile.get("release_gates", {})
    if isinstance(gates, dict):
        for gate_name, required in gates.items():
            if gate_name in {"must_pass", "global_must_pass"}:
                continue
            if not isinstance(required, list):
                raise ValueError(f"release gate {gate_name} must be a list")
            missing = sorted(str(item) for item in required if str(item) not in seen)
            if missing:
                raise ValueError(f"release gate {gate_name} references missing benchmark(s): {', '.join(missing)}")

    weights = profile.get("scoring_policy", {}).get("axis_weights")
    if isinstance(weights, dict):
        total = sum(float(value) for value in weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"axis weights must sum to 1.0, got {total:.6f}")


def normalize_mode(record: dict[str, Any], requested_mode: str, cycle: str) -> str:
    if requested_mode in {"smoke", "dry-run"}:
        return requested_mode
    command = command_for_record(record, cycle)
    return "command" if command else "dry-run"


def command_for_record(record: dict[str, Any], cycle: str) -> Any:
    command = record.get("command")
    commands = record.get("commands") if isinstance(record.get("commands"), dict) else {}
    return commands.get(cycle) or commands.get("smoke") or command


def select_records(profile: dict[str, Any], ids: list[str] | None) -> list[dict[str, Any]]:
    records = records_from_profile(profile)
    if not ids:
        return records
    wanted = set(ids)
    selected = [record for record in records if record["benchmark_id"] in wanted or record["adapter_id"] in wanted]
    found = {record["benchmark_id"] for record in selected} | {record["adapter_id"] for record in selected}
    missing = sorted(wanted - found)
    if missing:
        raise SystemExit(f"unknown benchmark id(s): {', '.join(missing)}")
    return selected


def select_adapters(profile: dict[str, Any], ids: list[str] | None) -> list[dict[str, Any]]:
    return select_records(profile, ids)


def resolve_out_dir(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def jsonl_append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def profile_meta(profile: dict[str, Any], profile_arg: str) -> dict[str, Any]:
    path = profile_path(profile_arg)
    return {
        "profile": str(path),
        "profile_name": profile.get("profile_name") or "benchmark_suite_2026",
        "profile_version": profile.get("version", "unknown"),
        "profile_sha256": file_sha256(path),
    }


def make_manifest_rows(
    profile: dict[str, Any],
    records: list[dict[str, Any]],
    mode: str,
    run_id: str,
    cycle: str,
    model: str,
    profile_arg: str,
) -> list[dict[str, Any]]:
    meta = profile_meta(profile, profile_arg)
    rows: list[dict[str, Any]] = []
    for record in records:
        rid = record["benchmark_id"]
        adapter_mode = normalize_mode(record, mode, cycle)
        command = command_for_record(record, cycle)
        row = {
            "type": "benchmark_manifest",
            "schema_version": SCHEMA_VERSION,
            "event_id": "",
            "suite_id": meta["profile_name"],
            "run_id": run_id,
            "created_at": utc_now(),
            "profile_version": meta["profile_version"],
            "profile_sha256": meta["profile_sha256"],
            "model": model,
            "benchmark_id": rid,
            "adapter_id": record["adapter_id"],
            "adapter_kind": record["adapter_kind"],
            "axis": record["axis"],
            "split": cycle,
            "phase": "plan",
            "task_id": f"{rid}:{cycle}:smoke",
            "task_revision": stable_hash({"source": record.get("source"), "smoke": record.get("smoke")})[:16],
            "task_format": record.get("task_format"),
            "modalities": record["modalities"],
            "mode": adapter_mode,
            "diagnostic_only": True,
            "official_score": False,
            "reportability_scope": "diagnostic_only",
            "smoke": record.get("smoke"),
            "source": record.get("source"),
            "metrics": record.get("metrics", []),
            "holdout_policy": record.get("holdout_policy", []),
            "contamination_class": "protected_eval",
            "no_heavy_downloads": True,
            "command": command if adapter_mode == "command" else None,
        }
        row["event_id"] = stable_hash({k: v for k, v in row.items() if k != "event_id"})
        row["manifest_hash"] = stable_hash(row)
        rows.append(row)
    return rows


def command_to_args(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command]
    if isinstance(command, str):
        return shlex.split(command, posix=(os.name != "nt"))
    return []


def run_command(command: Any, timeout_seconds: int) -> dict[str, Any]:
    args = command_to_args(command)
    if not args:
        return {"ok": False, "exit_code": None, "stdout": "", "stderr": "empty command"}
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "elapsed_seconds": round(elapsed, 6),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "exit_code": None,
            "elapsed_seconds": round(elapsed, 6),
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout_seconds}s",
        }


def smoke_result(manifest: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    started_at = utc_now()
    command_info: dict[str, Any] | None = None
    mode = manifest["mode"]
    diagnostic_score: float | None = None
    reportable_score = False
    diagnostic_only = True
    contract_only = True
    if mode == "command":
        command_info = run_command(manifest.get("command"), timeout_seconds)
        status = "passed" if command_info.get("ok") else "failed"
        diagnostic_score = 1.0 if status == "passed" else 0.0
    elif mode == "smoke":
        status = "passed"
    else:
        status = "skipped"

    result = {
        "type": "benchmark_result",
        "schema_version": SCHEMA_VERSION,
        "event_id": "",
        "suite_id": manifest["suite_id"],
        "run_id": manifest["run_id"],
        "benchmark_id": manifest["benchmark_id"],
        "adapter_id": manifest["adapter_id"],
        "adapter_kind": manifest["adapter_kind"],
        "axis": manifest["axis"],
        "task_id": manifest["task_id"],
        "task_revision": manifest["task_revision"],
        "split": manifest["split"],
        "phase": "scoring",
        "started_at": started_at,
        "finished_at": utc_now(),
        "mode": mode,
        "status": status,
        "diagnostic_only": diagnostic_only,
        "official_score": False,
        "reportability_scope": "diagnostic_only",
        "score": None,
        "score_json": {
            "canonical_score": None,
            "diagnostic_score": diagnostic_score,
            "score_claim_scope": "diagnostic_contract",
            "reportable_score": reportable_score,
            "contract_only": contract_only,
            "diagnostic_only": diagnostic_only,
            "official_score": False,
            "reportability_scope": "diagnostic_only",
        },
        "metrics": {
            "downloaded_bytes": 0,
            "heavy_downloads_allowed": False,
            "timeout_seconds": timeout_seconds,
            "contract_only": contract_only,
            "reportable_score": reportable_score,
            "diagnostic_only": diagnostic_only,
            "official_score": False,
        },
        "metrics_json": {
            "downloaded_bytes": 0,
            "timeout_seconds": timeout_seconds,
            "contract_only": contract_only,
            "diagnostic_only": diagnostic_only,
        },
        "artifact_refs": [],
        "input_sha256": stable_hash(
            {
                "benchmark_id": manifest["benchmark_id"],
                "split": manifest["split"],
                "smoke": manifest.get("smoke"),
            }
        ),
        "output_sha256": stable_hash(command_info or {"status": status, "contract_only": contract_only}),
        "manifest_hash": manifest["manifest_hash"],
        "command_result": command_info,
        "contamination": {
            "hidden_material_exposed": False,
            "trajectory_quarantine": status == "passed",
            "public_dev_allowed": True,
        },
    }
    result["event_id"] = stable_hash({k: v for k, v in result.items() if k != "event_id"})
    result["result_hash"] = stable_hash(result)
    return result


def normalize_answer(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = text.replace("(", " ").replace(")", " ").replace(".", " ")
    parts = [part for part in text.split() if part]
    if len(parts) == 1 and len(parts[0]) == 1 and parts[0].isalpha():
        return parts[0]
    if parts and len(parts[0]) == 1 and parts[0].isalpha() and parts[0] in {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}:
        return parts[0]
    return " ".join(parts)


def token_f1(prediction: Any, answer: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap: dict[str, int] = {}
    for token in gold_tokens:
        overlap[token] = overlap.get(token, 0) + 1
    hits = 0
    for token in pred_tokens:
        if overlap.get(token, 0) > 0:
            hits += 1
            overlap[token] -= 1
    if hits == 0:
        return 0.0
    precision = hits / len(pred_tokens)
    recall = hits / len(gold_tokens)
    return (2 * precision * recall) / max(1e-12, precision + recall)


def canonical_mcq_answer(value: Any, choices: Any) -> str:
    normalized = normalize_answer(value)
    if not isinstance(choices, list) or not choices:
        return normalized
    labels = [chr(ord("a") + index) for index in range(min(len(choices), 26))]
    if normalized in labels:
        return normalized
    try:
        index = int(str(value).strip())
        if 0 <= index < len(labels):
            return labels[index]
        if 1 <= index <= len(labels):
            return labels[index - 1]
    except Exception:
        pass
    for index, choice in enumerate(choices[: len(labels)]):
        if normalized and normalized == normalize_answer(choice):
            return labels[index]
    return normalized


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "passed", "pass", "ok", "resolved", "success"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(payload, dict):
                payload.setdefault("line_number", line_no)
                rows.append(payload)
    return rows


def resolve_optional_path(value: Any, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def configured_task_paths(profile: dict[str, Any], record: dict[str, Any], cycle: str, cli_tasks: list[str] | None) -> list[Path]:
    root = repo_root()
    paths: list[Path] = []
    for item in cli_tasks or []:
        path = resolve_optional_path(item, root)
        if path is not None:
            paths.append(path)
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    for key in ("reportable_tasks", "tasks_jsonl", f"{cycle}_tasks_jsonl"):
        path = resolve_optional_path(raw.get(key), root)
        if path is not None:
            paths.append(path)
    task_roots = profile.get("reportable_task_roots") if isinstance(profile.get("reportable_task_roots"), dict) else {}
    configured = task_roots.get(record["benchmark_id"]) or task_roots.get(record["adapter_id"])
    if isinstance(configured, list):
        for item in configured:
            path = resolve_optional_path(item, root)
            if path is not None:
                paths.append(path)
    elif configured:
        path = resolve_optional_path(configured, root)
        if path is not None:
            paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def snapshot_descriptors(profile: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("reportable_snapshots")
    if not isinstance(raw, dict):
        return []
    keys = [str(record.get("benchmark_id") or ""), str(record.get("adapter_id") or "")]
    descriptors: list[dict[str, Any]] = []
    for key in keys:
        value = raw.get(key)
        if isinstance(value, dict):
            descriptors.append(value)
        elif isinstance(value, list):
            descriptors.extend(item for item in value if isinstance(item, dict))
    return descriptors


def path_matches_snapshot(candidate: Path, descriptor: dict[str, Any], root: Path) -> bool:
    raw = descriptor.get("task_root") or descriptor.get("tasks_jsonl") or descriptor.get("path")
    if not raw:
        return False
    expected = resolve_optional_path(raw, root)
    if expected is None:
        return False
    try:
        candidate_resolved = candidate.resolve()
    except Exception:
        candidate_resolved = candidate
    try:
        expected_resolved = expected.resolve()
    except Exception:
        expected_resolved = expected
    if candidate_resolved == expected_resolved:
        return True
    if expected.exists() and expected.is_dir():
        try:
            candidate_resolved.relative_to(expected_resolved)
            return True
        except ValueError:
            return False
    return False


def snapshot_for_task_path(profile: dict[str, Any], record: dict[str, Any], candidate: Path) -> dict[str, Any] | None:
    root = repo_root()
    for descriptor in snapshot_descriptors(profile, record):
        if path_matches_snapshot(candidate, descriptor, root):
            return descriptor
    return None


def attach_snapshot_metadata(row: dict[str, Any], descriptor: dict[str, Any] | None, candidate: Path) -> dict[str, Any]:
    if not descriptor:
        row.setdefault("task_source_path", str(candidate))
        row.setdefault("task_row_sha256", stable_hash(row))
        return row
    for src, dst in (
        ("snapshot_id", "snapshot_id"),
        ("official_snapshot_id", "official_snapshot_id"),
        ("authorized_snapshot_id", "authorized_snapshot_id"),
        ("snapshot_sha256", "snapshot_sha256"),
        ("task_file_sha256", "task_file_sha256"),
        ("official_task_sha256", "official_task_sha256"),
        ("manifest_sha256", "manifest_sha256"),
        ("dataset_revision", "dataset_revision"),
        ("task_revision", "task_revision"),
        ("source", "source"),
        ("dataset_source", "dataset_source"),
        ("authorization_ref", "authorization_ref"),
        ("license_ref", "license_ref"),
        ("official_scorer_ref", "official_scorer_ref"),
        ("split", "split"),
    ):
        value = descriptor.get(src)
        if value not in (None, ""):
            row.setdefault(dst, value)
    if not any(
        row.get(key)
        for key in (
            "snapshot_sha256",
            "task_file_sha256",
            "official_task_sha256",
            "manifest_sha256",
            "source_sha256",
        )
    ):
        task_file_sha256 = file_sha256(candidate)
        if task_file_sha256:
            row.setdefault("task_file_sha256", task_file_sha256)
    auth = (
        descriptor.get("snapshot_authorization")
        or descriptor.get("authorization")
        or descriptor.get("dataset_source_kind")
        or descriptor.get("source_kind")
    )
    if auth not in (None, ""):
        row.setdefault("snapshot_authorization", auth)
    row.setdefault("task_source_path", str(candidate))
    row.setdefault("source_line", row.get("line_number"))
    row.setdefault("task_row_sha256", stable_hash(row))
    return row


def load_reportable_tasks(profile: dict[str, Any], record: dict[str, Any], cycle: str, cli_tasks: list[str] | None) -> list[dict[str, Any]]:
    wanted = {record["benchmark_id"], record["adapter_id"]}
    rows: list[dict[str, Any]] = []
    for path in configured_task_paths(profile, record, cycle, cli_tasks):
        if not path.exists():
            continue
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            snapshot = snapshot_for_task_path(profile, record, candidate)
            for row in read_jsonl(candidate):
                benchmark = str(row.get("benchmark_id") or row.get("adapter_id") or record["benchmark_id"])
                if benchmark not in wanted:
                    continue
                row.setdefault("benchmark_id", record["benchmark_id"])
                row.setdefault("task_id", row.get("id") or stable_hash({"path": str(candidate), "line": row.get("line_number")})[:16])
                row.setdefault("task_revision", row.get("dataset_revision") or row.get("revision") or "unknown")
                row.setdefault("source", row.get("dataset_source") or row.get("source") or "")
                row = attach_snapshot_metadata(row, snapshot, candidate)
                rows.append(row)
    return rows


def load_predictions(paths: list[str] | None) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for item in paths or []:
        path = Path(item)
        if not path.is_absolute():
            path = repo_root() / path
        for row in read_jsonl(path):
            benchmark = str(row.get("benchmark_id") or row.get("adapter_id") or "")
            task = str(row.get("task_id") or row.get("id") or "")
            if task:
                predictions[f"{benchmark}:{task}"] = row
                predictions.setdefault(task, row)
    return predictions


def task_prediction(task: dict[str, Any], predictions: dict[str, dict[str, Any]]) -> Any:
    benchmark = str(task.get("benchmark_id") or "")
    task_id = str(task.get("task_id") or task.get("id") or "")
    row = predictions.get(f"{benchmark}:{task_id}") or predictions.get(task_id)
    if row:
        for key in ("prediction", "output", "patch", "tool_call", "artifact_path"):
            if key in row:
                return row[key]
        for key in ("model_answer", "model_output", "model_patch", "model_actions", "output_path", "generated_artifact"):
            if key in row:
                return row[key]
    for key in ("prediction", "model_answer", "model_output", "output", "model_patch", "model_actions", "tool_call", "artifact_path", "output_path", "generated_artifact"):
        if key in task:
            return task[key]
    return None


def task_has_reportable_metadata(task: dict[str, Any]) -> bool:
    if task.get("reportable") is False:
        return False
    revision = str(task.get("dataset_revision") or task.get("task_revision") or task.get("revision") or "").strip()
    source = str(task.get("source") or task.get("dataset_source") or "").strip()
    snapshot = str(
        task.get("snapshot_id")
        or task.get("official_snapshot_id")
        or task.get("authorized_snapshot_id")
        or task.get("snapshot_sha256")
        or task.get("dataset_snapshot")
        or ""
    ).strip()
    source_hash = str(
        task.get("snapshot_sha256")
        or task.get("task_file_sha256")
        or task.get("official_task_sha256")
        or task.get("manifest_sha256")
        or task.get("source_sha256")
        or ""
    ).strip()
    license_ref = str(task.get("license_ref") or "").strip()
    scorer_ref = str(task.get("official_scorer_ref") or "").strip()
    authorization = str(
        task.get("snapshot_authorization")
        or task.get("authorization")
        or task.get("dataset_source_kind")
        or task.get("source_kind")
        or ""
    ).strip().lower()
    authorized_values = {
        "official",
        "official_release",
        "official_current_release",
        "authorized",
        "authorized_mirror",
        "authorized_private",
        "official_or_authorized",
        "official_or_authorized_current_release",
    }
    has_revision = revision.lower() not in {"", "unknown", "none", "null"}
    placeholder_hashes = {"", "unknown", "none", "null", "<operator_supplied_sha256>", "operator_required"}
    has_hash = source_hash.lower() not in placeholder_hashes and not source_hash.lower().startswith("operator_required")
    has_license = license_ref.lower() not in {"", "unknown", "none", "null"}
    has_scorer = scorer_ref.lower() not in {"", "unknown", "none", "null"}
    return bool(source) and has_revision and bool(snapshot) and has_hash and has_license and has_scorer and authorization in authorized_values


def task_has_model_output(task: dict[str, Any], prediction: Any) -> bool:
    if prediction is None:
        for key in (
            "prediction",
            "model_answer",
            "model_output",
            "model_patch",
            "model_actions",
            "trajectory",
            "response",
            "tool_call",
            "artifact_path",
            "output_path",
            "generated_artifact",
        ):
            value = task.get(key)
            if value not in (None, "", [], {}) and not model_output_quality_reason(value):
                return True
        return False
    return not model_output_quality_reason(prediction)


def score_mcq_task(task: dict[str, Any], prediction: Any) -> dict[str, Any]:
    answer = task.get("answer")
    if answer is None:
        answer = task.get("gold") or task.get("target")
    choices = task.get("choices")
    pred = canonical_mcq_answer(prediction, choices)
    gold = canonical_mcq_answer(answer, choices)
    exact = bool(pred and gold and pred == gold)
    return {"score": 1.0 if exact else 0.0, "metrics": {"exact_match": exact, "normalized_prediction": pred, "normalized_answer": gold}}


def score_qa_task(task: dict[str, Any], prediction: Any) -> dict[str, Any]:
    answer = task.get("answer") or task.get("gold") or task.get("target") or ""
    exact = normalize_answer(prediction) == normalize_answer(answer)
    f1 = token_f1(prediction, answer)
    return {"score": 1.0 if exact else f1, "metrics": {"exact_match": exact, "token_f1": round(f1, 6)}}


def score_swe_task(task: dict[str, Any], prediction: Any) -> dict[str, Any]:
    patch = prediction if isinstance(prediction, str) else task.get("model_patch") or task.get("patch") or ""
    patch_applies = boolish(task.get("patch_applies"))
    tests_pass = boolish(task.get("tests_pass") or task.get("hidden_tests_pass") or task.get("resolved"))
    if task.get("expected_patch_sha256"):
        patch_applies = stable_hash(str(patch)) == str(task.get("expected_patch_sha256"))
    resolved = patch_applies and tests_pass
    return {
        "score": 1.0 if resolved else 0.0,
        "metrics": {
            "resolved": resolved,
            "patch_applies": patch_applies,
            "tests_pass": tests_pass,
            "patch_chars": len(str(patch or "")),
        },
    }


def score_arc_agi3_task(task: dict[str, Any], prediction: Any) -> dict[str, Any]:
    del prediction
    success = boolish(task.get("success") or task.get("solved"))
    actions = max(1.0, float(task.get("actions") or task.get("agent_actions") or 1.0))
    human_actions = max(1.0, float(task.get("human_actions") or task.get("optimal_actions") or actions))
    rhae = (human_actions / actions) if success else 0.0
    rhae = max(0.0, min(1.0, rhae))
    return {"score": rhae, "metrics": {"success": success, "relative_human_action_efficiency": round(rhae, 6), "actions": actions, "human_actions": human_actions}}


def score_tool_task(task: dict[str, Any], prediction: Any) -> dict[str, Any]:
    expected = task.get("expected_tool_call") or task.get("expected") or {}
    predicted = prediction if isinstance(prediction, dict) else task.get("tool_call") or {}
    if not isinstance(expected, dict):
        expected = {"value": expected}
    if not isinstance(predicted, dict):
        predicted = {"value": predicted}
    expected_name = str(expected.get("name") or expected.get("tool_name") or "")
    predicted_name = str(predicted.get("name") or predicted.get("tool_name") or "")
    name_ok = expected_name == predicted_name if expected_name else bool(predicted_name)
    expected_args = expected.get("arguments") or expected.get("args") or expected.get("input") or {}
    predicted_args = predicted.get("arguments") or predicted.get("args") or predicted.get("input") or {}
    arg_score = 1.0 if expected_args == predicted_args else token_f1(expected_args, predicted_args)
    score = (0.55 if name_ok else 0.0) + (0.45 * arg_score)
    return {"score": score, "metrics": {"tool_name_match": name_ok, "argument_f1": round(arg_score, 6)}}


def score_media_task(task: dict[str, Any], prediction: Any) -> dict[str, Any]:
    artifact = prediction or task.get("artifact_path") or task.get("output_path")
    valid = False
    byte_size = 0
    sha256 = None
    if artifact:
        path = Path(str(artifact))
        if not path.is_absolute():
            path = repo_root() / path
        if path.exists() and path.is_file():
            byte_size = path.stat().st_size
            sha256 = file_sha256(path)
            valid = byte_size >= int(task.get("min_bytes") or 1)
    rubric = task.get("rubric_score")
    try:
        rubric_value = float(rubric) if rubric is not None else (1.0 if valid else 0.0)
    except (TypeError, ValueError):
        rubric_value = 1.0 if valid else 0.0
    score = (0.35 if valid else 0.0) + (0.65 * max(0.0, min(1.0, rubric_value)))
    return {"score": score, "metrics": {"artifact_valid": valid, "artifact_bytes": byte_size, "artifact_sha256": sha256, "rubric_score": rubric_value}}


def score_reportable_task(record: dict[str, Any], task: dict[str, Any], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prediction = task_prediction(task, predictions)
    has_output = task_has_model_output(task, prediction)
    text = f"{record['benchmark_id']} {record['adapter_kind']} {record['axis']} {record.get('task_format')}".lower()
    if not has_output:
        scored = {
            "score": 0.0,
            "metrics": {
                "missing_or_junk_model_output": True,
                "output_quality_reason": model_output_quality_reason(prediction),
            },
        }
    elif "arc_agi3" in text or "interactive" in text:
        scored = score_arc_agi3_task(task, prediction)
    elif "swe" in text or "patch" in text or "git" in text:
        scored = score_swe_task(task, prediction)
    elif "tool" in text or "bfcl" in text or "mcp" in text:
        scored = score_tool_task(task, prediction)
    elif "image_generation" in text or "video_generation" in text or "audio_generation" in text or "music_generation" in text or "media_generation" in text:
        scored = score_media_task(task, prediction)
    elif "mcq" in text or "mmmu" in text or task.get("choices"):
        scored = score_mcq_task(task, prediction)
    else:
        scored = score_qa_task(task, prediction)
    return {
        "task_id": str(task.get("task_id")),
        "task_revision": str(task.get("task_revision") or task.get("dataset_revision") or "unknown"),
        "score": round(float(scored["score"]), 6),
        "metrics": scored.get("metrics") or {},
        "reportable_metadata": task_has_reportable_metadata(task) and has_output,
        "has_model_output": has_output,
        "snapshot_id": str(task.get("snapshot_id") or task.get("official_snapshot_id") or task.get("authorized_snapshot_id") or ""),
        "input_sha256": stable_hash({k: task.get(k) for k in ("question", "prompt", "input", "image", "repo", "issue", "environment")}),
        "output_sha256": stable_hash(prediction),
    }


def reportable_result(
    profile: dict[str, Any],
    record: dict[str, Any],
    tasks: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    run_id: str,
    cycle: str,
    model: str,
    min_tasks: int,
) -> dict[str, Any]:
    started_at = utc_now()
    if not tasks:
        task_scores: list[dict[str, Any]] = []
        status = "skipped"
        score = None
        reportable_score = False
        reason = "no_reportable_tasks"
    else:
        task_scores = [score_reportable_task(record, task, predictions) for task in tasks]
        score_values = [float(item["score"]) for item in task_scores]
        score = sum(score_values) / len(score_values) if score_values else 0.0
        enough_tasks = len(task_scores) >= min_tasks
        metadata_ok = all(item["reportable_metadata"] for item in task_scores)
        metadata_contract_ok = enough_tasks and metadata_ok
        official_or_external_scorer = False
        reportable_score = metadata_contract_ok and official_or_external_scorer
        status = "passed" if reportable_score else ("contract_only" if metadata_contract_ok else "local_only")
        reason = "" if reportable_score else (
            "internal_contract_oracle_not_official_scorer" if metadata_contract_ok else "missing_reportable_metadata_or_min_task_count"
        )
    task_revisions = sorted({str(task.get("dataset_revision") or task.get("task_revision") or "unknown") for task in tasks})
    result = {
        "type": "benchmark_result",
        "schema_version": SCHEMA_VERSION,
        "event_id": "",
        "suite_id": profile.get("profile_name") or "benchmark_suite_2026",
        "run_id": run_id,
        "benchmark_id": record["benchmark_id"],
        "adapter_id": record["adapter_id"],
        "adapter_kind": record["adapter_kind"],
        "axis": record["axis"],
        "task_id": f"{record['benchmark_id']}:{cycle}:reportable",
        "task_revision": ",".join(task_revisions[:8]) if task_revisions else "none",
        "split": cycle,
        "phase": "reportable_scoring",
        "started_at": started_at,
        "finished_at": utc_now(),
        "mode": "reportable",
        "status": status,
        "reason": reason,
        "diagnostic_only": False,
        "official_score": False,
        "reportability_scope": "official_or_authorized_external_scorer" if reportable_score else "internal_contract_or_local_only",
        "model": model,
        "score": round(score, 6) if reportable_score and score is not None else None,
        "score_json": {
            "canonical_score": round(score, 6) if reportable_score and score is not None else None,
            "contract_score": round(score, 6) if score is not None else None,
            "score_claim_scope": "internal_contract_oracle",
            "reportable_score": reportable_score,
            "contract_only": bool(metadata_contract_ok and not reportable_score) if tasks else False,
            "diagnostic_only": False,
            "official_score": False,
            "scorer_kind": "authorized_contract_oracle",
            "task_count": len(task_scores),
            "min_tasks": min_tasks,
            "reportable_scope": "official_or_authorized_external_scorer" if reportable_score else "internal_contract_or_local_only",
            "reportability_scope": "official_or_authorized_external_scorer" if reportable_score else "internal_contract_or_local_only",
        },
        "metrics": {
            "task_count": len(task_scores),
            "reportable_task_count": sum(1 for item in task_scores if item["reportable_metadata"]),
            "mean_score": round(score, 6) if score is not None else None,
        },
        "metrics_json": {
            "task_scores": task_scores[:1000],
            "task_count": len(task_scores),
        },
        "artifact_refs": [],
        "input_sha256": stable_hash(tasks),
        "output_sha256": stable_hash(task_scores),
        "manifest_hash": stable_hash({"record": record, "cycle": cycle, "mode": "reportable"}),
        "command_result": None,
        "contamination": {
            "hidden_material_exposed": False,
            "trajectory_quarantine": bool(reportable_score),
            "public_dev_allowed": cycle != "release",
            "dataset_revision": task_revisions,
        },
    }
    result["event_id"] = stable_hash({k: v for k, v in result.items() if k != "event_id"})
    result["result_hash"] = stable_hash(result)
    return result


def row_score_json(row: dict[str, Any]) -> dict[str, Any]:
    score_json = row.get("score_json")
    return score_json if isinstance(score_json, dict) else {}


def row_is_diagnostic_only(row: dict[str, Any]) -> bool:
    score_json = row_score_json(row)
    if boolish(row.get("diagnostic_only") or score_json.get("diagnostic_only")):
        return True
    mode = str(row.get("mode") or "").lower()
    phase = str(row.get("phase") or "").lower()
    scope = str(row.get("reportability_scope") or score_json.get("reportability_scope") or "").lower()
    if scope == "diagnostic_only":
        return True
    return mode in {"smoke", "dry-run", "command"} and phase != "reportable_scoring"


def row_reportable_score(row: dict[str, Any]) -> bool:
    return boolish(row_score_json(row).get("reportable_score", False)) and not row_is_diagnostic_only(row)


def row_official_score(row: dict[str, Any]) -> bool:
    score_json = row_score_json(row)
    return (
        boolish(row.get("official_score") or score_json.get("official_score"))
        and row_reportable_score(row)
        and not row_is_diagnostic_only(row)
    )


def row_contract_only(row: dict[str, Any]) -> bool:
    return boolish(row_score_json(row).get("contract_only", False))


def row_canonical_score(row: dict[str, Any]) -> float | None:
    if not row_reportable_score(row):
        return None
    for value in (row.get("score"), row_score_json(row).get("canonical_score"), row.get("canonical_score")):
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def row_diagnostic_or_contract_score(row: dict[str, Any]) -> float | None:
    score_json = row_score_json(row)
    for value in (score_json.get("diagnostic_score"), score_json.get("contract_score")):
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def release_gate_report(profile: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    gates = profile.get("release_gates", {})
    if not isinstance(gates, dict):
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get("benchmark_id") or row.get("adapter_id"))] = row
    report: dict[str, Any] = {}
    for gate_name, required in gates.items():
        if gate_name in {"must_pass", "global_must_pass"} or not isinstance(required, list):
            continue
        required_ids = [str(item) for item in required]
        missing = [item for item in required_ids if item not in latest]
        failed = [item for item in required_ids if latest.get(item, {}).get("status") == "failed"]
        unscored = [
            item
            for item in required_ids
            if item in latest and not row_reportable_score(latest[item])
        ]
        if missing:
            status = "missing"
        elif failed:
            status = "failed"
        elif unscored:
            status = "contract_only"
        else:
            status = "passed"
        report[gate_name] = {
            "status": status,
            "required": required_ids,
            "missing": missing,
            "failed": failed,
            "unscored": unscored,
        }
    return report


def filter_ids(args: argparse.Namespace) -> list[str] | None:
    ids: list[str] = []
    for value in (getattr(args, "benchmark", None) or []):
        ids.append(value)
    for value in (getattr(args, "adapter", None) or []):
        ids.append(value)
    return ids or None


def cmd_validate(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    records = records_from_profile(profile)
    gates = profile.get("release_gates", {})
    gate_requirements = {
        str(name): [str(item) for item in required]
        for name, required in gates.items()
        if isinstance(gates, dict) and name not in {"must_pass", "global_must_pass"} and isinstance(required, list)
    }
    payload = {
        "status": "ok",
        "profile_version": profile.get("version", "unknown"),
        "benchmarks": len(records),
        "axes": sorted({record["axis"] for record in records}),
        "release_gate_requirements": gate_requirements,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    rows = []
    for record in select_records(profile, filter_ids(args)):
        row = {
            "benchmark_id": record["benchmark_id"],
            "adapter_id": record["adapter_id"],
            "axis": record["axis"],
            "adapter_kind": record["adapter_kind"],
            "kind": record["adapter_kind"],
            "smoke": record.get("smoke"),
            "source": record.get("source"),
            "has_command": bool(command_for_record(record, "smoke")),
        }
        rows.append(row)
    print(
        json.dumps(
            {
                "profile_version": profile.get("version", "unknown"),
                "benchmarks": rows,
                "adapters": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    run_id = args.run_id or f"plan-{int(time.time())}"
    rows = make_manifest_rows(
        profile,
        select_records(profile, filter_ids(args)),
        args.mode,
        run_id,
        args.cycle,
        args.model,
        args.profile,
    )
    out_dir = resolve_out_dir(args.out_dir)
    manifest_path = out_dir / "manifests.jsonl"
    jsonl_append(manifest_path, rows)
    summary = {
        "status": "ok",
        "manifest": str(manifest_path),
        "planned": len(rows),
        "run_id": run_id,
        "cycle": args.cycle,
    }
    write_json(out_dir / "plan_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def cmd_run_smoke(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    run_id = args.run_id or f"smoke-{int(time.time())}"
    records = select_records(profile, filter_ids(args))
    rows = make_manifest_rows(profile, records, args.mode, run_id, args.cycle, args.model, args.profile)
    out_dir = resolve_out_dir(args.out_dir)
    manifest_path = out_dir / "manifests.jsonl"
    results_path = out_dir / "results.jsonl"
    jsonl_append(manifest_path, rows)
    results = [smoke_result(row, args.timeout_seconds) for row in rows]
    jsonl_append(results_path, results)
    failed = sum(1 for row in results if row["status"] == "failed")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    summary = {
        "status": "ok" if failed == 0 else "failed",
        "run_id": run_id,
        "cycle": args.cycle,
        "manifest": str(manifest_path),
        "results": str(results_path),
        "ran": len(results),
        "failed": failed,
        "skipped": skipped,
        "release_gates": release_gate_report(profile, results),
    }
    write_json(out_dir / "summary.json", summary)
    if args.out:
        write_json(Path(args.out), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if failed == 0 else 1


def cmd_run_reportable(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    run_id = args.run_id or f"reportable-{int(time.time())}"
    records = select_records(profile, filter_ids(args))
    predictions = load_predictions(args.predictions)
    out_dir = resolve_out_dir(args.out_dir)
    results_path = out_dir / "reportable_results.jsonl"
    results: list[dict[str, Any]] = []
    for record in records:
        tasks = load_reportable_tasks(profile, record, args.cycle, args.tasks)
        results.append(reportable_result(profile, record, tasks, predictions, run_id, args.cycle, args.model, args.min_tasks))
    jsonl_append(results_path, results)
    failed = sum(1 for row in results if row["status"] == "failed")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    local_only = sum(1 for row in results if row["status"] == "local_only")
    reportable = sum(1 for row in results if row_reportable_score(row))
    status = "ok" if failed == 0 and skipped == 0 and local_only == 0 and reportable > 0 else "needs_data"
    policy = missing_reportable_policy(profile, args.missing_reportable_policy)
    gate_policy, gate_decision, blocked = reportability_decision(status, policy)
    summary = {
        "status": status,
        "run_id": run_id,
        "cycle": args.cycle,
        "results": str(results_path),
        "ran": len(results),
        "reportable": reportable,
        "official": sum(1 for row in results if row_official_score(row)),
        "failed": failed,
        "skipped": skipped,
        "local_only": local_only,
        "missing_reportable_policy": policy,
        "gate_policy": gate_policy,
        "gate_decision": gate_decision,
        "release_gates": release_gate_report(profile, results),
    }
    write_json(out_dir / "reportable_summary.json", summary)
    if args.out:
        write_json(Path(args.out), summary)
    print(json.dumps(summary, sort_keys=True))
    # Missing official metadata, task roots, or model outputs is a reportability
    # gate condition, not a CLI/process failure. Keep the release gate fail-closed
    # in the summary while returning success so automation can collect and publish
    # the local-only evidence for remediation.
    return 1 if failed > 0 else 0


def cmd_summarize(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = resolve_out_dir(args.out_dir) / args.results
    rows = jsonl_read(results_path)
    by_status: dict[str, int] = {}
    by_benchmark: dict[str, dict[str, Any]] = {}
    reportable_results = 0
    official_results = 0
    diagnostic_only_results = 0
    for row in rows:
        status = str(row.get("status", "unknown"))
        benchmark = str(row.get("benchmark_id") or row.get("adapter_id") or "unknown")
        reportable = (
            row.get("mode") == "reportable"
            and row.get("phase") == "reportable_scoring"
            and row_reportable_score(row)
            and not row_contract_only(row)
        )
        if reportable:
            reportable_results += 1
        if row_official_score(row):
            official_results += 1
        if row_is_diagnostic_only(row):
            diagnostic_only_results += 1
        by_status[status] = by_status.get(status, 0) + 1
        by_benchmark[benchmark] = {
            "latest_status": status,
            "latest_mode": row.get("mode"),
            "latest_run_id": row.get("run_id"),
            "latest_score": row_canonical_score(row),
            "latest_reportable_score": row_canonical_score(row),
            "latest_internal_score": row_diagnostic_or_contract_score(row),
            "score_claim_scope": row_score_json(row).get("score_claim_scope"),
            "reportable_score": reportable,
        }
    summary = {
        "type": "benchmark_summary",
        "schema_version": SCHEMA_VERSION,
        "results": str(results_path),
        "total_results": len(rows),
        "reportable_results": reportable_results,
        "official_results": official_results,
        "diagnostic_only_results": diagnostic_only_results,
        "contract_only_results": sum(1 for row in rows if row_contract_only(row)),
        "by_status": by_status,
        "by_benchmark": by_benchmark,
        "by_adapter": by_benchmark,
        "release_gates": release_gate_report(profile, rows),
    }
    if args.out:
        write_json(Path(args.out), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def add_common_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", action="append", help="Filter to one benchmark id; repeatable")
    parser.add_argument("--adapter", action="append", help="Alias for --benchmark; repeatable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JSONL-first Omnicoder 2026 benchmark suite")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Benchmark suite profile JSON")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for manifests/results JSONL")
    parser.add_argument("--out", default="", help="Optional summary JSON path")
    parser.add_argument("--model", default="unknown", help="Model path, model id, or serving route under evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_p = sub.add_parser("validate", help="Validate profile references and scoring weights")
    validate_p.set_defaults(func=cmd_validate)

    list_p = sub.add_parser("list", help="List registered benchmark adapters")
    add_common_selection(list_p)
    list_p.set_defaults(func=cmd_list)

    plan_p = sub.add_parser("plan", help="Append planned benchmark manifest rows")
    add_common_selection(plan_p)
    plan_p.add_argument("--mode", choices=["smoke", "dry-run", "command"], default="dry-run")
    plan_p.add_argument("--cycle", choices=["smoke", "nightly", "release"], default="smoke")
    plan_p.add_argument("--run-id", default="")
    plan_p.set_defaults(func=cmd_plan)

    smoke_p = sub.add_parser("run-smoke", help="Run smoke/dry-run/command adapters and append JSONL results")
    add_common_selection(smoke_p)
    smoke_p.add_argument("--mode", choices=["smoke", "dry-run", "command"], default="smoke")
    smoke_p.add_argument("--cycle", choices=["smoke", "nightly", "release"], default="smoke")
    smoke_p.add_argument("--run-id", default="")
    smoke_p.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    smoke_p.set_defaults(func=cmd_run_smoke)

    reportable_p = sub.add_parser("run-reportable", help="Score official/authorized task JSONL rows with real oracles")
    add_common_selection(reportable_p)
    reportable_p.add_argument("--tasks", action="append", help="JSONL file or directory with reportable task rows. Repeatable.")
    reportable_p.add_argument("--predictions", action="append", help="Optional JSONL predictions keyed by benchmark_id/task_id. Repeatable.")
    reportable_p.add_argument("--cycle", choices=["smoke", "nightly", "release"], default="smoke")
    reportable_p.add_argument("--run-id", default="")
    reportable_p.add_argument("--min-tasks", type=int, default=1)
    reportable_p.add_argument("--missing-reportable-policy", choices=["fail", "allow", "warn", "skip"], default="")
    reportable_p.set_defaults(func=cmd_run_reportable)

    sum_p = sub.add_parser("summarize", help="Summarize a results JSONL file")
    sum_p.add_argument("--results", default="results.jsonl", help="Results JSONL path or name under --out-dir")
    sum_p.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

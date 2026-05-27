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
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2026-05-24"
PREDICTION_SCHEMA = "omnicoder.reportable_prediction_2026.v1"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
AUTHORIZED_VALUES = {
    "official",
    "official_release",
    "official_current_release",
    "authorized",
    "authorized_mirror",
    "authorized_private",
    "official_or_authorized",
    "official_or_authorized_current_release",
}
MODEL_OUTPUT_KEYS = (
    "prediction",
    "model_patch",
    "model_actions",
    "tool_call",
    "artifact_path",
    "model_answer",
    "model_output",
    "output",
    "output_path",
    "generated_artifact",
)
JUNK_OUTPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"__OMNICODER_EMPTY_DECODE__",
        r"(?:_ph){3,}",
        r"^\W*$",
        r"^(.)\1{15,}$",
    )
)
SENSITIVE_TASK_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "gold_patch",
    "target",
    "targets",
    "label",
    "labels",
    "expected",
    "expected_answer",
    "expected_patch",
    "expected_patch_sha256",
    "expected_tool_call",
    "oracle",
    "oracle_patch",
    "rubric_score",
    "success",
    "solved",
    "tests_pass",
    "hidden_tests_pass",
    "resolved",
    "patch_applies",
    "prediction",
    "model_answer",
    "model_output",
    "model_patch",
    "model_actions",
    "tool_call",
    "artifact_path",
    "output_path",
    "generated_artifact",
}


class HarnessError(ValueError):
    """Raised when input tasks, model responses, or output rows are invalid."""


@dataclass(frozen=True)
class TaskRecord:
    benchmark_id: str
    task_id: str
    row: dict[str, Any]
    source_path: Path
    source_line: int


@dataclass(frozen=True)
class GenerateConfig:
    backend: str
    model: str
    max_output_tokens: int
    temperature: float
    timeout_seconds: int
    base_url: str
    api_key_env: str
    checkpoint_runner: str
    checkpoint_path: str


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def output_quality_reason(value: Any) -> str:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root() / path


def iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        raise HarnessError(f"task JSONL does not exist or is not a file: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HarnessError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            if not isinstance(payload, dict):
                raise HarnessError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append((line_number, payload))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise HarnessError(f"output already exists; pass --force to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":")) + "\n")


def task_paths(values: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = resolve_path(value)
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(candidate)
    if not resolved:
        raise HarnessError("at least one --tasks JSONL file or directory is required")
    return resolved


def require_text(row: dict[str, Any], keys: tuple[str, ...], label: str, source: Path, line: int) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return str(value).strip()
    raise HarnessError(f"{source}:{line}: missing required {label}: one of {', '.join(keys)}")


def validate_authorized_task(
    row: dict[str, Any],
    source: Path,
    line: int,
    *,
    allow_local_dev: bool = False,
) -> TaskRecord:
    benchmark_id = require_text(row, ("benchmark_id", "adapter_id"), "benchmark id", source, line)
    task_id = require_text(row, ("task_id", "id"), "task id", source, line)
    if allow_local_dev:
        return TaskRecord(benchmark_id=benchmark_id, task_id=task_id, row=row, source_path=source, source_line=line)
    if row.get("reportable") is False:
        raise HarnessError(f"{source}:{line}: reportable task row is explicitly marked reportable=false")

    revision = str(row.get("dataset_revision") or row.get("task_revision") or row.get("revision") or "").strip()
    if revision.lower() in {"", "unknown", "none", "null"}:
        raise HarnessError(f"{source}:{line}: authorized task row is missing dataset/task revision")
    if not str(row.get("source") or row.get("dataset_source") or "").strip():
        raise HarnessError(f"{source}:{line}: authorized task row is missing source/dataset_source")
    snapshot = str(
        row.get("snapshot_id")
        or row.get("official_snapshot_id")
        or row.get("authorized_snapshot_id")
        or row.get("snapshot_sha256")
        or row.get("dataset_snapshot")
        or ""
    ).strip()
    if not snapshot:
        raise HarnessError(f"{source}:{line}: authorized task row is missing snapshot metadata")
    authorization = str(
        row.get("snapshot_authorization")
        or row.get("authorization")
        or row.get("dataset_source_kind")
        or row.get("source_kind")
        or ""
    ).strip().lower()
    if authorization not in AUTHORIZED_VALUES:
        raise HarnessError(f"{source}:{line}: unsupported snapshot authorization: {authorization or '<missing>'}")
    return TaskRecord(benchmark_id=benchmark_id, task_id=task_id, row=row, source_path=source, source_line=line)


def load_tasks(paths: list[Path], *, allow_local_dev: bool = False) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    seen: set[str] = set()
    for path in paths:
        for line_number, row in iter_jsonl(path):
            task = validate_authorized_task(row, path, line_number, allow_local_dev=allow_local_dev)
            key = f"{task.benchmark_id}:{task.task_id}"
            if key in seen:
                raise HarnessError(f"{path}:{line_number}: duplicate benchmark/task id: {key}")
            seen.add(key)
            tasks.append(task)
    if not tasks:
        noun = "task" if allow_local_dev else "authorized task"
        raise HarnessError(f"no {noun} rows found")
    return tasks


def scrub_task(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in SENSITIVE_TASK_KEYS or str(key).startswith("fixture_"):
                continue
            cleaned[str(key)] = scrub_task(item)
        return cleaned
    if isinstance(value, list):
        return [scrub_task(item) for item in value]
    return value


def prompt_from_task(row: dict[str, Any]) -> str:
    parts: list[str] = []
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").lower()
            content = message.get("content")
            if role == "assistant" or not isinstance(content, str):
                continue
            if content.strip():
                parts.append(f"{role}: {content.strip()}")
    for key in ("prompt", "question", "input", "instruction", "instructions", "issue", "description", "task", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if row.get("choices") not in (None, "", [], {}):
        parts.append("Choices: " + json.dumps(row.get("choices"), ensure_ascii=True, sort_keys=True))
    if not parts:
        scrubbed = scrub_task(row)
        parts.append("Solve the authorized evaluation task:\n" + json.dumps(scrubbed, ensure_ascii=True, sort_keys=True))
    return "\n\n".join(parts)


def model_request(task: TaskRecord, cfg: GenerateConfig) -> dict[str, Any]:
    return {
        "schema": "omnicoder.reportable_prediction_request_2026.v1",
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": task.benchmark_id,
        "task_id": task.task_id,
        "model": cfg.model,
        "checkpoint_path": cfg.checkpoint_path,
        "prompt": prompt_from_task(task.row),
        "task": scrub_task(task.row),
        "max_output_tokens": cfg.max_output_tokens,
        "temperature": cfg.temperature,
    }


def validate_local_endpoint(base_url: str) -> str:
    if not base_url:
        raise HarnessError("--base-url is required for --backend openai-compatible")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise HarnessError("OpenAI-compatible endpoint must use http or https")
    host = (parsed.hostname or "").lower()
    if host not in LOCAL_HOSTS:
        raise HarnessError(f"OpenAI-compatible endpoint must be local, got host {host or '<missing>'}")
    return base_url.rstrip("/")


def openai_chat_url(base_url: str) -> str:
    base = validate_local_endpoint(base_url)
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def parse_model_payload(value: Any, fallback_field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        for key in MODEL_OUTPUT_KEYS:
            if value.get(key) not in (None, "", [], {}):
                return {key: value[key]}
        if value.get("content") not in (None, "", [], {}):
            return {fallback_field: value["content"]}
        raise HarnessError("model response JSON object does not contain a model output field")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise HarnessError("model response was empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {fallback_field: text}
        return parse_model_payload(parsed, fallback_field)
    if value not in (None, "", [], {}):
        return {fallback_field: value}
    raise HarnessError("model response was empty")


def output_field_for_task(task: TaskRecord) -> str:
    row = task.row
    text = " ".join(
        str(row.get(key) or "")
        for key in ("benchmark_id", "adapter_id", "adapter_kind", "axis", "task_format", "source")
    ).lower()
    if "swe" in text or "patch" in text or "git" in text or row.get("repo"):
        return "model_patch"
    if "tool" in text or "bfcl" in text or "mcp" in text:
        return "tool_call"
    if "arc_agi3" in text or "interactive" in text:
        return "model_actions"
    if "image_generation" in text or "video_generation" in text or "audio_generation" in text or "music_generation" in text:
        return "artifact_path"
    return "prediction"


def fixture_response(task: TaskRecord) -> dict[str, Any]:
    row = task.row
    fixture_map = (
        ("fixture_prediction", "prediction"),
        ("fixture_model_patch", "model_patch"),
        ("fixture_model_actions", "model_actions"),
        ("fixture_tool_call", "tool_call"),
        ("fixture_artifact_path", "artifact_path"),
        ("fixture_output", output_field_for_task(task)),
    )
    for source_key, output_key in fixture_map:
        value = row.get(source_key)
        if value not in (None, "", [], {}):
            return {output_key: value}
    if isinstance(row.get("choices"), list) and row["choices"]:
        return {"prediction": row["choices"][0]}
    return {output_field_for_task(task): f"fixture-prediction:{task.task_id}"}


def call_openai_compatible(task: TaskRecord, cfg: GenerateConfig) -> dict[str, Any]:
    request = model_request(task, cfg)
    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "system",
                "content": "Return only the model prediction for this authorized reportable evaluation task.",
            },
            {"role": "user", "content": request["prompt"]},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_output_tokens,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(cfg.api_key_env, "") if cfg.api_key_env else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = urllib.request.Request(
        openai_chat_url(cfg.base_url),
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=cfg.timeout_seconds) as response:
        body = response.read().decode("utf-8")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"OpenAI-compatible endpoint returned invalid JSON: {exc}") from exc
    content = None
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
            if content is None:
                content = first.get("text")
    if content is None and isinstance(data, dict):
        content = data.get("output_text") or data.get("response") or data.get("prediction")
    return parse_model_payload(content, output_field_for_task(task))


def command_args(command: str) -> list[str]:
    args = shlex.split(command, posix=(os.name != "nt"))
    if not args:
        raise HarnessError("--checkpoint-runner command is empty")
    return args


def call_checkpoint_runner(task: TaskRecord, cfg: GenerateConfig) -> dict[str, Any]:
    if not cfg.checkpoint_runner:
        raise HarnessError("--checkpoint-runner is required for --backend checkpoint-runner")
    request = model_request(task, cfg)
    started = time.perf_counter()
    proc = subprocess.run(
        command_args(cfg.checkpoint_runner),
        input=json.dumps(request, ensure_ascii=True),
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        timeout=cfg.timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise HarnessError(
            f"checkpoint runner failed for {task.benchmark_id}:{task.task_id} "
            f"with exit {proc.returncode}: {proc.stderr[-1000:]}"
        )
    if not proc.stdout.strip():
        raise HarnessError(f"checkpoint runner produced no stdout after {elapsed:.3f}s")
    return parse_model_payload(proc.stdout, output_field_for_task(task))


def call_backend(task: TaskRecord, cfg: GenerateConfig) -> dict[str, Any]:
    if cfg.backend == "fixture":
        return fixture_response(task)
    if cfg.backend == "openai-compatible":
        return call_openai_compatible(task, cfg)
    if cfg.backend == "checkpoint-runner":
        return call_checkpoint_runner(task, cfg)
    raise HarnessError(f"unknown backend: {cfg.backend}")


def prediction_output_quality_rejections(row: dict[str, Any]) -> list[str]:
    rejected_outputs: list[str] = []
    for key in MODEL_OUTPUT_KEYS:
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        reason = output_quality_reason(value)
        if reason:
            rejected_outputs.append(f"{key}:{reason}")
    return rejected_outputs


def validate_prediction_row(row: dict[str, Any], *, allow_rejected_model_output: bool = False) -> None:
    for key in ("schema", "schema_version", "benchmark_id", "task_id", "model", "backend"):
        if row.get(key) in (None, "", [], {}):
            raise HarnessError(f"prediction row is missing {key}: {row!r}")
    outputs: list[str] = []
    for key in MODEL_OUTPUT_KEYS:
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        reason = output_quality_reason(value)
        if not reason:
            outputs.append(key)
    rejected_outputs = prediction_output_quality_rejections(row)
    if rejected_outputs:
        if allow_rejected_model_output:
            return
        raise HarnessError(f"prediction row has rejected model output {rejected_outputs}: {row!r}")
    if not outputs:
        raise HarnessError(f"prediction row has no model output field: {row!r}")


def prediction_row(task: TaskRecord, cfg: GenerateConfig) -> dict[str, Any]:
    started = time.perf_counter()
    output = call_backend(task, cfg)
    elapsed = round(time.perf_counter() - started, 6)
    if not isinstance(output, dict):
        output = parse_model_payload(output, output_field_for_task(task))
    row = {
        "schema": PREDICTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "benchmark_id": task.benchmark_id,
        "task_id": task.task_id,
        "task_revision": str(task.row.get("task_revision") or task.row.get("dataset_revision") or "unknown"),
        "dataset_revision": str(task.row.get("dataset_revision") or task.row.get("task_revision") or "unknown"),
        "snapshot_id": str(
            task.row.get("snapshot_id")
            or task.row.get("official_snapshot_id")
            or task.row.get("authorized_snapshot_id")
            or task.row.get("snapshot_sha256")
            or ""
        ),
        "model": cfg.model,
        "backend": cfg.backend,
        "source_task_path": str(task.source_path),
        "source_line": task.source_line,
        "task_row_sha256": stable_hash(task.row),
        "task_file_sha256": file_sha256(task.source_path),
        "request_sha256": stable_hash(model_request(task, cfg)),
        "latency_seconds": elapsed,
    }
    row.update(output)
    row["prediction_id"] = stable_hash({key: value for key, value in row.items() if key != "prediction_id"})
    validate_prediction_row(row)
    return row


def run_generation(
    task_inputs: list[str],
    out_path: str,
    cfg: GenerateConfig,
    *,
    force: bool,
    allow_local_dev: bool = False,
) -> dict[str, Any]:
    paths = task_paths(task_inputs)
    tasks = load_tasks(paths, allow_local_dev=allow_local_dev)
    rows = [prediction_row(task, cfg) for task in tasks]
    out = resolve_path(out_path)
    write_jsonl(out, rows, force=force)
    by_backend = {cfg.backend: len(rows)}
    task_mode = "local_public_dev" if allow_local_dev else "authorized_reportable"
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "predictions": str(out),
        "records": len(rows),
        "tasks": [str(path) for path in paths],
        "task_mode": task_mode,
        "official_score": False,
        "model": cfg.model,
        "backend_counts": by_backend,
        "prediction_sha256": file_sha256(out),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reportable benchmark prediction JSONL from authorized tasks")
    parser.add_argument("--tasks", action="append", required=True, help="Authorized task JSONL file or directory; repeatable")
    parser.add_argument("--out", required=True, help="Prediction JSONL path for benchmark-suite-2026 run-reportable")
    parser.add_argument(
        "--backend",
        choices=["fixture", "openai-compatible", "checkpoint-runner"],
        required=True,
        help="Prediction backend. fixture is deterministic and performs no network I/O.",
    )
    parser.add_argument("--model", default="unknown", help="Model id, checkpoint label, or local serving route")
    parser.add_argument("--base-url", default="", help="Local OpenAI-compatible /v1 endpoint base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding local endpoint API key")
    parser.add_argument("--checkpoint-runner", default="", help="Local command that reads one JSON request on stdin and writes JSON/text prediction")
    parser.add_argument("--checkpoint-path", default="", help="Optional local checkpoint path passed through to checkpoint runners")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--allow-one-token-canary",
        action="store_true",
        help="Permit --max-output-tokens <= 1 only for explicit non-reportable canary probes.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--summary", default="", help="Optional summary JSON path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing --out")
    parser.add_argument(
        "--allow-local-dev-tasks",
        action="store_true",
        help=(
            "Accept public-dev/local-regression task rows such as reportable=false. "
            "Predictions produced with this flag are never official/reportable scores."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GenerateConfig:
    if args.max_output_tokens <= 0:
        raise HarnessError("--max-output-tokens must be positive")
    if args.max_output_tokens <= 1 and not bool(args.allow_one_token_canary):
        raise HarnessError(
            "--max-output-tokens <= 1 is canary-only; pass --allow-one-token-canary "
            "only for explicit non-reportable smoke runs"
        )
    if args.timeout_seconds <= 0:
        raise HarnessError("--timeout-seconds must be positive")
    if args.backend == "openai-compatible":
        validate_local_endpoint(args.base_url)
    if args.backend == "checkpoint-runner" and not str(args.checkpoint_runner).strip():
        raise HarnessError("--checkpoint-runner is required for checkpoint-runner backend")
    return GenerateConfig(
        backend=str(args.backend),
        model=str(args.model or "unknown"),
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
        timeout_seconds=int(args.timeout_seconds),
        base_url=str(args.base_url or ""),
        api_key_env=str(args.api_key_env or ""),
        checkpoint_runner=str(args.checkpoint_runner or ""),
        checkpoint_path=str(args.checkpoint_path or ""),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_generation(
            args.tasks,
            args.out,
            config_from_args(args),
            force=bool(args.force),
            allow_local_dev=bool(args.allow_local_dev_tasks),
        )
        if args.summary:
            write_json(resolve_path(args.summary), summary)
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 0
    except (HarnessError, subprocess.TimeoutExpired, OSError) as exc:
        payload = {"status": "error", "error": str(exc), "schema_version": SCHEMA_VERSION}
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

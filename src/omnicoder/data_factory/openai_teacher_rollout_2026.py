from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a teacher model producing concise, structured distillation targets "
    "for Omnicoder's dense omnimodal student. Return useful corrections, tool "
    "plans, verifier notes, and reward labels. Return one JSON object whenever "
    "possible with keys: corrected_response, corrected_tool_calls, chosen, "
    "rejected, reward, reward_components, verifier_labels, process_labels, "
    "safety_notes."
)


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                row = {"text": line.rstrip("\n")}
            if isinstance(row, dict):
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n")


def existing_jsonl_rows(path: str | Path) -> int:
    source = Path(path)
    if not source.exists() or not source.is_file():
        return 0
    rows = 0
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def extract_prompt(row: dict[str, Any]) -> str:
    payload = row.get("input_json") if isinstance(row.get("input_json"), dict) else row
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                role = str(message.get("role") or "user")
                parts.append(f"{role}: {message['content']}")
        if parts:
            return "\n".join(parts)
    for key in ("prompt", "text", "content", "normalized_text", "completion", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(row, ensure_ascii=True, sort_keys=True)[:4000]


def post_chat(base_url: str, model: str, prompt: str, timeout: int, max_tokens: int, temperature: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=int(timeout)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def teacher_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                content = message["content"]
                if content.strip():
                    return content
                for key in ("reasoning_content", "reasoning", "analysis"):
                    fallback = message.get(key)
                    if isinstance(fallback, str) and fallback.strip():
                        return fallback
            if isinstance(first.get("text"), str):
                return first["text"]
    return json.dumps(response, ensure_ascii=True, sort_keys=True)


def parse_teacher_signal(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    probes = [content.strip()]
    if "```" in content:
        import re

        probes.extend(re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE))
    for probe in probes:
        starts = [idx for idx, char in enumerate(probe) if char == "{"]
        if 0 not in starts:
            starts.insert(0, 0)
        for start in starts:
            try:
                value, _ = decoder.raw_decode(probe[start:])
            except Exception:
                continue
            if isinstance(value, dict):
                return value
    return {}


def gpu_temperature(index: str) -> int | None:
    if not index:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    try:
        return int(float(line.strip()))
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible teacher rollout writer for Omnicoder 2026 distillation data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--model", default="qwen3.6-27b-q4")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--record-kind", default="qwen36_p40_teacher_rollout")
    parser.add_argument("--thermal-gpu-index", default="", help="Optional nvidia-smi GPU index to guard before each request")
    parser.add_argument("--max-gpu-temp", type=int, default=0, help="Stop before dispatch when guarded GPU reaches this Celsius temperature")
    parser.add_argument("--resume", action="store_true", help="Skip rows already written to the output JSONL so interrupted rollouts can continue")
    args = parser.parse_args(argv)

    rows = read_jsonl(args.input, limit=int(args.limit))
    skipped_existing = existing_jsonl_rows(args.out) if args.resume else 0
    if skipped_existing:
        rows_to_process = rows[skipped_existing:]
    else:
        rows_to_process = rows
    emitted: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stopped_for_thermal = False
    for relative_index, row in enumerate(rows_to_process, 1):
        index = skipped_existing + relative_index
        if args.max_gpu_temp > 0 and str(args.thermal_gpu_index).strip():
            temperature = gpu_temperature(str(args.thermal_gpu_index).strip())
            if temperature is not None and temperature >= int(args.max_gpu_temp):
                stopped_for_thermal = True
                emitted.append(
                    {
                        "schema": "omnicoder.openai_teacher_rollout_2026.v1",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "run_started_at": started,
                        "index": index,
                        "status": "stopped",
                        "error": f"thermal_guard_gpu_{args.thermal_gpu_index}_temp_{temperature}_ge_{args.max_gpu_temp}",
                        "teacher": args.model,
                        "base_url": args.base_url,
                        "record_kind": args.record_kind,
                        "input_json": {"source_record": row},
                        "target_json": {"content": "", "teacher_status": "stopped"},
                        "modalities": ["text", "tool"],
                        "split": "train",
                        "quality_score": 0.0,
                    }
                )
                break
        prompt = extract_prompt(row)
        try:
            response = post_chat(args.base_url, args.model, prompt, args.timeout, args.max_tokens, args.temperature)
            content = teacher_text(response)
            parsed_signal = parse_teacher_signal(content)
            if content.strip():
                status = "ok"
                error = ""
            else:
                status = "failed"
                error = "empty_teacher_content"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            content = ""
            parsed_signal = {}
            status = "failed"
            error = repr(exc)
        emitted.append(
            {
                "schema": "omnicoder.openai_teacher_rollout_2026.v1",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_started_at": started,
                "index": index,
                "status": status,
                "error": error,
                "teacher": args.model,
                "base_url": args.base_url,
                "record_kind": args.record_kind,
                "input_json": {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "source_record": row,
                },
                "target_json": {
                    "content": content,
                    "teacher_status": status,
                    "teacher_signal": parsed_signal,
                },
                "modalities": row.get("modalities") if isinstance(row.get("modalities"), list) else ["text", "tool"],
                "split": "train",
                "quality_score": 0.75 if status == "ok" else 0.0,
            }
        )
        if len(emitted) >= 8:
            write_jsonl(args.out, emitted)
            emitted.clear()
        if args.sleep:
            time.sleep(float(args.sleep))
    if emitted:
        write_jsonl(args.out, emitted)
    status = "stopped_thermal_guard" if stopped_for_thermal else "ok"
    print(
        json.dumps(
            {
                "status": status,
                "input": args.input,
                "out": args.out,
                "records": len(rows),
                "processed": len(rows_to_process),
                "skipped_existing": skipped_existing,
                "resume": bool(args.resume),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

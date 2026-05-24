from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from omnicoder.data_factory.postgres import claim_work, complete_work


STAGE_MODULES = {
    "ingest_agent_memory": "omnicoder.data_factory.ingest_agent_memory",
    "ingest_codex_transcripts": "omnicoder.data_factory.ingest_codex_transcripts",
    "ingest_comfyui_outputs": "omnicoder.data_factory.ingest_comfyui_outputs",
    "quality_scoring": "omnicoder.data_factory.quality_scoring",
    "contamination": "omnicoder.data_factory.contamination",
    "export_sft_jsonl": "omnicoder.data_factory.export_sft_jsonl",
}


UNDERSCORE_FLAGS = {"dataset_name", "source_date"}


def payload_to_argv(payload: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in payload.items():
        if key in {"stage", "module"}:
            continue
        flag_name = key if key in UNDERSCORE_FLAGS else key.replace("_", "-")
        flag = "--" + flag_name
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            argv.extend([flag, json.dumps(value, ensure_ascii=True)])
        else:
            argv.extend([flag, str(value)])
    return argv


def run_stage(stage: str, payload: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    module = STAGE_MODULES.get(stage)
    if module is None:
        raise ValueError(f"unknown data_factory stage: {stage}")
    command = [sys.executable, "-m", module, *payload_to_argv(payload)]
    if dry_run:
        return {"command": command, "dry_run": True}
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }


def parse_payload(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    path = Path(value)
    if path.exists() and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    stripped = value.strip().strip("{}")
    payload: dict[str, Any] = {}
    for pair in stripped.split(","):
        if ":" not in pair:
            continue
        key, raw = pair.split(":", 1)
        payload[key.strip().strip("'\"")] = raw.strip().strip("'\"")
    if payload:
        return payload
    raise ValueError("offline payload must be a JSON object, path to a JSON object, or {key:value} pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim and run raw-PostgreSQL data-factory work_queue jobs")
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_MODULES))
    parser.add_argument("--worker-id", default="data_factory_worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-payload", default=None, help="JSON object to run without claiming PostgreSQL work")
    args = parser.parse_args()

    if args.offline_payload:
        payload = parse_payload(args.offline_payload)
        print(json.dumps(run_stage(args.stage, payload, args.dry_run), ensure_ascii=True))
        return

    while True:
        job = claim_work(args.stage, args.worker_id)
        if job is None:
            print(json.dumps({"status": "idle", "stage": args.stage}, ensure_ascii=True))
            return
        try:
            result = run_stage(args.stage, job["payload"], args.dry_run)
            status = "done" if result.get("returncode", 0) == 0 else "failed"
            complete_work(job["work_id"], status, result)
            print(json.dumps({"status": status, "work_id": job["work_id"], "result": result}, ensure_ascii=True))
        except Exception as exc:
            complete_work(job["work_id"], "failed", {"error": str(exc)})
            print(json.dumps({"status": "failed", "work_id": job["work_id"], "error": str(exc)}, ensure_ascii=True))
        if args.once:
            return


if __name__ == "__main__":
    main()

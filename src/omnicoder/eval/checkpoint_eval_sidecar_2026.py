from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "omnicoder.checkpoint_eval_sidecar_2026.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":")) + "\n")


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def checkpoint_fingerprint(checkpoint: Path) -> dict[str, Any]:
    manifest = checkpoint / "manifest.json"
    complete = checkpoint / ".complete.json"
    rank_files = sorted(checkpoint.glob("rank*.pt"))
    h = hashlib.sha256()
    for path in [manifest, complete, *rank_files]:
        if path.exists():
            h.update(path.name.encode("utf-8"))
            h.update(str(path.stat().st_size).encode("utf-8"))
            h.update(str(int(path.stat().st_mtime_ns)).encode("utf-8"))
    return {
        "checkpoint": str(checkpoint),
        "complete": bool(complete.exists()),
        "manifest": bool(manifest.exists()),
        "rank_files": [path.name for path in rank_files],
        "rank_file_count": len(rank_files),
        "fingerprint": f"sha256:{h.hexdigest()}",
    }


def _torchrun_prefix(nproc: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(max(1, int(nproc))),
        "--max_restarts",
        "0",
    ]


def write_decode_sanity_tasks(out_dir: Path) -> Path:
    tasks = [
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "text_probe",
            "reportable": False,
            "prompt": "Write two concise sentences explaining why phase timing helps optimize model training.",
            "expected_signal": "coherent_text",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "code_probe",
            "reportable": False,
            "prompt": "Write a Python function add(a, b) and include one assert-based test.",
            "expected_signal": "coherent_code",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "tool_probe",
            "reportable": False,
            "prompt": "Return only a JSON object for a tool call that runs pytest on tests/test_pipeline_pretrain_2026.py.",
            "expected_signal": "tool_call_json",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "math_probe",
            "reportable": False,
            "prompt": "Solve 17 * 19 and include one sentence explaining the arithmetic.",
            "expected_signal": "coherent_math",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "image_route_probe",
            "reportable": False,
            "prompt": "Return an image artifact route descriptor for a 512x512 diagnostic image of a blue cube on a white background.",
            "expected_signal": "image_route_or_artifact_descriptor",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "video_route_probe",
            "reportable": False,
            "prompt": "Return a video artifact route descriptor for a three-second diagnostic clip of a cube rotating on a white background.",
            "expected_signal": "video_route_or_artifact_descriptor",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "audio_route_probe",
            "reportable": False,
            "prompt": "Return an audio artifact route descriptor for a one-second diagnostic sine sweep.",
            "expected_signal": "audio_route_or_artifact_descriptor",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "music_route_probe",
            "reportable": False,
            "prompt": "Return a music artifact route descriptor for a short bright piano cadence.",
            "expected_signal": "music_route_or_artifact_descriptor",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "tts_route_probe",
            "reportable": False,
            "prompt": "Return a TTS artifact route descriptor for the phrase 'OmniCoder diagnostics are online.'",
            "expected_signal": "tts_route_or_artifact_descriptor",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "ocr_probe",
            "reportable": False,
            "prompt": "Return OCR text extracted from an imagined document containing the words 'target coverage ok'.",
            "expected_signal": "ocr_text",
        },
        {
            "benchmark_id": "local_decode_sanity",
            "task_id": "long_context_probe",
            "reportable": False,
            "prompt": "Repeat the key phrase 'needle-ready' once, then summarize why long-context recall gates matter.",
            "expected_signal": "long_context_recall_text",
        },
    ]
    path = out_dir / "decode_sanity_tasks.jsonl"
    _write_jsonl(path, tasks)
    return path


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint = str(args.checkpoint)
    out_dir = Path(args.out_dir)
    data_args: list[str] = []
    for item in args.data or []:
        data_args.extend(["--data", str(item)])
    if args.data_dir:
        data_args.extend(["--data-dir", str(args.data_dir)])
    common_pipeline = [
        "--checkpoint",
        checkpoint,
        "--preset",
        str(args.preset),
        "--rank-device-map",
        str(args.rank_device_map),
        "--placement-layer-counts",
        str(args.placement_layer_counts),
        "--precision",
        str(args.precision),
        "--init-dtype",
        str(args.init_dtype),
    ]
    if args.fake_quant:
        common_pipeline.append("--fake-quant")
    if int(args.fake_quant_chunk_rows or 0) > 0:
        common_pipeline.extend(["--fake-quant-chunk-rows", str(int(args.fake_quant_chunk_rows))])
    if int(args.fake_quant_max_full_elements or 0) > 0:
        common_pipeline.extend(["--fake-quant-max-full-elements", str(int(args.fake_quant_max_full_elements))])
    if args.require_target_contract:
        common_pipeline.append("--require-target-contract")
    decode_tasks = write_decode_sanity_tasks(out_dir)

    jobs: list[dict[str, Any]] = []
    jobs.append(
        {
            "name": "media_route_probe",
            "required": True,
            "artifact": out_dir / "media_route_probe.json",
            "cmd": [
                sys.executable,
                "-m",
                "omnicoder.eval.media_route_probe_2026",
                "--out",
                str(out_dir / "media_route_probe.json"),
            ],
        }
    )
    jobs.append(
        {
            "name": "target_token_diagnostics",
            "required": bool(data_args),
            "artifact": out_dir / "target_token_diagnostics.json",
            "cmd": _torchrun_prefix(args.nproc_per_node)
            + [
                "-m",
                "omnicoder.eval.pipeline_target_token_diagnostics_2026",
                *common_pipeline,
                *data_args,
                "--seq-len",
                str(int(args.seq_len)),
                "--max-records-per-file",
                str(int(args.max_records_per_file)),
                "--out",
                str(out_dir / "target_token_diagnostics.json"),
            ],
        }
    )
    jobs.append(
        {
            "name": "heldout_pipeline_sample_loss",
            "required": bool(data_args),
            "artifact": out_dir / "heldout_pipeline_sample_loss.json",
            "cmd": _torchrun_prefix(args.nproc_per_node)
            + [
                "-m",
                "omnicoder.eval.pipeline_sample_loss_2026",
                *common_pipeline,
                *data_args,
                "--seq-len",
                str(int(args.seq_len)),
                "--max-records-per-file",
                str(int(args.max_records_per_file)),
                "--out",
                str(out_dir / "heldout_pipeline_sample_loss.json"),
            ],
        }
    )
    jobs.append(
        {
            "name": "token_topk_probe",
            "required": True,
            "artifact": out_dir / "token_topk_probe.json",
            "cmd": _torchrun_prefix(args.nproc_per_node)
            + [
                "-m",
                "omnicoder.eval.pipeline_token_topk_probe_2026",
                *common_pipeline,
                "--out",
                str(out_dir / "token_topk_probe.json"),
            ],
        }
    )
    jobs.append(
        {
            "name": "decode_sanity_predictions",
            "required": True,
            "artifact": out_dir / "decode_sanity_predictions.jsonl",
            "cmd": [
                sys.executable,
                "-m",
                "omnicoder.eval.pipeline_checkpoint_batch_predict_2026",
                "--checkpoint",
                checkpoint,
                "--tasks",
                str(decode_tasks),
                "--out",
                str(out_dir / "decode_sanity_predictions.jsonl"),
                "--summary",
                str(out_dir / "decode_sanity_summary.json"),
                "--preset",
                str(args.preset),
                "--rank-device-map",
                str(args.rank_device_map),
                "--placement-layer-counts",
                str(args.placement_layer_counts),
                "--precision",
                str(args.precision),
                "--init-dtype",
                str(args.init_dtype),
                "--nproc-per-node",
                str(max(1, int(args.nproc_per_node))),
                "--max-prompt-tokens",
                str(int(args.decode_max_prompt_tokens)),
                "--max-output-tokens",
                str(int(args.decode_max_output_tokens)),
                "--allow-local-dev-tasks",
                "--allow-media-route-text-proof",
                "--force",
            ]
            + (["--fake-quant"] if args.fake_quant else [])
            + (
                ["--fake-quant-chunk-rows", str(int(args.fake_quant_chunk_rows))]
                if int(args.fake_quant_chunk_rows or 0) > 0
                else []
            )
            + (
                ["--fake-quant-max-full-elements", str(int(args.fake_quant_max_full_elements))]
                if int(args.fake_quant_max_full_elements or 0) > 0
                else []
            )
            + (["--require-target-contract"] if args.require_target_contract else []),
        }
    )
    return jobs


def run_job(job: dict[str, Any], *, timeout_seconds: int, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    artifact = Path(job["artifact"])
    result: dict[str, Any] = {
        "name": job["name"],
        "required": bool(job.get("required")),
        "artifact": str(artifact),
        "cmd": [str(item) for item in job["cmd"]],
        "started_at": started,
        "dry_run": bool(dry_run),
    }
    if not bool(job.get("required")):
        result.update({"status": "skipped", "reason": "no_data_bound"})
        return result
    if dry_run:
        result.update({"status": "planned", "finished_at": time.time(), "duration_sec": 0.0})
        return result
    artifact.parent.mkdir(parents=True, exist_ok=True)
    log_path = artifact.with_suffix(artifact.suffix + ".log")
    finished = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(job["cmd"], stdout=log, stderr=subprocess.STDOUT, timeout=int(timeout_seconds), text=True)
        returncode = int(proc.returncode)
        status = "passed" if returncode == 0 and artifact.exists() else "failed"
        error = ""
    except subprocess.TimeoutExpired as exc:
        finished = time.time()
        returncode = None
        status = "failed"
        error = f"timeout_expired:{exc}"
    except (OSError, subprocess.SubprocessError) as exc:
        finished = time.time()
        returncode = None
        status = "failed"
        error = f"subprocess_error:{exc}"
    result.update(
        {
            "status": status,
            "returncode": returncode,
            "finished_at": finished,
            "duration_sec": float(finished - started),
            "log": str(log_path),
            "artifact_exists": artifact.exists(),
            "artifact_sha256": _file_sha256(artifact),
        }
    )
    if error:
        result["error"] = error
    return result


def run_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = checkpoint_fingerprint(checkpoint)
    jobs = build_jobs(args)
    results = [run_job(job, timeout_seconds=int(args.timeout_seconds), dry_run=bool(args.dry_run)) for job in jobs]
    failed = [item for item in results if item.get("required") and item.get("status") not in {"passed", "planned"}]
    manifest = {
        "schema": SCHEMA,
        "status": "failed" if failed else ("planned" if args.dry_run else "passed"),
        "created_at": time.time(),
        "checkpoint": fingerprint,
        "out_dir": str(out_dir),
        "jobs": results,
        "failed_jobs": [str(item.get("name")) for item in failed],
    }
    _write_json(out_dir / "checkpoint_eval_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded eval sidecar jobs for a complete Omnicoder sharded checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--preset", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRESET", "omnicoder2026_20b_1m"))
    parser.add_argument("--rank-device-map", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_RANK_DEVICE_MAP", ""))
    parser.add_argument("--placement-layer-counts", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PLACEMENT_LAYER_COUNTS", ""))
    parser.add_argument("--precision", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_PRECISION", "fp16"), choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--init-dtype", default=os.getenv("OMNICODER2026_PIPELINE_BATCH_INIT_DTYPE", "auto"), choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--nproc-per-node", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_EXPECTED_WORLD_SIZE", "1") or 1))
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-records-per-file", type=int, default=4)
    parser.add_argument("--decode-max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--decode-max-output-tokens", type=int, default=64)
    parser.add_argument("--fake-quant", action="store_true")
    parser.add_argument("--fake-quant-chunk-rows", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_FAKE_QUANT_CHUNK_ROWS", "0") or 0))
    parser.add_argument("--fake-quant-max-full-elements", type=int, default=int(os.getenv("OMNICODER2026_PIPELINE_BATCH_FAKE_QUANT_MAX_FULL_ELEMENTS", "0") or 0))
    parser.add_argument("--require-target-contract", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_sidecar(args)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["status"] in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

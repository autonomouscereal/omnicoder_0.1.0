from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


SCHEMA = "omnicoder.omnimodal_overfit_proof_2026.v1"
PROOF_GROUPS = ("text", "code_tool", "image_ocr", "video", "audio_tts_music", "ledger_all")
DEFAULT_MAX_RELOAD_SAMPLE_LOSS = 0.05
_TOKENIZER_CACHE: Any | None = None


class _ProofFallbackTokenizer:
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            code = ord(ch)
            ids.append((code - 32) + 2 if 32 <= code < 127 else 1)
        return ids or [1]


def _get_proof_tokenizer(*, prefer_hf: bool = True) -> Any:
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE
    try:
        from omnicoder.training.simple_tokenizer import get_text_tokenizer

        _TOKENIZER_CACHE = get_text_tokenizer(prefer_hf=prefer_hf)
    except Exception:
        _TOKENIZER_CACHE = _ProofFallbackTokenizer()
    return _TOKENIZER_CACHE


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return float(default)
    return float(raw)


def _add_failure(report: dict[str, Any], reason: str) -> None:
    failures = report.setdefault("failures", [])
    if reason not in failures:
        failures.append(reason)
    report.setdefault("failure", reason)


def _bucket_loss(bucket: Any) -> float | None:
    if not isinstance(bucket, dict):
        return None
    for key in ("loss", "avg_loss"):
        value = bucket.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    tokens = int(bucket.get("tokens") or 0)
    loss_sum = bucket.get("loss_sum")
    if tokens > 0 and loss_sum not in (None, ""):
        return float(loss_sum) / float(tokens)
    return None


def _sample_loss_failures(loss_json: Any, *, max_loss: float) -> list[dict[str, Any]]:
    if not isinstance(loss_json, dict):
        return [{"reason": "invalid_sample_loss", "bucket": "root"}]
    checks: list[tuple[str, Any]] = [("overall", loss_json.get("overall"))]
    modalities = loss_json.get("modalities")
    if isinstance(modalities, dict):
        for name, bucket in sorted(modalities.items()):
            checks.append((f"modality:{name}", bucket))
    failures: list[dict[str, Any]] = []
    for bucket_name, bucket in checks:
        if not isinstance(bucket, dict):
            failures.append({"reason": "missing_sample_loss_bucket", "bucket": bucket_name})
            continue
        tokens = int(bucket.get("tokens") or 0)
        loss = _bucket_loss(bucket)
        if tokens <= 0 or loss is None:
            failures.append({"reason": "missing_sample_loss", "bucket": bucket_name, "tokens": tokens})
            continue
        if not math.isfinite(float(loss)):
            failures.append({"reason": "nonfinite_sample_loss", "bucket": bucket_name, "loss": loss, "tokens": tokens})
            continue
        if float(loss) > float(max_loss):
            failures.append(
                {
                    "reason": "high_sample_loss",
                    "bucket": bucket_name,
                    "loss": float(loss),
                    "max_loss": float(max_loss),
                    "tokens": tokens,
                }
            )
    return failures


def _target_token_count(target_json: Any) -> int:
    if not isinstance(target_json, dict):
        return 0
    for key in ("target_tokens", "target_token_count"):
        value = target_json.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    overall = target_json.get("overall")
    if isinstance(overall, dict):
        for key in ("target_tokens", "target_token_count"):
            value = overall.get(key)
            if value not in (None, ""):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
    return 0


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids: list[int] = []
    for raw in tokenizer.encode(text):
        token_id = int(raw)
        ids.append(token_id if 0 <= token_id < 128_000 else 1)
    return ids or [1]


def _span(family: str, seed: int, count: int = 8) -> list[int]:
    lo, hi = DEFAULT_LEDGER.as_config_ranges()[family]
    width = int(hi) - int(lo)
    return [int(lo) + ((int(seed) * 37 + index * 11) % width) for index in range(count)]


def _row(
    *,
    group: str,
    modality: str,
    prompt: str,
    target_ids: list[int],
    target_family: str,
    target_text: str = "",
) -> dict[str, Any]:
    tokenizer = _get_proof_tokenizer(prefer_hf=True)
    prompt_ids = _encode(tokenizer, prompt)
    row = {
        "schema": SCHEMA,
        "group": group,
        "modality": modality,
        "prompt": prompt,
        "target": target_text or f"{target_family} proof target",
        "prompt_token_ids": prompt_ids,
        "target_token_ids": [int(token_id) for token_id in target_ids],
        "source_id": f"omnimodal_overfit_{group}_{target_family}",
        "source_uri": "local://omnicoder/overfit_proof_2026",
        "source_date": "2026-05-28",
        "license": "internal diagnostic proof rows",
        "split": "train",
        "quality_score": 1.0,
        "contamination_status": "clean",
        "task_type": "overfit_trainability_proof",
        "diagnostic_only": True,
        "target_ledger_family": target_family,
        "valid_target_tokens": len(target_ids),
    }
    if target_family in {"vision_semantic", "vision_residual"}:
        row["target_json"] = {"content": target_text or "image | {\"artifact\":\"proof\"}", "artifact_tokens": "<image_proof_tokens>"}
        row["artifact_token_ids"] = [int(token_id) for token_id in target_ids]
    if target_family in {"speech_tts", "audio_music", "music_control"}:
        row["target_json"] = {"content": target_text or "music | {\"artifact\":\"proof\"}", "artifact_tokens": "<audio_proof_tokens>"}
        row["artifact_token_ids"] = [int(token_id) for token_id in target_ids]
    return row


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    data_dir = out_dir / "data"
    tasks_dir = out_dir / "tasks"
    tokenizer = _get_proof_tokenizer(prefer_hf=True)
    examples = max(1, int(args.examples_per_modality))
    ranges = DEFAULT_LEDGER.as_config_ranges()
    groups: dict[str, list[dict[str, Any]]] = {}

    groups["text"] = [
        _row(
            group="text",
            modality="text",
            prompt=f"user: remember text proof {index}\nassistant:",
            target_ids=_encode(tokenizer, f" text_proof_answer_{index} stable"),
            target_text=f"text_proof_answer_{index} stable",
            target_family="text",
        )
        for index in range(examples)
    ]
    groups["code_tool"] = []
    for index in range(examples):
        if index % 2 == 0:
            groups["code_tool"].append(
                _row(
                    group="code_tool",
                    modality="code",
                    prompt=f"user: code proof {index}\nassistant:",
                    target_ids=_encode(tokenizer, f" def proof_{index}(): return {index}"),
                    target_text=f"def proof_{index}(): return {index}",
                    target_family="text",
                )
            )
        else:
            groups["code_tool"].append(
                _row(
                    group="code_tool",
                    modality="tool",
                    prompt=f"user: tool proof {index}\nassistant:",
                    target_ids=_span("tool_agent", index),
                    target_family="tool_agent",
                )
            )
    groups["image_ocr"] = []
    for index in range(examples):
        if index % 2 == 0:
            groups["image_ocr"].append(
                _row(
                    group="image_ocr",
                    modality="image",
                    prompt=f"user: image artifact proof {index}\nassistant:",
                    target_ids=_span("vision_semantic", index, 6) + _span("vision_residual", index, 4),
                    target_text=f"image | {{\"proof_id\": {index}}}",
                    target_family="vision_semantic",
                )
            )
        else:
            groups["image_ocr"].append(
                _row(
                    group="image_ocr",
                    modality="ocr",
                    prompt=f"user: OCR proof {index}\nassistant:",
                    target_ids=_encode(tokenizer, f" OCR_TEXT_PROOF_{index}"),
                    target_text=f"OCR_TEXT_PROOF_{index}",
                    target_family="text",
                )
            )
    groups["video"] = [
        _row(
            group="video",
            modality="video",
            prompt=f"user: video temporal proof {index}\nassistant:",
            target_ids=_span("vision_semantic", index, 5) + _span("time_space", index, 5),
            target_text=f"video | {{\"proof_id\": {index}, \"frames\": 4}}",
            target_family="time_space",
        )
        for index in range(examples)
    ]
    audio_families = ("speech_tts", "audio_music", "music_control")
    groups["audio_tts_music"] = [
        _row(
            group="audio_tts_music",
            modality=("tts" if index % 3 == 0 else "audio" if index % 3 == 1 else "music"),
            prompt=f"user: audio tts music proof {index}\nassistant:",
            target_ids=_span(audio_families[index % len(audio_families)], index, 8),
            target_text=f"{'speech' if index % 3 == 0 else 'music'} | {{\"proof_id\": {index}}}",
            target_family=audio_families[index % len(audio_families)],
        )
        for index in range(examples)
    ]
    ledger_names = list(ranges)
    groups["ledger_all"] = [
        _row(
            group="ledger_all",
            modality=("text" if family == "control" else "tool" if family == "tool_agent" else "image"),
            prompt=f"user: ledger family {family} proof\nassistant:",
            target_ids=_span(family, index, 8),
            target_family=family,
        )
        for index, family in enumerate(ledger_names[:examples])
    ]

    manifest_rows: list[dict[str, Any]] = []
    for group, rows in groups.items():
        _write_jsonl(data_dir / f"{group}.jsonl", rows)
        task_rows = [
            {
                "benchmark_id": f"local_{group}_overfit",
                "task_id": f"{group}_{index:02d}",
                "reportable": False,
                "prompt": row["prompt"],
                "source": "local_overfit_2026",
                "task_format": "local_overfit",
                "output_modality": row["modality"],
                "target_modality": row["modality"],
                "output_field": "prediction" if row["modality"] in {"text", "code", "tool", "ocr"} else "generated_artifact",
            }
            for index, row in enumerate(rows)
        ]
        _write_jsonl(tasks_dir / f"{group}.jsonl", task_rows)
        families = Counter(str(row.get("target_ledger_family") or "unknown") for row in rows)
        manifest_rows.append(
            {
                "group": group,
                "rows": len(rows),
                "data": str(data_dir / f"{group}.jsonl"),
                "tasks": str(tasks_dir / f"{group}.jsonl"),
                "target_families": dict(sorted(families.items())),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "examples_per_modality": examples,
        "groups": manifest_rows,
        "ledger": DEFAULT_LEDGER.as_metadata(),
        "commands": {
            "train_module": "omnicoder.training.pipeline_pretrain_2026_dense",
            "sample_loss_module": "omnicoder.eval.pipeline_sample_loss_2026",
            "target_diagnostics_module": "omnicoder.eval.pipeline_target_token_diagnostics_2026",
            "batch_predict_module": "omnicoder.eval.pipeline_checkpoint_batch_predict_2026",
        },
    }
    _write_json(out_dir / "omnimodal_overfit_manifest.json", manifest)
    return {"status": "ok", "manifest": str(out_dir / "omnimodal_overfit_manifest.json"), "groups": len(manifest_rows)}


def summary(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run)
    data_dir = run_dir / "data"
    eval_dir = run_dir / "eval"
    max_reload_sample_loss = float(getattr(args, "max_reload_sample_loss", DEFAULT_MAX_RELOAD_SAMPLE_LOSS))
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "run": str(run_dir),
        "groups": {},
        "status": "passed",
        "max_reload_sample_loss": max_reload_sample_loss,
    }
    for group in PROOF_GROUPS:
        data_path = data_dir / f"{group}.jsonl"
        loss_path = eval_dir / f"{group}.loss.json"
        target_path = eval_dir / f"{group}.targets.json"
        group_report: dict[str, Any] = {"data": str(data_path), "loss": str(loss_path), "targets": str(target_path)}
        if data_path.exists():
            rows = _read_jsonl(data_path)
            group_report["rows"] = len(rows)
            group_report["target_families"] = dict(sorted(Counter(str(row.get("target_ledger_family") or "unknown") for row in rows).items()))
        if loss_path.exists():
            try:
                loss_json = json.loads(loss_path.read_text(encoding="utf-8"))
                group_report["loss_json"] = loss_json
                sample_loss_failures = _sample_loss_failures(loss_json, max_loss=max_reload_sample_loss)
                if sample_loss_failures:
                    group_report["sample_loss_failures"] = sample_loss_failures
                    for failure in sample_loss_failures:
                        _add_failure(group_report, str(failure.get("reason") or "sample_loss_failed"))
            except Exception as exc:
                group_report["loss_error"] = repr(exc)
                _add_failure(group_report, "invalid_sample_loss")
        else:
            _add_failure(group_report, "missing_sample_loss")
        if target_path.exists():
            try:
                target_json = json.loads(target_path.read_text(encoding="utf-8"))
                group_report["target_json"] = target_json
                if _target_token_count(target_json) <= 0:
                    _add_failure(group_report, "no_target_tokens")
            except Exception as exc:
                group_report["target_error"] = repr(exc)
                _add_failure(group_report, "invalid_target_diagnostics")
        else:
            _add_failure(group_report, "missing_target_diagnostics")
        if group_report.get("failure") or group_report.get("rows", 0) <= 0:
            report["status"] = "failed"
        report["groups"][group] = group_report
    _write_json(Path(args.out) if args.out else run_dir / "omnimodal_overfit_summary.json", report)
    return {"status": report["status"], "out": str(Path(args.out) if args.out else run_dir / "omnimodal_overfit_summary.json")}


def train_plan(args: argparse.Namespace) -> dict[str, Any]:
    run = str(Path(args.run))
    commands = []
    for group in PROOF_GROUPS:
        commands.append(
            " ".join(
                [
                    'python -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-1}"',
                    "-m omnicoder.training.pipeline_pretrain_2026_dense",
                    f'--data "{run}/data/{group}.jsonl"',
                    f'--out "{run}/ckpt/{group}"',
                    f'--log_file "{run}/logs/{group}.train.jsonl"',
                    f'--train_diagnostics_file "{run}/logs/{group}.diag.rank{{rank}}.jsonl"',
                    "--preset ledger_probe --allow_probe --placement_layer_counts \"${PLACEMENT:-4}\"",
                    "--rank_device_map \"${RANK_MAP:-}\" --pipeline_schedule gpipe --pipeline_microbatches 1",
                    "--batch_size 1 --seq_len 128 --steps 600 --lr 8e-4 --max_records 10",
                    "--precision \"${PREC:-fp32}\" --init_dtype \"${INIT:-fp32}\"",
                    "--optimizer_in_backward_update lowmem_adafactor --lm_loss_chunk_tokens 64",
                    "--target_boundary_weight 2 --target_prefix_weight 2 --target_prefix_tokens 2 --no_shuffle",
                ]
            )
        )
    payload = {"schema": SCHEMA, "run": run, "commands": commands}
    if args.out:
        _write_json(Path(args.out), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize and summarize Omnicoder 2026 per-modality overfit proof data.")
    sub = parser.add_subparsers(dest="command", required=True)
    mat = sub.add_parser("materialize")
    mat.add_argument("--out", required=True)
    mat.add_argument("--examples-per-modality", type=int, default=10)
    plan = sub.add_parser("train-plan")
    plan.add_argument("--run", required=True)
    plan.add_argument("--out", default="")
    summ = sub.add_parser("summary")
    summ.add_argument("--run", required=True)
    summ.add_argument("--out", default="")
    summ.add_argument(
        "--max-reload-sample-loss",
        "--max_reload_sample_loss",
        dest="max_reload_sample_loss",
        type=float,
        default=_env_float("OMNICODER_OVERFIT_MAX_RELOAD_SAMPLE_LOSS", DEFAULT_MAX_RELOAD_SAMPLE_LOSS),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize(args)
    elif args.command == "train-plan":
        result = train_plan(args)
    elif args.command == "summary":
        result = summary(args)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

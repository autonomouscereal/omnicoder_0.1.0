from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OUTPUT_FIELDS = (
    "prediction",
    "answer",
    "output",
    "model_output",
    "model_answer",
    "patch",
    "model_patch",
    "tool_call",
    "artifact_path",
    "generated_artifact",
    "output_path",
    "image_path",
    "video_path",
    "audio_path",
    "music_path",
    "ocr_text",
)
MEDIA_FIELDS = {"artifact_path", "generated_artifact", "output_path", "image_path", "video_path", "audio_path", "music_path"}
MEDIA_MODALITIES = {"image", "video", "audio", "music"}
JUNK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"__OMNICODER_EMPTY_DECODE__",
        r"(?:_ph){3,}",
        r"^\W*$",
        r"^(.)\1{15,}$",
        r"^(?:\w{1,4}[\s_,-]*){1,3}$",
    )
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                row.setdefault("_line_number", line_number)
                yield row


def output_value(row: dict[str, Any]) -> tuple[str, Any]:
    for field in OUTPUT_FIELDS:
        value = row.get(field)
        if value is not None:
            return field, value
    return "", None


def infer_modality(row: dict[str, Any], field: str, value: Any) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("benchmark_id", "task_id", "domain", "modality", "task_type", "source_task_path")
    ).lower()
    if field in {"image_path"} or any(marker in text for marker in ("image", "ocr", "vision", "mmmu", "qwen_image")):
        return "image"
    if field in {"video_path"} or "video" in text or "ltx" in text:
        return "video"
    if field in {"audio_path"} or any(marker in text for marker in ("audio", "speech", "tts", "asr")):
        return "audio"
    if field in {"music_path"} or "music" in text or "song" in text:
        return "music"
    if "tool" in text or field == "tool_call":
        return "tool"
    if "code" in text or "swe" in text:
        return "code"
    return "text"


def is_junk_text(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, (dict, list)) else str(value or "")
    stripped = text.strip()
    if not stripped:
        return True
    return any(pattern.search(stripped) for pattern in JUNK_PATTERNS)


def ffprobe_ok(path: Path, modality: str) -> tuple[bool, dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        if os.getenv("OMNICODER_ALLOW_MISSING_FFPROBE_MEDIA_GATE", "").lower() in {"1", "true", "yes"}:
            return True, {"ffprobe": "missing_debug_allowed", "file_size": path.stat().st_size}
        return False, {"reason": "ffprobe_missing", "file_size": path.stat().st_size}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=60)
    if proc.returncode != 0:
        return False, {"ffprobe_error": proc.stderr.strip()[:500]}
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        payload = {}
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    duration = float(fmt.get("duration") or 0.0)
    size = int(float(fmt.get("size") or path.stat().st_size or 0))
    min_duration = 0.15 if modality in {"audio", "music"} else 0.05
    ok = size > 0 and (modality not in {"video", "audio", "music"} or duration >= min_duration)
    return ok, {"duration": duration, "size": size}


def media_artifact_ok(value: Any, modality: str) -> tuple[bool, dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return False, {"reason": "missing_artifact_path"}
    path = Path(value.strip())
    if not path.is_absolute():
        return False, {"reason": "artifact_path_not_absolute", "path": str(path)}
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False, {"reason": "artifact_missing_or_empty", "path": str(path)}
    if modality == "image":
        header = path.read_bytes()[:16]
        image_magic = (
            header.startswith(b"\x89PNG\r\n\x1a\n")
            or header.startswith(b"\xff\xd8\xff")
            or header.startswith(b"GIF87a")
            or header.startswith(b"GIF89a")
            or header.startswith(b"RIFF") and header[8:12] == b"WEBP"
            or header.startswith(b"BM")
        )
        if not image_magic:
            return False, {"reason": "image_artifact_magic_mismatch", "path": str(path), "size": path.stat().st_size}
    if modality in {"video", "audio", "music"}:
        ok, details = ffprobe_ok(path, modality)
        details["path"] = str(path)
        return ok, details
    return True, {"path": str(path), "size": path.stat().st_size}


def validate_prediction(row: dict[str, Any], min_output_tokens: int) -> dict[str, Any]:
    field, value = output_value(row)
    modality = infer_modality(row, field, value)
    metadata = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
    generated_tokens = int(metadata.get("generated_tokens") or row.get("generated_tokens") or 0)
    reasons: list[str] = []
    details: dict[str, Any] = {"field": field, "modality": modality, "generated_tokens": generated_tokens}
    if not field:
        reasons.append("missing_output_field")
    if generated_tokens and generated_tokens < min_output_tokens:
        reasons.append("too_few_generated_tokens")
    if modality in MEDIA_MODALITIES:
        if field not in MEDIA_FIELDS:
            reasons.append("missing_media_artifact_field")
        else:
            ok, artifact_details = media_artifact_ok(value, modality)
            details["artifact"] = artifact_details
            if not ok:
                reasons.append("invalid_media_artifact")
    elif is_junk_text(value):
        reasons.append("junk_or_empty_text")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "details": details,
        "task_id": row.get("task_id"),
        "benchmark_id": row.get("benchmark_id"),
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for pred_path in args.predictions:
        for row in iter_jsonl(Path(pred_path)):
            result = validate_prediction(row, int(args.min_output_tokens))
            rows.append(result)
            counts["accepted" if result["accepted"] else "rejected"] += 1
            counts[f"modality_{result['details']['modality']}"] += 1
            for reason in result.get("reasons") or []:
                counts[f"reason_{reason}"] += 1
    required = {item.strip() for item in args.require_modalities.split(",") if item.strip()}
    present = {str(row["details"]["modality"]) for row in rows if row["accepted"]}
    missing = sorted(required - present)
    status = "passed" if counts["rejected"] == 0 and not missing and rows else "failed"
    report = {
        "schema": "omnicoder.omnimodal_release_gate_2026.v1",
        "status": status,
        "predictions": [str(Path(item)) for item in args.predictions],
        "counts": dict(sorted(counts.items())),
        "required_modalities": sorted(required),
        "accepted_modalities": sorted(present),
        "missing_required_modalities": missing,
        "min_output_tokens": int(args.min_output_tokens),
        "rows": rows[: int(args.max_report_rows)],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real Omnicoder predictions and generated media artifacts for release gating.")
    parser.add_argument("--predictions", action="append", required=True, help="Prediction JSONL. Repeatable.")
    parser.add_argument("--out", default="")
    parser.add_argument("--require-modalities", default="text,code,tool,image,video,audio,music")
    parser.add_argument("--min-output-tokens", type=int, default=16)
    parser.add_argument("--max-report-rows", type=int, default=200)
    args = parser.parse_args(argv)
    report = run_gate(args)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

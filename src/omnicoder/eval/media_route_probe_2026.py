from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omnicoder.inference.output_router_2026 import route_for_output, route_manifest


SCHEMA = "omnicoder.media_route_probe_2026.v1"
DEFAULT_MEDIA_MODALITIES = ("image", "video", "audio", "music", "tts", "ocr")
DEFAULT_ROUTE_MODALITIES = ("text", "code", "math", "tool", *DEFAULT_MEDIA_MODALITIES)


def _probe_row_for_modality(modality: str) -> tuple[dict[str, Any], str]:
    normalized = str(modality or "").strip().lower()
    if normalized == "code":
        return (
            {
                "benchmark_id": "route_probe_code_2026",
                "target_modality": "code",
                "modality": "code",
                "output_modality": "code",
                "task_format": "code_patch",
            },
            "model_patch",
        )
    if normalized == "tool":
        return (
            {
                "benchmark_id": "route_probe_tool_2026",
                "target_modality": "tool",
                "modality": "tool",
                "output_modality": "tool",
                "task_format": "tool_call_json",
            },
            "tool_call",
        )
    if normalized == "math":
        return (
            {
                "benchmark_id": "route_probe_math_2026",
                "target_modality": "math",
                "modality": "math",
                "axis": "reasoning",
                "task_format": "math_answer",
                "output_modality": "text",
            },
            "prediction",
        )
    if normalized == "text":
        return (
            {
                "benchmark_id": "route_probe_text_2026",
                "target_modality": "text",
                "modality": "text",
                "output_modality": "text",
                "task_format": "text_answer",
            },
            "prediction",
        )
    field = "prediction" if normalized == "ocr" else "artifact_path"
    row: dict[str, Any] = {
        "benchmark_id": f"media_route_probe_{normalized}",
        "target_modality": normalized,
        "modality": normalized,
    }
    if normalized == "ocr":
        row["output_modality"] = "text"
        row["task_format"] = "ocr_text"
    else:
        row["output_modality"] = normalized
    return row, field


def build_media_route_probe(*, model_vocab_size: int = 330000, modalities: tuple[str, ...] = DEFAULT_ROUTE_MODALITIES) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for modality in modalities:
        row, field = _probe_row_for_modality(modality)
        route = route_for_output(row=row, output_field=field, tokenizer=None, model_vocab_size=int(model_vocab_size))
        rows.append(
            {
                "modality": modality,
                "target_modality": row.get("target_modality", modality),
                "task_format": row.get("task_format", ""),
                "output_field": field,
                "output_route": route_manifest(route),
            }
        )
    covered = sorted({str(row["target_modality"]) for row in rows})
    return {
        "schema": SCHEMA,
        "status": "ok",
        "model_vocab_size": int(model_vocab_size),
        "training_invoked": False,
        "probe_scope": "text_code_tool_math_media_route_readiness",
        "covered_target_modalities": covered,
        "required_route_modalities": list(DEFAULT_ROUTE_MODALITIES),
        "rows": rows,
        "routes": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate static media output-route diagnostics for checkpoint readiness.")
    parser.add_argument("--model-vocab-size", type=int, default=330000)
    parser.add_argument("--modalities", default=",".join(DEFAULT_ROUTE_MODALITIES))
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modalities = tuple(item.strip().lower() for item in str(args.modalities or "").split(",") if item.strip())
    report = build_media_route_probe(model_vocab_size=int(args.model_vocab_size), modalities=modalities or DEFAULT_MEDIA_MODALITIES)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

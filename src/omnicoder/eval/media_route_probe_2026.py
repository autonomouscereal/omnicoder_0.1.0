from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omnicoder.inference.output_router_2026 import route_for_output, route_manifest


SCHEMA = "omnicoder.media_route_probe_2026.v1"
DEFAULT_MEDIA_MODALITIES = ("image", "video", "audio", "music", "tts", "ocr")


def build_media_route_probe(*, model_vocab_size: int = 330000, modalities: tuple[str, ...] = DEFAULT_MEDIA_MODALITIES) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for modality in modalities:
        field = "prediction" if modality == "ocr" else "artifact_path"
        row = {
            "benchmark_id": f"media_route_probe_{modality}",
            "target_modality": modality,
            "modality": modality,
        }
        if modality == "ocr":
            row["output_modality"] = "text"
            row["task_format"] = "ocr_text"
        else:
            row["output_modality"] = modality
        route = route_for_output(row=row, output_field=field, tokenizer=None, model_vocab_size=int(model_vocab_size))
        rows.append(
            {
                "modality": modality,
                "output_field": field,
                "output_route": route_manifest(route),
            }
        )
    return {
        "schema": SCHEMA,
        "status": "ok",
        "model_vocab_size": int(model_vocab_size),
        "training_invoked": False,
        "rows": rows,
        "routes": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate static media output-route diagnostics for checkpoint readiness.")
    parser.add_argument("--model-vocab-size", type=int, default=330000)
    parser.add_argument("--modalities", default=",".join(DEFAULT_MEDIA_MODALITIES))
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

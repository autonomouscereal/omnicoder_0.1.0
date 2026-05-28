from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.tokenization.text_range_2026 import effective_text_token_range


TEXT_FIELDS = {"prediction", "model_patch", "tool_call", "model_actions", "model_answer", "model_output", "output"}
MEDIA_FIELDS = {"artifact_path", "generated_artifact", "output_path", "image_path", "video_path", "audio_path", "music_path"}
MODEL_ROUTE_PREFIXES = {
    "image": "image",
    "video": "video",
    "music": "music",
    "tts": "speech",
    "speech": "speech",
    "ocr": "text",
}


@dataclass(frozen=True)
class OutputRoute:
    name: str
    output_field: str
    output_modality: str
    token_ranges: tuple[tuple[str, int, int], ...]
    requires_artifact_decoder: bool
    artifact_kind: str
    notes: tuple[str, ...] = ()

    def numeric_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(begin), int(end)) for _name, begin, end in self.token_ranges)


def _task_text(row: dict[str, Any], output_field: str) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "benchmark_id",
            "adapter_id",
            "adapter_kind",
            "axis",
            "task_format",
            "source",
            "modality",
            "output_modality",
            "target_modality",
            output_field,
        )
    ).lower()


def _ledger_ranges(names: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    ranges = DEFAULT_LEDGER.as_config_ranges()
    out: list[tuple[str, int, int]] = []
    for name in names:
        begin, end = ranges[name]
        out.append((name, int(begin), int(end)))
    return tuple(out)


def infer_output_modality(row: dict[str, Any], output_field: str) -> str:
    explicit = row.get("output_modality") or row.get("target_modality") or row.get("modality")
    if isinstance(explicit, str):
        value = explicit.strip().lower()
        if value in {"image", "video", "audio", "music", "speech", "tts", "text", "code", "tool", "agent"}:
            return "speech" if value == "tts" else value
    text = _task_text(row, output_field)
    if "image" in text:
        return "image"
    if "video" in text:
        return "video"
    if "music" in text:
        return "music"
    if "speech" in text or "tts" in text:
        return "speech"
    if "audio" in text:
        return "audio"
    if output_field == "tool_call":
        return "tool"
    if output_field == "model_actions":
        return "agent"
    if output_field == "model_patch" or "code" in text or "swe" in text:
        return "code"
    return "text"


def parse_model_output_route(text: str) -> tuple[str, str]:
    """Parse the assistant-boundary route token used for media output training.

    The trunk emits ordinary text tokens such as ``image |`` before structured
    JSON/artifact tokens. Runtime code can strip that visible route marker and
    use it to choose the media decoder without adding in-trunk adapters.
    """

    stripped = str(text or "").lstrip()
    for prefix, modality in MODEL_ROUTE_PREFIXES.items():
        for separator in (" | ", "|\n", "|\r\n", "|"):
            marker = f"{prefix}{separator}"
            if stripped.lower().startswith(marker):
                return modality, stripped[len(marker):].lstrip()
    return "", str(text or "")


def route_for_model_output_text(
    *,
    text: str,
    row: dict[str, Any],
    output_field: str,
    tokenizer: Any | None,
    model_vocab_size: int,
) -> tuple[OutputRoute, str]:
    modality, cleaned = parse_model_output_route(text)
    route_row = dict(row)
    if modality:
        route_row["output_modality"] = modality
        if modality in {"image", "video", "music", "speech"}:
            output_field = "generated_artifact"
    return route_for_output(row=route_row, output_field=output_field, tokenizer=tokenizer, model_vocab_size=model_vocab_size), cleaned


def route_for_output(
    *,
    row: dict[str, Any],
    output_field: str,
    tokenizer: Any | None,
    model_vocab_size: int,
) -> OutputRoute:
    modality = infer_output_modality(row, output_field)
    text_begin, text_end = effective_text_token_range(tokenizer=tokenizer, model_vocab_size=model_vocab_size)
    text_range = (("text", int(text_begin), int(text_end)),)
    if output_field in MEDIA_FIELDS or modality == "image":
        if modality == "image":
            return OutputRoute(
                name="image_artifact",
                output_field=output_field,
                output_modality="image",
                token_ranges=_ledger_ranges(("vision_semantic", "vision_residual", "time_space", "flow")),
                requires_artifact_decoder=True,
                artifact_kind="image",
                notes=("Requires an edge image decoder/renderer to turn ledger tokens into pixels.",),
            )
        if modality == "video":
            return OutputRoute(
                name="video_artifact",
                output_field=output_field,
                output_modality="video",
                token_ranges=_ledger_ranges(("vision_semantic", "vision_residual", "audio_music", "time_space", "flow")),
                requires_artifact_decoder=True,
                artifact_kind="video",
                notes=("Requires an edge video/audio decoder or renderer to turn ledger tokens into media.",),
            )
        if modality == "music":
            return OutputRoute(
                name="music_artifact",
                output_field=output_field,
                output_modality="music",
                token_ranges=_ledger_ranges(("audio_music", "music_control", "time_space", "flow")),
                requires_artifact_decoder=True,
                artifact_kind="music",
                notes=("Requires an edge music/audio decoder to turn ledger tokens into audio.",),
            )
        if modality in {"audio", "speech"}:
            ranges = ("speech_tts", "time_space", "flow") if modality == "speech" else ("audio_music", "time_space", "flow")
            return OutputRoute(
                name=f"{modality}_artifact",
                output_field=output_field,
                output_modality=modality,
                token_ranges=_ledger_ranges(ranges),
                requires_artifact_decoder=True,
                artifact_kind="audio",
                notes=("Requires an edge audio/TTS decoder to turn ledger tokens into waveform audio.",),
            )
    if modality in {"tool", "agent"}:
        return OutputRoute(
            name=f"{modality}_text_structured",
            output_field=output_field,
            output_modality=modality,
            token_ranges=text_range,
            requires_artifact_decoder=False,
            artifact_kind="",
            notes=("Current evaluation decodes tool/action outputs as structured text JSON.",),
        )
    return OutputRoute(
        name="text",
        output_field=output_field,
        output_modality=modality,
        token_ranges=text_range,
        requires_artifact_decoder=False,
        artifact_kind="",
    )


def route_manifest(route: OutputRoute) -> dict[str, Any]:
    return {
        "name": route.name,
        "output_field": route.output_field,
        "output_modality": route.output_modality,
        "token_ranges": [
            {"name": name, "begin": int(begin), "end": int(end)}
            for name, begin, end in route.token_ranges
        ],
        "requires_artifact_decoder": bool(route.requires_artifact_decoder),
        "artifact_kind": route.artifact_kind,
        "notes": list(route.notes),
    }

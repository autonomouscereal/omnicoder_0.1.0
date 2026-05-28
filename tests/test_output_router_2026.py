from __future__ import annotations

from omnicoder.inference.output_router_2026 import parse_model_output_route, route_for_model_output_text, route_for_output, route_manifest
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


class _Tokenizer:
    vocab_size = 270_592


def test_text_route_uses_tokenizer_text_range() -> None:
    route = route_for_output(
        row={"benchmark_id": "coding_livecodebench_2026"},
        output_field="model_patch",
        tokenizer=_Tokenizer(),
        model_vocab_size=330_000,
    )

    assert route.requires_artifact_decoder is False
    assert route.output_modality == "code"
    assert route.token_ranges == (("text", 0, 270_592),)


def test_image_route_targets_media_ledger_ranges_and_requires_decoder() -> None:
    route = route_for_output(
        row={"benchmark_id": "image_generation_2026"},
        output_field="artifact_path",
        tokenizer=_Tokenizer(),
        model_vocab_size=330_000,
    )
    ranges = DEFAULT_LEDGER.as_config_ranges()
    manifest = route_manifest(route)

    assert route.requires_artifact_decoder is True
    assert route.artifact_kind == "image"
    assert ("vision_semantic", *ranges["vision_semantic"]) in route.token_ranges
    assert ("flow", *ranges["flow"]) in route.token_ranges
    assert manifest["token_ranges"][0]["name"] == "vision_semantic"


def test_music_route_targets_audio_music_and_music_control() -> None:
    route = route_for_output(
        row={"benchmark_id": "music_generation_2026"},
        output_field="artifact_path",
        tokenizer=_Tokenizer(),
        model_vocab_size=330_000,
    )

    names = [item[0] for item in route.token_ranges]
    assert route.output_modality == "music"
    assert route.requires_artifact_decoder is True
    assert "audio_music" in names
    assert "music_control" in names


def test_model_output_route_prefix_selects_media_decoder_and_strips_marker() -> None:
    route, cleaned = route_for_model_output_text(
        text='image | {"output_modality":"image","artifact_tokens":"<image_begin> proof <image_end>"}',
        row={"benchmark_id": "local_media_probe"},
        output_field="prediction",
        tokenizer=_Tokenizer(),
        model_vocab_size=330_000,
    )

    assert route.output_modality == "image"
    assert route.requires_artifact_decoder is True
    assert cleaned.startswith('{"output_modality":"image"')


def test_ocr_route_prefix_remains_text_output() -> None:
    modality, cleaned = parse_model_output_route('ocr | {"output_modality":"text","task":"ocr"}')

    assert modality == "text"
    assert cleaned.startswith('{"output_modality":"text"')

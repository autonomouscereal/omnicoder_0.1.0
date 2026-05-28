from __future__ import annotations

import pytest

from omnicoder.tokenization.native_segments_2026 import (
    NATIVE_SEGMENT_SPECS,
    build_native_segment_packet,
    native_segment_manifest,
    segment_count,
    spec_for_modality,
)
from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


def test_native_segment_specs_cover_all_output_modalities() -> None:
    expected = {"image", "video", "audio", "music", "tts", "ocr"}
    manifest = native_segment_manifest()

    assert set(NATIVE_SEGMENT_SPECS) == expected
    assert set(manifest["specs"]) == expected
    assert "no modality-specific learned adapter" in manifest["adapter_policy"]


@pytest.mark.parametrize(
    ("modality", "shape", "expected_segments"),
    [
        ("image", (65, 65), 9),
        ("video", (3, 33, 33), 12),
        ("audio", (4097,), 3),
        ("music", (8193,), 3),
        ("tts", (2048,), 1),
        ("ocr", (2, 33, 33), 8),
    ],
)
def test_native_segment_packet_is_edge_only_and_ledger_aligned(
    modality: str,
    shape: tuple[int, ...],
    expected_segments: int,
) -> None:
    packet = build_native_segment_packet(modality, shape, feature_dim=24)
    spec = spec_for_modality(modality)
    begin, end = DEFAULT_LEDGER.as_config_ranges()[spec.ledger_range]

    assert packet["segment_count"] == expected_segments
    assert packet["feature_dim"] == 24
    assert packet["edge_only"] is True
    assert packet["learned_in_trunk_adapter"] is False
    assert packet["requires_aligned_input_ids"] is True
    assert packet["type_ids"] == [spec.type_id] * expected_segments
    assert len(packet["positions"]) == expected_segments
    assert all(begin <= token_id < end for token_id in packet["token_ids"])


def test_native_segment_shape_rank_must_match_patch_rank() -> None:
    with pytest.raises(ValueError, match="shape rank"):
        segment_count((32, 32), (1, 32, 32))


def test_native_segment_unknown_modality_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        spec_for_modality("depth_camera")

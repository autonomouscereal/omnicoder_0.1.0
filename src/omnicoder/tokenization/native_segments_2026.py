from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


@dataclass(frozen=True)
class NativeSegmentSpec:
    """Edge-only continuous segment contract for the shared trunk.

    These specs describe patch/segment geometry and metadata. They are not
    learned encoders. The only learned path is the shared trunk bridge in
    ``NativeContinuousMediaBridge``.
    """

    modality: str
    type_id: int
    ledger_range: str
    patch_axes: tuple[str, ...]
    default_patch: tuple[int, ...]
    feature_dim: int = 3072
    generation: str = "continuous_flow_reconstruction"


NATIVE_SEGMENT_SPECS: dict[str, NativeSegmentSpec] = {
    "image": NativeSegmentSpec("image", 1, "vision_residual", ("height", "width"), (32, 32)),
    "video": NativeSegmentSpec("video", 2, "vision_residual", ("frames", "height", "width"), (1, 32, 32)),
    "audio": NativeSegmentSpec("audio", 3, "audio_music", ("samples",), (2048,)),
    "music": NativeSegmentSpec("music", 4, "audio_music", ("samples",), (4096,)),
    "tts": NativeSegmentSpec("tts", 5, "speech_tts", ("samples",), (2048,)),
    "ocr": NativeSegmentSpec("ocr", 6, "vision_semantic", ("pages", "height", "width"), (1, 32, 32)),
}


def spec_for_modality(modality: str) -> NativeSegmentSpec:
    key = str(modality or "").strip().lower().replace("-", "_")
    if key not in NATIVE_SEGMENT_SPECS:
        raise ValueError(f"unsupported native segment modality: {modality!r}")
    return NATIVE_SEGMENT_SPECS[key]


def segment_count(shape: tuple[int, ...] | list[int], patch: tuple[int, ...]) -> int:
    dims = [max(1, int(value)) for value in shape]
    patch_dims = [max(1, int(value)) for value in patch]
    if len(dims) != len(patch_dims):
        raise ValueError(f"shape rank {len(dims)} does not match patch rank {len(patch_dims)}")
    total = 1
    for dim, patch_dim in zip(dims, patch_dims, strict=True):
        total *= int(math.ceil(float(dim) / float(patch_dim)))
    return max(1, int(total))


def ledger_token_ids(modality: str, count: int) -> list[int]:
    spec = spec_for_modality(modality)
    begin, end = DEFAULT_LEDGER.as_config_ranges()[spec.ledger_range]
    width = max(1, int(end) - int(begin))
    return [int(begin) + (index % width) for index in range(max(1, int(count)))]


def normalized_positions(count: int, axes: int = 4) -> list[list[float]]:
    total = max(1, int(count))
    axis_count = max(1, int(axes))
    rows: list[list[float]] = []
    for index in range(total):
        base = float(index) / float(max(1, total - 1))
        rows.append([base if axis == 0 else 0.0 for axis in range(axis_count)])
    return rows


def build_native_segment_packet(
    modality: str,
    shape: tuple[int, ...] | list[int],
    *,
    feature_dim: int = 3072,
    patch: tuple[int, ...] | None = None,
    source_ref: str = "",
) -> dict[str, Any]:
    spec = spec_for_modality(modality)
    chosen_patch = tuple(int(value) for value in (patch or spec.default_patch))
    count = segment_count(shape, chosen_patch)
    return {
        "schema": "omnicoder.native_continuous_segments_2026.v1",
        "modality": spec.modality,
        "type_id": int(spec.type_id),
        "ledger_range": spec.ledger_range,
        "shape": [int(value) for value in shape],
        "patch": [int(value) for value in chosen_patch],
        "segment_count": int(count),
        "feature_dim": int(feature_dim),
        "token_ids": ledger_token_ids(spec.modality, count),
        "type_ids": [int(spec.type_id)] * count,
        "positions": normalized_positions(count),
        "source_ref": source_ref,
        "edge_only": True,
        "learned_in_trunk_adapter": False,
        "requires_aligned_input_ids": True,
        "alignment_rule": "token_ids occupy the trunk sequence positions; continuous features add shared metadata at those same positions",
        "target_kind": spec.generation,
    }


def native_segment_manifest() -> dict[str, Any]:
    return {
        "schema": "omnicoder.native_continuous_segments_manifest_2026.v1",
        "trunk_rule": "image, video, audio, music, TTS, and OCR use edge patchify/segmentize only, then one shared trunk bridge",
        "adapter_policy": "no modality-specific learned adapter inside the trunk; patchify/segmentize is edge preprocessing only",
        "specs": {
            key: {
                "type_id": value.type_id,
                "ledger_range": value.ledger_range,
                "patch_axes": list(value.patch_axes),
                "default_patch": list(value.default_patch),
                "feature_dim": value.feature_dim,
                "generation": value.generation,
            }
            for key, value in sorted(NATIVE_SEGMENT_SPECS.items())
        },
    }

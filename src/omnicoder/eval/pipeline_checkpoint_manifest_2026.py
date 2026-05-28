from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PipelineCheckpointManifestError(ValueError):
    """Raised when a sharded pipeline checkpoint manifest is incomplete or inconsistent."""


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineCheckpointManifestError(f"{path} must contain one JSON object")
    return payload


def rank_files(checkpoint: Path) -> list[Path]:
    return sorted(path for path in checkpoint.glob("rank*.pt") if path.is_file())


def positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value or 0)
    except Exception as exc:
        raise PipelineCheckpointManifestError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise PipelineCheckpointManifestError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def resolve_expected_world_size(
    checkpoint: Path,
    manifest: dict[str, Any],
    explicit_world_size: int | None = None,
) -> int:
    try:
        explicit = int(explicit_world_size or 0)
    except Exception as exc:
        raise PipelineCheckpointManifestError(
            f"explicit world size must be an integer, got {explicit_world_size!r}"
        ) from exc
    if explicit < 0:
        raise PipelineCheckpointManifestError(f"explicit world size must be non-negative, got {explicit}")
    if explicit > 0:
        return explicit
    raw = manifest.get("world_size")
    if raw is not None:
        return positive_int(raw, "manifest world_size")
    rank_count = len(rank_files(checkpoint))
    if rank_count <= 0:
        raise PipelineCheckpointManifestError(f"pipeline checkpoint has no rank*.pt files: {checkpoint}")
    return int(rank_count)


def load_pipeline_manifest(checkpoint: Path, expected_world_size: int | None = None) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    complete_path = checkpoint / ".complete.json"
    if not manifest_path.exists():
        raise PipelineCheckpointManifestError(f"pipeline checkpoint is missing manifest.json: {checkpoint}")
    if not complete_path.exists():
        raise PipelineCheckpointManifestError(f"pipeline checkpoint is missing .complete.json: {checkpoint}")
    manifest = read_json(manifest_path)
    files = rank_files(checkpoint)
    world_size = resolve_expected_world_size(checkpoint, manifest, expected_world_size)
    if len(files) != world_size:
        raise PipelineCheckpointManifestError(
            f"batch predictor expects exactly {world_size} pipeline shards, found {len(files)} in {checkpoint}"
        )
    expected = [checkpoint / f"rank{rank:05d}.pt" for rank in range(world_size)]
    missing = [path.name for path in expected if not path.exists()]
    if missing:
        raise PipelineCheckpointManifestError(f"pipeline checkpoint rank files are not contiguous; missing: {missing}")
    marker_missing = [
        f"rank{rank:05d}.pt.complete.json"
        for rank in range(world_size)
        if not (checkpoint / f"rank{rank:05d}.pt.complete.json").exists()
    ]
    if marker_missing:
        raise PipelineCheckpointManifestError(
            f"pipeline checkpoint is missing rank completion markers: {marker_missing}"
        )
    return manifest

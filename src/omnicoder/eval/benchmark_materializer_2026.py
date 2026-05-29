from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from omnicoder.data_factory import benchmark_materializer_2026 as _base
from omnicoder.data_factory.benchmark_materializer_2026 import *  # noqa: F401,F403


_BASE_NORMALIZE_TASK = _base.normalize_task
_BASE_AUDIT_PROFILE = _base.audit_profile
_BASE_MATERIALIZE = _base.materialize
_BASE_MAIN = _base.main

HASH_METADATA_KEYS = (
    "snapshot_sha256",
    "snapshot_hash",
    "task_file_sha256",
    "official_task_sha256",
    "manifest_sha256",
    "source_sha256",
)
PLACEHOLDER_VALUES = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "todo",
    "tbd",
    "placeholder",
    "<operator_supplied_sha256>",
    "<operator-supplied-sha256>",
    "operator_required",
    "operator-supplied",
    "operator_supplied",
    "sha256:descriptor",
    "sha256:placeholder",
    "sha256:operator_required",
}
PLACEHOLDER_PREFIXES = (
    "operator_required",
    "operator supplied",
    "operator-supplied",
    "operator_supplied",
    "<operator",
    "placeholder:",
    "todo:",
    "tbd:",
)

EXPLICIT_MEDIA_FIELD_TYPES = (
    ("query_images", "image"),
    ("query_image", "image"),
    ("example_images", "image"),
    ("example_image", "image"),
    ("global_images", "image"),
    ("global_image", "image"),
    ("prompt_images", "image"),
    ("response_a_images", "image"),
    ("response_b_images", "image"),
    ("image_list", "image"),
    ("needle_image_list", "image"),
    ("query_video", "video"),
    ("query_videos", "video"),
    ("example_video", "video"),
    ("example_videos", "video"),
    ("global_video", "video"),
    ("global_videos", "video"),
    ("query_audio", "audio"),
    ("query_audios", "audio"),
    ("example_audio", "audio"),
    ("example_audios", "audio"),
    ("global_audio", "audio"),
    ("global_audios", "audio"),
    ("query_document", "document"),
    ("query_documents", "document"),
    ("example_document", "document"),
    ("example_documents", "document"),
    ("global_document", "document"),
    ("global_documents", "document"),
)
GENERIC_MEDIA_FIELDS = ("query_media", "example_media", "global_media", "media")
NESTED_MEDIA_KEYS = (
    ("image", "image"),
    ("images", "image"),
    ("image_path", "image"),
    ("image_paths", "image"),
    ("image_url", "image"),
    ("image_urls", "image"),
    ("prompt_images", "image"),
    ("video", "video"),
    ("videos", "video"),
    ("video_path", "video"),
    ("video_paths", "video"),
    ("video_url", "video"),
    ("video_urls", "video"),
    ("audio", "audio"),
    ("audios", "audio"),
    ("audio_path", "audio"),
    ("audio_paths", "audio"),
    ("audio_url", "audio"),
    ("audio_urls", "audio"),
    ("wav", "audio"),
    ("wav_path", "audio"),
    ("document", "document"),
    ("documents", "document"),
    ("document_path", "document"),
    ("document_paths", "document"),
    ("pdf", "document"),
    ("pdf_path", "document"),
)


def _non_placeholder(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    text = str(value).strip()
    lowered = text.lower()
    if lowered in PLACEHOLDER_VALUES:
        return False
    return not any(lowered.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def _append_canonical_media(media: list[dict[str, Any]], value: Any, media_type: str, field: str) -> int:
    before = len(media)
    _base.append_media(media, value, media_type, field)
    return len(media) - before


def _append_generic_media(media: list[dict[str, Any]], value: Any, field: str) -> int:
    if not _base.has_value(value):
        return 0
    appended = 0
    if isinstance(value, list):
        for index, item in enumerate(value):
            appended += _append_generic_media(media, item, f"{field}[{index}]")
        return appended
    if isinstance(value, dict):
        for key, media_type in NESTED_MEDIA_KEYS:
            if _base.has_value(value.get(key)):
                appended += _append_canonical_media(media, value.get(key), media_type, f"{field}.{key}")
        if appended:
            return appended
    return _append_canonical_media(media, value, "media", field)


def _canonical_media(raw: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    media = list(_base.normalized_media(raw, row))
    for field, media_type in EXPLICIT_MEDIA_FIELD_TYPES:
        _append_canonical_media(media, raw.get(field), media_type, field)
        _append_canonical_media(media, row.get(field), media_type, field)
    for field in GENERIC_MEDIA_FIELDS:
        _append_generic_media(media, raw.get(field), field)
        _append_generic_media(media, row.get(field), field)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in media:
        if not isinstance(item, dict) or not _base.has_value(item):
            continue
        key = _base.stable_hash(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _canonicalize_hellaswag_task_id(benchmark_id: str, raw: dict[str, Any], row: dict[str, Any]) -> None:
    if benchmark_id != "reasoning_hellaswag_full_2026":
        return
    ind = _base.first_value(raw, ("ind",))
    if _base.has_value(ind):
        row["task_id"] = f"hellaswag-{str(ind).strip()}"


def _refresh_task_hash(row: dict[str, Any]) -> None:
    row.pop("task_row_sha256", None)
    row["task_row_sha256"] = _base.stable_hash(row)


def normalize_task(
    benchmark_id: str,
    raw: dict[str, Any],
    spec: dict[str, Any],
    profile_record: dict[str, Any],
    snapshot: dict[str, Any],
    mode: str,
    source_ref: str,
    index: int,
) -> dict[str, Any] | None:
    row = _BASE_NORMALIZE_TASK(benchmark_id, raw, spec, profile_record, snapshot, mode, source_ref, index)
    if row is None:
        return None
    _canonicalize_hellaswag_task_id(benchmark_id, raw, row)
    media = _canonical_media(raw, row)
    if media:
        row["media"] = media
    _refresh_task_hash(row)
    return row


def _snapshot_metadata_gaps(profile: dict[str, Any], args: Any) -> list[dict[str, Any]]:
    snapshots = profile.get("reportable_snapshots") if isinstance(profile.get("reportable_snapshots"), dict) else {}
    selected = _base.selected_benchmark_ids(profile, args)
    gaps: list[dict[str, Any]] = []
    for benchmark_id in selected:
        snapshot = snapshots.get(benchmark_id)
        if not isinstance(snapshot, dict):
            continue
        missing: list[str] = []
        if not _non_placeholder(snapshot.get("license_ref")):
            missing.append("license_ref")
        if not _non_placeholder(snapshot.get("official_scorer_ref")):
            missing.append("official_scorer_ref")
        if not any(_non_placeholder(snapshot.get(key)) for key in HASH_METADATA_KEYS):
            missing.append("snapshot_sha256_or_task_file_sha256")
        if missing:
            gaps.append(
                {
                    "benchmark_id": benchmark_id,
                    "missing": missing,
                    "snapshot_id": str(
                        snapshot.get("snapshot_id")
                        or snapshot.get("official_snapshot_id")
                        or snapshot.get("authorized_snapshot_id")
                        or ""
                    ),
                }
            )
    return gaps


def audit_profile(args: Any) -> dict[str, Any]:
    report = _BASE_AUDIT_PROFILE(args)
    profile = _base.load_profile(args.profile)
    gaps = _snapshot_metadata_gaps(profile, args)
    report["snapshot_metadata_gaps"] = gaps
    report["official_reportable_snapshot_metadata_ok"] = not bool(gaps)
    if gaps:
        reasons = list(report.get("fail_reasons") or [])
        reason = "reportable_snapshots_missing_official_metadata"
        if reason not in reasons:
            reasons.append(reason)
        report["fail_reasons"] = reasons
        report["status"] = "failed"
    return report


def materialize(args: Any) -> dict[str, Any]:
    with _patched_base():
        return _BASE_MATERIALIZE(args)


@contextmanager
def _patched_base() -> Iterator[None]:
    old_normalize_task = _base.normalize_task
    old_audit_profile = _base.audit_profile
    old_materialize = _base.materialize
    try:
        _base.normalize_task = normalize_task
        _base.audit_profile = audit_profile
        _base.materialize = materialize
        yield
    finally:
        _base.normalize_task = old_normalize_task
        _base.audit_profile = old_audit_profile
        _base.materialize = old_materialize


def main(argv: list[str] | None = None) -> int:
    with _patched_base():
        return _BASE_MAIN(argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

from omnicoder.eval import omnimodal_release_gate_2026 as gate


def test_media_generation_requires_artifact_field() -> None:
    result = gate.validate_prediction(
        {
            "benchmark_id": "image_generation_2026",
            "task_id": "image-1",
            "prediction": "Generated an image.",
            "generation_metadata": {"generated_tokens": 32},
        },
        min_output_tokens=16,
    )

    assert result["accepted"] is False
    assert "missing_media_artifact_field" in result["reasons"]


def test_text_prediction_requires_generated_token_metadata() -> None:
    result = gate.validate_prediction(
        {
            "benchmark_id": "coding_livecodebench_2026",
            "task_id": "code-1",
            "prediction": "def add(a, b): return a + b",
            "generation_metadata": {},
        },
        min_output_tokens=16,
    )

    assert result["accepted"] is False
    assert "missing_generated_tokens" in result["reasons"]
    assert "too_few_generated_tokens" in result["reasons"]


def test_image_artifact_rejects_text_file_named_png(tmp_path: Path) -> None:
    artifact = tmp_path / "fake.png"
    artifact.write_text("this is not a png", encoding="utf-8")

    result = gate.validate_prediction(
        {
            "benchmark_id": "image_generation_2026",
            "task_id": "image-1",
            "image_path": str(artifact),
            "generation_metadata": {"generated_tokens": 32},
        },
        min_output_tokens=16,
    )

    assert result["accepted"] is False
    assert "invalid_media_artifact" in result["reasons"]
    assert result["details"]["artifact"]["reason"] == "image_artifact_magic_mismatch"


def test_missing_ffprobe_fails_closed_for_audio(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "audio.wav"
    artifact.write_bytes(b"RIFF" + b"\x00" * 128)
    monkeypatch.setattr(gate.shutil, "which", lambda _name: None)
    monkeypatch.delenv("OMNICODER_ALLOW_MISSING_FFPROBE_MEDIA_GATE", raising=False)

    result = gate.validate_prediction(
        {
            "benchmark_id": "audio_speech_2026",
            "task_id": "audio-1",
            "audio_path": str(artifact),
            "generation_metadata": {"generated_tokens": 32},
        },
        min_output_tokens=16,
    )

    assert result["accepted"] is False
    assert "invalid_media_artifact" in result["reasons"]
    assert result["details"]["artifact"]["reason"] == "ffprobe_missing"


def test_ocr_prediction_is_text_extraction_not_image_generation() -> None:
    result = gate.validate_prediction(
        {
            "benchmark_id": "ocr_document_text_2026",
            "task_id": "ocr-1",
            "prediction": "Extracted text: invoice total 42 dollars.",
            "generation_metadata": {"generated_tokens": 18},
        },
        min_output_tokens=8,
    )

    assert result["accepted"] is True
    assert result["details"]["modality"] == "ocr"


def test_tts_prediction_requires_audio_artifact_field() -> None:
    result = gate.validate_prediction(
        {
            "benchmark_id": "tts_generation_2026",
            "task_id": "tts-1",
            "prediction": "speech artifact ready",
            "generation_metadata": {"generated_tokens": 32},
        },
        min_output_tokens=8,
    )

    assert result["accepted"] is False
    assert result["details"]["modality"] == "tts"
    assert "missing_media_artifact_field" in result["reasons"]


def test_diagnostic_generated_media_artifact_dict_counts_as_local_proof(tmp_path: Path) -> None:
    artifact = tmp_path / "media-token-proof.json"
    artifact.write_text('{"tokens":[1,2,3]}', encoding="utf-8")

    result = gate.validate_prediction(
        {
            "benchmark_id": "video_generation_decode_sanity",
            "task_id": "video-1",
            "generated_artifact": {
                "schema": "omnicoder.diagnostic_native_media_artifact_2026.v1",
                "backend": "diagnostic_native_media_token_artifact",
                "diagnostic_only": True,
                "path": str(artifact),
                "modality": "video",
                "sha256": "x",
                "byte_size": artifact.stat().st_size,
                "token_count": 3,
                "reportable_quality": "not_reportable_without_real_codec_backend",
            },
            "generation_metadata": {"generated_tokens": 3},
        },
        min_output_tokens=1,
    )

    assert result["accepted"] is True
    assert result["details"]["artifact"]["diagnostic_only"] is True

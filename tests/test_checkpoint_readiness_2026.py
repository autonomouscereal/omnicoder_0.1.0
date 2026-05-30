from __future__ import annotations

import json
from pathlib import Path

from omnicoder.eval.checkpoint_readiness_2026 import ReadinessThresholds, checkpoint_readiness
from omnicoder.eval.media_route_probe_2026 import build_media_route_probe


def _topk(text: str = "def add returns the sum", checkpoint: str = "") -> dict:
    return {
        "schema": "omnicoder.pipeline_token_topk_probe_2026.v1",
        "status": "ok",
        "checkpoint": checkpoint,
        "model_vocab_size": 128,
        "text_range": [0, 100],
        "tokenizer": {"vocab_size": 100},
        "generated_token_ids": [12, 13, 14],
        "generated_text": text,
        "rank_reports": [
            {
                "rank": 0,
                "block_tensor_count": 8,
                "tensors": {
                    "embed.weight": {"std_sample": 0.02, "finite_sample": True},
                },
            },
            {
                "rank": 1,
                "block_tensor_count": 8,
                "tensors": {},
            },
            {
                "rank": 2,
                "block_tensor_count": 8,
                "tensors": {
                    "lm_head.weight": {"std_sample": 0.02, "finite_sample": True},
                },
            },
        ],
        "steps": [
            {
                "step": 1,
                "text_topk": [
                    {"token_id": 12, "logit": 4.2, "decoded": "def"},
                    {"token_id": 13, "logit": 3.1, "decoded": "add"},
                ],
            }
        ],
    }


def _sample_loss(avg_loss: float = 1.25, perplexity: float = 3.49, tokens: int = 42, checkpoint: str = "") -> dict:
    return {
        "schema": "omnicoder.pipeline_sample_loss_2026.v1",
        "checkpoint": checkpoint,
        "overall": {
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "tokens": tokens,
            "samples": 2,
            "records": 1,
        },
    }


def _media_route_probe() -> dict:
    return build_media_route_probe(model_vocab_size=330000)


def test_checkpoint_readiness_passes_valid_diagnostics() -> None:
    report = checkpoint_readiness(_topk(), _sample_loss(), _media_route_probe())

    assert report["status"] == "passed"
    assert report["passed"] is True
    assert report["reasons"] == []


def test_checkpoint_readiness_rejects_punctuation_topk_and_bad_tokenizer_range() -> None:
    topk = _topk("!!!!!,,,,,.....")
    topk["text_range"] = [100, 100]

    report = checkpoint_readiness(topk, _sample_loss(), _media_route_probe())

    assert report["status"] == "failed"
    assert "topk_generated_text_punctuation_only" in report["reasons"]
    assert "tokenizer_text_range_invalid" in report["reasons"]


def test_checkpoint_readiness_rejects_missing_or_over_threshold_sample_loss() -> None:
    sample = {"overall": {"avg_loss": None, "perplexity": 100.0, "tokens": 8}}

    report = checkpoint_readiness(
        _topk(),
        sample,
        _media_route_probe(),
        thresholds=ReadinessThresholds(max_avg_loss=2.0, max_perplexity=10.0, min_tokens=16),
    )

    assert report["status"] == "failed"
    assert "heldout_avg_loss_missing" in report["reasons"]
    assert "heldout_perplexity_over_threshold" in report["reasons"]
    assert "heldout_tokens_below_threshold" in report["reasons"]


def test_checkpoint_readiness_rejects_missing_media_router_metadata() -> None:
    report = checkpoint_readiness(_topk(), _sample_loss(), {"routes": []})

    assert report["status"] == "failed"
    assert "media_router_metadata_missing" in report["reasons"]


def test_checkpoint_readiness_requires_every_core_media_route() -> None:
    probe = build_media_route_probe(model_vocab_size=330000, modalities=("image", "video", "music", "tts", "ocr"))

    report = checkpoint_readiness(_topk(), _sample_loss(), probe)

    assert report["status"] == "failed"
    assert "media_route_missing_required_modality_audio" in report["reasons"]


def test_checkpoint_readiness_rejects_bad_weight_scale() -> None:
    topk = _topk()
    topk["rank_reports"][0]["tensors"]["embed.weight"]["std_sample"] = 1.0

    report = checkpoint_readiness(topk, _sample_loss(), _media_route_probe())

    assert report["status"] == "failed"
    assert "embed.weight:std_over_threshold" in report["reasons"]
    assert "checkpoint_weight_stats_invalid" in report["reasons"]


def test_checkpoint_readiness_accepts_json_paths(tmp_path: Path) -> None:
    topk_path = tmp_path / "topk.json"
    sample_path = tmp_path / "sample.json"
    route_path = tmp_path / "route.json"
    topk_path.write_text(json.dumps(_topk()), encoding="utf-8")
    sample_path.write_text(json.dumps(_sample_loss()), encoding="utf-8")
    route_path.write_text(json.dumps(_media_route_probe()), encoding="utf-8")

    report = checkpoint_readiness(topk_path, sample_path, route_path)

    assert report["status"] == "passed"


def test_checkpoint_readiness_binds_diagnostics_to_expected_checkpoint() -> None:
    checkpoint = "/workspace/weights/run/checkpoints/posttrain/step608"
    expected = "/home/cereal/omnicoder_2026_work/weights/run/checkpoints/posttrain/step608"

    report = checkpoint_readiness(
        _topk(checkpoint=checkpoint),
        _sample_loss(checkpoint=checkpoint),
        _media_route_probe(),
        expected_checkpoint=expected,
        expected_world_size=3,
    )

    assert report["status"] == "passed"
    assert report["checks"]["checkpoint_binding"]["status"] == "passed"


def test_checkpoint_readiness_rejects_stale_checkpoint_diagnostics() -> None:
    report = checkpoint_readiness(
        _topk(checkpoint="/workspace/weights/old/checkpoint"),
        _sample_loss(checkpoint="/workspace/weights/old/checkpoint"),
        _media_route_probe(),
        expected_checkpoint="/workspace/weights/new/checkpoint",
        expected_world_size=3,
    )

    assert report["status"] == "failed"
    assert "checkpoint_binding_invalid" in report["reasons"]
    assert "topk_probe_checkpoint_mismatch" in report["reasons"]


def test_checkpoint_readiness_rejects_missing_expected_rank_reports() -> None:
    topk = _topk(checkpoint="/workspace/weights/checkpoint")
    topk["rank_reports"] = topk["rank_reports"][:2]

    report = checkpoint_readiness(
        topk,
        _sample_loss(checkpoint="/workspace/weights/checkpoint"),
        _media_route_probe(),
        expected_checkpoint="/workspace/weights/checkpoint",
        expected_world_size=3,
    )

    assert report["status"] == "failed"
    assert "rank_weight_stats_world_size_mismatch" in report["reasons"]
    assert "rank_weight_stats_rank_ids_incomplete" in report["reasons"]


def test_media_route_probe_generates_all_core_media_routes() -> None:
    report = build_media_route_probe(model_vocab_size=330000)
    readiness = checkpoint_readiness(_topk(), _sample_loss(), report)

    assert readiness["checks"]["media_route_probe"]["status"] == "passed"
    routes = {row["modality"]: row["output_route"] for row in report["rows"]}
    assert {"image", "video", "audio", "music", "tts", "ocr"} <= set(routes)
    assert routes["image"]["requires_artifact_decoder"] is True
    assert routes["video"]["requires_artifact_decoder"] is True
    assert routes["audio"]["requires_artifact_decoder"] is True
    assert routes["music"]["requires_artifact_decoder"] is True
    assert routes["tts"]["output_modality"] == "speech"
    assert routes["tts"]["requires_artifact_decoder"] is True
    assert routes["ocr"]["output_field"] == "prediction"
    assert routes["ocr"]["output_modality"] == "text"
    assert routes["ocr"]["requires_artifact_decoder"] is False


def test_route_probe_generates_text_code_tool_math_and_media_routes() -> None:
    report = build_media_route_probe(model_vocab_size=330000)

    assert report["probe_scope"] == "text_code_tool_math_media_route_readiness"
    assert {
        "text",
        "code",
        "math",
        "tool",
        "image",
        "video",
        "audio",
        "music",
        "tts",
        "ocr",
    } <= set(report["covered_target_modalities"])
    routes = {row["modality"]: row["output_route"] for row in report["rows"]}

    assert routes["text"]["name"] == "text"
    assert routes["text"]["requires_artifact_decoder"] is False
    assert routes["code"]["output_modality"] == "code"
    assert routes["code"]["token_ranges"][0]["name"] == "text"
    assert routes["math"]["output_modality"] == "text"
    assert routes["math"]["token_ranges"][0]["name"] == "text"
    assert routes["tool"]["name"] == "tool_text_structured"
    assert routes["tool"]["requires_artifact_decoder"] is False
    assert routes["image"]["requires_artifact_decoder"] is True

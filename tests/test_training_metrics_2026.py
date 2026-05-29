from __future__ import annotations

import json
from pathlib import Path

from omnicoder.training import metrics_2026


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_summarize_train_diagnostics_groups_modalities_and_token_families(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "train_diagnostics.jsonl",
        [
            {
                "schema": "omnicoder.train_diagnostics_2026.v1",
                "event": "train_step",
                "rank": 0,
                "world_size": 2,
                "global_step": 10,
                "local_step": 1,
                "loss": {
                    "total_ce": 4.0,
                    "ce_by_modality": {"text": 2.0, "vision": 3.0},
                    "ce_by_token_family": {"text": 2.0, "vision_semantic": 3.0},
                },
                "targets": {
                    "by_modality": {"text": 5, "vision": 3},
                    "optimized_by_modality": {"text": 4, "vision": 2},
                    "by_token_family": {"text": 5, "vision_semantic": 3},
                    "optimized_by_token_family": {"text": 4, "vision_semantic": 2},
                },
                "runtime": {"tokens": 100, "elapsed_sec": 2.0, "tokens_per_sec": 50.0},
            },
            {
                "schema": "omnicoder.train_diagnostics_2026.v1",
                "event": "train_step",
                "rank": 1,
                "world_size": 2,
                "global_step": 11,
                "local_step": 2,
                "loss": {
                    "total_ce": 3.0,
                    "ce_by_modality": {"text": 1.0, "audio_music": 2.5},
                    "ce_by_token_family": {"text": 1.0, "audio_music": 2.5},
                },
                "targets": {
                    "by_modality": {"text": 6, "audio_music": 7},
                    "optimized_by_modality": {"text": 4, "audio_music": 6},
                    "by_token_family": {"text": 6, "audio_music": 7},
                    "optimized_by_token_family": {"text": 4, "audio_music": 6},
                },
                "runtime": {"tokens": 120, "elapsed_sec": 3.0, "tokens_per_sec": 40.0},
            },
            {"event": "not_a_train_step", "loss": 99.0},
        ],
    )

    summary = metrics_2026.summarize_train_diagnostics_log(path)

    assert summary["diagnostic_events"] == 2
    assert summary["ranks"] == [0, 1]
    assert summary["world_sizes"] == [2]
    assert summary["global_step_first"] == 10
    assert summary["global_step_last"] == 11
    assert summary["loss_total_ce"]["first"] == 4.0
    assert summary["loss_total_ce"]["last"] == 3.0
    assert summary["total_target_tokens"] == 21
    assert summary["total_optimized_target_tokens"] == 16
    assert summary["optimized_target_coverage"] == 16 / 21
    assert summary["by_modality"]["text"]["target_tokens"] == 11
    assert summary["by_modality"]["text"]["optimized_target_tokens"] == 8
    assert summary["by_modality"]["text"]["optimized_target_coverage"] == 8 / 11
    assert summary["by_modality"]["text"]["loss_quality"]["positive_count"] == 2
    assert summary["by_modality"]["text"]["ce"]["mean"] == 1.5
    assert summary["by_modality"]["vision"]["ce"]["last"] == 3.0
    assert summary["by_modality"]["audio_music"]["target_tokens"] == 7
    assert summary["by_token_family"]["vision_semantic"]["optimized_target_tokens"] == 2
    assert summary["trainability"]["coverage"]["modalities"]["covered_target"] == ["audio_music", "text", "vision"]
    assert summary["runtime"]["tokens"] == 220
    assert summary["runtime"]["elapsed_sec"] == 5.0
    assert summary["runtime"]["tokens_per_sec"]["last"] == 40.0


def test_summarize_train_diagnostics_flags_fake_or_broken_trainability_metrics(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "bad_train_diagnostics.jsonl",
        [
            {
                "schema": "omnicoder.train_diagnostics_2026.v1",
                "event": "train_step",
                "rank": 0,
                "world_size": 1,
                "global_step": 1,
                "loss": {
                    "total_ce": 0.0,
                    "ce_by_modality": {"text": 0.0, "vision": -0.25, "tts": "not-a-loss"},
                    "ce_by_token_family": {"text": 0.0, "vision_semantic": -0.25},
                },
                "targets": {
                    "by_modality": {"text": 5, "vision": 3, "tts": 2},
                    "optimized_by_modality": {"text": 0, "vision": 4, "tts": 2},
                    "by_token_family": {"text": 5, "vision_semantic": 3, "speech_tts": 2},
                    "optimized_by_token_family": {"text": 0, "vision_semantic": 4, "speech_tts": 2},
                },
            }
        ],
    )

    summary = metrics_2026.summarize_train_diagnostics_log(path)
    reasons = set(summary["trainability"]["reasons"])

    assert summary["trainability"]["status"] == "failed"
    assert summary["loss_quality_total_ce"]["non_positive_count"] == 1
    assert summary["by_modality"]["text"]["loss_quality"]["non_positive_count"] == 1
    assert summary["by_modality"]["vision"]["optimized_exceeds_target_tokens"] is True
    assert summary["by_modality"]["vision"]["optimized_target_coverage"] == 4 / 3
    assert summary["by_modality"]["tts"]["loss_quality"]["non_numeric_count"] == 1
    assert "total_ce_non_positive" in reasons
    assert "modality:text:missing_optimized_target_tokens" in reasons
    assert "modality:text:non_positive_ce" in reasons
    assert "modality:vision:optimized_target_tokens_exceed_target_tokens" in reasons
    assert "modality:tts:non_numeric_ce" in reasons
    assert "token_family:text:missing_optimized_target_tokens" in reasons
    assert "token_family:vision_semantic:optimized_target_tokens_exceed_target_tokens" in reasons


def test_summarize_training_log_embeds_train_diagnostics_when_present(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "mixed.log",
        [
            {"step": 1, "loss": 5.0},
            {
                "event": "train_step",
                "global_step": 2,
                "loss": {"total_ce": 4.0, "ce_by_modality": {"tool": 1.25}},
                "targets": {"by_modality": {"tool": 9}, "optimized_by_modality": {"tool": 8}},
            },
        ],
    )

    summary = metrics_2026.summarize_training_log(path)

    assert summary["json_events"] == 2
    assert summary["steps"] == 1
    assert summary["loss_last"] == 5.0
    assert summary["train_diagnostics"]["diagnostic_events"] == 1
    assert summary["train_diagnostics"]["by_modality"]["tool"]["target_tokens"] == 9


def test_summarize_training_log_reports_checkpoint_contracts_and_eval_summaries(tmp_path: Path) -> None:
    path = tmp_path / "eval_and_checkpoint_summary.json"
    path.write_text(
        json.dumps(
            [
                {
                    "schema": "omnicoder.pipeline_sample_loss_2026.v1",
                    "status": "ok",
                    "checkpoint": "/workspace/weights/checkpoints/step42",
                    "overall": {"avg_loss": 1.25, "perplexity": 3.49, "tokens": 128, "records": 4},
                    "modalities": {
                        "text": {"loss": 1.1, "tokens": 64},
                        "vision": {"loss": 1.4, "tokens": 64},
                    },
                },
                {
                    "checkpoint_eval_artifact_contract": {
                        "schema": "omnicoder.checkpoint_eval_artifact_contract_2026.v1",
                        "status": "required_after_checkpoint_save",
                        "training_invoked": False,
                        "checkpoint_dir": "/workspace/weights/checkpoints/step42",
                        "artifacts": [
                            {
                                "name": "heldout_sample_loss_by_modality",
                                "required": True,
                                "path": "/workspace/weights/checkpoints/step42/evals/heldout_pipeline_sample_loss.json",
                                "schema": "omnicoder.pipeline_sample_loss_2026.v1",
                                "must_include": ["overall", "modalities"],
                            }
                        ],
                    }
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = metrics_2026.summarize_training_log(path)

    assert summary["json_events"] == 2
    assert summary["eval_summaries"]["count"] == 1
    assert summary["eval_summaries"]["schemas"] == {"omnicoder.pipeline_sample_loss_2026.v1": 1}
    assert summary["eval_summaries"]["checkpoints"] == ["/workspace/weights/checkpoints/step42"]
    assert summary["eval_summaries"]["items"][0]["overall"]["tokens"] == 128.0
    assert summary["eval_summaries"]["items"][0]["modalities"] == ["text", "vision"]
    assert summary["checkpoint_reports"]["eval_artifact_contract_count"] == 1
    assert summary["checkpoint_reports"]["eval_artifact_contracts"][0]["required_artifact_count"] == 1
    assert summary["checkpoint_reports"]["eval_artifact_contracts"][0]["artifacts"][0]["must_include"] == ["overall", "modalities"]

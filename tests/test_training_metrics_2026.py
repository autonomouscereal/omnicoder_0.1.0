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
    assert summary["by_modality"]["text"]["target_tokens"] == 11
    assert summary["by_modality"]["text"]["optimized_target_tokens"] == 8
    assert summary["by_modality"]["text"]["ce"]["mean"] == 1.5
    assert summary["by_modality"]["vision"]["ce"]["last"] == 3.0
    assert summary["by_modality"]["audio_music"]["target_tokens"] == 7
    assert summary["by_token_family"]["vision_semantic"]["optimized_target_tokens"] == 2
    assert summary["runtime"]["tokens"] == 220
    assert summary["runtime"]["elapsed_sec"] == 5.0
    assert summary["runtime"]["tokens_per_sec"]["last"] == 40.0


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

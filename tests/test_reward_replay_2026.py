from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from omnicoder.training import reward_replay_2026


class TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [index + 1 for index, _ in enumerate(text.split() or ["empty"])]


def test_record_to_text_treats_suffix_rlvr_as_rlvr_and_preserves_contract_fields() -> None:
    text, weight = reward_replay_2026.record_to_text_and_weight(
        {
            "training_kind": "terminal_rlvr",
            "prompt": "Run the terminal verifier.",
            "chosen": "This preference-shaped field should not change RLVR handling.",
            "rejected": "Bad answer.",
            "verifier": {"checks": ["exit_code_zero"], "reward": 0.75},
            "environment": {"kind": "terminal", "timeout_seconds": 30},
            "reward": 0.9,
            "reward_components": {"exit_code_zero": 1.0},
            "tool_calls": [{"tool": "shell", "arguments": {"command": "pytest"}}],
            "tool_results": [{"exit_code": 0}],
        }
    )

    payload = json.loads(text.split("assistant: ", 1)[1])
    assert payload["verifier"]["checks"] == ["exit_code_zero"]
    assert payload["environment"]["kind"] == "terminal"
    assert payload["reward"] == 0.9
    assert payload["reward_components"] == {"exit_code_zero": 1.0}
    assert payload["tool_results"] == [{"exit_code": 0}]
    assert weight > 1.0


def test_empty_reward_replay_dataset_fails_without_explicit_smoke(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="reward replay dataset is empty"):
        reward_replay_2026.RewardReplayDataset([str(empty)], TinyTokenizer(), seq_len=8)


def test_empty_reward_replay_dataset_allows_explicit_smoke(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    dataset = reward_replay_2026.RewardReplayDataset([str(empty)], TinyTokenizer(), seq_len=8, allow_empty=True)

    assert dataset.empty_source is True
    assert len(dataset) == 1
    _, weight, sample_id = dataset[0]
    assert float(weight) == pytest.approx(0.05)
    assert sample_id == "empty_smoke_fallback"

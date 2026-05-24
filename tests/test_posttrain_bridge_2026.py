from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from omnicoder.training import posttrain_bridge_2026


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _args(train_jsonl: str | None, *, dry_run: bool = False, smoke: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        algorithm="grpo",
        model="student",
        train_jsonl=train_jsonl,
        eval_jsonl=None,
        out_dir="out",
        max_seq_len=4096,
        max_steps=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        lora_r=8,
        lora_alpha=16,
        load_in_4bit=False,
        reward_batch_size=2,
        num_generations=2,
        kl_beta=0.02,
        temperature=0.8,
        dry_run=dry_run,
        smoke=smoke,
        check_deps=False,
    )


def test_inspect_dataset_treats_any_suffix_rlvr_as_tool_trajectory(tmp_path: Path) -> None:
    train = _write_jsonl(
        tmp_path / "terminal_rlvr.jsonl",
        [
            {
                "training_kind": "terminal_rlvr",
                "messages": [{"role": "user", "content": "This still follows the RLVR training_kind."}],
                "prompt": "Verify the shell outcome.",
                "verifier": {"checks": ["exit_code_zero"]},
                "environment": {"kind": "terminal"},
                "reward": 1.0,
            }
        ],
    )

    dataset = posttrain_bridge_2026.inspect_dataset(train)

    assert dataset["records_sampled"] == 1
    assert dataset["schemas"]["tool_trajectory"] == 1


def test_build_manifest_rejects_empty_dataset_without_explicit_smoke_or_dry_run(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        posttrain_bridge_2026.build_manifest(_args(str(empty)))

    payload = json.loads(str(exc.value))
    assert payload["error"] == "empty_dataset"
    assert payload["train_jsonl"] == str(empty)


def test_build_manifest_allows_empty_dataset_for_explicit_dry_run(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    manifest = posttrain_bridge_2026.build_manifest(_args(str(empty), dry_run=True))

    assert manifest["status"] == "dry_run_ok"
    assert manifest["dataset"]["records_sampled"] == 0
    assert manifest["empty_dataset_allowed_by"] == "dry_run"

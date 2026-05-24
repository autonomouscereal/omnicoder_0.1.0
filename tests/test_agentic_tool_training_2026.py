from __future__ import annotations

import json
from pathlib import Path

from omnicoder.training import agentic_tool_training_2026 as tooltrain
from omnicoder.training import full_harness_2026, posttrain_bridge_2026


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_agentic_tool_training_builds_all_training_splits(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "traces.jsonl",
        [
            {
                "source_date": "2026-05-23",
                "messages": [
                    {"role": "user", "content": "Check the dashboard health with the ai server tool."},
                    {
                        "role": "assistant",
                        "content": "{\"tool\":\"server_manager\",\"arguments\":{\"server\":\"ai\",\"command\":\"curl -s http://127.0.0.1:4000/health\"}}",
                    },
                    {"role": "tool", "content": "{\"exit_code\":0,\"status\":\"ok\"}"},
                    {"role": "assistant", "content": "Health check passed."},
                ],
                "quality": {"score": 0.91},
                "lineage": {"trace_id": "trace-health"},
            }
        ],
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "agentic_tool_training": {
                    "input_jsonl": str(source),
                    "out_dir": str(tmp_path / "out"),
                    "min_quality": 0.1,
                    "stages": ["tool_sft", "tool_reward", "tool_preference", "tool_rlvr"],
                    "reward_axes": ["tool_schema_valid", "task_outcome"],
                }
            }
        ),
        encoding="utf-8",
    )

    assert tooltrain.main(["--profile", str(profile), "build", "--dry-run"]) == 0

    manifest = json.loads((tmp_path / "out" / "agentic_tool_training_manifest.json").read_text())
    assert manifest["counts"]["sft"] == 1
    assert manifest["counts"]["preference"] == 1
    assert manifest["counts"]["reward"] == 1
    assert manifest["counts"]["rlvr"] == 1
    assert Path(manifest["posttrain_manifests"]["grpo"]).exists()
    sft_row = json.loads((tmp_path / "out" / "tool_sft.jsonl").read_text().splitlines()[0])
    assert sft_row["training_kind"] == "tool_sft"
    assert sft_row["metadata"]["tool_schema_masking"] is True


def test_posttrain_bridge_detects_tool_trajectory_dataset(tmp_path: Path) -> None:
    train_path = _write_jsonl(
        tmp_path / "tool_reward.jsonl",
        [{"training_kind": "tool_reward", "prompt": "use tool", "reward": 1.0}],
    )

    manifest = posttrain_bridge_2026.build_manifest(
        type(
            "Args",
            (),
            {
                "algorithm": "grpo",
                "model": "student",
                "train_jsonl": str(train_path),
                "eval_jsonl": None,
                "out_dir": str(tmp_path / "out"),
                "max_seq_len": 4096,
                "max_steps": 1,
                "learning_rate": 1e-6,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lora_r": 8,
                "lora_alpha": 16,
                "load_in_4bit": True,
                "reward_batch_size": 2,
                "num_generations": 2,
                "kl_beta": 0.02,
                "temperature": 0.8,
                "dry_run": True,
                "check_deps": False,
            },
        )()
    )

    assert manifest["dataset"]["schemas"]["tool_trajectory"] == 1
    assert "BFCL-style function-call rewards" in manifest["reward_contract"]["agentic_extensions"]


def test_full_harness_default_includes_agentic_tool_stage() -> None:
    stages = full_harness_2026.stage_list("all")

    assert "export_sft" in stages
    assert "agentic_tool_training" in stages
    assert stages.index("export_sft") < stages.index("agentic_tool_training") < stages.index("teacher_jobs")

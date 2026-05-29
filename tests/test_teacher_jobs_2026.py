from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import teacher_jobs_2026 as jobs


def test_teacher_jobs_preserve_external_training_record_payload(tmp_path: Path) -> None:
    records = tmp_path / "external.jsonl"
    record = {
        "input_json": {"messages": [{"role": "user", "content": "Fix the failing pytest."}]},
        "target_json": {"content": "Run pytest, patch the bug, and re-run tests."},
        "dataset_name": "Scale-SWE",
        "dataset_family": "coding_agentic",
        "training_bucket": "train",
        "license_tier": "permissive_attribution",
        "use_policy": "train",
        "curriculum_axes": ["swe", "test_verified"],
        "modalities": ["text", "tool"],
        "modality": "tool",
        "tool_calls": [{"tool": "shell_command", "arguments": {"command": "pytest"}}],
        "tool_results": [{"exit_code": 0, "stdout": "1 failed before patch; 1 passed after patch"}],
        "quality": {"score": 0.9},
        "contamination": {"status": "clean"},
        "contamination_status": "clean",
        "source_payload": {"source_id": "row-1", "dataset_name": "Scale-SWE"},
        "token_ids": [1, 2, 3, 4],
    }
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")

    built = jobs.build_jobs(str(records), "qwen3.6_27b_q4_local", "coding_agent_trajectory_critique", limit=1)
    payload = built[0]["input_json"]

    assert payload["prompt"] == "Fix the failing pytest."
    assert payload["target_text"] == "Run pytest, patch the bug, and re-run tests."
    assert payload["dataset"]["name"] == "Scale-SWE"
    assert payload["dataset"]["family"] == "coding_agentic"
    assert payload["dataset"]["training_bucket"] == "train"
    assert payload["curriculum_axes"] == ["swe", "test_verified"]
    assert payload["token_count"] == 4
    assert payload["token_id_sample"] == [1, 2, 3, 4]


def test_teacher_jobs_reject_toy_or_unknown_contamination_rows(tmp_path: Path) -> None:
    records = tmp_path / "external.jsonl"
    rows = [
        {
            "input_json": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
            "target_json": {"content": "4"},
            "modality": "text",
            "quality": {"score": 1.0},
            "contamination_status": "clean",
        },
        {
            "input_json": {"messages": [{"role": "user", "content": "Summarize the agent trace."}]},
            "target_json": {"content": "This should fail closed because contamination is unknown even though the answer is longer."},
            "modality": "text",
            "quality": {"score": 1.0},
        },
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    built = jobs.build_jobs(str(records), "qwen3.6_27b_q4_local", "trace_critique")

    assert built == []


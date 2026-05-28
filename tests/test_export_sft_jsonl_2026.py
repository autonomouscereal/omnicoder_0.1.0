from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import export_sft_jsonl as export_sft


def test_export_trace_conversations_orders_tool_events(tmp_path: Path) -> None:
    source = tmp_path / "curated.jsonl"
    rows = [
        {
            "input_json": {"messages": [{"role": "assistant", "content": "Now I can answer."}]},
            "target_json": {"content": "Done."},
            "quality": {"score": 1.0},
            "contamination_status": "clean",
            "lineage": {"trace_id": "trace-1", "step_index": 3},
        },
        {
            "input_json": {"messages": [{"role": "user", "content": "Run pytest."}]},
            "target_json": {"content": "I will run the tests."},
            "quality": {"score": 1.0},
            "contamination_status": "clean",
            "tool_calls": [{"tool": "shell_command", "arguments": {"command": "pytest"}}],
            "lineage": {"trace_id": "trace-1", "step_index": 1},
        },
        {
            "input_json": {"messages": [{"role": "tool", "content": "1 passed"}]},
            "target_json": {"content": "The tests passed."},
            "quality": {"score": 1.0},
            "contamination_status": "clean",
            "tool_results": [{"exit_code": 0, "stdout": "1 passed"}],
            "lineage": {"trace_id": "trace-1", "step_index": 2},
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "sft.jsonl"

    count = export_sft.export_trace_conversations(source, out, min_quality=0.0, allow_contaminated=False)
    payload = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    messages = payload["messages"]

    assert count == 1
    assert messages[0]["role"] == "user"
    assert "Run pytest" in messages[0]["content"]
    assert any(message["role"] == "tool" and "1 passed" in message["content"] for message in messages)
    assert messages[-1]["content"] == "Done."


def test_export_trace_conversations_emits_normalized_tool_fields(tmp_path: Path) -> None:
    source = tmp_path / "curated.jsonl"
    row = {
        "input_json": {
            "messages": [{"role": "assistant", "content": "I will inspect the file."}],
            "tool_name": "shell_command",
            "tool_input": {"command": "rg TODO src"},
        },
        "target_json": {
            "content": "The search completed.",
            "tool_output": {"exit_code": 0, "stdout": "src/app.py:TODO"},
        },
        "quality": {"score": 1.0},
        "contamination_status": "clean",
        "lineage": {"trace_id": "trace-tool-fields", "step_index": 1},
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "sft.jsonl"

    count = export_sft.export_trace_conversations(source, out, min_quality=0.0, allow_contaminated=False)
    messages = json.loads(out.read_text(encoding="utf-8").splitlines()[0])["messages"]

    assert count == 1
    assert any(message["role"] == "assistant" and "tool_call" in message["content"] for message in messages)
    assert any(message["role"] == "tool" and "src/app.py:TODO" in message["content"] for message in messages)


def test_export_trace_conversations_rejects_unknown_contamination_for_whole_trace(tmp_path: Path) -> None:
    source = tmp_path / "curated.jsonl"
    rows = [
        {
            "input_json": {"messages": [{"role": "user", "content": "Safe clean turn."}]},
            "target_json": {"content": "Clean answer."},
            "quality": {"score": 1.0},
            "contamination_status": "clean",
            "lineage": {"trace_id": "trace-mixed", "step_index": 1},
        },
        {
            "input_json": {"messages": [{"role": "user", "content": "Unknown provenance turn."}]},
            "target_json": {"content": "Should quarantine the whole trace."},
            "quality": {"score": 1.0},
            "lineage": {"trace_id": "trace-mixed", "step_index": 2},
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "sft.jsonl"

    count = export_sft.export_trace_conversations(source, out, min_quality=0.0, allow_contaminated=False)

    assert count == 0
    assert out.read_text(encoding="utf-8") == ""

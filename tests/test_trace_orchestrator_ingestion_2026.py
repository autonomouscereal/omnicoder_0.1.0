from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import curated_dataset_builder_2026 as builder
from omnicoder.data_factory import trace_orchestrator_2026 as traces


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def test_trace_orchestrator_enriches_tool_math_code_domains(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    source = root / "data" / "raw" / "local_agent.jsonl"
    _write_jsonl(
        source,
        [
            {
                "session_id": "trace-tool-1",
                "created_at": "2026-05-24T10:00:00Z",
                "event_type": "assistant_tool_call",
                "role": "assistant",
                "content": "Solve the equation, patch def add, and run pytest.",
                "tool_name": "shell_command",
                "tool_input": {"command": "pytest tests/test_math.py"},
                "tool_output": {"exit_code": 0, "stdout": "1 passed"},
            }
        ],
    )
    profile = {
        "profile_name": "unit_trace",
        "work_dir": str(root / "weights" / "trace"),
        "source_date": "2026-05-24",
        "trace_inputs": {"sources": [{"path": str(source), "harness": "local_agent"}], "patterns": ["*.jsonl"]},
        "data": {"bucket": "agentic_trace_sft_2026", "split": "train"},
    }
    _write_json(root / "profile.json", profile)
    monkeypatch.setattr(traces, "repo_root", lambda: root)

    manifest = traces.run_pipeline(root / "profile.json")
    normalized_path = Path(manifest["outputs"]["normalized"])
    row = json.loads(normalized_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["trace_features"]["source_harness"] == "local_agent"
    assert row["trace_features"]["has_tool_call"] is True
    assert row["trace_features"]["has_tool_result"] is True
    assert {"code", "math", "terminal", "tool"}.issubset(set(row["domains"]))
    assert row["tool_calls"][0]["tool"] == "shell_command"
    assert row["tool_results"][0]["exit_code"] == 0


def test_trace_orchestrator_collects_comfyui_media_directory(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    media_dir = root / "comfyui" / "output"
    image = media_dir / "qwen_trace.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not a real png but enough for hashing")
    _write_json(
        image.with_suffix(image.suffix + ".json"),
        {
            "prompt": "Generate an image of a terminal dashboard after a successful code test.",
            "caption": "A generated dashboard image.",
            "workflow": {"nodes": [{"class_type": "KSampler"}]},
        },
    )
    profile = {
        "profile_name": "unit_comfyui",
        "work_dir": str(root / "weights" / "trace"),
        "source_date": "2026-05-24",
        "trace_inputs": {"sources": [{"path": str(media_dir), "harness": "comfyui"}], "patterns": ["*.jsonl"]},
        "data": {"bucket": "comfyui_multimodal_trace_2026", "split": "train"},
    }
    _write_json(root / "profile.json", profile)
    monkeypatch.setattr(traces, "repo_root", lambda: root)

    manifest = traces.run_pipeline(root / "profile.json")
    assert manifest["stages"]["collect"]["files"] == 1
    assert manifest["stages"]["normalize"]["records"] == 1
    row = json.loads(Path(manifest["outputs"]["normalized"]).read_text(encoding="utf-8").splitlines()[0])

    assert row["trace_features"]["source_harness"] == "comfyui"
    assert row["trace_features"]["has_multimodal"] is True
    assert "image" in row["modalities"]
    assert "multimodal" in row["domains"]
    assert row["media_refs"][0]["path"] == str(image)


def test_trace_orchestrator_collects_comfyui_manifest_directory(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    manifest_dir = root / "data" / "raw" / "comfyui_outputs_2026"
    image = root / "comfyui" / "output" / "qwen_image.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image bytes")
    _write_jsonl(
        manifest_dir / "comfyui_run.jsonl",
        [
            {
                "bucket": "multimodal_media",
                "split": "train",
                "source_date": "2026-05-24",
                "input_json": {
                    "messages": [{"role": "user", "content": "Generate a code dashboard image."}],
                    "modality": "image",
                    "workflow": {"nodes": [{"class_type": "KSampler"}]},
                },
                "target_json": {
                    "artifact_path": str(image),
                    "media_type": "image/png",
                    "sha256": "abc123",
                    "caption": "dashboard image",
                },
                "lineage": {"source": "comfyui_output"},
            }
        ],
    )
    profile = {
        "profile_name": "unit_comfyui_manifest",
        "work_dir": str(root / "weights" / "trace"),
        "source_date": "2026-05-24",
        "trace_inputs": {"sources": [{"path": str(manifest_dir), "harness": "comfyui"}], "patterns": ["*.jsonl"]},
        "data": {"bucket": "comfyui_multimodal_trace_2026", "split": "train"},
    }
    _write_json(root / "profile.json", profile)
    monkeypatch.setattr(traces, "repo_root", lambda: root)

    manifest = traces.run_pipeline(root / "profile.json")
    row = json.loads(Path(manifest["outputs"]["normalized"]).read_text(encoding="utf-8").splitlines()[0])

    assert manifest["stages"]["collect"]["files"] == 1
    assert row["trace_features"]["source_harness"] == "comfyui"
    assert "image" in row["modalities"]
    assert row["media_refs"][0]["sha256"] == "abc123"


def test_agent_memory_export_uses_server_or_workstation_script_candidates(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    script = root / "agent_memory.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'rows': [{'event_type': 'PostToolUse', 'content': 'pytest passed'}]}))\n",
        encoding="utf-8",
    )
    profile = {
        "builder_2026": {
            "out_dir": str(root / "weights" / "curated"),
            "agent_memory_cli_export": {
                "enabled": True,
                "script": "C:/missing/agent_memory.py",
                "script_candidates": ["/missing/linux/agent_memory.py", str(script)],
                "out": "data/raw/agent_memory_events_2026.jsonl",
                "limit": 5,
                "all_spaces": True,
            },
        }
    }
    _write_json(root / "profile.json", profile)
    monkeypatch.setattr(builder, "repo_root", lambda: root)

    result = builder.export_agent_memory_only(root / "profile.json", root / "weights" / "curated")
    out = root / "data" / "raw" / "agent_memory_events_2026.jsonl"

    assert result["status"] == "ok"
    assert result["records"] == 1
    assert "pytest passed" in out.read_text(encoding="utf-8")


def test_builder_prefers_raw_postgres_agent_memory_export(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    profile = {
        "agent_memory_postgres_export": {
            "enabled": True,
            "out": "data/raw/agent_memory_events_2026.jsonl",
            "date_floor": "2025-01-01",
        },
        "builder_2026": {
            "out_dir": str(root / "weights" / "curated"),
            "agent_memory_cli_export": {
                "enabled": True,
                "script": "C:/missing/agent_memory.py",
            },
        },
    }
    _write_json(root / "profile.json", profile)
    monkeypatch.setattr(builder, "repo_root", lambda: root)

    def fake_export(cfg, out_path):
        assert cfg["date_floor"] == "2025-01-01"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"event_type": "PostToolUse", "content": "raw pg event"}) + "\n", encoding="utf-8")
        return {"status": "ok", "out": str(out_path), "records": 1}

    monkeypatch.setattr(builder.export_agent_memory_postgres_2026, "export_rows", fake_export)
    result = builder.export_agent_memory_only(root / "profile.json", root / "weights" / "curated")

    assert result["status"] == "ok"
    assert result["path"] == "raw_postgresql"
    assert "raw pg event" in (root / "data" / "raw" / "agent_memory_events_2026.jsonl").read_text(encoding="utf-8")


def test_trace_orchestrator_assigns_source_step_indices(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    source = root / "data" / "raw" / "local_agent.jsonl"
    _write_jsonl(
        source,
        [
            {"session_id": "s1", "role": "user", "content": "Run the test."},
            {"session_id": "s1", "role": "assistant", "content": "I will run pytest.", "tool_name": "shell_command"},
        ],
    )
    profile = {
        "profile_name": "unit_trace_order",
        "work_dir": str(root / "weights" / "trace"),
        "source_date": "2026-05-24",
        "trace_inputs": {"sources": [{"path": str(source), "harness": "local_agent"}], "patterns": ["*.jsonl"]},
        "data": {"bucket": "agentic_trace_sft_2026", "split": "train"},
    }
    _write_json(root / "profile.json", profile)
    monkeypatch.setattr(traces, "repo_root", lambda: root)

    manifest = traces.run_pipeline(root / "profile.json")
    rows = [json.loads(line) for line in Path(manifest["outputs"]["normalized"]).read_text(encoding="utf-8").splitlines()]

    assert [row["lineage"]["source_index"] for row in rows] == [1, 2]
    assert [row["lineage"]["step_index"] for row in rows] == [1, 2]

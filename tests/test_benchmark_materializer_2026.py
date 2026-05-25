from __future__ import annotations

import json
from pathlib import Path

from omnicoder.data_factory import benchmark_materializer_2026 as materializer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _profile(path: Path) -> Path:
    _write_json(
        path,
        {
            "benchmarks": [
                {
                    "benchmark_id": "multimodal_mmmu_pro_2026",
                    "adapter_kind": "expert_multimodal_reasoning",
                    "axis": "multimodal_understanding",
                    "source": "https://huggingface.co/datasets/MMMU/MMMU_Pro",
                    "splits": {"smoke": "one public-dev item"},
                }
            ],
            "reportable_task_roots": {
                "multimodal_mmmu_pro_2026": ["data/eval/reportable_2026/mmmu_pro_authorized.jsonl"]
            },
            "reportable_snapshots": {
                "multimodal_mmmu_pro_2026": {
                    "snapshot_id": "mmmu-pro-authorized-2026-current",
                    "snapshot_authorization": "official_or_authorized_current_release",
                    "dataset_revision": "mmmu-pro-authorized-2026-current",
                    "source": "https://huggingface.co/datasets/MMMU/MMMU_Pro",
                    "authorization_ref": "operator_authorized_snapshot_manifest",
                    "task_root": "data/eval/reportable_2026/mmmu_pro_authorized.jsonl",
                }
            },
        },
    )
    return path


def test_materializer_writes_local_public_dev_rows_without_network(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "profile.json")
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"id": "q1", "question": "What is shown?", "choices": ["A", "B"], "answer": "A"},
            {"id": "q1", "question": "duplicate should be deduped", "answer": "B"},
        ],
    )
    out_root = tmp_path / "materialized"

    assert (
        materializer.main(
            [
                "--profile",
                str(profile),
                "--out-root",
                str(out_root),
                "--run-id",
                "run_a",
                "--benchmark",
                "multimodal_mmmu_pro_2026",
                "--source-override",
                f"multimodal_mmmu_pro_2026={source}",
                "--limit",
                "8",
                "materialize",
            ]
        )
        == 0
    )

    rows = _read_jsonl(out_root / "local_2026" / "multimodal_mmmu_pro_2026_public_dev.jsonl")
    manifest = _read_json(out_root / "manifests" / "benchmark_materialization_manifest.json")
    assert len(rows) == 1
    assert rows[0]["reportable"] is False
    assert rows[0]["local_only"] is True
    assert rows[0]["benchmark_id"] == "multimodal_mmmu_pro_2026"
    assert manifest["rows"] == 1
    assert manifest["records"][0]["local_only"] is True


def test_materializer_writes_run_scoped_authorized_rows(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "profile.json")
    source = tmp_path / "authorized.jsonl"
    _write_jsonl(source, [{"task_id": "auth-1", "question": "Choose A", "choices": ["A", "B"], "answer": "A"}])
    out_root = tmp_path / "materialized"

    assert (
        materializer.main(
            [
                "--profile",
                str(profile),
                "--out-root",
                str(out_root),
                "--run-id",
                "run_b",
                "--benchmark",
                "multimodal_mmmu_pro_2026",
                "--mode",
                "reportable",
                "--source-override",
                f"multimodal_mmmu_pro_2026={source}",
                "materialize",
            ]
        )
        == 0
    )

    rows = _read_jsonl(out_root / "reportable_2026" / "multimodal_mmmu_pro_2026_authorized.jsonl")
    manifest = _read_json(out_root / "manifests" / "benchmark_materialization_manifest.json")
    assert rows[0]["reportable"] is True
    assert rows[0]["local_only"] is False
    assert rows[0]["snapshot_id"] == "mmmu-pro-authorized-2026-current"
    assert rows[0]["snapshot_authorization"] == "official_or_authorized_current_release"
    assert manifest["records"][0]["reportable"] is True


def test_materializer_reads_terminal_task_toml_and_instruction(tmp_path: Path) -> None:
    root = tmp_path / "terminal"
    task_dir = root / "repair-cli"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('timeout = 300\ncategory = "shell"\n', encoding="utf-8")
    (task_dir / "instruction.md").write_text("Fix the CLI and make the tests pass.", encoding="utf-8")

    rows, errors = materializer.scan_local_source(root, 8)
    task = materializer.normalize_task(
        "agent_terminal_bench_2026",
        rows[0],
        {"kind": "terminal", "source": "fixture"},
        {"adapter_kind": "container_terminal_task"},
        {},
        "public-dev",
        str(root),
        0,
    )

    assert errors == []
    assert task is not None
    assert task["task_id"] == "repair-cli"
    assert "Fix the CLI" in task["prompt"]


def test_materializer_reads_mcpmark_meta_json(tmp_path: Path) -> None:
    meta = tmp_path / "mcp" / "tasks" / "notion" / "easy" / "task_a" / "meta.json"
    _write_json(meta, {"task_id": "task_a", "description": "Move the Notion cards.", "mcp": ["notion"]})

    rows, errors = materializer.scan_local_source(tmp_path / "mcp", 8)
    task = materializer.normalize_task(
        "agent_mcp_workflows_2026",
        rows[0],
        {"kind": "tool", "source": "fixture"},
        {"adapter_kind": "mcp_fixture_adapter"},
        {},
        "public-dev",
        str(tmp_path / "mcp"),
        0,
    )

    assert errors == []
    assert task is not None
    assert task["task_id"] == "task_a"
    assert task["prompt"] == "Move the Notion cards."

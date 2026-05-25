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


def test_materializer_reads_mcp_bench_runner_tasks(tmp_path: Path) -> None:
    task_file = tmp_path / "mcp-bench" / "tasks" / "mcpbench_tasks_multi_2server_runner_format.json"
    _write_json(
        task_file,
        {
            "server_tasks": [
                {
                    "server_name": "Google Maps+Weather Data",
                    "servers": ["Google Maps", "Weather Data"],
                    "combination_name": "Travel Planning Suite",
                    "combination_type": "two_server_combinations",
                    "tasks": [
                        {
                            "task_id": "trip-001",
                            "task_description": "Plan a trip using map and weather tools.",
                            "fuzzy_description": "Plan a trip.",
                        }
                    ],
                }
            ],
            "total_tasks": 0,
        },
    )

    rows, errors = materializer.scan_local_source(tmp_path / "mcp-bench", 8)
    task = materializer.normalize_task(
        "agent_mcp_bench_2026",
        rows[0],
        {"kind": "tool", "source": "fixture"},
        {"adapter_kind": "mcp_agent_workflow_eval"},
        {},
        "public-dev",
        str(tmp_path / "mcp-bench"),
        0,
    )

    assert errors == []
    assert task is not None
    assert task["task_id"] == "trip-001"
    assert task["prompt"] == "Plan a trip using map and weather tools."
    assert task["tools"] == [{"name": "Google Maps"}, {"name": "Weather Data"}]
    assert task["server_name"] == "Google Maps+Weather Data"


def test_materializer_reads_mcp_universe_single_task_json(tmp_path: Path) -> None:
    task_file = tmp_path / "MCP-Universe" / "mcpuniverse" / "benchmark" / "configs" / "mcpuniverse" / "maps" / "task_0001.json"
    _write_json(
        task_file,
        {
            "category": "general",
            "question": "Plan a multi-city route with rest stops.",
            "mcp_servers": [{"name": "google-maps"}],
            "output_format": {"routes": ["..."]},
            "evaluators": [{"type": "json"}],
        },
    )

    rows, errors = materializer.scan_local_source(tmp_path / "MCP-Universe", 8)
    task = materializer.normalize_task(
        "agent_mcp_universe_2026",
        rows[0],
        {"kind": "tool", "source": "fixture"},
        {"adapter_kind": "mcp_server_universe_eval"},
        {},
        "public-dev",
        str(tmp_path / "MCP-Universe"),
        0,
    )

    assert errors == []
    assert task is not None
    assert task["task_id"] == "task_0001"
    assert task["prompt"] == "Plan a multi-city route with rest stops."
    assert task["tools"] == [{"name": "google-maps"}]
    assert task["expected_tool_call"] == {"routes": ["..."]}


def test_materializer_reads_agent_company_scenarios_with_task_prompt(tmp_path: Path) -> None:
    task_dir = tmp_path / "TheAgentCompany" / "workspaces" / "tasks" / "admin-ask-for-meeting-feedback"
    _write_json(
        task_dir / "scenarios.json",
        {
            "Huang Jie": {
                "extra_info": "Someone will ask you about the meeting.",
                "strategy_hint": "Give concise feedback.",
            }
        },
    )
    (task_dir / "task.md").write_text("Collect meeting feedback from the right coworkers.", encoding="utf-8")
    (task_dir / "checkpoints.md").write_text("- feedback collected", encoding="utf-8")
    (task_dir / "dependencies.yml").write_text("services:\n  - rocketchat\n  - owncloud\n", encoding="utf-8")

    rows, errors = materializer.scan_local_source(tmp_path / "TheAgentCompany", 8)
    task = materializer.normalize_task(
        "agent_theagentcompany_enterprise_2026",
        rows[0],
        {"kind": "agent_tool", "source": "fixture"},
        {"adapter_kind": "enterprise_workflow_eval"},
        {},
        "public-dev",
        str(tmp_path / "TheAgentCompany"),
        0,
    )

    assert errors == []
    assert task is not None
    assert task["task_id"] == "admin-ask-for-meeting-feedback:Huang_Jie"
    assert task["prompt"] == "Collect meeting feedback from the right coworkers."
    assert task["answer"] == "Give concise feedback."
    assert task["scenario_extra_info"] == "Someone will ask you about the meeting."
    assert task["checkpoints"] == "- feedback collected"


def test_materializer_normalizes_long_context_prompt_aliases() -> None:
    task = materializer.normalize_task(
        "long_context_longproc_2026",
        {
            "id": "html_to_tsv_0.5k_001",
            "input_prompt": "Extract the requested rows from the long HTML document.",
            "reference_output": "name\tprice\nexample\t1",
        },
        {"kind": "long_context", "source": "fixture"},
        {"adapter_kind": "long_input_long_output_process_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )

    assert task is not None
    assert task["prompt"].startswith("Extract the requested")
    assert task["answer"] == "name\tprice\nexample\t1"


def test_materializer_tracks_2026_official_source_mirrors() -> None:
    required = {
        "agent_mcp_bench_2026",
        "agent_mcp_universe_2026",
        "agent_clawbench_browser_2026",
        "agent_browsecomp_long_context_2026",
        "agent_theagentcompany_enterprise_2026",
        "agent_paperbench_2026",
        "agent_gdpval_2026",
        "coding_swe_lancer_2026",
        "coding_swe_rebench_v2_2026",
        "reasoning_hle_rolling_2026",
        "reasoning_matharena_2026",
        "reasoning_rlvr_linearity_math_2026",
        "coding_nous_rlvr_coding_2026",
        "long_context_longproc_2026",
        "long_context_nolima_1m_2026",
        "multimodal_audiobench_mmau_2026",
        "multimodal_audiomarathon_2026",
        "multimodal_mmar_audio_music_reasoning_2026",
        "multimodal_rewardbench2_2026",
        "generation_audio_speech_2026",
        "generation_oneig_bench_2026",
    }
    assert required.issubset(materializer.KNOWN_BENCHMARKS)
    voicebench = materializer.KNOWN_BENCHMARKS["generation_audio_speech_2026"]["hf"][0]
    assert voicebench["id"] == "hlt-lab/voicebench"
    assert voicebench["config"] == "ifeval"
    assert materializer.KNOWN_BENCHMARKS["agent_clawbench_browser_2026"]["hf"][0] == "TIGER-Lab/ClawBench"
    assert materializer.KNOWN_BENCHMARKS["long_context_nolima_1m_2026"]["hf"] == ["amodaresi/NoLiMa"]
    assert materializer.KNOWN_BENCHMARKS["multimodal_audiomarathon_2026"]["hf"][0] == "AudioMarathon/AudioMarathon"
    assert materializer.KNOWN_BENCHMARKS["coding_swe_rebench_v2_2026"]["splits"] == ["train"]
    rewardbench = materializer.KNOWN_BENCHMARKS["multimodal_rewardbench2_2026"]["hf"][0]
    assert rewardbench["id"] == "rl-research/multimodal-rewardbench-2"
    assert rewardbench["config"] == "edit"
    oneig = materializer.KNOWN_BENCHMARKS["generation_oneig_bench_2026"]["hf"][0]
    assert oneig["config"] == "OneIG-Bench"

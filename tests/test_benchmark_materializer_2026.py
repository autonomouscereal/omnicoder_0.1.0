from __future__ import annotations

import argparse
import json
import sys
import types
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


def test_stable_hash_accepts_mixed_type_nested_keys() -> None:
    mixed = {"outer": {True: "bool key", "true": "string key", 7: {"x": "nested"}}, "rows": [("a", "b")]}

    assert materializer.stable_hash(mixed) == materializer.stable_hash(mixed)
    assert len(materializer.stable_hash(mixed)) == 64


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


def test_materializer_normalizes_countdown_rewardbench_and_oneig_aliases() -> None:
    countdown = materializer.normalize_task(
        "long_context_longproc_2026",
        {"nums": [21, 16, 17, 26], "target": 46, "solution_text": "21 + 16 = 37\n37 - 17 = 20\n26 + 20 = 46"},
        {"kind": "long_context", "source": "fixture"},
        {"adapter_kind": "long_input_long_output_process_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    reward_pair = materializer.normalize_task(
        "multimodal_rewardbench2_2026",
        {
            "pair_id": "pair-1",
            "prompt_text": "<image_0> Make this product photo look cinematic.",
            "response_a_text": "<image_0>",
            "response_b_text": "<image_0>",
            "chosen": "a",
        },
        {"kind": "multimodal_mcq", "source": "fixture"},
        {"adapter_kind": "multimodal_reward_pair_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    oneig = materializer.normalize_task(
        "generation_oneig_bench_2026",
        {"id": "001", "prompt_en": "A crisp studio photo of a transparent smartwatch."},
        {"kind": "image_generation", "source": "fixture"},
        {"adapter_kind": "image_generation_instruction_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )

    assert countdown is not None
    assert "target 46" in countdown["prompt"]
    assert countdown["answer"].startswith("21 + 16")
    assert reward_pair is not None
    assert reward_pair["prompt"].startswith("<image_0>")
    assert reward_pair["choices"] == ["<image_0>", "<image_0>"]
    assert reward_pair["answer"] == "a"
    assert oneig is not None
    assert oneig["prompt"].startswith("A crisp studio")


def test_materializer_normalizes_mcp_atlas_uppercase_fields() -> None:
    task = materializer.normalize_task(
        "agent_mcp_atlas_2026",
        {
            "TASK": "689f4d693e212e8ef3390731",
            "PROMPT": "Use GitHub and WHOIS tools to compare repository and domain dates.",
            "ENABLED_TOOLS": '["github_search_repositories","whois_whois_domain"]',
            "GTFA_CLAIMS": '["The repository was created in 2013.","The domain was registered in 2006."]',
        },
        {"kind": "mcp_real_server_tool_eval", "source": "fixture"},
        {"adapter_kind": "mcp_real_server_tool_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )

    assert task is not None
    assert task["task_id"] == "689f4d693e212e8ef3390731"
    assert task["prompt"].startswith("Use GitHub")
    assert task["tools"] == ["github_search_repositories", "whois_whois_domain"]
    assert "repository was created" in task["answer"]
    assert "domain was registered" in task["expected_tool_call"][1]


def test_materializer_normalizes_ttsds2_listening_rows() -> None:
    task = materializer.normalize_task(
        "generation_ttsds2_2026",
        {
            "id": "valle_v2_1",
            "audio": "noisy/valle_v2/example.wav",
            "dataset": "Noisy",
            "system": "valle_v2",
            "rating_type": "mos",
            "value": 4,
            "annotator": "10de3a4ff444823253dfb8fc9037856b",
        },
        {"kind": "audio_generation", "source": "fixture"},
        {"adapter_kind": "tts_generation_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )

    assert task is not None
    assert task["task_id"] == "valle_v2_1:10de3a4ff444823253dfb8fc9037856b"
    assert task["audio"] == "noisy/valle_v2/example.wav"
    assert task["answer"] == 4
    assert task["expected_artifact_kind"] == "audio_generation"


def test_materializer_normalizes_next_wave_2026_aliases() -> None:
    octo = materializer.normalize_task(
        "coding_octocodingbench_2026",
        {
            "instance_id": "octo-1",
            "user_query": "Update the CLI while following AGENTS.md.",
            "checklist": ["tests pass"],
            "system_prompt": "You are in a repo.",
            "scaffold": {"files": ["AGENTS.md"]},
            "workspace_abs_path": "/work/repo",
        },
        {"kind": "agent_tool", "source": "fixture"},
        {"adapter_kind": "agent_tool"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    veri = materializer.normalize_task(
        "coding_verisoftbench_2026",
        {
            "id": "veri-1",
            "thm_stmt": "theorem add_zero (n : Nat) : n + 0 = n := by",
            "ground_truth_proof": "simp",
            "lean_root": "Mathlib",
            "rel_path": "Demo.lean",
            "imports": ["Mathlib"],
        },
        {"kind": "formal_verification", "source": "fixture"},
        {"adapter_kind": "lean_formal_verification"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    imo = materializer.normalize_task(
        "reasoning_imo_bench_2026",
        {"Problem ID": "imo-1", "Problem": "Find all integers n such that n^2 = n.", "Short Answer": "0 and 1"},
        {"kind": "math", "source": "fixture"},
        {"adapter_kind": "math_reasoning"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    graph = materializer.normalize_task(
        "long_context_graphwalks_2026",
        {"prompt": "Walk the graph from A.", "answer_nodes": ["B", "C"], "problem_type": "walk", "date_added": "2026-05-01"},
        {"kind": "long_context", "source": "fixture"},
        {"adapter_kind": "long_context_graph_reasoning"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    mmlong = materializer.normalize_task(
        "multimodal_mmlongbench_2026",
        {
            "id": "mmlong-1",
            "question": "Which figure contains the answer?",
            "answer": "figure 7",
            "image_list": ["page7.png"],
            "needle_image_list": ["needle.png"],
            "ctxs": [{"text": "long doc"}],
        },
        {"kind": "multimodal_long_context", "source": "fixture"},
        {"adapter_kind": "multimodal_long_context"},
        {},
        "public-dev",
        "fixture",
        0,
    )

    assert octo is not None and octo["prompt"].startswith("Update the CLI")
    assert octo["expected_tool_call"] == ["tests pass"]
    assert octo["system_prompt"] == "You are in a repo."
    assert veri is not None and veri["prompt"].startswith("theorem add_zero")
    assert veri["answer"] == "simp"
    assert veri["lean_root"] == "Mathlib"
    assert imo is not None and imo["task_id"] == "imo-1" and imo["answer"] == "0 and 1"
    assert graph is not None and graph["answer"] == ["B", "C"] and graph["problem_type"] == "walk"
    assert mmlong is not None and mmlong["images"] == ["page7.png"]
    assert mmlong["needle_image_list"] == ["needle.png"]
    assert mmlong["ctxs"] == [{"text": "long doc"}]


def test_materializer_normalizes_agentic_omni_benchmark_wave_aliases() -> None:
    swe_mera = materializer.normalize_task(
        "coding_swe_mera_2026",
        {
            "instance_id": "mera-1",
            "problem_statement": "Fix the failing parser.",
            "repo": "demo/repo",
            "base_commit": "abc123",
            "test_patch": "diff --git a/test_parser.py b/test_parser.py",
            "FAIL_TO_PASS": ["tests/test_parser.py::test_cli"],
            "command_test": "pytest tests/test_parser.py",
        },
        {"kind": "swe", "source": "fixture"},
        {"adapter_kind": "dynamic_swe_repo_patch_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    mathnet = materializer.normalize_task(
        "multimodal_mathnet_2026",
        {
            "uid": "mathnet-1",
            "problem_markdown": "Use the diagram to prove the angle claim.",
            "final_answer": "42 degrees",
            "images": ["diagram.png"],
            "competition": "Olympiad",
            "country": "US",
            "topics_flat": ["geometry"],
        },
        {"kind": "multimodal_math", "source": "fixture"},
        {"adapter_kind": "multimodal_olympiad_math_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    sonic = materializer.normalize_task(
        "multimodal_sonic_o1_2026",
        {
            "annotation_id": "sonic-1",
            "question_text": "When does the speaker laugh?",
            "ground_truth": "00:03-00:05",
            "video_url": "clip.mp4",
            "audio_url": "clip.wav",
            "start_time": 3.0,
            "end_time": 5.0,
            "duration": 10.0,
            "rationale": "The laugh begins after the door closes.",
        },
        {"kind": "video_audio", "source": "fixture"},
        {"adapter_kind": "audio_video_o1_reasoning_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )
    emergent_tts = materializer.normalize_task(
        "generation_emergent_tts_eval_2026",
        {
            "item_id": "tts-1",
            "text_to_synthesize": "Whisper the line with relief.",
            "reference_audio": "voice.wav",
            "language": "en",
            "evolution_depth": 3,
            "rubric": {"prosody": "relieved whisper"},
        },
        {"kind": "audio_generation", "source": "fixture"},
        {"adapter_kind": "expressive_tts_instruction_eval"},
        {},
        "public-dev",
        "fixture",
        0,
    )

    assert swe_mera is not None and swe_mera["repo"] == "demo/repo"
    assert swe_mera["FAIL_TO_PASS"] == ["tests/test_parser.py::test_cli"]
    assert swe_mera["command_test"] == "pytest tests/test_parser.py"
    assert mathnet is not None and mathnet["prompt"].startswith("Use the diagram")
    assert mathnet["answer"] == "42 degrees"
    assert mathnet["country"] == "US"
    assert sonic is not None and sonic["prompt"].startswith("When does")
    assert sonic["video_url"] == "clip.mp4"
    assert sonic["audio_url"] == "clip.wav"
    assert sonic["start_time"] == 3.0 and sonic["end_time"] == 5.0
    assert emergent_tts is not None and emergent_tts["expected_artifact_kind"] == "audio_generation"
    assert emergent_tts["reference_audio"] == "voice.wav"
    assert emergent_tts["rubric"] == {"prosody": "relieved whisper"}


def test_hf_rows_falls_back_to_raw_hub_files(monkeypatch, tmp_path: Path) -> None:
    remote_file = tmp_path / "mmlong.jsonl"
    _write_jsonl(
        remote_file,
        [
            {
                "id": "mmlb-1",
                "question": "Find the image needle.",
                "answer": "page 4",
                "image_list": ["page4.png"],
            }
        ],
    )

    datasets_module = types.SimpleNamespace(
        Audio=type("Audio", (), {"__init__": lambda self, decode=False: None}),
        load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("generator failed")),
    )
    hub_module = types.SimpleNamespace(
        list_repo_files=lambda repo_id, repo_type="dataset": [
            "README.md",
            "mmlb_data_example/NIAH/retrieval-image_test_K128_dep6.jsonl",
        ],
        hf_hub_download=lambda repo_id, filename, repo_type="dataset", cache_dir=None: str(remote_file),
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    rows, errors = materializer.hf_rows(
        {
            "hf": [
                {
                    "id": "ZhaoweiWang/MMLongBench",
                    "splits": ["test"],
                    "files": ["mmlb_data_example/**/*.jsonl"],
                }
            ]
        },
        tmp_path / "cache",
        8,
    )

    assert len(rows) == 1
    assert rows[0]["_hf_file"] == "mmlb_data_example/NIAH/retrieval-image_test_K128_dep6.jsonl"
    assert rows[0]["image_list"] == ["page4.png"]
    assert not errors


def test_materializer_tracks_2026_official_source_mirrors() -> None:
    required = {
        "agent_mcp_bench_2026",
        "agent_mcp_atlas_2026",
        "agent_mcp_universe_2026",
        "agent_clawbench_browser_2026",
        "agent_browsecomp_long_context_2026",
        "agent_theagentcompany_enterprise_2026",
        "agent_paperbench_2026",
        "agent_gdpval_2026",
        "reasoning_arc_agi2_2026",
        "reasoning_arc_agi3_2026",
        "coding_swe_lancer_2026",
        "coding_swe_rebench_v2_2026",
        "coding_swe_polybench_2026",
        "coding_swe_smith_2026",
        "coding_octocodingbench_2026",
        "coding_gittaskbench_2026",
        "coding_verisoftbench_2026",
        "coding_swe_mera_2026",
        "coding_ale_bench_2026",
        "reasoning_hle_rolling_2026",
        "reasoning_imo_bench_2026",
        "reasoning_matharena_2026",
        "reasoning_rlvr_linearity_math_2026",
        "coding_nous_rlvr_coding_2026",
        "coding_multi_swe_bench_2026",
        "coding_swe_bench_plus_2026",
        "long_context_helmet_longproc_2026",
        "long_context_longcodebench_2026",
        "long_context_graphwalks_2026",
        "long_context_longproc_2026",
        "long_context_nolima_1m_2026",
        "multimodal_audiobench_mmau_2026",
        "multimodal_mmau_pro_2026",
        "multimodal_mmlongbench_2026",
        "multimodal_mathnet_2026",
        "multimodal_video_morse500_2026",
        "multimodal_sonic_o1_2026",
        "multimodal_mme_unify_2026",
        "multimodal_longspeech_2026",
        "multimodal_audiomarathon_2026",
        "multimodal_mmar_audio_music_reasoning_2026",
        "multimodal_rewardbench2_2026",
        "generation_audio_speech_2026",
        "generation_ttsds2_2026",
        "generation_emergent_tts_eval_2026",
        "generation_long_tts_eval_2026",
        "generation_tta_bench_2026",
        "generation_nv_bench_2026",
        "generation_oneig_bench_2026",
        "agent_agencybench_2026",
        "agent_locobench_agent_2026",
        "safety_tool_security_2026",
        "deployment_turboquant_kv_1m_2026",
        "deployment_performance_2026",
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
    ttsds2 = materializer.KNOWN_BENCHMARKS["generation_ttsds2_2026"]["hf"][0]
    assert ttsds2["id"] == "ttsds/listening_test"
    assert ttsds2["splits"] == ["test"]
    octo = materializer.KNOWN_BENCHMARKS["coding_octocodingbench_2026"]["hf"][0]
    assert octo["id"] == "MiniMaxAI/OctoCodingBench"
    graphwalks = materializer.KNOWN_BENCHMARKS["long_context_graphwalks_2026"]["hf"][0]
    assert graphwalks["id"] == "openai/graphwalks"
    assert materializer.KNOWN_BENCHMARKS["agent_terminal_bench_2026"]["hf"] == ["harborframework/terminal-bench-2.0"]
    assert materializer.KNOWN_BENCHMARKS["coding_swe_mera_2026"]["hf"][0]["id"] == "MERA-evaluation/SWE-MERA"
    assert materializer.KNOWN_BENCHMARKS["multimodal_mme_unify_2026"]["hf"][0]["id"] == "wulin222/MME-Unify"
    assert materializer.KNOWN_BENCHMARKS["generation_long_tts_eval_2026"]["hf"][0]["id"] == "wcy1122/Long-TTS-Eval"


def test_audit_profile_reports_materializer_and_core25_gaps(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _write_json(
        profile,
        {
            "benchmarks": [
                {
                    "benchmark_id": "agent_bfcl_v4_2026",
                    "adapter_kind": "tool_call_state_scorer",
                    "axis": "agent_tool",
                    "source": "fixture",
                    "splits": {"smoke": "fixture"},
                },
                {
                    "benchmark_id": "fresh_missing_2026",
                    "adapter_kind": "new_eval",
                    "axis": "reasoning",
                    "source": "fixture",
                    "splits": {"smoke": "fixture"},
                },
            ],
            "reportable_core_25": ["agent_bfcl_v4_2026", "fresh_missing_2026"],
            "reportable_task_roots": {
                "agent_bfcl_v4_2026": ["data/eval/reportable_2026/bfcl_v4_authorized.jsonl"]
            },
            "reportable_snapshots": {
                "agent_bfcl_v4_2026": {
                    "snapshot_id": "bfcl-authorized",
                    "snapshot_authorization": "official_or_authorized_current_release",
                    "dataset_revision": "bfcl-authorized",
                    "source": "fixture",
                }
            },
        },
    )

    report = materializer.audit_profile(
        argparse.Namespace(
            profile=str(profile),
            benchmark=None,
            suite="profile",
            fail_core25=True,
            fail_missing_materializers=True,
            fail_known_not_profile=False,
        )
    )

    assert report["status"] == "failed"
    assert "fresh_missing_2026" in report["missing_materializer_or_snapshot"]
    assert "fresh_missing_2026" in report["core25"]["missing_reportable_task_root"]
    assert "fresh_missing_2026" in report["core25"]["missing_reportable_snapshot"]


def test_audit_profile_cli_exits_nonzero_on_core25_gap(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _write_json(
        profile,
        {
            "benchmarks": [
                {
                    "benchmark_id": "agent_bfcl_v4_2026",
                    "adapter_kind": "tool_call_state_scorer",
                    "axis": "agent_tool",
                    "source": "fixture",
                    "splits": {"smoke": "fixture"},
                }
            ],
            "reportable_core_25": ["agent_bfcl_v4_2026"],
            "reportable_task_roots": {},
            "reportable_snapshots": {},
        },
    )

    assert materializer.main(["--profile", str(profile), "--suite", "profile", "audit-profile", "--fail-core25"]) == 5

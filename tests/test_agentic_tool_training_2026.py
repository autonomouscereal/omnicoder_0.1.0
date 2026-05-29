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
                    {
                        "role": "assistant",
                        "content": "The dashboard health check returned a successful JSON response from the AI server, so the trace includes a real tool call, observation, and grounded final report.",
                    },
                ],
                "quality": {"score": 0.91},
                "contamination_status": "clean",
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
    assert manifest["counts"]["terminal_rlvr"] == 1
    assert manifest["counts"]["browser_rlvr"] == 1
    assert manifest["counts"]["tool_rlvr"] == 1
    assert Path(manifest["posttrain_manifests"]["grpo"]).exists()
    assert Path(manifest["posttrain_manifests"]["terminal_rlvr"]).exists()
    sft_row = json.loads((tmp_path / "out" / "tool_sft.jsonl").read_text().splitlines()[0])
    assert sft_row["training_kind"] == "tool_sft"
    assert sft_row["metadata"]["tool_schema_masking"] is True
    terminal_row = json.loads((tmp_path / "out" / "terminal_rlvr.jsonl").read_text().splitlines()[0])
    assert terminal_row["domains"] == ["terminal"]
    assert "exit_code_zero" in terminal_row["verifier"]["checks"]


def test_posttrain_bridge_detects_tool_trajectory_dataset(tmp_path: Path) -> None:
    train_path = _write_jsonl(
        tmp_path / "tool_reward.jsonl",
        [
            {
                "training_kind": "tool_reward",
                "prompt": "Use the server manager tool to check service health and report the observed result.",
                "target": "The tool call completed and returned a healthy service response, so the task outcome reward is positive.",
                "reward": 1.0,
                "contamination_status": "clean",
                "quality_score": 0.95,
            }
        ],
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


def _capture_full_harness_commands(tmp_path: Path, monkeypatch, profile: dict, source: Path) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], log_path: Path, cwd: Path, env: dict | None = None) -> tuple[int, dict]:
        commands.append(cmd)
        return 0, {}

    monkeypatch.setattr(full_harness_2026, "run_command", fake_run_command)
    registry = full_harness_2026.JsonlRunRegistry(tmp_path / "registry")
    run_id = "dry-run-hardening"
    registry.create_run("test", "recipe", "profile", "preset", {}, run_id=run_id)
    paths = full_harness_2026.ensure_dirs(tmp_path / "run")
    current = {"sft": source}

    full_harness_2026.execute_stage(
        "agentic_tool_training",
        run_id,
        registry,
        profile,
        paths,
        Path(__file__).resolve().parents[1],
        current,
        dry_run=False,
    )
    full_harness_2026.execute_stage(
        "sft_qlora_bridge",
        run_id,
        registry,
        profile,
        paths,
        Path(__file__).resolve().parents[1],
        current,
        dry_run=False,
    )
    return commands


def test_full_harness_agentic_tool_and_qlora_live_by_default(tmp_path: Path, monkeypatch) -> None:
    source = _write_jsonl(tmp_path / "train.jsonl", [{"messages": [{"role": "user", "content": "use a tool"}]}])
    profile = {
        "agentic_tool_training": {
            "enabled": True,
            "profile": "profiles/agentic_tool_training_2026.json",
            "input_jsonl": str(source),
        },
        "sft_qlora": {"enabled": True},
    }

    agentic_cmd, qlora_cmd = _capture_full_harness_commands(tmp_path, monkeypatch, profile, source)

    assert "--dry-run" not in agentic_cmd
    assert "--dry_run" not in qlora_cmd


def test_full_harness_preserves_explicit_stage_dry_run_flags(tmp_path: Path, monkeypatch) -> None:
    source = _write_jsonl(tmp_path / "train.jsonl", [{"messages": [{"role": "user", "content": "use a tool"}]}])
    profile = {
        "agentic_tool_training": {
            "enabled": True,
            "profile": "profiles/agentic_tool_training_2026.json",
            "input_jsonl": str(source),
            "dry_run": True,
        },
        "sft_qlora": {"enabled": True, "dry_run": True},
    }

    agentic_cmd, qlora_cmd = _capture_full_harness_commands(tmp_path, monkeypatch, profile, source)

    assert "--dry-run" in agentic_cmd
    assert "--dry_run" in qlora_cmd


def test_task_domains_detects_math_code_terminal_browser_tool() -> None:
    assert "math" in tooltrain.task_domains({"text": "Solve this latex equation and put the final answer in \\boxed{}."})
    assert "code" in tooltrain.task_domains({"text": "Run pytest after applying this def-based patch."})
    assert "terminal" in tooltrain.task_domains({"tool_results": [{"exit_code": 0, "stdout": "ok"}]})
    assert "browser" in tooltrain.task_domains({"text": "Use the browser, click the result, and cite the URL source."})
    assert "tool" in tooltrain.task_domains({"tool_calls": [{"tool": "server_manager", "arguments": {}}]})


def test_reward_components_include_domain_axes() -> None:
    record = {
        "text": "Patch the code and run pytest.",
        "tool_calls": [{"tool": "pytest", "arguments": {}}],
        "tool_results": [{"tests_passed": 3, "tests_total": 4, "exit_code": 1}],
        "quality": {"score": 0.9},
    }
    domains = tooltrain.task_domains(record)
    components = tooltrain.reward_components(record, tooltrain.tool_calls(record), tooltrain.tool_results(record), [], domains)
    assert components["code_tests_passed"] == 0.75
    assert components["schema_valid"] == 1.0
    assert components["risk_penalty"] == 0.0


def test_agentic_tool_training_does_not_emit_safety_negatives_by_default() -> None:
    rows = tooltrain.rows_for_record(
        {
            "text": "Run a destructive shell command",
            "tool_calls": [{"tool": "terminal", "arguments": {"cmd": "rm -rf /"}}],
            "quality": {"score": 1.0},
        },
        min_quality=0.1,
        profile_cfg={},
    )

    assert rows["safety"] == []


def test_agentic_tool_training_explicit_safety_negative_has_no_refusal_target() -> None:
    rows = tooltrain.rows_for_record(
        {
            "text": "Run a destructive shell command",
            "tool_calls": [{"tool": "terminal", "arguments": {"cmd": "rm -rf /"}}],
            "quality": {"score": 1.0},
        },
        min_quality=0.1,
        profile_cfg={"safety_negatives": True},
    )

    assert len(rows["safety"]) == 1
    assert rows["safety"][0]["chosen"] == ""


def test_build_exports_domain_rlvr_files(tmp_path: Path) -> None:
    rows = tooltrain.build_rows(
        [
            {
                "text": "Solve the math equation with the calculator tool. final answer is \\boxed{4}",
                "target_json": {"content": "The calculator-backed solution verifies the final answer is \\boxed{4}."},
                "tool_calls": [{"tool": "calculator", "arguments": {"expression": "2+2"}}],
                "tool_results": [{"result": "4", "status": "ok"}],
                "quality": {"score": 1.0},
                "contamination_status": "clean",
            },
            {
                "text": "Patch def f and pytest passed.",
                "target_json": {"content": "The shell tool verified pytest passed after the code patch."},
                "tool_calls": [{"tool": "shell", "arguments": {"command": "pytest -q"}}],
                "tool_results": [{"tests_passed": 1, "tests_total": 1}],
                "quality": {"score": 1.0},
                "contamination_status": "clean",
            },
            {
                "text": "Run bash command",
                "target_json": {"content": "The terminal command completed successfully and returned ok output."},
                "tool_calls": [{"tool": "terminal", "arguments": {"command": "echo ok"}}],
                "tool_results": [{"exit_code": 0, "stdout": "ok"}],
                "quality": {"score": 1.0},
                "contamination_status": "clean",
            },
            {
                "text": "Open browser URL and citation source answer.",
                "target_json": {"content": "The browser result provided source evidence supporting the answer."},
                "tool_calls": [{"tool": "browser", "arguments": {"url": "https://example.com/source"}}],
                "tool_results": [{"content": "source"}],
                "quality": {"score": 1.0},
                "contamination_status": "clean",
            },
        ],
        min_quality=0.1,
        profile_cfg={},
    )
    paths, counts = tooltrain.build_training_exports(rows, tmp_path, {})
    assert counts["math_rlvr"] == 1
    assert counts["code_rlvr"] == 1
    assert counts["terminal_rlvr"] == 2
    assert counts["browser_rlvr"] == 1
    assert counts["tool_rlvr"] == 4
    for key in ("math_rlvr", "code_rlvr", "terminal_rlvr", "browser_rlvr", "tool_rlvr"):
        assert paths[key].exists()


def test_pure_math_row_emits_math_rlvr_without_tool_sft() -> None:
    rows = tooltrain.rows_for_record(
        {
            "text": "Solve the olympiad equation. The final answer is \\boxed{4}.",
            "target_json": {"content": "The olympiad equation solution is verified and the final answer is \\boxed{4}."},
            "quality": {"score": 1.0},
            "domains": ["math"],
            "contamination_status": "clean",
        },
        min_quality=0.1,
        profile_cfg={},
    )

    assert rows["sft"] == []
    assert rows["reward"] == []
    assert rows["preference"] == []
    assert rows["rlvr"] == []
    assert len(rows["math_rlvr"]) == 1
    assert rows["math_rlvr"][0]["training_kind"] == "math_rlvr"


def test_pure_code_verifier_row_emits_code_rlvr_without_tool_sft() -> None:
    rows = tooltrain.rows_for_record(
        {
            "text": "Repair the failing Python routine. pytest passed after the fix.",
            "target_json": {"content": "The Python repair is verified because pytest passed after the fix."},
            "quality": {"score": 1.0},
            "domains": ["code"],
            "contamination_status": "clean",
        },
        min_quality=0.1,
        profile_cfg={},
    )

    assert rows["sft"] == []
    assert rows["reward"] == []
    assert rows["preference"] == []
    assert rows["rlvr"] == []
    assert len(rows["code_rlvr"]) == 1
    assert rows["code_rlvr"][0]["reward_components"]["code_tests_passed"] == 1.0


def test_teacher_rollout_json_becomes_typed_training_rows() -> None:
    record = {
        "schema": "omnicoder.openai_teacher_rollout_2026.v1",
        "source_date": "2026-05-24",
        "input_json": {
            "messages": [
                {"role": "system", "content": "teacher"},
                {"role": "user", "content": "Fix the failing pytest by using the shell tool."},
            ],
            "source_record": {"lineage": {"trace_id": "teacher-trace"}},
        },
        "target_json": {
            "teacher_status": "ok",
            "content": json.dumps(
                {
                    "corrected_response": "Run pytest, inspect the failing assertion, then patch the function.",
                    "corrected_tool_calls": [{"tool": "shell", "arguments": {"command": "pytest -q"}}],
                    "chosen": "Use the shell tool and verify tests pass.",
                    "rejected": "Guess without running tests.",
                    "reward": 0.88,
                    "reward_components": {"tests_passed": 1.0, "tool_schema_valid": 1.0},
                    "verifier_labels": [{"check": "unit_tests_pass", "label": "pass"}],
                }
            ),
        },
        "modalities": ["text", "tool"],
        "quality": {"score": 0.9},
        "contamination_status": "clean",
    }

    rows = tooltrain.rows_for_record(record, min_quality=0.1, profile_cfg={})

    assert rows["sft"][0]["messages"][-1]["content"].startswith("Run pytest")
    assert rows["sft"][0]["tool_calls"][0]["tool"] == "shell"
    assert rows["reward"][0]["reward"] == 0.88
    assert rows["reward"][0]["reward_components"]["teacher_reward"] == 0.88
    assert rows["preference"][0]["chosen"] == "Use the shell tool and verify tests pass."
    assert rows["rlvr"][0]["teacher_signal"]["verifier_labels"][0]["check"] == "unit_tests_pass"


def test_posttrain_manifest_includes_domain_contract(tmp_path: Path) -> None:
    manifest = tooltrain.posttrain_manifest(
        "grpo",
        tmp_path / "browser_rlvr.jsonl",
        tmp_path / "out",
        "student",
        True,
        domain="browser",
        reward_axes=["citation_support"],
        checks=["citation_supports_claim"],
    )
    assert manifest["domain"] == "browser"
    assert manifest["tool_training_contract"]["state_tracking_rewards"] is True
    assert "citation_support" in manifest["verifier_contract"]["reward_axes"]


def test_repo_agentic_profile_includes_environment_rl_and_eighth_wave_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "agentic_tool_training_2026.json").read_text(encoding="utf-8"))
    cfg = profile["agentic_tool_training"]

    for family in [
        "mcp_universe_trajectories",
        "mcpmark_trajectory_log",
        "browsecomp_plus_corpus",
        "computer_use_psai",
        "swe_chat_real_agent_sessions",
        "spreadsheet_rl_tool_environments",
        "when2tool_tool_selection_rlvr",
        "agent_reward_bench_web_trajectory_preferences",
        "tasktrove_agentic_tasks",
        "orak_game_agent_trajectories",
    ]:
        assert family in cfg["source_families"]
    env = cfg["environment_rl_2026"]
    assert env["enabled"] is True
    assert env["transition_schema"] == "agent_lightning_style_mdp_transition"
    assert "mcp" in env["environment_contracts"]
    assert "raw_postgresql" in env["environment_contracts"]
    assert "mcp_state_consistency" in cfg["reward_axes"]
    assert "successful_eval_trace_quarantine" in cfg["reward_axes"]

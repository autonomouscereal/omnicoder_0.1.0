from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnicoder.eval import benchmark_suite_2026 as runner
from omnicoder.eval import official_harness_adapters_2026 as official_adapters


def _write_profile(path: Path, profile: dict[str, Any]) -> Path:
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def _minimal_profile() -> dict[str, Any]:
    return {
        "version": "2026-05-23.test",
        "benchmarks": [
            {
                "benchmark_id": "local_alpha",
                "adapter_kind": "local_fixture",
                "axis": "agent_tool",
                "source": "fixture",
                "splits": {"smoke": "alpha_probe"},
                "holdout_policy": ["hidden_fixture"],
            },
            {
                "benchmark_id": "local_beta",
                "adapter_kind": "local_fixture",
                "axis": "agent_tool",
                "source": "fixture",
                "splits": {"smoke": "beta_probe"},
                "holdout_policy": [],
            },
        ],
        "release_gates": {
            "must_pass": ["no_network", "no_database"],
            "local_release": ["local_alpha", "local_beta"],
        },
    }


def test_mcq_scoring_accepts_index_letter_or_choice_text() -> None:
    task = {
        "benchmark_id": "reasoning_hellaswag_full_2026",
        "choices": ["opens the door.", "pulls out a tray.", "walks away."],
        "answer": "1",
    }

    assert runner.score_mcq_task(task, "B")["score"] == 1.0
    assert runner.score_mcq_task(task, "pulls out a tray.")["score"] == 1.0
    assert runner.score_mcq_task({**task, "answer": "2"}, "C")["score"] == 1.0
    assert runner.score_mcq_task(task, "A")["score"] == 0.0


def _json_from_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_profile_valid(profile: dict[str, Any]) -> None:
    adapters = profile.get("benchmarks") or profile.get("adapters")
    assert isinstance(adapters, list) and adapters
    adapter_ids = set()
    for adapter in adapters:
        assert isinstance(adapter, dict)
        adapter_key = adapter.get("benchmark_id") or adapter.get("id")
        kind = adapter.get("adapter_kind") or adapter.get("kind")
        smoke = adapter.get("smoke") or adapter.get("splits", {}).get("smoke")
        assert isinstance(adapter_key, str) and adapter_key
        assert isinstance(kind, str) and kind
        assert isinstance(smoke, str) and smoke
        assert adapter_key not in adapter_ids
        adapter_ids.add(adapter_key)

    for gate_name, required in profile.get("release_gates", {}).items():
        if gate_name in {"must_pass", "global_must_pass"}:
            continue
        missing = sorted(set(required) - adapter_ids)
        assert not missing, f"{gate_name} references missing adapter(s): {missing}"


def test_registry_profile_validates_release_gate_adapter_references() -> None:
    profile = runner.load_profile(runner.DEFAULT_PROFILE)

    _assert_profile_valid(profile)


def test_release_gate_registry_includes_fresh_rlvr_and_media_preference_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "benchmark_registry_2026.json").read_text(encoding="utf-8"))

    _assert_profile_valid(profile)
    adapters = {adapter["id"]: adapter for adapter in profile["adapters"]}
    for adapter_id in [
        "rlvr_linearity_math_2026",
        "nous_rlvr_coding_2026",
        "editreward_bench_2026",
        "iesbench_image_edit_safety_2026",
        "svi_benchmark_2026",
        "text_to_audio_pref_bench_2026",
    ]:
        assert adapter_id in adapters
    assert "rlvr_linearity_math_2026" in profile["release_gates"]["coding"]
    assert "text_to_audio_pref_bench_2026" in profile["release_gates"]["omnimodal_reasoning"]
    assert "editreward_bench_2026" in profile["release_gates"]["generation"]
    assert adapters["editreward_bench_2026"]["kind"] == "image_edit_reward_model_eval"


def test_suite_profile_includes_fresh_rlvr_and_media_preference_adapters() -> None:
    profile = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in profile["benchmarks"]}

    for adapter_id in [
        "reasoning_rlvr_linearity_math_2026",
        "coding_nous_rlvr_coding_2026",
        "generation_editreward_bench_2026",
        "safety_iesbench_image_edit_2026",
        "generation_svi_benchmark_2026",
        "generation_text_to_audio_pref_2026",
    ]:
        assert adapter_id in adapters
    assert "reasoning_rlvr_linearity_math_2026" in profile["release_gates"]["reasoning_release"]
    assert "coding_nous_rlvr_coding_2026" in profile["release_gates"]["coding_release"]
    assert "generation_text_to_audio_pref_2026" in profile["release_gates"]["generation_release"]
    assert adapters["safety_iesbench_image_edit_2026"]["axis"] == "safety_tool_security"


def test_profiles_include_seventh_wave_agentic_omni_release_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "profiles" / "benchmark_registry_2026.json").read_text(encoding="utf-8"))
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    registry_adapters = {adapter["id"]: adapter for adapter in registry["adapters"]}
    suite_adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    for adapter_id in [
        "arc_agi3_interactive_2026",
        "terminal_bench_2_1_2026",
        "browsergym_webarena_verified_2026",
        "osworld_desktop_2026",
        "livebench_math_2026",
        "mmmu_pro_split_2026",
        "video_mme_v2_grouped_2026",
        "audiobench_mmau_2026",
        "vbench2_intrinsic_faithfulness_2026",
        "music_arena_2026",
        "browsecomp_long_context_2026",
        "theagentcompany_enterprise_2026",
    ]:
        assert adapter_id in registry_adapters
    for adapter_id in [
        "reasoning_arc_agi3_interactive_2026",
        "agent_terminal_bench_2_1_2026",
        "agent_browsergym_webarena_verified_2026",
        "agent_osworld_desktop_2026",
        "reasoning_livebench_math_2026",
        "multimodal_mmmu_pro_standard_2026",
        "multimodal_video_mme_v2_grouped_2026",
        "multimodal_audiobench_mmau_2026",
        "generation_vbench2_intrinsic_faithfulness_2026",
        "generation_music_arena_2026",
        "agent_browsecomp_long_context_2026",
        "agent_theagentcompany_enterprise_2026",
    ]:
        assert adapter_id in suite_adapters
    assert "agent_terminal_bench_2_1_2026" in suite["release_gates"]["agent_tool_release"]
    assert "agent_browsecomp_long_context_2026" in suite["release_gates"]["agent_tool_release"]
    assert "agent_theagentcompany_enterprise_2026" in suite["release_gates"]["agent_tool_release"]
    assert "multimodal_video_mme_v2_grouped_2026" in suite["release_gates"]["multimodal_understanding_release"]
    assert "generation_music_arena_2026" in suite["release_gates"]["generation_release"]


def test_profiles_include_ninth_wave_agentic_multimodal_generation_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    for adapter_id in [
        "agent_agencybench_2026",
        "agent_locobench_agent_2026",
        "coding_swe_mera_2026",
        "coding_ale_bench_2026",
        "multimodal_mathnet_2026",
        "multimodal_video_morse500_2026",
        "multimodal_sonic_o1_2026",
        "multimodal_mme_unify_2026",
        "multimodal_longspeech_2026",
        "generation_emergent_tts_eval_2026",
        "generation_long_tts_eval_2026",
        "generation_tta_bench_2026",
        "generation_nv_bench_2026",
    ]:
        assert adapter_id in adapters

    assert "agent_agencybench_2026" in suite["release_gates"]["agent_tool_release"]
    assert "agent_locobench_agent_2026" in suite["release_gates"]["agent_tool_release"]
    assert "coding_swe_mera_2026" in suite["release_gates"]["coding_release"]
    assert "coding_ale_bench_2026" in suite["release_gates"]["coding_release"]
    assert "multimodal_mme_unify_2026" in suite["release_gates"]["multimodal_understanding_release"]
    assert "multimodal_sonic_o1_2026" in suite["release_gates"]["multimodal_understanding_release"]
    assert "generation_emergent_tts_eval_2026" in suite["release_gates"]["generation_release"]
    assert "generation_long_tts_eval_2026" in suite["release_gates"]["generation_release"]


def test_suite_profile_includes_tenth_wave_curated_benchmark_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_mcptoolbenchpp_2026": "agent_tool_release",
        "agent_webbench_2026": "agent_tool_release",
        "reasoning_maime2025_2026": "reasoning_release",
        "long_context_sagascale_2026": "long_context_release",
        "long_context_academiceval_2026": "long_context_release",
        "multimodal_mpbench_2026": "multimodal_understanding_release",
        "multimodal_cmi_bench_music_2026": "multimodal_understanding_release",
        "multimodal_muse_music_2026": "multimodal_understanding_release",
        "multimodal_rtv_bench_2026": "multimodal_understanding_release",
        "multimodal_maverix_av_reasoning_2026": "multimodal_understanding_release",
        "multimodal_river_video_interaction_2026": "multimodal_understanding_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]

    assert adapters["agent_mcptoolbenchpp_2026"]["source"] == "https://huggingface.co/MCPToolBench"
    assert adapters["long_context_sagascale_2026"]["context_windows"][-1] == 1048576
    assert adapters["multimodal_rtv_bench_2026"]["adapter_kind"] == "real_time_video_multitimestamp_eval"
    assert adapters["multimodal_river_video_interaction_2026"]["source"] == "https://github.com/OpenGVLab/RIVER"
    assert adapters["multimodal_cmi_bench_music_2026"]["modalities"] == ["audio", "music", "text"]


def test_suite_profile_includes_eleventh_wave_agentic_omni_benchmark_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_livemcpbench_2026": "agent_tool_release",
        "agent_sra_bench_2026": "agent_tool_release",
        "agent_skillret_2026": "agent_tool_release",
        "long_context_memoryagentbench_2026": "long_context_release",
        "multimodal_omnigaia_2026": "multimodal_understanding_release",
        "multimodal_omnirag_agent_2026": "multimodal_understanding_release",
        "multimodal_vstat_visual_state_tracking_2026": "multimodal_understanding_release",
        "generation_tricky_tts_2026": "generation_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]

    assert adapters["agent_livemcpbench_2026"]["source"] == "https://huggingface.co/datasets/ICIP/LiveMCPBench"
    assert adapters["agent_skillret_2026"]["adapter_kind"] == "agent_skill_retrieval_eval"
    assert adapters["long_context_memoryagentbench_2026"]["context_windows"][-1] == 1048576
    assert "tool" in adapters["multimodal_omnigaia_2026"]["modalities"]
    assert adapters["generation_tricky_tts_2026"]["axis"] == "audio_generation"


def test_suite_profile_includes_twelfth_wave_agent_memory_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_state_bench_2026": "agent_tool_release",
        "long_context_ama_bench_2026": "long_context_release",
        "multimodal_smmbench_2026": "multimodal_understanding_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]

    assert adapters["agent_state_bench_2026"]["source"] == "https://github.com/microsoft/STATE-Bench"
    assert adapters["long_context_ama_bench_2026"]["context_windows"][-1] == 1048576
    assert "agent_memory" in adapters["multimodal_smmbench_2026"]["modalities"]


def test_suite_profile_includes_thirteenth_wave_agentic_math_multimodal_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_tobench_mm_toolbench_2026": "agent_tool_release",
        "multimodal_agentic_mme_2026": "multimodal_understanding_release",
        "coding_abc_bench_2026": "coding_release",
        "long_context_longbench_pro_2026": "long_context_release",
        "multimodal_megabench_2026": "multimodal_understanding_release",
        "multimodal_stepeval_audio_360_2026": "multimodal_understanding_release",
        "reasoning_indimathbench_2026": "reasoning_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]

    assert adapters["multimodal_agentic_mme_2026"]["source"] == "https://huggingface.co/datasets/Agentic-MME/Agentic-MME"
    assert adapters["coding_abc_bench_2026"]["adapter_kind"] == "backend_coding_agent_eval"
    assert adapters["long_context_longbench_pro_2026"]["context_windows"][-2] == 262144
    assert "audio" in adapters["multimodal_stepeval_audio_360_2026"]["modalities"]
    assert adapters["reasoning_indimathbench_2026"]["task_format"] == "jsonl_lean4_formal_math_task"


def test_suite_profile_includes_fourteenth_wave_agentic_gui_video_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_mcpverse_2026": "agent_tool_release",
        "agent_ui_vision_2026": "agent_tool_release",
        "multimodal_vimul_bench_2026": "multimodal_understanding_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]

    assert adapters["agent_mcpverse_2026"]["source"] == "https://github.com/hailsham/mcpverse"
    assert "desktop" in adapters["agent_ui_vision_2026"]["modalities"]
    assert "video" in adapters["multimodal_vimul_bench_2026"]["modalities"]


def test_suite_profile_includes_fifteenth_wave_agentic_coding_audio_document_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_world_model_rl_2026": "agent_tool_release",
        "agent_tool_genesis_2026": "agent_tool_release",
        "agent_agentif_2025": "agent_tool_release",
        "agent_webgym_tasks_2026": "agent_tool_release",
        "agent_omniagentbench_2026": "agent_tool_release",
        "safety_mcp_security_bench_2026": "safety_security_release",
        "coding_beyondswe_2026": "coding_release",
        "coding_contextbench_2026": "coding_release",
        "coding_ccbench_2026": "coding_release",
        "coding_computeeval_cuda_2026": "coding_release",
        "long_context_officeqa_2026": "long_context_release",
        "multimodal_parsebench_2026": "multimodal_understanding_release",
        "multimodal_audiomcq_strongac_2026": "multimodal_understanding_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]

    assert adapters["agent_webgym_tasks_2026"]["adapter_kind"] == "browsergym_visual_web_agent_eval"
    assert adapters["coding_computeeval_cuda_2026"]["source"] == "https://huggingface.co/datasets/nvidia/compute-eval"
    assert "document" in adapters["multimodal_parsebench_2026"]["modalities"]
    assert "audio" in adapters["multimodal_audiomcq_strongac_2026"]["modalities"]
    assert adapters["long_context_officeqa_2026"]["context_windows"][-1] == 1048576


def test_suite_profile_includes_sixteenth_wave_video_gui_av_and_edit_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "agent_videowebarena_2026": "agent_tool_release",
        "agent_osuniverse_gui_2026": "agent_tool_release",
        "multimodal_avatar_av_localization_2026": "multimodal_understanding_release",
        "coding_swe_bench_multimodal_2026": "coding_release",
        "long_context_loft_2026": "long_context_release",
        "reasoning_frontiermath_2026": "reasoning_release",
        "generation_gie_bench_2026": "generation_release",
        "generation_editinspector_2026": "generation_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]
        assert adapter_id in suite["reportable_snapshots"]
        assert adapter_id in suite["reportable_task_roots"]

    assert "video" in adapters["agent_videowebarena_2026"]["modalities"]
    assert "desktop_gui" in adapters["agent_osuniverse_gui_2026"]["modalities"]
    assert "audio" in adapters["multimodal_avatar_av_localization_2026"]["modalities"]
    assert adapters["long_context_loft_2026"]["source"] == "https://github.com/google-deepmind/loft"


def test_reportable_core25_keeps_math_and_hash_license_requirements() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    core25 = suite["reportable_core_25"]
    requirements = set(suite["reportability_policy"]["reportable_score_requires"])

    assert len(core25) == 25
    assert "reasoning_matharena_2026" in core25
    assert "snapshot_sha256_or_task_file_sha256" in requirements
    assert "license_ref" in requirements
    assert "official_scorer_ref" in requirements


def test_suite_profile_includes_seventeenth_wave_feature_and_generation_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected = {
        "coding_fea_bench_2026": "coding_release",
        "generation_worldgenbench_2026": "generation_release",
        "generation_omnigenbench_2026": "generation_release",
    }
    for adapter_id, gate in expected.items():
        assert adapter_id in adapters
        assert adapter_id in suite["release_gates"][gate]
        assert adapter_id in suite["reportable_snapshots"]
        assert adapter_id in suite["reportable_task_roots"]

    assert "git" in adapters["coding_fea_bench_2026"]["modalities"]
    assert "image" in adapters["generation_worldgenbench_2026"]["modalities"]
    assert adapters["generation_omnigenbench_2026"]["source"] == "https://arxiv.org/abs/2505.18775"


def test_profile_validation_fails_when_release_gate_references_missing_adapter() -> None:
    profile = _minimal_profile()
    profile["release_gates"]["local_release"].append("missing_adapter")

    with pytest.raises(AssertionError, match="missing_adapter"):
        _assert_profile_valid(profile)


def test_list_outputs_profile_adapters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", _minimal_profile())

    assert runner.main(["--profile", str(profile_path), "list"]) == 0

    payload = _json_from_stdout(capsys)
    assert payload["profile_version"] == "2026-05-23.test"
    assert [row["adapter_id"] for row in payload["adapters"]] == ["local_alpha", "local_beta"]
    assert payload["adapters"][0] | {
        "adapter_id": "local_alpha",
        "benchmark_id": "local_alpha",
        "axis": "agent_tool",
        "kind": "local_fixture",
        "smoke": "alpha_probe",
        "has_command": False,
    } == payload["adapters"][0]


def test_plan_writes_manifest_jsonl_with_expected_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", _minimal_profile())
    out_dir = tmp_path / "out"

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(out_dir),
                "plan",
                "--adapter",
                "local_alpha",
                "--mode",
                "dry-run",
                "--run-id",
                "plan-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    manifest_path = Path(payload["manifest"])
    rows = _jsonl_rows(manifest_path)
    assert payload == {
        "status": "ok",
        "manifest": str(out_dir / "manifests.jsonl"),
        "planned": 1,
        "run_id": "plan-fixture",
        "cycle": "smoke",
    }
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "benchmark_manifest"
    assert row["schema_version"] == "2026-05-23"
    assert row["run_id"] == "plan-fixture"
    assert row["adapter_id"] == "local_alpha"
    assert row["adapter_kind"] == "local_fixture"
    assert row["benchmark_id"] == "local_alpha"
    assert row["axis"] == "agent_tool"
    assert row["mode"] == "dry-run"
    assert row["smoke"] == "alpha_probe"
    assert row["no_heavy_downloads"] is True
    assert row["command"] is None
    assert isinstance(row["manifest_hash"], str) and len(row["manifest_hash"]) == 64


def test_run_smoke_writes_result_jsonl_without_network_database_or_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", _minimal_profile())
    out_dir = tmp_path / "out"

    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access is not allowed in runner smoke tests")

    def fail_subprocess(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess commands are not allowed for command-free smoke fixtures")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(subprocess, "run", fail_subprocess)

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(out_dir),
                "run-smoke",
                "--adapter",
                "local_alpha",
                "--run-id",
                "smoke-fixture",
                "--timeout-seconds",
                "1",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    results_path = Path(payload["results"])
    rows = _jsonl_rows(results_path)
    assert payload["status"] == "ok"
    assert payload["ran"] == 1
    assert payload["failed"] == 0
    assert len(rows) == 1
    result = rows[0]
    assert result["type"] == "benchmark_result"
    assert result["schema_version"] == "2026-05-23"
    assert result["run_id"] == "smoke-fixture"
    assert result["adapter_id"] == "local_alpha"
    assert result["adapter_kind"] == "local_fixture"
    assert result["benchmark_id"] == "local_alpha"
    assert result["axis"] == "agent_tool"
    assert result["mode"] == "smoke"
    assert result["status"] == "passed"
    assert result["score"] is None
    assert result["metrics"]["downloaded_bytes"] == 0
    assert result["metrics"]["heavy_downloads_allowed"] is False
    assert result["metrics"]["timeout_seconds"] == 1
    assert result["metrics"]["contract_only"] is True
    assert result["metrics"]["reportable_score"] is False
    assert result["score_json"]["reportable_score"] is False
    assert result["command_result"] is None
    assert isinstance(result["manifest_hash"], str) and len(result["manifest_hash"]) == 64
    assert isinstance(result["result_hash"], str) and len(result["result_hash"]) == 64


def test_smoke_command_result_is_diagnostic_only_even_when_command_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _minimal_profile()
    profile["benchmarks"][0]["command"] = [sys.executable, "-c", "print('ok')"]
    profile_path = _write_profile(tmp_path / "profile.json", profile)

    monkeypatch.setattr(
        runner,
        "run_command",
        lambda *_args, **_kwargs: {
            "ok": True,
            "exit_code": 0,
            "elapsed_seconds": 0.001,
            "stdout": "ok\n",
            "stderr": "",
        },
    )

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-smoke",
                "--adapter",
                "local_alpha",
                "--mode",
                "command",
            ]
        )
        == 0
    )

    result = _jsonl_rows(tmp_path / "out" / "results.jsonl")[0]
    assert result["status"] == "passed"
    assert result["score"] is None
    assert result["score_json"]["canonical_score"] is None
    assert result["score_json"]["diagnostic_score"] == 1.0
    assert result["score_json"]["score_claim_scope"] == "diagnostic_contract"
    assert result["diagnostic_only"] is True
    assert result["official_score"] is False
    assert result["reportability_scope"] == "diagnostic_only"
    assert result["score_json"]["reportable_score"] is False
    assert runner.row_reportable_score(result) is False


def test_official_harness_manifest_marks_required_adapters_and_audio_placeholders() -> None:
    manifest = official_adapters.build_manifest()
    adapters = {row["adapter_id"]: row for row in manifest["adapters"]}

    for adapter_id in [
        "lm_eval",
        "helm",
        "swe_bench",
        "terminal_bench",
        "bfcl",
        "mmmu",
        "ruler",
        "longbench",
        "nolima",
        "vbench",
    ]:
        assert adapters[adapter_id]["claim_scope"] == "official_harness"
        assert adapters[adapter_id]["requires_official_snapshot"] is True
        assert adapters[adapter_id]["heavy_run"] is True
        assert adapters[adapter_id]["official_score"] is False

    for adapter_id in ["fad", "clap", "mos"]:
        assert adapters[adapter_id]["claim_scope"] == "official_metric_unavailable"
        assert adapters[adapter_id]["placeholder_until_installed"] is True
        assert adapters[adapter_id]["official_score"] is False
        assert adapters[adapter_id]["availability"]["status"] in {
            "unavailable_until_official_package_installed",
            "available_not_executed",
        }


def test_run_smoke_returns_failure_for_missing_configured_adapter(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", _minimal_profile())

    with pytest.raises(SystemExit, match="unknown benchmark id\\(s\\): missing_adapter"):
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-smoke",
                "--adapter",
                "missing_adapter",
            ]
        )


def test_summarize_reports_status_counts_and_latest_adapter_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "type": "benchmark_result",
            "schema_version": "2026-05-23",
            "run_id": "old",
            "adapter_id": "local_alpha",
            "adapter_kind": "local_fixture",
            "mode": "smoke",
            "status": "failed",
            "score": None,
            "score_json": {"reportable_score": False},
        },
        {
            "type": "benchmark_result",
            "schema_version": "2026-05-23",
            "run_id": "new",
            "adapter_id": "local_alpha",
            "adapter_kind": "local_fixture",
            "mode": "smoke",
            "status": "passed",
            "score": 1.0,
            "score_json": {"reportable_score": True},
        },
        {
            "type": "benchmark_result",
            "schema_version": "2026-05-23",
            "run_id": "reportable",
            "adapter_id": "reasoning_arc_agi3_2026",
            "benchmark_id": "reasoning_arc_agi3_2026",
            "adapter_kind": "interactive_reasoning_env",
            "mode": "reportable",
            "phase": "reportable_scoring",
            "status": "passed",
            "score": 1.0,
            "score_json": {"reportable_score": True, "contract_only": False},
        },
    ]
    results_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert runner.main(["--out-dir", str(tmp_path), "summarize", "--results", str(results_path)]) == 0

    summary = _json_from_stdout(capsys)
    assert summary["type"] == "benchmark_summary"
    assert summary["schema_version"] == "2026-05-23"
    assert summary["total_results"] == 3
    assert summary["by_status"] == {"failed": 1, "passed": 2}
    assert summary["by_adapter"]["local_alpha"] == {
            "latest_status": "passed",
            "latest_mode": "smoke",
            "latest_run_id": "new",
            "latest_score": None,
            "latest_reportable_score": None,
            "latest_internal_score": None,
            "score_claim_scope": None,
            "reportable_score": False,
        }
    assert summary["reportable_results"] == 1
    assert summary["contract_only_results"] == 0


def _reportable_profile(tmp_path: Path) -> tuple[Path, Path]:
    tasks = tmp_path / "reportable_tasks.jsonl"
    rows = [
        {
            "benchmark_id": "reasoning_arc_agi3_2026",
            "task_id": "arc-env-1",
            "dataset_revision": "arc-agi3-authorized-2026-05",
            "snapshot_id": "arc-agi3-authorized-2026-05-smoke",
            "snapshot_authorization": "authorized_private",
            "snapshot_sha256": "sha256:arc-agi3-smoke",
            "authorization_ref": "internal-authorized-eval-ledger",
            "license_ref": "authorized-eval-ledger:arc-agi3",
            "official_scorer_ref": "arc-agi3-official-scorer-2026",
            "source": "https://arcprize.org/arc-agi/3",
            "reportable": True,
            "success": True,
            "actions": 4,
            "human_actions": 2,
            "model_actions": ["inspect_grid", "submit_solution"],
        },
        {
            "benchmark_id": "coding_swe_bench_live_2026",
            "task_id": "swe-live-1",
            "dataset_revision": "swe-live-authorized-2026-05",
            "snapshot_id": "swe-live-authorized-2026-05-smoke",
            "snapshot_authorization": "authorized_private",
            "snapshot_sha256": "sha256:swe-live-smoke",
            "authorization_ref": "internal-authorized-eval-ledger",
            "license_ref": "authorized-eval-ledger:swe-live",
            "official_scorer_ref": "swe-bench-live-official-eval-2026",
            "source": "https://arxiv.org/abs/2505.23419",
            "reportable": True,
            "patch": "diff --git a/a.py b/a.py",
            "model_patch": "diff --git a/a.py b/a.py",
            "patch_applies": True,
            "tests_pass": True,
        },
        {
            "benchmark_id": "multimodal_mmmu_pro_2026",
            "task_id": "mmmu-pro-1",
            "dataset_revision": "mmmu-pro-authorized-2026-05",
            "snapshot_id": "mmmu-pro-authorized-2026-05-smoke",
            "snapshot_authorization": "authorized_private",
            "snapshot_sha256": "sha256:mmmu-pro-smoke",
            "authorization_ref": "internal-authorized-eval-ledger",
            "license_ref": "authorized-eval-ledger:mmmu-pro",
            "official_scorer_ref": "mmmu-pro-official-eval-2026",
            "source": "https://mmmu-benchmark.github.io/",
            "reportable": True,
            "question": "Which option matches the diagram?",
            "choices": ["A", "B", "C", "D"],
            "answer": "C",
            "prediction": "C",
        },
    ]
    tasks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    profile = {
        "version": "2026-05-23.reportable-test",
        "benchmarks": [
            {
                "benchmark_id": "reasoning_arc_agi3_2026",
                "adapter_kind": "interactive_env",
                "axis": "reasoning",
                "source": "https://arcprize.org/arc-agi/3",
                "task_format": "jsonl_interactive_env_result",
                "splits": {"smoke": "authorized fixture"},
                "metrics": ["relative_human_action_efficiency"],
                "holdout_policy": ["hide_private_envs"],
            },
            {
                "benchmark_id": "coding_swe_bench_live_2026",
                "adapter_kind": "fresh_git_container_patch",
                "axis": "coding",
                "source": "https://arxiv.org/abs/2505.23419",
                "task_format": "jsonl_patch_task",
                "splits": {"smoke": "authorized fixture"},
                "metrics": ["resolved_rate"],
                "holdout_policy": ["hide_hidden_tests"],
            },
            {
                "benchmark_id": "multimodal_mmmu_pro_2026",
                "adapter_kind": "multimodal_mcq",
                "axis": "multimodal_understanding",
                "source": "https://mmmu-benchmark.github.io/",
                "task_format": "jsonl_multimodal_mcq",
                "splits": {"smoke": "authorized fixture"},
                "metrics": ["accuracy"],
                "holdout_policy": ["hide_answers"],
            },
        ],
        "release_gates": {
            "reasoning_release": ["reasoning_arc_agi3_2026"],
            "coding_release": ["coding_swe_bench_live_2026"],
            "multimodal_understanding_release": ["multimodal_mmmu_pro_2026"],
        },
    }
    profile_path = _write_profile(tmp_path / "profile.json", profile)
    return profile_path, tasks


def test_run_reportable_scores_arc_swe_and_mmmu_with_real_oracles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    out_dir = tmp_path / "out"

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(out_dir),
                "run-reportable",
                "--tasks",
                str(tasks),
                "--run-id",
                "reportable-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    rows = _jsonl_rows(Path(payload["results"]))
    assert payload["status"] == "needs_data"
    assert payload["reportable"] == 0
    assert len(rows) == 3
    by_id = {row["benchmark_id"]: row for row in rows}
    assert by_id["reasoning_arc_agi3_2026"]["score"] is None
    assert by_id["reasoning_arc_agi3_2026"]["score_json"]["contract_score"] == 0.5
    assert by_id["coding_swe_bench_live_2026"]["score_json"]["contract_score"] == 1.0
    assert by_id["multimodal_mmmu_pro_2026"]["score_json"]["contract_score"] == 1.0
    for row in rows:
        assert row["mode"] == "reportable"
        assert row["status"] == "contract_only"
        assert row["score_json"]["reportable_score"] is False
        assert row["score_json"]["contract_only"] is True


def test_run_reportable_rejects_reportable_true_without_authorized_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    for key in ("snapshot_id", "snapshot_authorization", "snapshot_sha256", "authorization_ref"):
        rows[0].pop(key, None)
    tasks.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "reasoning_arc_agi3_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "missing-snapshot-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert result["score_json"]["reportable_score"] is False


def test_run_reportable_rejects_missing_license_and_scorer_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    rows[0].pop("license_ref")
    rows[0].pop("official_scorer_ref")
    tasks.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "reasoning_arc_agi3_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "missing-license-scorer-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert result["score_json"]["reportable_score"] is False


def test_run_reportable_accepts_authorized_snapshot_descriptor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    arc = rows[0]
    for key in ("snapshot_id", "snapshot_authorization", "snapshot_sha256", "authorization_ref"):
        arc.pop(key, None)
    tasks.write_text(json.dumps(arc) + "\n", encoding="utf-8")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["reportable_snapshots"] = {
        "reasoning_arc_agi3_2026": {
            "snapshot_id": "arc-agi3-authorized-descriptor",
            "snapshot_authorization": "authorized_private",
            "snapshot_sha256": "sha256:descriptor",
            "authorization_ref": "authorized-eval-ledger",
            "license_ref": "authorized-eval-ledger:arc-agi3",
            "official_scorer_ref": "arc-agi3-official-scorer-2026",
            "dataset_revision": "arc-agi3-authorized-2026-05",
            "source": "https://arcprize.org/arc-agi/3",
            "task_root": str(tasks),
        }
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "reasoning_arc_agi3_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "descriptor-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    task_score = result["metrics_json"]["task_scores"][0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "contract_only"
    assert result["score_json"]["reportable_score"] is False
    assert result["score_json"]["contract_only"] is True
    assert task_score["snapshot_id"] == "arc-agi3-authorized-descriptor"


def test_run_reportable_marks_missing_official_metadata_as_local_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    rows[0].pop("dataset_revision")
    tasks.write_text("\n".join(json.dumps(row) for row in rows[:1]) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "reasoning_arc_agi3_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "local-only-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert result["score_json"]["reportable_score"] is False
    assert result["score_json"]["contract_only"] is False


def test_run_reportable_requires_task_level_source_not_profile_backfill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    rows[0].pop("source")
    tasks.write_text("\n".join(json.dumps(row) for row in rows[:1]) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "reasoning_arc_agi3_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "missing-source-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    task_score = result["metrics_json"]["task_scores"][0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert task_score["reportable_metadata"] is False


def test_run_reportable_requires_model_output_not_only_gold_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    mmmu = rows[2]
    mmmu.pop("prediction")
    tasks.write_text(json.dumps(mmmu) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "multimodal_mmmu_pro_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "missing-output-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    task_score = result["metrics_json"]["task_scores"][0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert task_score["has_model_output"] is False
    assert task_score["reportable_metadata"] is False


def test_run_reportable_requires_prediction_not_oracle_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    arc = rows[0]
    arc.pop("model_actions", None)
    tasks.write_text(json.dumps(arc) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "reasoning_arc_agi3_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "oracle-only-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    task_score = result["metrics_json"]["task_scores"][0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert task_score["has_model_output"] is False


def test_run_reportable_requires_model_patch_not_oracle_patch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, tasks = _reportable_profile(tmp_path)
    rows = _jsonl_rows(tasks)
    swe = rows[1]
    swe.pop("model_patch", None)
    tasks.write_text(json.dumps(swe) + "\n", encoding="utf-8")

    assert (
        runner.main(
            [
                "--profile",
                str(profile_path),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                "coding_swe_bench_live_2026",
                "--tasks",
                str(tasks),
                "--run-id",
                "oracle-patch-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    task_score = result["metrics_json"]["task_scores"][0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "local_only"
    assert task_score["has_model_output"] is False


def test_suite_profile_includes_nineteenth_wave_factuality_doc_agent_gates() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    expected_gates = {
        "factuality_simpleqa_verified_2026": "factuality_release",
        "factuality_facts_grounding_2026": "factuality_release",
        "agent_mc_search_mmrag_2026": "agent_tool_release",
        "agent_metr_time_horizon_hcast_2026": "agent_tool_release",
        "agent_agent2_rl_bench_2026": "agent_tool_release",
        "agent_memgym_2026": "agent_tool_release",
        "multimodal_ocrbench_v2_2026": "multimodal_understanding_release",
        "multimodal_omnidocbench_2026": "multimodal_understanding_release",
        "multimodal_cc_ocr_v2_2026": "multimodal_understanding_release",
        "multimodal_real5_omnidocbench_2026": "multimodal_understanding_release",
    }
    for benchmark_id, gate in expected_gates.items():
        assert benchmark_id in adapters
        assert benchmark_id in suite["release_gates"][gate]
        assert benchmark_id in suite["reportable_snapshots"]
        assert benchmark_id in suite["reportable_task_roots"]

    assert adapters["factuality_simpleqa_verified_2026"]["source"] == (
        "https://huggingface.co/datasets/google/simpleqa-verified"
    )
    assert adapters["agent_mc_search_mmrag_2026"]["source"] == "https://mc-search-project.github.io/"
    assert "document_vqa_accuracy" in adapters["multimodal_ocrbench_v2_2026"]["metrics"]


def test_suite_profile_includes_terminalworld_agent_gate() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    benchmark_id = "agent_terminalworld_2026"
    assert benchmark_id in adapters
    assert benchmark_id in suite["release_gates"]["agent_tool_release"]
    assert benchmark_id in suite["reportable_snapshots"]
    assert benchmark_id in suite["reportable_task_roots"]
    assert adapters[benchmark_id]["source"] == "https://github.com/EuniAI/TerminalWorld"
    assert "terminal" in adapters[benchmark_id]["modalities"]


def test_suite_profile_includes_hellaswag_learning_signal_gate() -> None:
    suite = runner.load_profile(runner.DEFAULT_PROFILE)
    adapters = {adapter["benchmark_id"]: adapter for adapter in suite["benchmarks"]}

    benchmark_id = "reasoning_hellaswag_full_2026"
    adapter = adapters[benchmark_id]
    snapshot = suite["reportable_snapshots"][benchmark_id]
    assert benchmark_id in suite["release_gates"]["reasoning_release"]
    assert benchmark_id in suite["reportable_snapshots"]
    assert benchmark_id in suite["reportable_task_roots"]
    assert suite["reportable_task_roots"][benchmark_id] == ["data/eval/reportable_2026/hellaswag_authorized.jsonl"]
    assert snapshot["license_ref"] == "authorized-eval-ledger:hellaswag"
    assert snapshot["official_scorer_ref"] == "hellaswag-official-eval-2026"
    assert adapter["adapter_kind"] == "commonsense_completion_mcq"
    assert adapter["task_format"] == "jsonl_multiple_choice_completion"
    assert "multiple_choice" in adapter["modalities"]
    assert "normalized_accuracy" in adapter["metrics"]
    assert adapter["source"] == "https://huggingface.co/datasets/Rowan/hellaswag"


def test_run_reportable_scores_hellaswag_with_descriptor_file_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    benchmark_id = "reasoning_hellaswag_full_2026"
    tasks = tmp_path / "hellaswag_authorized.jsonl"
    _write_profile(
        tmp_path / "profile.json",
        {
            "version": "2026-05-23.hellaswag-test",
            "benchmarks": [
                {
                    "benchmark_id": benchmark_id,
                    "adapter_kind": "commonsense_completion_mcq",
                    "axis": "reasoning",
                    "source": "https://huggingface.co/datasets/Rowan/hellaswag",
                    "task_format": "jsonl_multiple_choice_completion",
                    "splits": {"smoke": "authorized HellaSwag fixture"},
                    "metrics": ["accuracy", "normalized_accuracy"],
                    "holdout_policy": ["hide_answer_keys"],
                }
            ],
            "reportable_task_roots": {benchmark_id: [str(tasks)]},
            "reportable_snapshots": {
                benchmark_id: {
                    "snapshot_id": "hellaswag-authorized-2026-current",
                    "snapshot_authorization": "official_or_authorized_current_release",
                    "dataset_revision": "hellaswag-authorized-2026-current",
                    "source": "https://huggingface.co/datasets/Rowan/hellaswag",
                    "authorization_ref": "operator_supplied_authorized_snapshot_manifest",
                    "license_ref": "authorized-eval-ledger:hellaswag",
                    "official_scorer_ref": "hellaswag-official-eval-2026",
                    "task_root": str(tasks),
                }
            },
        },
    )
    tasks.write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "task_id": "hellaswag-1",
                "question": "A person opens the oven door and",
                "choices": ["sits on the couch.", "pulls out a tray.", "drives away.", "prints a receipt."],
                "answer": "1",
                "prediction": "B",
                "source": "https://huggingface.co/datasets/Rowan/hellaswag",
                "reportable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        runner.main(
            [
                "--profile",
                str(tmp_path / "profile.json"),
                "--out-dir",
                str(tmp_path / "out"),
                "run-reportable",
                "--adapter",
                benchmark_id,
                "--run-id",
                "hellaswag-reportable-fixture",
            ]
        )
        == 0
    )

    payload = _json_from_stdout(capsys)
    result = _jsonl_rows(Path(payload["results"]))[0]
    task_score = result["metrics_json"]["task_scores"][0]
    assert payload["status"] == "needs_data"
    assert result["status"] == "contract_only"
    assert result["score"] is None
    assert result["score_json"]["contract_score"] == 1.0
    assert result["score_json"]["reportable_score"] is False
    assert result["score_json"]["contract_only"] is True
    assert task_score["reportable_metadata"] is True
    assert task_score["has_model_output"] is True

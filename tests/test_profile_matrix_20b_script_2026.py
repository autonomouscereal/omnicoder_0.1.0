from __future__ import annotations

import json
import runpy
from types import SimpleNamespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ai_server_profile_matrix_20b.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_profile_matrix_summary_aggregates_phase_timings_and_target_coverage(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    summarize_run = namespace["summarize_run"]

    out_dir = tmp_path / "run"
    _write_jsonl(
        out_dir / "logs" / "profile_loss.jsonl",
        [
            {
                "step": 1,
                "loss": 2.5,
                "optimized_target_tokens": 6,
                "valid_target_tokens": 8,
                "seq_len": 32,
                "batch_size": 2,
            }
        ],
    )
    _write_jsonl(
        out_dir / "diagnostics" / "profile_step_timing.rank00000.jsonl",
        [
            {
                "event": "pipeline_step_timing",
                "total_sec": 4.0,
                "rank_skew_sec": 0.25,
                "spans": {
                    "batch_fetch_sec": 0.5,
                    "host_to_device_sec": 0.25,
                    "schedule_step_sec": 2.0,
                    "optimizer_step_sec": 1.0,
                },
                "optimizer_diagnostics": {"step": 1, "hook_step_sec": 0.75},
                "lm_loss_timing": {
                    "total_sec": 1.5,
                    "spans": {
                        "selected_lm_head_ce_sec": 1.25,
                        "selected_position_scan_sec": 0.1,
                    },
                },
            },
            {
                "event": "pipeline_step_timing",
                "total_sec": 5.0,
                "rank_skew_sec": 0.5,
                "spans": {
                    "batch_fetch_sec": 0.75,
                    "host_to_device_sec": 0.5,
                    "schedule_step_sec": 2.5,
                    "optimizer_step_sec": 1.25,
                },
                "optimizer_diagnostics": {"step": 2, "hook_step_sec": 1.25},
                "lm_loss_timing": {
                    "total_sec": 2.0,
                    "spans": {
                        "selected_lm_head_ce_sec": 1.75,
                        "selected_position_scan_sec": 0.15,
                    },
                },
            },
        ],
    )

    class _Proc:
        stdout = "tail"
        stderr = ""
        returncode = 0

    summarize_run.__globals__["run"] = lambda *_args, **_kwargs: _Proc()

    summary = summarize_run(out_dir, "container", tmp_path)

    assert summary["last_target_token_coverage"] == 0.75
    assert summary["sequence_tokens_per_sec"] == 6.4
    assert summary["training_tokens_per_sec"] == 12.8
    assert summary["no_checkpoint_written"] is True
    assert summary["phase_timing_summary"]["batch_fetch_sec"]["count"] == 2
    assert summary["phase_timing_summary"]["batch_fetch_sec"]["max_sec"] == 0.75
    assert summary["phase_timing_summary"]["schedule_step_sec"]["mean_sec"] == 2.25
    assert summary["rank_skew_summary"]["max_sec"] == 0.5
    assert summary["rank_timing"][0]["phase_spans"]["host_to_device_sec"] == 0.5
    assert summary["rank_timing"][0]["lm_loss_timing"]["spans"]["selected_lm_head_ce_sec"] == 1.75
    assert summary["lm_loss_timing_summary"]["selected_lm_head_ce_sec"]["max_sec"] == 1.75
    assert summary["lm_loss_timing_summary"]["total_sec"]["mean_sec"] == 1.75
    assert summary["optimizer_diagnostics_summary"]["hook_step_sec"]["max_sec"] == 1.25
    assert summary["schedule_step_skew_ratio"] == 1.0


def test_profile_matrix_summary_flags_checkpoint_files(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    summarize_run = namespace["summarize_run"]
    out_dir = tmp_path / "run"
    (out_dir / "checkpoints" / "posttrain" / "step0001").mkdir(parents=True)
    (out_dir / "checkpoints" / "posttrain" / "step0001" / ".complete.json").write_text(
        '{"status":"complete"}\n',
        encoding="utf-8",
    )

    class _Proc:
        stdout = ""
        stderr = ""
        returncode = 0

    summarize_run.__globals__["run"] = lambda *_args, **_kwargs: _Proc()

    summary = summarize_run(out_dir, "container", tmp_path)

    assert summary["no_checkpoint_written"] is False
    assert summary["checkpoint_complete_files"]


def test_profile_matrix_stops_timed_out_container_when_cleanup_enabled(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    launch_variant = namespace["launch_variant"]
    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd: list[str], **_kwargs):
        calls.append(cmd)
        if cmd and cmd[0] == "bash":
            return _Proc(stdout="container=profile_timeout\nhost_out_dir=/tmp/profile_timeout\n")
        return _Proc(stdout="ok\n")

    launch_variant.__globals__["run"] = fake_run
    launch_variant.__globals__["wait_container"] = lambda *_args, **_kwargs: {"exists": True, "running": True, "timed_out": True}
    launch_variant.__globals__["inspect_container"] = lambda *_args, **_kwargs: {"exists": True, "running": False, "exit_code": 137, "oom_killed": False}
    launch_variant.__globals__["summarize_run"] = lambda *_args, **_kwargs: {"checkpoint_complete_files": [], "no_checkpoint_written": True}

    args = SimpleNamespace(
        timeout_seconds=1,
        poll_seconds=1,
        profile=None,
        curation_manifest="/tmp/curation_manifest.json",
        profile_corpus=False,
        mode="target_20b_native_1m_q4",
        seq_len=128,
        steps=1,
        cleanup_containers=True,
    )

    result = launch_variant(tmp_path, tmp_path, "timeout_probe", {"name": "variant", "env": {}, "steps": 1}, args)

    assert result["status"] == "timed_out"
    assert result["timeout_stop"]["returncode"] == 0
    assert result["container_state_after_timeout_stop"]["running"] is False
    assert ["docker", "stop", "profile_timeout"] in calls
    assert ["docker", "rm", "profile_timeout"] in calls


def test_profile_matrix_default_selection_skips_opt_in_risky_variants() -> None:
    namespace = runpy.run_path(str(SCRIPT))
    select_variants = namespace["select_variants"]

    selected, missing = select_variants(set())
    names = {str(variant["name"]) for variant in selected}

    assert missing == []
    assert "fakequant_chunk2048_loss64" in names
    assert "fakequant_chunk4096_loss64" not in names
    assert "fakequant_chunk8192_loss64" not in names
    assert "headroom_16_16_32_q4_chunk2048_loss64" not in names
    assert "headroom_16_16_32_q4_chunk8192_loss64" not in names
    assert "gdn2_compiled_fakequant_chunk256_loss64" not in names
    assert "gdn2_jit_q4_loss64" not in names
    assert "fakequant_chunk2048_loss64_diagnostics" not in names
    assert "block_timing_q4_chunk2048_loss64" not in names
    assert "checkpoint_segment2_q4_chunk2048_loss64" not in names
    assert "checkpoint_segment4_q4_chunk2048_loss64" not in names
    assert "reasoning_effort2_q4_chunk2048_loss64" not in names
    assert "reasoning_efforthigh_q4_chunk2048_loss64" not in names
    assert "p2p_on_ffn_chunk1024_headroom_q4_chunk8192_loss64" not in names
    assert "gdn2_compiled_headroom_q4_chunk8192_ffn1024_loss64" not in names
    assert "gdn2_jit_headroom_q4_chunk8192_ffn1024_loss64" not in names
    assert "gpipe_mb2_q4_chunk2048_loss64" not in names
    assert "onef1b_mb2_q4_chunk2048_loss64" not in names
    assert "actckpt_off_q4_chunk2048_loss64" not in names
    assert "actckpt_off_q4_loss64" not in names

    selected, missing = select_variants({"gdn2_jit_q4_loss64", "fakequant_chunk4096_loss64", "headroom_16_16_32_q4_chunk8192_loss64", "block_timing_q4_chunk2048_loss64", "checkpoint_segment2_q4_chunk2048_loss64", "reasoning_effort2_q4_chunk2048_loss64", "p2p_on_ffn_chunk1024_headroom_q4_chunk8192_loss64", "gdn2_compiled_headroom_q4_chunk8192_ffn1024_loss64", "gdn2_jit_headroom_q4_chunk8192_ffn1024_loss64", "gpipe_mb2_q4_chunk2048_loss64", "missing_variant"})
    assert [variant["name"] for variant in selected] == [
        "gdn2_jit_q4_loss64",
        "fakequant_chunk4096_loss64",
        "headroom_16_16_32_q4_chunk8192_loss64",
        "p2p_on_ffn_chunk1024_headroom_q4_chunk8192_loss64",
        "gdn2_compiled_headroom_q4_chunk8192_ffn1024_loss64",
        "gdn2_jit_headroom_q4_chunk8192_ffn1024_loss64",
        "block_timing_q4_chunk2048_loss64",
        "checkpoint_segment2_q4_chunk2048_loss64",
        "reasoning_effort2_q4_chunk2048_loss64",
        "gpipe_mb2_q4_chunk2048_loss64",
    ]
    assert missing == ["missing_variant"]

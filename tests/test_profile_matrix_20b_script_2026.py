from __future__ import annotations

import json
import runpy
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
                "optimizer_diagnostics": {"step": 1},
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
                "optimizer_diagnostics": {"step": 2},
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
    assert summary["no_checkpoint_written"] is True
    assert summary["phase_timing_summary"]["batch_fetch_sec"]["count"] == 2
    assert summary["phase_timing_summary"]["batch_fetch_sec"]["max_sec"] == 0.75
    assert summary["phase_timing_summary"]["schedule_step_sec"]["mean_sec"] == 2.25
    assert summary["rank_skew_summary"]["max_sec"] == 0.5
    assert summary["rank_timing"][0]["phase_spans"]["host_to_device_sec"] == 0.5


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

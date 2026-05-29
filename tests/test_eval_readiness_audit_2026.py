from __future__ import annotations

import json
from pathlib import Path

from omnicoder.eval.eval_readiness_audit_2026 import build_audit, main


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _profile(tmp_path: Path, *, reportable_roots: dict | None = None) -> Path:
    benchmarks = [
        {"benchmark_id": "reasoning_hellaswag_full_2026", "axis": "reasoning"},
        {"benchmark_id": "reasoning_arc_agi3_2026", "axis": "reasoning"},
        {"benchmark_id": "reasoning_frontiermath_2026", "axis": "reasoning"},
        {"benchmark_id": "coding_swe_bench_live_2026", "axis": "coding"},
        {"benchmark_id": "agent_bfcl_v4_2026", "axis": "agent_tool"},
        {"benchmark_id": "multimodal_ocrbench_v2_2026", "axis": "multimodal_understanding"},
        {"benchmark_id": "generation_image_edit_2026", "axis": "image_generation"},
        {"benchmark_id": "generation_video_2026", "axis": "video_generation"},
        {"benchmark_id": "generation_audio_speech_2026", "axis": "audio_generation"},
        {"benchmark_id": "generation_music_2026", "axis": "music_generation"},
        {"benchmark_id": "long_context_nolima_1m_2026", "axis": "long_context"},
    ]
    return _write_json(
        tmp_path / "profiles" / "benchmark_suite_2026.json",
        {
            "benchmarks": benchmarks,
            "reportable_task_roots": reportable_roots or {},
            "reportable_snapshots": {},
            "reportable_core_25": ["reasoning_hellaswag_full_2026"],
        },
    )


def test_eval_readiness_blocks_public_dev_without_reportable_roots_or_diagnostics(tmp_path: Path) -> None:
    profile_path = _profile(tmp_path)
    mat = tmp_path / "weights" / "data_factory" / "runs" / "benchmark_materialization" / "hellaswag" / "manifests"
    _write_json(
        mat / "benchmark_materialization_manifest.json",
        {
            "records": [
                {
                    "benchmark_id": "reasoning_hellaswag_full_2026",
                    "status": "materialized",
                    "rows": 10042,
                }
            ]
        },
    )

    report = build_audit(
        repo_root=tmp_path,
        weights_root=tmp_path / "weights",
        profile_path=profile_path,
        materialization_root=tmp_path / "weights" / "data_factory" / "runs" / "benchmark_materialization",
        diagnostic_roots=[tmp_path / "weights" / "diagnostics"],
        score_roots=[tmp_path / "weights" / "benchmarks"],
    )

    assert report["status"] == "blocked"
    assert "declared_reportable_task_roots_missing_or_empty" in report["blockers"]
    assert "no_official_reportable_scorer_results" in report["blockers"]
    assert "decode_sanity_missing_or_unusable" in report["blockers"]
    assert report["materialized_public_dev"]["required_groups"]["hellaswag"]["status"] == "diagnostic_materialized"
    assert report["reportable_task_roots"]["declared_files"] == 0


def test_eval_readiness_passes_when_reportable_scores_and_core_diagnostics_exist(tmp_path: Path) -> None:
    reportable = _write_jsonl(
        tmp_path / "data" / "eval" / "reportable_2026" / "hellaswag_authorized.jsonl",
        [{"benchmark_id": "reasoning_hellaswag_full_2026", "task_id": "hs-1"}],
    )
    profile_path = _profile(
        tmp_path,
        reportable_roots={"reasoning_hellaswag_full_2026": [str(reportable.relative_to(tmp_path))]},
    )
    mat = tmp_path / "weights" / "data_factory" / "runs" / "benchmark_materialization" / "hellaswag" / "manifests"
    _write_json(
        mat / "benchmark_materialization_manifest.json",
        {"records": [{"benchmark_id": "reasoning_hellaswag_full_2026", "status": "materialized", "rows": 8}]},
    )
    diag = tmp_path / "weights" / "diagnostics" / "run_a"
    _write_json(diag / "heldout_sample_loss.json", {"overall": {"avg_loss": 1.2, "tokens": 10}})
    _write_json(diag / "target_mask_coverage.json", {"schema": "coverage", "status": "ok"})
    _write_json(diag / "decode_sanity.json", {"schema": "decode", "status": "ok", "reasons": []})
    _write_json(
        tmp_path / "weights" / "benchmarks" / "reportable" / "reportable_summary.json",
        {"status": "ok", "reportable": 1, "official": 1, "failed": 0},
    )

    report = build_audit(
        repo_root=tmp_path,
        weights_root=tmp_path / "weights",
        profile_path=profile_path,
        materialization_root=tmp_path / "weights" / "data_factory" / "runs" / "benchmark_materialization",
        diagnostic_roots=[tmp_path / "weights" / "diagnostics"],
        score_roots=[tmp_path / "weights" / "benchmarks"],
    )

    assert report["status"] == "ready"
    assert report["ready_for_full_training"] is True
    assert report["blockers"] == []


def test_eval_readiness_cli_sanitizes_nonfinite_old_loss_artifacts(tmp_path: Path, capsys) -> None:
    profile_path = _profile(tmp_path)
    _write_json(tmp_path / "weights" / "diagnostics" / "bad_sample_loss.json", {"overall": {"avg_loss": float("inf"), "tokens": 10}})

    code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--weights-root",
            str(tmp_path / "weights"),
            "--profile",
            str(profile_path),
            "--diagnostic-root",
            str(tmp_path / "weights" / "diagnostics"),
        ]
    )

    assert code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"

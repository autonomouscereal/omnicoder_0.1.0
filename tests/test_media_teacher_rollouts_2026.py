from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omnicoder.data_factory import media_teacher_rollouts_2026 as rollouts


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dry_run_classifies_modality_teacher_jobs(tmp_path: Path) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    _jsonl(
        source,
        [
            {"teacher_name": "qwen_image_generate", "job_type": "image_reward_label", "input_json": {"prompt": "make a microscope image"}},
            {"teacher_name": "ltx_2_3", "job_type": "image_to_video_plan", "input_json": {"prompt": "animate this reference"}},
            {"teacher_name": "ace_step_1_5", "job_type": "music_plan", "input_json": {"prompt": "bright synth bumper"}},
        ],
    )

    code = rollouts.main(["--input", str(source), "--out-dir", str(out_dir), "--mode", "dry-run"])

    assert code == 0
    rows = _rows(out_dir / "media_teacher_rollouts.jsonl")
    assert [row["status"] for row in rows] == ["planned", "planned", "planned"]
    assert [row["media_family"] for row in rows] == ["qwen_image", "ltx_video", "ace_music"]
    assert (out_dir / "qwen_image_rollouts.jsonl").exists()
    assert (out_dir / "ltx_video_rollouts.jsonl").exists()
    assert (out_dir / "ace_music_rollouts.jsonl").exists()
    manifest = json.loads((out_dir / "media_teacher_rollout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["counts"]["media_teacher_rollouts.jsonl"] == 3


def test_live_subprocess_runner_records_artifact_metadata(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "qwen_result.png"
    artifact.write_bytes(b"not really a png but enough for metadata")
    _jsonl(
        source,
        [
            {
                "teacher_name": "qwen_image_generate",
                "job_type": "image_reward_label",
                "contamination_status": "clean",
                "input_json": {"prompt": "city at night"},
            },
        ],
    )
    calls: list[str] = []

    def fake_run(command, shell, check, stdout, stderr, text, timeout, env):  # noqa: ANN001
        calls.append(command)
        assert env["OMNICODER_MEDIA_TEACHER_TEST"] == "qwen_t2i"
        payload = {"ok": True, "files": [str(artifact)]}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(rollouts.subprocess, "run", fake_run)

    code = rollouts.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--mode",
            "live",
            "--artifact-root",
            str(artifact_root),
            "--runner-command",
            "python runner.py --job {job_json} --test {test}",
        ]
    )

    assert code == 0
    assert calls == [f"python runner.py --job {out_dir / 'jobs' / 'media_teacher_job_00000001.json'} --test qwen_t2i"]
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "ok"
    assert row["split"] == "train"
    assert row["artifact_metadata"][0]["exists"] is True
    assert row["artifact_metadata"][0]["byte_size"] == artifact.stat().st_size
    assert row["artifact_metadata"][0]["sha256"]


def test_resume_retries_failed_rows_but_skips_ok_rows(tmp_path: Path, monkeypatch) -> None:
    retry_job = {
        "teacher_name": "qwen_image_generate",
        "job_type": "image_reward_label",
        "contamination_status": "clean",
        "input_json": {"prompt": "retry this failed teacher job"},
    }
    completed_job = {
        "teacher_name": "qwen_image_generate",
        "job_type": "image_reward_label",
        "contamination_status": "clean",
        "input_json": {"prompt": "do not rerun this completed teacher job"},
    }
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "retry_result.png"
    artifact.write_bytes(b"stub artifact")
    _jsonl(source, [retry_job, completed_job])
    _jsonl(
        out_dir / "media_teacher_rollouts.jsonl",
        [
            {"status": "failed", "job_hash": rollouts.stable_hash(retry_job), "error": "previous_failure"},
            {"status": "ok", "job_hash": rollouts.stable_hash(completed_job)},
        ],
    )
    calls: list[str] = []

    def fake_run(command, shell, check, stdout, stderr, text, timeout, env):  # noqa: ANN001
        calls.append(command)
        job_payload = json.loads(Path(env["OMNICODER_MEDIA_TEACHER_JOB"]).read_text(encoding="utf-8"))
        assert job_payload == retry_job
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "files": [str(artifact)]}), stderr="")

    monkeypatch.setattr(rollouts.subprocess, "run", fake_run)

    code = rollouts.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--mode",
            "live",
            "--resume",
            "--artifact-root",
            str(artifact_root),
            "--runner-command",
            "python runner.py --job {job_json} --test {test}",
        ]
    )

    assert code == 0
    assert calls == [f"python runner.py --job {out_dir / 'jobs' / 'media_teacher_job_00000001.json'} --test qwen_t2i"]
    rows = _rows(out_dir / "media_teacher_rollouts.jsonl")
    assert [row["status"] for row in rows] == ["failed", "ok", "ok"]
    assert rows[-1]["job_hash"] == rollouts.stable_hash(retry_job)
    manifest = json.loads((out_dir / "media_teacher_rollout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["written"] == 1
    assert manifest["counts"]["media_teacher_rollouts.jsonl"] == 2


def test_live_http_runner_uses_embedded_workflow(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    _jsonl(
        source,
        [
            {
                "teacher_name": "ltx_2_3",
                "job_type": "temporal_reward_label",
                "contamination_status": "clean",
                "input_json": {"prompt": "slow camera move", "workflow": {"1": {"class_type": "SaveAnimatedWEBP"}}},
            }
        ],
    )
    calls: list[tuple[str, str, dict | None]] = []

    def fake_request_json(method: str, url: str, payload=None, timeout: int = 60) -> dict:  # noqa: ANN001
        calls.append((method, url, payload))
        if method == "POST":
            assert payload == {"prompt": {"1": {"class_type": "SaveAnimatedWEBP"}}}
            return {"prompt_id": "abc"}
        return {
            "abc": {
                "status": {"completed": True},
                "outputs": {"9": {"gifs": [{"filename": "ltx_result.webp"}]}},
            }
        }

    monkeypatch.setattr(rollouts, "request_json", fake_request_json)

    def fake_artifact_metadata(path, root=None):  # noqa: ANN001
        filename = path["filename"] if isinstance(path, dict) else str(path)
        return {"path": str(Path(root or "") / filename), "exists": True, "byte_size": 10}

    monkeypatch.setattr(
        rollouts,
        "artifact_metadata",
        fake_artifact_metadata,
    )

    code = rollouts.main(["--input", str(source), "--out-dir", str(out_dir), "--mode", "live", "--comfyui-url", "http://127.0.0.1:18188"])

    assert code == 0
    assert calls[0][0] == "POST"
    assert calls[1][0] == "GET"
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "ok"
    assert row["split"] == "train"
    assert row["target_json"]["rollout_result"]["prompt_id"] == "abc"
    assert row["artifact_metadata"][0]["path"].endswith("ltx_result.webp")


def test_live_artifact_with_unknown_source_contamination_is_quarantined(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "qwen_result.png"
    artifact.write_bytes(b"valid enough artifact bytes")
    _jsonl(
        source,
        [
            {
                "teacher_name": "qwen_image_generate",
                "job_type": "image_reward_label",
                "input_json": {
                    "prompt": "city at night",
                    "source_payload": {"contamination": {"status": "unknown"}},
                },
            },
        ],
    )

    def fake_run(command, shell, check, stdout, stderr, text, timeout, env):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "files": [str(artifact)]}), stderr="")

    monkeypatch.setattr(rollouts.subprocess, "run", fake_run)

    code = rollouts.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--mode",
            "live",
            "--artifact-root",
            str(artifact_root),
            "--runner-command",
            "python runner.py --job {job_json} --test {test}",
        ]
    )

    assert code == 0
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "ok"
    assert row["artifact_metadata"][0]["exists"] is True
    assert row["split"] == "blocked_until_review"
    assert row["training_bucket"] == "blocked_until_review"
    assert row["contamination_status"] == "quarantine"
    assert row["source_contamination_status"] == "unknown"
    assert "source_contamination_status_not_clean:unknown" in row["train_quarantine_reasons"]
    assert row["quality_score"] == 0.0


def test_live_artifact_with_clean_curation_policy_is_trainable(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "qwen_result.png"
    artifact.write_bytes(b"valid enough artifact bytes")
    _jsonl(
        source,
        [
            {
                "teacher": "qwen_image_generate",
                "job_type": "qwen_image_prompt_reward",
                "curation_policy_2026": {
                    "accepted": True,
                    "dataset_integrity_2026": {"accepted": True, "issues": [], "reasons": []},
                },
                "input_json": {"prompt": "clean curated image prompt"},
            },
        ],
    )

    def fake_run(command, shell, check, stdout, stderr, text, timeout, env):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "files": [str(artifact)]}), stderr="")

    monkeypatch.setattr(rollouts.subprocess, "run", fake_run)

    code = rollouts.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--mode",
            "live",
            "--artifact-root",
            str(artifact_root),
            "--runner-command",
            "python runner.py --job {job_json} --test {test}",
        ]
    )

    assert code == 0
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "ok"
    assert row["teacher"] == "qwen_image_generate"
    assert row["split"] == "train"
    assert row["source_contamination_status"] == "clean"
    assert row["train_quarantine_reasons"] == []


def test_clean_curation_policy_does_not_override_embedded_unknown_source_prompt(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = artifact_root / "qwen_result.png"
    artifact.write_bytes(b"valid enough artifact bytes")
    _jsonl(
        source,
        [
            {
                "teacher": "qwen_image_generate",
                "job_type": "qwen_image_prompt_reward",
                "curation_policy_2026": {
                    "accepted": True,
                    "dataset_integrity_2026": {"accepted": True, "issues": [], "reasons": []},
                },
                "input_json": {
                    "prompt": "Generate image from {'contamination': {'status': 'unknown'}, 'prompt': 'bad embedded metadata'}",
                },
            },
        ],
    )

    def fake_run(command, shell, check, stdout, stderr, text, timeout, env):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "files": [str(artifact)]}), stderr="")

    monkeypatch.setattr(rollouts.subprocess, "run", fake_run)

    code = rollouts.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--mode",
            "live",
            "--artifact-root",
            str(artifact_root),
            "--runner-command",
            "python runner.py --job {job_json} --test {test}",
        ]
    )

    assert code == 0
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "ok"
    assert row["split"] == "blocked_until_review"
    assert row["source_contamination_status"] == "unknown"
    assert "source_contamination_status_not_clean:unknown" in row["train_quarantine_reasons"]


def test_artifact_metadata_maps_comfyui_container_output_path_to_host_root(tmp_path: Path) -> None:
    artifact = tmp_path / "nested" / "result.png"
    artifact.parent.mkdir()
    artifact.write_bytes(b"real-ish image bytes")

    meta = rollouts.artifact_metadata("/opt/ComfyUI/output/nested/result.png", tmp_path)

    assert meta["path"] == str(artifact)
    assert meta["exists"] is True
    assert meta["byte_size"] == artifact.stat().st_size
    assert meta["sha256"]


def test_live_strict_returns_nonzero_on_missing_runner(tmp_path: Path) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    _jsonl(source, [{"teacher_name": "ace_step_1_5", "job_type": "music_plan", "input_json": {"prompt": "tts-like vocal"}}])

    code = rollouts.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(out_dir),
            "--mode",
            "live",
            "--modal-script",
            str(tmp_path / "missing.py"),
            "--strict-live",
        ]
    )

    assert code == 6
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "failed"
    assert "missing_runner_script" in row["error"]

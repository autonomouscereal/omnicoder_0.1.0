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
            {"teacher_name": "qwen_image_generate", "job_type": "image_reward_label", "input_json": {"prompt": "city at night"}},
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
    assert row["artifact_metadata"][0]["exists"] is True
    assert row["artifact_metadata"][0]["byte_size"] == artifact.stat().st_size
    assert row["artifact_metadata"][0]["sha256"]


def test_live_http_runner_uses_embedded_workflow(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "jobs.jsonl"
    out_dir = tmp_path / "rollouts"
    _jsonl(
        source,
        [
            {
                "teacher_name": "ltx_2_3",
                "job_type": "temporal_reward_label",
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
    monkeypatch.setattr(
        rollouts,
        "artifact_metadata",
        lambda path, root=None: {"path": str(Path(root or "") / str(path)), "exists": True, "byte_size": 10},
    )

    code = rollouts.main(["--input", str(source), "--out-dir", str(out_dir), "--mode", "live", "--comfyui-url", "http://127.0.0.1:18188"])

    assert code == 0
    assert calls[0][0] == "POST"
    assert calls[1][0] == "GET"
    row = _rows(out_dir / "media_teacher_rollouts.jsonl")[0]
    assert row["status"] == "ok"
    assert row["target_json"]["rollout_result"]["prompt_id"] == "abc"
    assert row["artifact_metadata"][0]["path"].endswith("ltx_result.webp")


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

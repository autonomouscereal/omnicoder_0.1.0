from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnicoder.tools import gpu_sidecar_2026 as sidecar


def _profile(tmp_path: Path) -> dict:
    return {
        "training_plan": {
            "distributed_training": {
                "main_gpu_devices": ["0", "4", "6"],
                "p40_sidecar_devices": ["1", "2", "3", "5"],
            }
        },
        "gpu_utilization_sidecar": {
            "enabled": True,
            "devices": ["1", "2", "3", "5"],
            "main_training_devices": ["0", "4", "6"],
            "out_root": str(tmp_path / "sidecar"),
            "jobs": [
                {
                    "job_id": "dataset",
                    "job_type": "dataset_materialization",
                    "device": "1",
                },
                {
                    "job_id": "teacher",
                    "job_type": "teacher_distillation",
                    "device": "2",
                    "records": str(tmp_path / "curated.jsonl"),
                    "limit": 7,
                    "skip_if_missing": [str(tmp_path / "curated.jsonl")],
                },
                {
                    "job_id": "external",
                    "job_type": "external_dataset_expansion",
                    "device": "cpu",
                    "profile": "profiles/dataset_curation_2026.json",
                    "out_dir": str(tmp_path / "external" / "{run_id}"),
                    "max_records_per_dataset": 32,
                },
                {
                    "job_id": "bench",
                    "job_type": "benchmark_canary",
                    "device": "3",
                    "model": "probe",
                    "timeout_seconds": 11,
                },
                {
                    "job_id": "train_probe",
                    "job_type": "training_run",
                    "device": "5",
                    "steps_per_stage": 2,
                    "seq_len": 96,
                    "batch_size": 1,
                    "fake_quant": True,
                    "allow_verifier_preset": True,
                },
            ],
        },
    }


def _args(tmp_path: Path, **updates) -> argparse.Namespace:
    payload = {
        "out_root": str(tmp_path / "sidecar"),
        "job": None,
        "dry_run": True,
        "wait": False,
    }
    payload.update(updates)
    return argparse.Namespace(**payload)


def test_sidecar_devices_do_not_overlap_main_fsdp_devices(tmp_path):
    profile = _profile(tmp_path)
    result = sidecar.validate_device_isolation(profile)
    assert result["status"] == "ok"
    assert result["sidecar_devices"] == ["1", "2", "3", "5"]
    assert result["main_training_devices"] == ["0", "4", "6"]
    assert result["overlap"] == []
    assert result["job_device_overlaps"] == []
    assert result["invalid_job_devices"] == []


def test_sidecar_validation_fails_on_device_overlap(tmp_path):
    profile = _profile(tmp_path)
    profile["gpu_utilization_sidecar"]["devices"] = ["1", "4"]
    result = sidecar.validate_device_isolation(profile)
    assert result["status"] == "failed"
    assert result["overlap"] == ["4"]


def test_sidecar_validation_fails_on_job_pinned_to_main_device(tmp_path):
    profile = _profile(tmp_path)
    profile["gpu_utilization_sidecar"]["jobs"][0]["device"] = "4"
    result = sidecar.validate_device_isolation(profile)
    assert result["status"] == "failed"
    assert result["job_device_overlaps"] == [{"job_id": "dataset", "device": "4"}]


def test_materialized_jobs_pin_cuda_and_write_to_sidecar_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "repo_root", lambda: tmp_path)
    profile = _profile(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    jobs = sidecar.materialize_jobs(profile_path, profile, _args(tmp_path))
    by_id = {job["job_id"]: job for job in jobs}

    assert by_id["dataset"]["device"] == "1"
    assert by_id["dataset"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.training.training_orchestration_2026"]
    assert str(tmp_path / "sidecar" / "dataset") in by_id["dataset"]["command"]
    assert by_id["teacher"]["status"] == "skipped"
    assert "missing required path" in by_id["teacher"]["skip_reason"]
    assert by_id["external"]["device"] == "cpu"
    assert by_id["external"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.data_factory.dataset_expansion_2026"]
    assert "--download" in by_id["external"]["command"]
    assert by_id["external"]["command"][by_id["external"]["command"].index("--max-records-per-dataset") + 1] == "32"
    assert by_id["bench"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.eval.benchmark_suite_2026"]
    assert by_id["train_probe"]["device"] == "5"
    assert by_id["train_probe"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.training.training_orchestration_2026"]
    assert "run-real" in by_id["train_probe"]["command"]
    assert by_id["train_probe"]["command"][by_id["train_probe"]["command"].index("--device") + 1] == "cuda"
    assert "--fake-quant" in by_id["train_probe"]["command"]

    env = sidecar.launch_env(by_id["bench"], profile)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["OMNICODER_SIDECAR_JOB_ID"] == "bench"
    assert env["OMNICODER_SIDECAR_OUT_DIR"] == str(tmp_path / "sidecar" / "bench")

    cpu_env = sidecar.launch_env(by_id["external"], profile)
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""


def test_openai_teacher_rollout_expands_multidevice_job(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "repo_root", lambda: tmp_path)
    profile = _profile(tmp_path)
    profile["gpu_utilization_sidecar"]["jobs"] = [
        {
            "job_id": "rollout",
            "job_type": "openai_compatible_teacher_rollout",
            "device": "1,2,3",
            "records": str(tmp_path / "teacher_jobs" / "shard_gpu*.jsonl"),
            "output": str(tmp_path / "teacher_rollouts" / "qwen36.jsonl"),
            "base_urls": [
                "http://127.0.0.1:18084/v1",
                "http://127.0.0.1:18082/v1",
                "http://127.0.0.1:18085/v1",
            ],
            "limit": 12,
            "thermal_guard_celsius": 78,
        }
    ]
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    jobs = sidecar.materialize_jobs(profile_path, profile, _args(tmp_path))

    assert [job["job_id"] for job in jobs] == ["rollout_gpu1", "rollout_gpu2", "rollout_gpu3"]
    assert [job["device"] for job in jobs] == ["1", "2", "3"]
    for device, job in zip(("1", "2", "3"), jobs):
        command = job["command"]
        assert command[:3] == [sidecar.sys.executable, "-m", "omnicoder.data_factory.openai_teacher_rollout_2026"]
        assert command[command.index("--input") + 1].endswith(f"shard_gpu{device}.jsonl")
        assert command[command.index("--out") + 1].endswith(f"qwen36_gpu{device}.jsonl")
        assert command[command.index("--thermal-gpu-index") + 1] == device
        assert "--resume" in command


def test_dry_run_launch_writes_manifest_without_spawning_heavy_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "repo_root", lambda: tmp_path)
    profile = _profile(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    result = sidecar.launch_jobs(profile_path, profile, _args(tmp_path))

    assert result["status"] == "ok"
    manifest = Path(result["manifest"])
    event_log = Path(result["event_log"])
    assert manifest.exists()
    assert event_log.exists()
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    assert rows["dry_run"] is True
    assert any(job["status"] == "dry_run" for job in rows["jobs"])
    assert any(job["status"] == "skipped" for job in rows["jobs"])

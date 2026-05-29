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
            "qwen_reserved_devices": ["1", "2", "3"],
            "comfyui_modality_devices": ["5"],
            "out_root": str(tmp_path / "sidecar"),
            "jobs": [
                {
                    "job_id": "dataset",
                    "job_type": "dataset_materialization",
                    "device": "cpu",
                },
                {
                    "job_id": "teacher",
                    "job_type": "teacher_distillation",
                    "device": "cpu",
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
                    "device": "cpu",
                    "model": "probe",
                    "timeout_seconds": 11,
                },
                {
                    "job_id": "media",
                    "job_type": "media_teacher_rollout",
                    "device": "5",
                    "records": str(tmp_path / "media_jobs.jsonl"),
                    "limit": 2,
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
    assert result["qwen_reserved_devices"] == ["1", "2", "3"]
    assert result["comfyui_modality_devices"] == ["5"]
    assert result["overlap"] == []
    assert result["job_device_overlaps"] == []
    assert result["invalid_job_devices"] == []
    assert result["role_device_violations"] == []


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

    assert by_id["dataset"]["device"] == "cpu"
    assert by_id["dataset"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.training.training_orchestration_2026"]
    assert str(tmp_path / "sidecar" / "dataset") in by_id["dataset"]["command"]
    assert by_id["teacher"]["status"] == "skipped"
    assert "missing required path" in by_id["teacher"]["skip_reason"]
    assert by_id["external"]["device"] == "cpu"
    assert by_id["external"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.data_factory.dataset_expansion_2026"]
    assert "--download" in by_id["external"]["command"]
    assert by_id["external"]["command"][by_id["external"]["command"].index("--max-records-per-dataset") + 1] == "32"
    assert by_id["bench"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.eval.benchmark_suite_2026"]
    assert by_id["media"]["device"] == "5"
    assert by_id["media"]["command"][:3] == [sidecar.sys.executable, "-m", "omnicoder.data_factory.media_teacher_rollouts_2026"]
    assert by_id["media"]["command"][by_id["media"]["command"].index("--comfyui-url") + 1] == "http://127.0.0.1:27189"

    env = sidecar.launch_env(by_id["bench"], profile)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
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
                "http://127.0.0.1:1234/v1",
                "http://127.0.0.1:1234/v1",
                "http://127.0.0.1:1234/v1",
            ],
            "model_ids": [
                "qwen/qwen3.6-27b",
                "qwen/qwen3.6-27b2",
                "qwen/qwen3.6-27b3",
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
        assert command[command.index("--base-url") + 1] == "http://127.0.0.1:1234/v1"
        assert command[command.index("--model") + 1] == f"qwen/qwen3.6-27b{'' if device == '1' else device}"
        assert command[command.index("--thermal-gpu-index") + 1] == device
        assert "--resume" in command
        env = sidecar.launch_env(job, profile)
        assert env["CUDA_VISIBLE_DEVICES"] == ""


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


def test_role_policy_blocks_non_qwen_on_reserved_p40(tmp_path):
    profile = _profile(tmp_path)
    profile["gpu_utilization_sidecar"]["jobs"][0]["device"] = "1"
    result = sidecar.validate_device_isolation(profile)
    assert result["status"] == "failed"
    assert result["role_device_violations"] == [
        {"job_id": "dataset", "device": "1", "reason": "protected_p40_requires_explicit_qwen_or_comfyui_role"}
    ]


def test_role_policy_blocks_qwen_on_fast_or_comfyui_lane(tmp_path):
    profile = _profile(tmp_path)
    profile["gpu_utilization_sidecar"]["jobs"] = [
        {"job_id": "bad_qwen", "job_type": "openai_compatible_teacher_rollout", "device": "5"}
    ]
    result = sidecar.validate_device_isolation(profile)
    assert result["status"] == "failed"
    assert result["role_device_violations"] == [
        {"job_id": "bad_qwen", "device": "5", "reason": "qwen_distillation_must_use_qwen_reserved_p40"},
        {"job_id": "bad_qwen", "device": "5", "reason": "qwen_distillation_requires_explicit_model_ids"},
    ]


def test_role_policy_requires_qwen_model_ids_match_multidevice_count(tmp_path):
    profile = _profile(tmp_path)
    profile["gpu_utilization_sidecar"]["jobs"] = [
        {
            "job_id": "collapsed_qwen",
            "job_type": "openai_compatible_teacher_rollout",
            "device": "1,2,3",
            "model_ids": ["qwen/qwen3.6-27b"],
        }
    ]
    result = sidecar.validate_device_isolation(profile)
    assert result["status"] == "failed"
    assert result["role_device_violations"] == [
        {
            "job_id": "collapsed_qwen",
            "device": "1,2,3",
            "reason": "qwen_model_ids_must_match_device_count",
        }
    ]

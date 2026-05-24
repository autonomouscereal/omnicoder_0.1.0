from __future__ import annotations

import json
import argparse
import os
from pathlib import Path

import omnicoder.training.training_orchestration_2026 as orch


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _profile(root: Path) -> dict:
    image_path = root / "data" / "images" / "one.jpg"
    audio_path = root / "data" / "audio" / "one.wav"
    video_path = root / "data" / "video" / "one.mp4"
    music_path = root / "data" / "music" / "music_song.mp3"
    for path, payload in (
        (image_path, b"real image bytes"),
        (audio_path, b"real audio bytes"),
        (video_path, b"real video bytes"),
        (music_path, b"real music bytes"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _write_jsonl(
        root / "data" / "trace.jsonl",
        [
            {
                "event_type": "tool_call",
                "tool_input": {"cmd": "status --full"},
                "tool_output": {
                    "ok": True,
                    "content": "The orchestrator collected a long real trace span with tool arguments, observations, state deltas, retry decisions, and final verification evidence.",
                },
            }
        ],
    )
    _write_jsonl(root / "data" / "text.jsonl", [{"prompt": "Explain the run.", "answer": "The run trained across real records."}])
    _write_jsonl(root / "data" / "code.jsonl", [{"content": "def ok():\n    return True\n"}])
    _write_jsonl(root / "data" / "images.jsonl", [{"image": str(image_path), "text": "A real image fixture."}])
    _write_jsonl(root / "data" / "videos.jsonl", [{"id": "video-fixture-1", "video": str(video_path), "caption": "A real video fixture."}])
    _write_jsonl(root / "data" / "music.jsonl", [{"source_id": "music-fixture-1", "music": str(music_path), "caption": "A real music fixture."}])
    return {
        "profile_name": "unit",
        "work_dir": str(root / "weights"),
        "modalities": {name: {"enabled": True} for name in orch.DEFAULT_STAGE_ORDER},
        "real_sources": {
            "trace_jsonl": ["data/trace.jsonl"],
            "text_jsonl": ["data/text.jsonl"],
            "code_jsonl": ["data/code.jsonl"],
            "image_jsonl": ["data/images.jsonl"],
            "video_jsonl": ["data/videos.jsonl"],
            "music_jsonl": ["data/music.jsonl"],
            "audio_roots": ["data/audio"],
            "video_roots": ["data/video"],
            "music_roots": ["data/music"],
            "media_roots": [],
        },
        "training_plan": {
            "max_records_per_modality": 2,
            "min_records_per_modality": 1,
            "artifact_token_count": {"image": 4, "video": 4, "audio": 4, "music": 4, "tool": 4, "long_context": 4},
            "max_hash_bytes": 1024,
            "max_media_bytes": 1024 * 1024,
            "min_media_bytes": 1,
            "text_token_limit": 64,
            "target_text_chars": 256,
        },
        "learning_checks": {"min_loss_points": 2, "min_relative_loss_drop": 0.001},
        "record_contracts": {
            "training_record_required_fields": [
                "record_id",
                "source_id",
                "source_date",
                "modalities",
                "split",
                "quality_score",
                "contamination_status",
                "payload_sha256",
                "artifact_refs",
            ]
        },
    }


def test_real_corpus_builder_covers_modalities(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    orch.validate_profile(profile)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    assert manifest["records"] >= 8
    for modality in orch.DEFAULT_STAGE_ORDER:
        assert manifest["modalities"][modality] >= 1
        assert Path(manifest["per_modality_jsonl"][modality]).exists()


def test_media_records_include_ledger_token_ids(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    image_rows = list(orch.iter_jsonl(manifest["per_modality_jsonl"]["image"]))
    assert image_rows
    assert image_rows[0]["artifact_refs"][0]["hash_scope"] == "full"
    assert max(image_rows[0]["token_ids"]) < orch.DEFAULT_LEDGER.vocab_size
    assert image_rows[0]["artifact_refs"][0]["byte_size"] > 0
    assert image_rows[0]["artifact_refs"][0]["created_at"]


def test_curated_records_satisfy_required_training_contract_fields(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    required = set(profile["record_contracts"]["training_record_required_fields"])
    rows = list(orch.iter_jsonl(manifest["curated_jsonl"]))
    assert rows
    for row in rows:
        assert required.issubset(row.keys())
        assert row["source_id"]
        assert row["split"] in {"train", "eval", "test"}
        assert isinstance(row["quality_score"], float)
        assert row["contamination_status"]


def test_video_and_music_jsonl_manifests_are_ingested(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    video_rows = list(orch.iter_jsonl(manifest["curated_jsonl"]))
    assert any(row.get("modality") == "video" and row.get("source_id") == "video-fixture-1" for row in video_rows)
    assert any(row.get("modality") == "music" and row.get("source_id") == "music-fixture-1" for row in video_rows)


def test_curation_manifests_and_posttraining_exports_are_written(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    cleaned = json.loads(Path(manifest["cleaned_dataset_manifest"]).read_text(encoding="utf-8"))
    blend = json.loads(Path(manifest["dataset_blend_manifest"]).read_text(encoding="utf-8"))
    artifacts = list(orch.iter_jsonl(manifest["artifact_manifest_jsonl"]))
    source_files = list(orch.iter_jsonl(manifest["source_files_manifest_jsonl"]))
    posttraining = manifest["posttraining_curation_exports"]
    assert cleaned["status"] == "passed"
    assert cleaned["missing_required_field_counts"] == {}
    assert blend["records"]["train"] == manifest["records"]
    assert artifacts
    assert source_files
    assert Path(posttraining["manifest"]).exists()
    assert posttraining["counts"]["sft"] == manifest["records"]
    assert posttraining["counts"]["reward"] == manifest["records"]
    assert posttraining["counts"]["rlvr"] == manifest["records"]


def test_deterministic_splits_are_repeatable():
    rows = [{"record_id": f"row-{index}", "modality": "text", "payload_sha256": f"sha-{index}"} for index in range(20)]
    plan = {"eval_holdout_ratio": 0.10, "test_holdout_ratio": 0.10}
    first = orch.assign_deterministic_splits(rows, "text", plan)
    second = orch.assign_deterministic_splits(list(reversed(rows)), "text", plan)
    assert {key: [row["record_id"] for row in value] for key, value in first.items()} == {
        key: [row["record_id"] for row in value] for key, value in second.items()
    }
    assert {key: len(value) for key, value in first.items()} == {"train": 16, "eval": 2, "test": 2}


def test_split_bucket_counts_tiny_modalities():
    assert orch.split_bucket_counts(0, 0.10, 0.10) == {"train": 0, "eval": 0, "test": 0}
    assert orch.split_bucket_counts(1, 0.10, 0.10) == {"train": 1, "eval": 0, "test": 0}
    assert orch.split_bucket_counts(2, 0.10, 0.10) == {"train": 1, "eval": 0, "test": 1}
    assert orch.split_bucket_counts(3, 0.10, 0.10) == {"train": 1, "eval": 1, "test": 1}


def test_manifest_keeps_train_compatibility_and_split_paths(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    assert manifest["eval_all_jsonl"]
    assert manifest["test_all_jsonl"]
    for modality in orch.DEFAULT_STAGE_ORDER:
        split_paths = manifest["per_modality_split_jsonl"][modality]
        assert manifest["per_modality_jsonl"][modality] == split_paths["train"]
        assert Path(split_paths["train"]).exists()
        assert Path(split_paths["eval"]).exists()
        assert Path(split_paths["test"]).exists()


def test_learning_report_requires_loss_drop():
    passed = orch.learning_report([10.0, 9.9, 9.8], min_relative_drop=0.001, min_points=2)
    failed = orch.learning_report([10.0, 10.1, 10.2], min_relative_drop=0.001, min_points=2)
    assert passed["status"] == "passed"
    assert failed["status"] == "failed"


def _runtime_args(**overrides):
    values = {
        "distributed": "",
        "nproc_per_node": 0,
        "precision": "",
        "init_dtype": "",
        "optimizer": "",
        "optimizer_in_backward": False,
        "optimizer_in_backward_update": "",
        "optimizer_in_backward_grad_clip": 0.0,
        "optimizer_in_backward_clip_mode": "",
        "optimizer_in_backward_adafactor_chunk_rows": 0,
        "optimizer_in_backward_adafactor_clip_threshold": 0.0,
        "optimizer_in_backward_adafactor_decay_rate": 0.0,
        "optimizer_in_backward_adafactor_eps1": 0.0,
        "rank_device_map": "",
        "activation_checkpointing": False,
        "cpu_offload": False,
        "fake_quant_chunk_rows": 0,
        "fake_quant_max_full_elements": 0,
        "placement": "",
        "placement_devices": "",
        "placement_layer_counts": "",
        "placement_head_device": -1,
        "placement_schedule": "",
        "pipeline_microbatches": 0,
        "pipeline_stage_schedule": "",
        "pipeline_async_streams": None,
        "allow_verifier_preset": False,
        "heldout_max_records_per_file": 0,
        "benchmark_max_records_per_file": 0,
        "heldout_sample_loss_timeout_seconds": 0,
        "benchmark_sample_loss_timeout_seconds": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_pipeline_stage_launcher_uses_torchrun_without_dense_only_flags():
    cfg = {
        "training_plan": {
            "distributed_training": {
                "mode": "pipeline_stage",
                "nproc_per_node": 3,
                "rank_device_map": ["0", "1", "2"],
                "placement_devices": ["0", "1", "2"],
                "placement_head_device": 2,
                "placement_layer_counts": [16, 16, 32],
                "precision": "fp16",
                "init_dtype": "fp16",
                "optimizer": "adafactor",
                "optimizer_in_backward": True,
                "optimizer_in_backward_update": "lowmem_adafactor",
                "pipeline_stage_schedule": "gpipe",
                "pipeline_microbatches": 1,
                "activation_checkpointing": True,
            }
        }
    }
    args = _runtime_args()
    cmd = orch.pretrain_launcher(cfg, args)
    cmd.extend(["--preset", "omnicoder2026_20b_1m", "--data", "train.jsonl", "--out", "ckpt"])
    orch.append_pretrain_runtime_args(cmd, cfg, args)
    assert "omnicoder.training.pipeline_pretrain_2026_dense" in cmd
    assert "--distributed" not in cmd
    assert "--device" not in cmd
    assert "--aux_probe" not in cmd
    assert cmd[cmd.index("--nproc_per_node") + 1] == "3"
    assert cmd[cmd.index("--rank_device_map") + 1] == "0,1,2"
    assert cmd[cmd.index("--placement_layer_counts") + 1] == "16,16,32"
    assert cmd[cmd.index("--pipeline_schedule") + 1] == "gpipe"
    assert cmd[cmd.index("--pipeline_microbatches") + 1] == "1"


def test_checkpoint_complete_marker_supports_sharded_directories(tmp_path):
    single = tmp_path / "native.pt"
    single.write_bytes(b"checkpoint")
    Path(str(single) + ".complete.json").write_text("{}", encoding="utf-8")
    assert orch.checkpoint_is_complete(single)
    assert orch.checkpoint_complete_marker(single).name == "native.pt.complete.json"

    truncated = tmp_path / "pipeline_truncated"
    truncated.mkdir()
    (truncated / ".complete.json").write_text("{}", encoding="utf-8")
    (truncated / "manifest.json").write_text(json.dumps({"rank_files": ["rank00000.pt"]}), encoding="utf-8")
    (truncated / "rank00000.pt").write_bytes(b"checkpoint")
    (truncated / "rank00000.pt.complete.json").write_text("{}", encoding="utf-8")
    assert not orch.checkpoint_is_complete(truncated)

    sharded = tmp_path / "pipeline_dir"
    sharded.mkdir()
    (sharded / ".complete.json").write_text("{}", encoding="utf-8")
    (sharded / "manifest.json").write_text(
        json.dumps({"world_size": 3, "rank_files": ["rank00000.pt", "rank00001.pt", "rank00002.pt"]}),
        encoding="utf-8",
    )
    for rank in range(3):
        rank_path = sharded / f"rank{rank:05d}.pt"
        rank_path.write_bytes(b"checkpoint")
        Path(str(rank_path) + ".complete.json").write_text("{}", encoding="utf-8")
    assert orch.checkpoint_is_complete(sharded, expected_world_size=3)
    assert not orch.checkpoint_is_complete(sharded, expected_world_size=4)
    assert orch.checkpoint_complete_marker(sharded) == sharded / ".complete.json"


def test_pipeline_checkpoint_sample_loss_uses_distributed_eval(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pipeline_ckpt"
    checkpoint.mkdir()
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(eval_path, [{"text": "hello world", "modality": "text"}])
    cfg = {
        "training_plan": {
            "fake_quant": True,
            "distributed_training": {
                "mode": "pipeline_stage",
                "nproc_per_node": 3,
                "rank_device_map": ["0", "1", "2"],
                "placement_layer_counts": [16, 16, 32],
                "precision": "fp16",
                "init_dtype": "fp16",
                "pipeline_stage_schedule": "gpipe",
                "pipeline_microbatches": 1,
                "fake_quant_chunk_rows": 64,
                "fake_quant_max_full_elements": 1024,
            }
        }
    }
    args = _runtime_args()
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], log_path: Path, timeout_seconds: int = 0) -> int:
        commands.append(cmd)
        assert timeout_seconds == 3600
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"overall": {"avg_loss": 1.0}}), encoding="utf-8")
        return 0

    monkeypatch.setattr(orch, "run_command", fake_run_command)
    result = orch.run_sample_loss_eval(
        checkpoint,
        "text",
        {"eval": str(eval_path)},
        tmp_path,
        cfg=cfg,
        args=args,
        preset="omnicoder2026_20b_1m",
        device="cuda",
        seq_len=128,
    )
    cmd = commands[0]
    assert "omnicoder.eval.pipeline_sample_loss_2026" in cmd
    assert cmd[cmd.index("--nproc_per_node") + 1] == "3"
    assert cmd[cmd.index("--rank_device_map") + 1] == "0,1,2"
    assert cmd[cmd.index("--placement_layer_counts") + 1] == "16,16,32"
    assert cmd[cmd.index("--max-records-per-file") + 1] == "32"
    assert "--require_target_contract" in cmd
    assert "--placement-devices" not in cmd
    assert "--placement-head-device" not in cmd
    assert "--activation-checkpointing" not in cmd
    assert result["returncode"] == 0


def test_live_posttraining_runs_native_reward_replay_not_dry_run(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile["reinforcement_learning"] = {
        "enabled": True,
        "offline_reward_replay": {
            "inputs": [str(tmp_path / "out" / "agentic_tool_training_2026" / "tool_sft.jsonl")],
            "algorithms_represented": ["reward_weighted_sft_replay"],
        },
    }
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    manifest = orch.build_real_corpus(profile, tmp_path / "out")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], log_path: Path) -> int:
        commands.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if "omnicoder.training.posttrain_bridge_2026" in cmd:
            manifest_path = Path(cmd[cmd.index("--manifest") + 1])
            bridge_dir = Path(cmd[cmd.index("--out_dir") + 1])
            out_path = bridge_dir / "checkpoints" / "sft_live_replay.pt"
            replay_log = bridge_dir / "logs" / "sft_live_replay.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            replay_log.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"replayed")
            replay_log.write_text('{"loss": 2.0}\n{"loss": 1.5}\n', encoding="utf-8")
            orch.write_json(
                manifest_path,
                {
                    "status": "live_training_passed",
                    "execution": {
                        "status": "passed",
                        "executor": "reward_replay_2026",
                        "returncode": 0,
                        "checkpoint": str(out_path),
                        "loss_log": str(replay_log),
                    },
                },
            )
        return 0

    monkeypatch.setattr(orch, "run_command", fake_run_command)
    args = argparse.Namespace(
        live_posttraining=True,
        preset="ledger_probe",
        device="cpu",
        seq_len=16,
        batch_size=1,
        posttrain_steps=2,
        posttrain_lr=1e-6,
        posttrain_max_records=0,
    )
    result = orch.run_posttraining_stages(profile, tmp_path / "out", {"status": "passed", "final_checkpoint": str(checkpoint)}, args)
    assert result["status"] == "passed"
    assert result["mode"] == "posttrain_bridge_live_optimizer"
    assert result["final_checkpoint"].endswith("sft_live_replay.pt")
    bridge_cmd = next(cmd for cmd in commands if "omnicoder.training.posttrain_bridge_2026" in cmd)
    assert "--dry_run" not in bridge_cmd
    assert not any("omnicoder.training.reward_replay_2026" in cmd for cmd in commands)


def test_live_posttraining_runs_pipeline_reward_replay_for_sharded_checkpoint(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile["training_plan"]["distributed_training"] = {
        "mode": "pipeline_stage",
        "nproc_per_node": 3,
        "rank_device_map": ["0", "1", "2"],
        "placement_layer_counts": [16, 16, 32],
        "pipeline_stage_schedule": "gpipe",
        "pipeline_microbatches": 1,
    }
    profile["reinforcement_learning"] = {
        "enabled": True,
        "offline_reward_replay": {
            "inputs": [str(tmp_path / "out" / "agentic_tool_training_2026" / "tool_sft.jsonl")],
            "algorithms_represented": ["reward_weighted_sft_replay"],
        },
    }
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    orch.build_real_corpus(profile, tmp_path / "out")
    checkpoint = tmp_path / "pipeline_checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".complete.json").write_text("{}", encoding="utf-8")
    (checkpoint / "manifest.json").write_text(json.dumps({"world_size": 3, "rank_files": ["rank00000.pt", "rank00001.pt", "rank00002.pt"]}), encoding="utf-8")
    for rank in range(3):
        rank_path = checkpoint / f"rank{rank:05d}.pt"
        rank_path.write_bytes(b"checkpoint")
        Path(str(rank_path) + ".complete.json").write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], log_path: Path) -> int:
        commands.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if "omnicoder.training.posttrain_bridge_2026" in cmd:
            manifest_path = Path(cmd[cmd.index("--manifest") + 1])
            orch.write_json(
                manifest_path,
                {
                    "status": "live_optimizer_deferred",
                    "execution": {"status": "deferred", "executor": "distributed_pipeline_reward_replay"},
                },
            )
        if "omnicoder.training.pipeline_pretrain_2026_dense" in cmd:
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.mkdir(parents=True, exist_ok=True)
            orch.write_json(out_path / ".complete.json", {"status": "complete"})
            orch.write_json(out_path / "manifest.json", {"world_size": 3, "rank_files": ["rank00000.pt", "rank00001.pt", "rank00002.pt"]})
            for rank in range(3):
                rank_path = out_path / f"rank{rank:05d}.pt"
                rank_path.write_bytes(b"checkpoint")
                Path(str(rank_path) + ".complete.json").write_text("{}", encoding="utf-8")
            replay_log = Path(cmd[cmd.index("--log_file") + 1])
            replay_log.parent.mkdir(parents=True, exist_ok=True)
            replay_log.write_text('{"loss": 3.0}\n{"loss": 2.5}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(orch, "run_command", fake_run_command)
    args = argparse.Namespace(
        live_posttraining=True,
        preset="ledger_probe",
        device="cpu",
        seq_len=16,
        batch_size=1,
        posttrain_steps=2,
        posttrain_lr=1e-6,
        posttrain_max_records=0,
        nproc_per_node=3,
        distributed="pipeline_stage",
        rank_device_map="0,1,2",
        placement_layer_counts="16,16,32",
        pipeline_stage_schedule="gpipe",
        pipeline_microbatches=1,
        precision="fp16",
        init_dtype="fp16",
        optimizer="adafactor",
        optimizer_in_backward=False,
        optimizer_in_backward_update="",
        optimizer_in_backward_grad_clip=0.0,
        optimizer_in_backward_clip_mode="",
        optimizer_in_backward_adafactor_chunk_rows=0,
        optimizer_in_backward_adafactor_clip_threshold=0.0,
        optimizer_in_backward_adafactor_decay_rate=0.0,
        optimizer_in_backward_adafactor_eps1=0.0,
        activation_checkpointing=False,
        fake_quant=False,
        fake_quant_chunk_rows=0,
        fake_quant_max_full_elements=0,
    )
    result = orch.run_posttraining_stages(profile, tmp_path / "out", {"status": "passed", "final_checkpoint": str(checkpoint)}, args)
    assert result["status"] == "passed"
    assert result["stages"][0]["mode"] == "distributed_pipeline_reward_replay"
    assert result["final_checkpoint"].endswith("01_reward_weighted_sft_replay_pipeline")
    pipeline_cmd = next(cmd for cmd in commands if "omnicoder.training.pipeline_pretrain_2026_dense" in cmd)
    assert pipeline_cmd[pipeline_cmd.index("--resume") + 1] == str(checkpoint)
    assert pipeline_cmd[pipeline_cmd.index("--nproc_per_node") + 1] == "3"
    bridge_cmd = next(cmd for cmd in commands if "omnicoder.training.posttrain_bridge_2026" in cmd)
    assert "--defer_optimizer" in bridge_cmd
    assert "--dry_run" not in bridge_cmd
    assert not any("omnicoder.training.reward_replay_2026" in cmd for cmd in commands)


def test_live_posttraining_stops_after_failed_pipeline_replay(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile["training_plan"]["distributed_training"] = {
        "mode": "pipeline_stage",
        "nproc_per_node": 3,
        "rank_device_map": ["0", "1", "2"],
        "placement_layer_counts": [16, 16, 32],
        "pipeline_stage_schedule": "gpipe",
        "pipeline_microbatches": 1,
    }
    profile["reinforcement_learning"] = {
        "enabled": True,
        "offline_reward_replay": {
            "inputs": [str(tmp_path / "out" / "agentic_tool_training_2026" / "tool_sft.jsonl")],
            "algorithms_represented": ["reward_weighted_sft_replay", "dpo_pair_replay"],
        },
        "stop_on_posttrain_failure": True,
    }
    monkeypatch.setattr(orch, "repo_root", lambda: tmp_path)
    orch.build_real_corpus(profile, tmp_path / "out")
    checkpoint = tmp_path / "pipeline_checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".complete.json").write_text("{}", encoding="utf-8")
    (checkpoint / "manifest.json").write_text(json.dumps({"world_size": 3, "rank_files": ["rank00000.pt", "rank00001.pt", "rank00002.pt"]}), encoding="utf-8")
    for rank in range(3):
        rank_path = checkpoint / f"rank{rank:05d}.pt"
        rank_path.write_bytes(b"checkpoint")
        Path(str(rank_path) + ".complete.json").write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], log_path: Path) -> int:
        commands.append(cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if "omnicoder.training.posttrain_bridge_2026" in cmd:
            manifest_path = Path(cmd[cmd.index("--manifest") + 1])
            orch.write_json(
                manifest_path,
                {
                    "status": "live_optimizer_deferred",
                    "execution": {"status": "deferred", "executor": "distributed_pipeline_reward_replay"},
                },
            )
            return 0
        if "omnicoder.training.pipeline_pretrain_2026_dense" in cmd:
            replay_log = Path(cmd[cmd.index("--log_file") + 1])
            replay_log.parent.mkdir(parents=True, exist_ok=True)
            replay_log.write_text('{"loss": 3.0}\n{"loss": 2.9}\n', encoding="utf-8")
            return 1
        return 0

    monkeypatch.setattr(orch, "run_command", fake_run_command)
    args = argparse.Namespace(
        live_posttraining=True,
        preset="ledger_probe",
        device="cpu",
        seq_len=16,
        batch_size=1,
        posttrain_steps=2,
        posttrain_lr=1e-6,
        posttrain_max_records=0,
        nproc_per_node=3,
        distributed="pipeline_stage",
        rank_device_map="0,1,2",
        placement_layer_counts="16,16,32",
        pipeline_stage_schedule="gpipe",
        pipeline_microbatches=1,
        precision="fp16",
        init_dtype="fp16",
        optimizer="adafactor",
        optimizer_in_backward=False,
        optimizer_in_backward_update="",
        optimizer_in_backward_grad_clip=0.0,
        optimizer_in_backward_clip_mode="",
        optimizer_in_backward_adafactor_chunk_rows=0,
        optimizer_in_backward_adafactor_clip_threshold=0.0,
        optimizer_in_backward_adafactor_decay_rate=0.0,
        optimizer_in_backward_adafactor_eps1=0.0,
        activation_checkpointing=False,
        fake_quant=False,
        fake_quant_chunk_rows=0,
        fake_quant_max_full_elements=0,
    )
    result = orch.run_posttraining_stages(profile, tmp_path / "out", {"status": "passed", "final_checkpoint": str(checkpoint)}, args)
    assert result["status"] == "failed"
    assert result["final_checkpoint"] == str(checkpoint)
    assert result["stages"][0]["status"] == "failed"
    assert result["stages"][1] == {
        "requested_algorithm": "dpo_pair_replay",
        "status": "skipped",
        "reason": "previous_posttraining_stage_failed",
        "blocked_by": "reward_weighted_sft_replay",
    }
    bridge_commands = [cmd for cmd in commands if "omnicoder.training.posttrain_bridge_2026" in cmd]
    assert len(bridge_commands) == 1


def test_posttraining_checkpoint_retention_prunes_old_complete_dirs(tmp_path):
    root = tmp_path / "out" / "checkpoints" / "posttrain"
    keep = root / "03_keep_pipeline"
    old_one = root / "01_old_pipeline"
    old_two = root / "02_old_pipeline"
    incomplete = root / "00_incomplete_pipeline"
    for index, path in enumerate((incomplete, old_one, old_two, keep), 1):
        path.mkdir(parents=True)
        (path / "rank00000.pt").write_bytes(b"checkpoint")
        if path is not incomplete:
            orch.write_json(path / ".complete.json", {"status": "complete"})
        os_time = 1_700_000_000 + index
        path.touch()
        os.utime(path, (os_time, os_time))

    report = orch.prune_posttrain_checkpoints(
        tmp_path / "out",
        [keep],
        {"enabled": True, "keep_last_successful": 1, "delete_incomplete": True},
    )
    assert report["status"] == "passed"
    assert keep.exists()
    assert not old_one.exists()
    assert not old_two.exists()
    assert not incomplete.exists()


def test_run_full_summarizes_all_major_training_phases(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    out_dir = tmp_path / "out"
    final_checkpoint = out_dir / "checkpoints" / "99_final_all_modality_finetune.pt"

    monkeypatch.setattr(
        orch,
        "build_real_corpus",
        lambda loaded, out: {
            "status": "ok",
            "records": 8,
            "curated_jsonl": str(out_dir / "jsonl" / "curated_records.jsonl"),
            "train_all_jsonl": str(out_dir / "jsonl" / "train_all_modalities.jsonl"),
            "eval_all_jsonl": str(out_dir / "jsonl" / "eval_all_modalities.jsonl"),
            "test_all_jsonl": str(out_dir / "jsonl" / "test_all_modalities.jsonl"),
        },
    )
    monkeypatch.setattr(
        orch,
        "run_training_stages",
        lambda loaded, manifest, out, args: {"status": "passed", "final_checkpoint": str(out / "checkpoints" / "08_long_context.pt")},
    )
    monkeypatch.setattr(
        orch,
        "run_distillation_curriculum_stage",
        lambda loaded, manifest, out, checkpoint, args: {"status": "passed", "final_checkpoint": str(out / "checkpoints" / "09_distillation_replay.pt")},
    )

    def fake_posttraining(loaded, out, training, args):
        assert args.live_posttraining is True
        return {"status": "passed", "final_checkpoint": str(out / "checkpoints" / "posttrain" / "01_reward_weighted_sft_replay.pt")}

    monkeypatch.setattr(orch, "run_posttraining_stages", fake_posttraining)
    monkeypatch.setattr(
        orch,
        "run_final_finetune_stage",
        lambda loaded, manifest, out, checkpoint, args: {"status": "passed", "final_checkpoint": str(final_checkpoint)},
    )
    monkeypatch.setattr(
        orch,
        "run_checkpoint_benchmark_gate",
        lambda loaded, manifest, out, checkpoint, phase, args: {"status": "passed", "phase": phase, "checkpoint": str(checkpoint)},
    )
    args = argparse.Namespace(
        profile=str(profile_path),
        out_dir=str(out_dir),
        steps_per_stage=0,
        seq_len=0,
        batch_size=0,
        lr=0.0,
        preset="",
        device="",
        fake_quant=False,
        resume_checkpoint="",
    )
    result = orch.run_full(args)
    assert result["status"] == "passed"
    assert result["pretraining"]["status"] == "passed"
    assert result["distillation"]["status"] == "passed"
    assert result["posttraining"]["status"] == "passed"
    assert result["finetune"]["status"] == "passed"
    assert result["benchmark_gates"]["status"] == "passed"
    assert result["final_checkpoint"] == str(final_checkpoint)
    assert result["artifacts"]["final_checkpoint"] == str(final_checkpoint)
    assert (out_dir / "full_training_summary.json").exists()


def test_run_real_cli_wires_live_posttraining_args(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    captured: dict[str, argparse.Namespace] = {}

    monkeypatch.setattr(orch, "build_real_corpus", lambda loaded, out: {"status": "ok"})
    monkeypatch.setattr(orch, "run_training_stages", lambda loaded, manifest, out, args: {"status": "passed", "final_checkpoint": str(tmp_path / "ckpt.pt")})

    def fake_posttraining(loaded, out, training, args):
        captured["args"] = args
        return {"status": "passed", "mode": "native_reward_replay", "final_checkpoint": training["final_checkpoint"]}

    monkeypatch.setattr(orch, "run_posttraining_stages", fake_posttraining)
    code = orch.main(
        [
            "--profile",
            str(profile_path),
            "--out-dir",
            str(tmp_path / "out"),
            "run-real",
            "--live-posttraining",
            "--posttrain-steps",
            "2",
            "--posttrain-lr",
            "0.000001",
        ]
    )
    assert code == 0
    assert captured["args"].live_posttraining is True
    assert captured["args"].posttrain_steps == 2


def test_adaptive_mixture_plan_reweights_sparse_modalities(tmp_path):
    profile = _profile(tmp_path)
    profile["adaptive_training_scheduler_2026"] = {
        "enabled": True,
        "sample_weight_bounds": [0.25, 4.0],
        "context_ladder": [8192, 32768, 1048576],
        "promotion_gates": {"require_nonzero_all_modalities": True},
    }
    out_dir = tmp_path / "out"
    manifests = out_dir / "manifests"
    manifests.mkdir(parents=True)
    curation_manifest = manifests / "curation_manifest.json"
    curation_manifest.write_text(
        json.dumps(
            {
                "records": 18,
                "modalities": {"text": 12, "code": 4, "tool": 2, "image": 0, "video": 0, "audio": 0, "music": 0, "long_context": 0},
            }
        ),
        encoding="utf-8",
    )
    external_manifest = tmp_path / "external.json"
    external_manifest.write_text(json.dumps({"records": {"train": 6}, "modalities": {"image": 2, "video": 1, "audio": 1, "music": 1, "long_context": 1}}), encoding="utf-8")

    plan = orch.build_adaptive_mixture_plan(
        profile,
        out_dir,
        curation_manifest_path=curation_manifest,
        external_manifest_path=external_manifest,
        agentic_manifest_path=tmp_path / "missing_agentic.json",
        teacher_manifest_path=tmp_path / "missing_teacher.json",
    )

    weights = {row["stage"]: row["weight"] for row in plan["stage_weights"]}
    assert plan["status"] == "passed"
    assert weights["video"] > weights["text"]
    assert weights["long_context"] > weights["text"]
    assert plan["context_schedule"][-1]["context_length"] == 1048576
    assert Path(plan["path"]).exists()


def test_repo_training_profile_enables_adaptive_scheduler() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "training_orchestration_2026.json").read_text(encoding="utf-8"))
    scheduler = profile["adaptive_training_scheduler_2026"]

    assert scheduler["enabled"] is True
    assert scheduler["context_ladder"][-1] == 1048576
    assert "modality_coverage_deficit" in scheduler["signals"]
    assert profile["artifacts"]["mixture_plan"].endswith("mixture_plan.json")


def test_configured_reportable_roots_ignore_core_benchmark_ids(tmp_path) -> None:
    benchmark_profile = tmp_path / "benchmark_profile.json"
    benchmark_profile.write_text(
        json.dumps(
            {
                "reportable_core_25": ["reasoning_arc_agi3_2026"],
                "reportable_task_roots": {
                    "reasoning_arc_agi3_2026": ["data/eval/reportable_2026/arc_agi3_authorized.jsonl"]
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "reportable_core_25": ["should_not_be_a_path"],
        "benchmark_gates": {"reportable_core_25": ["also_not_a_path"]},
    }

    roots, sources = orch.configured_reportable_roots(cfg, str(benchmark_profile))

    assert "should_not_be_a_path" not in roots
    assert "also_not_a_path" not in roots
    assert "reasoning_arc_agi3_2026" not in roots
    assert roots == ["data/eval/reportable_2026/arc_agi3_authorized.jsonl"]
    assert sources == [f"{benchmark_profile}.reportable_task_roots"]


def test_reportable_prediction_seed_uses_explicit_model_outputs_only(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_jsonl(
        tasks,
        [
            {
                "benchmark_id": "reasoning_arc_agi3_2026",
                "task_id": "oracle-only",
                "success": True,
                "actions": 3,
            },
            {
                "benchmark_id": "coding_swe_bench_live_2026",
                "task_id": "model-patch",
                "model_patch": "diff --git a/a.py b/a.py",
            },
        ],
    )

    result = orch.write_reportable_prediction_seed([tasks], tmp_path / "predictions.jsonl")
    rows = list(orch.iter_jsonl(result["path"]))

    assert result["records"] == 1
    assert rows[0]["task_id"] == "model-patch"
    assert rows[0]["prediction"] == "diff --git a/a.py b/a.py"


def test_training_data_eval_layers_do_not_import_forbidden_database_libraries():
    forbidden = (
        "pydantic",
        "sqlalchemy",
        "sqlite3",
        "chromadb",
        "BaseModel",
        "create_engine",
        "sessionmaker",
        "declarative_base",
    )
    root = Path(orch.__file__).resolve().parents[1]
    scoped_dirs = [root / "training", root / "data_factory", root / "eval"]
    hits: list[str] = []
    for scoped in scoped_dirs:
        for path in scoped.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in forbidden:
                if marker in text:
                    hits.append(f"{path}:{marker}")
    assert hits == []

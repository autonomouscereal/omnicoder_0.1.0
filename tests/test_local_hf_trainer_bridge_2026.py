from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnicoder.training import full_harness_2026
from omnicoder.training import local_hf_trainer_bridge_2026 as bridge


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _args(train: Path, out: Path, **overrides) -> argparse.Namespace:
    payload = {
        "command": "sft",
        "backend": "unsloth",
        "model": "Qwen/Qwen3-4B",
        "train_jsonl": str(train),
        "eval_jsonl": None,
        "out_dir": str(out),
        "manifest": str(out / "manifest.json"),
        "max_seq_len": 4096,
        "max_steps": 4,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_steps": 10,
        "eval_steps": 10,
        "logging_steps": 1,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "load_in_4bit": True,
        "dtype": "auto",
        "packing": True,
        "assistant_only_loss": True,
        "unsloth_tiled_mlp": False,
        "unsloth_gradient_checkpointing": "unsloth",
        "device_map": "",
        "host_gpu_ids": "",
        "protected_gpus": "0,4,6",
        "allow_protected_gpus": False,
        "allow_nontrain_rows": False,
        "allow_missing_backend": False,
        "check_deps": False,
        "dry_run": True,
        "limit": 0,
        "eval_limit": 0,
        "seed": 3407,
        "save_gguf": False,
        "gguf_quantization": "q4_k_m",
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_dataset_inspector_rejects_holdout_reportable_and_contaminated_rows(tmp_path: Path) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"split": "train", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}]},
            {"split": "eval_holdout", "prompt": "do not train", "completion": "leak"},
            {"split": "train", "prompt": "secret", "completion": "bad", "secret_rejected": True},
            {"split": "train", "prompt": "contam", "completion": "bad", "contamination": {"contaminated": True}},
            {"split": "train", "prompt": "bench", "completion": "bad", "reportable_task": True},
        ],
    )

    inspected = bridge.inspect_sft_dataset(train)

    assert inspected["accepted_records"] == 1
    assert inspected["rejected"]["non_train_split"] == 1
    assert inspected["rejected"]["secret_rejected"] == 1
    assert inspected["rejected"]["contaminated"] == 1
    assert inspected["rejected"]["reportable_task"] == 1


def test_dry_run_manifest_allows_missing_unsloth_but_records_backend(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [{"split": "train", "prompt": "solve", "completion": "ok"}])
    args = _args(train, tmp_path / "out")

    code, manifest = bridge.execute(args)

    assert code == 0
    assert manifest["status"] in {"dry_run_ok", "dry_run_ok_missing_backend"}
    assert manifest["backend"] == "unsloth"
    assert manifest["dataset"]["accepted_records"] == 1
    assert manifest["benchmark"]["backend"] == "checkpoint-runner"
    assert "local_hf_adapter_runner_2026" in manifest["benchmark"]["checkpoint_runner_command"]
    assert (tmp_path / "out" / "manifest.json").exists()


def test_normalizer_accepts_omnicoder_input_target_training_rows(tmp_path: Path) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "training_bucket": "train",
                "use_policy": "train",
                "input_json": {"messages": [{"role": "user", "content": "Solve 2+2."}]},
                "target_json": {"content": "4"},
                "schema": "omnicoder.real_multimodal_training_2026.v1",
            }
        ],
    )

    rows, rejected, total = bridge.normalize_sft_rows(train)

    assert total == 1
    assert rejected == {}
    assert rows == [{"messages": [{"role": "user", "content": "Solve 2+2."}, {"role": "assistant", "content": "4"}], "metadata": {}}]


def test_live_manifest_fails_closed_when_backend_missing(tmp_path: Path, monkeypatch) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [{"split": "train", "prompt": "solve", "completion": "ok"}])
    args = _args(train, tmp_path / "out", dry_run=False)
    monkeypatch.setattr(bridge, "missing_deps", lambda backend, load_in_4bit: ["unsloth"])

    code, manifest = bridge.execute(args)

    assert code == 2
    assert manifest["status"] == "missing_dependencies"
    assert "unsloth" in manifest["missing_dependencies"]


def test_protected_gpu_overlap_fails_closed(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [{"split": "train", "prompt": "solve", "completion": "ok"}])
    args = _args(train, tmp_path / "out", host_gpu_ids="0")

    code, manifest = bridge.execute(args)

    assert code == 2
    assert manifest["status"] == "protected_gpu_overlap"
    assert manifest["gpu_guard"]["overlap"] == ["0"]


def test_full_harness_can_emit_local_hf_trainer_stage(tmp_path: Path, monkeypatch) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [{"split": "train", "messages": [{"role": "user", "content": "use tool"}]}])
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], log_path: Path, cwd: Path, env: dict | None = None) -> tuple[int, dict]:
        commands.append(cmd)
        manifest = Path(cmd[cmd.index("--manifest") + 1])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"status": "dry_run_ok"}) + "\n", encoding="utf-8")
        return 0, {}

    monkeypatch.setattr(full_harness_2026, "run_command", fake_run_command)
    registry = full_harness_2026.JsonlRunRegistry(tmp_path / "registry")
    run_id = "local-hf-stage"
    registry.create_run("test", "recipe", "profile", "preset", {}, run_id=run_id)
    current = {"sft": train}
    profile = {"local_hf_trainer": {"enabled": True, "backend": "unsloth", "dry_run": True}}

    full_harness_2026.execute_stage(
        "local_hf_trainer",
        run_id,
        registry,
        profile,
        full_harness_2026.ensure_dirs(tmp_path / "run"),
        Path(__file__).resolve().parents[1],
        current,
        dry_run=False,
    )

    assert commands
    cmd = commands[0]
    assert "omnicoder.training.local_hf_trainer_bridge_2026" in cmd
    assert "--dry-run" in cmd
    assert current["local_hf_trainer_manifest"].exists()

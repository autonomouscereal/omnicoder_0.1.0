from __future__ import annotations

import json
from pathlib import Path

import torch

from omnicoder.eval.sample_loss_2026 import _candidate_data_files, compare_baseline, evaluate_files, load_native_checkpoint
from omnicoder.modeling.omnicoder2026 import OmniCoder2026, OmniCoder2026Config


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def _tiny_config() -> dict:
    return {
        "vocab_size": 64,
        "n_layers": 1,
        "d_model": 16,
        "n_heads": 2,
        "head_dim": 8,
        "num_key_value_heads": 1,
        "mlp_dim": 32,
        "max_seq_len": 32,
        "local_window": 16,
        "csa_block_size": 16,
        "csa_top_k_blocks": 2,
        "hca_block_size": 16,
        "latent_dim": 8,
        "rope_dim": 8,
        "sink_tokens": 1,
        "q_lora_rank": 8,
        "o_lora_rank": 8,
        "o_groups": 1,
        "index_head_dim": 8,
        "flow_latent_dim": 8,
        "layer_pattern": ("kda",),
    }


def test_load_native_checkpoint_accepts_config_key(tmp_path):
    cfg = OmniCoder2026Config(**_tiny_config())
    model = OmniCoder2026(cfg)
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": _tiny_config(), "model_state_dict": model.state_dict()}, checkpoint)

    loaded = load_native_checkpoint(checkpoint, "ledger_probe", torch.device("cpu"))

    assert loaded.vocab_size == 64
    assert loaded.max_seq_len == 32


def test_load_native_checkpoint_reshapes_legacy_scalar_residual_scales(tmp_path):
    cfg = OmniCoder2026Config(**_tiny_config())
    model = OmniCoder2026(cfg)
    state = model.state_dict()
    for name, value in list(state.items()):
        if name.endswith("_residual.scale") and value.numel() == 1:
            state[name] = value.reshape(())
    checkpoint = tmp_path / "tiny_legacy_residual.pt"
    torch.save({"config": _tiny_config(), "model_state_dict": state}, checkpoint)

    loaded = load_native_checkpoint(checkpoint, "ledger_probe", torch.device("cpu"))

    assert loaded.vocab_size == 64


def test_load_native_checkpoint_supports_weighted_placement_on_cpu(tmp_path):
    config = {**_tiny_config(), "n_layers": 2, "layer_pattern": ("kda", "csa")}
    cfg = OmniCoder2026Config(**config)
    model = OmniCoder2026(cfg)
    checkpoint = tmp_path / "tiny_weighted.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint)

    loaded = load_native_checkpoint(
        checkpoint,
        "ledger_probe",
        torch.device("cpu"),
        placement="weighted_layers",
        placement_devices="cpu,cpu",
        placement_layer_counts="1,1",
        placement_head_device=0,
    )

    summary = getattr(loaded, "_eval_placement_summary")
    assert summary["mode"] == "weighted_layers"
    assert summary["requested_counts"] == [1, 1]
    assert summary["embed_device"] == "cpu"
    assert loaded(torch.tensor([[1, 2, 3]], dtype=torch.long), return_hidden=False)["logits"].shape[:2] == (1, 3)


def test_sample_loss_aggregates_by_file_and_modality(tmp_path):
    cfg = OmniCoder2026Config(**_tiny_config())
    model = OmniCoder2026(cfg)
    data = tmp_path / "jsonl"
    text_path = data / "train_text.jsonl"
    code_path = data / "train_code.jsonl"
    _write_jsonl(text_path, [{"modality": "text", "token_ids": [2, 3, 4, 5]}, {"modality": "text", "token_ids": [6, 7, 8]}])
    _write_jsonl(code_path, [{"modality": "code", "token_ids": [9, 10, 11, 12]}])

    files = _candidate_data_files([], str(data))
    result = evaluate_files(model, files, seq_len=4, max_records_per_file=1, device=torch.device("cpu"))

    assert result["overall"]["records"] == 2
    assert result["overall"]["tokens"] == 6
    assert result["overall"]["avg_loss"] is not None
    assert result["modalities"]["text"]["records"] == 1
    assert result["modalities"]["code"]["records"] == 1
    assert len(result["files"]) == 2


def test_candidate_data_files_can_exclude_data_dir_aggregate_jsonl(tmp_path):
    data = tmp_path / "jsonl"
    keep_path = data / "train_text.jsonl"
    curated_path = data / "curated_records.jsonl"
    all_modalities_path = data / "nested" / "train_all_modalities.jsonl"
    _write_jsonl(keep_path, [{"token_ids": [2, 3]}])
    _write_jsonl(curated_path, [{"token_ids": [3, 4]}])
    _write_jsonl(all_modalities_path, [{"token_ids": [4, 5]}])

    default_files = _candidate_data_files([], str(data))
    filtered_files = _candidate_data_files([], str(data), exclude_aggregate_jsonl=True)
    explicit_files = _candidate_data_files([str(curated_path)], str(data), exclude_aggregate_jsonl=True)

    assert {path.name for path in default_files} == {
        "curated_records.jsonl",
        "train_all_modalities.jsonl",
        "train_text.jsonl",
    }
    assert filtered_files == [keep_path]
    assert explicit_files == [keep_path, curated_path]


def test_compare_baseline_reports_overall_and_modality_deltas(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline = {
        "overall": {"avg_loss": 2.5},
        "modalities": {
            "code": {"avg_loss": 3.0},
            "image": {"avg_loss": 4.0},
        },
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    result = {
        "overall": {"avg_loss": 2.0},
        "modalities": {
            "code": {"avg_loss": 2.75},
            "text": {"avg_loss": 1.5},
        },
    }

    comparison = compare_baseline(result, baseline_path)

    assert comparison["baseline_path"] == str(baseline_path)
    assert comparison["overall"]["delta_avg_loss"] == -0.5
    assert comparison["modalities"]["code"]["delta_avg_loss"] == -0.25
    assert comparison["modalities"]["text"]["baseline_avg_loss"] is None
    assert comparison["modalities"]["text"]["delta_avg_loss"] is None
    assert comparison["modalities"]["image"]["current_avg_loss"] is None

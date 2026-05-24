from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from omnicoder.eval import fsdp_checkpoint_2026 as fsdp_ckpt
from omnicoder.eval.harness_2026 import hash_file
from omnicoder.eval.sample_loss_2026 import load_native_checkpoint


def _write_rank_local_dir(path: Path, world_size: int = 2) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rank_files = []
    for rank in range(world_size):
        rank_file = path / f"rank{rank:05d}.pt"
        torch.save(
            {
                "format": fsdp_ckpt.FSDP_LOCAL_FORMAT,
                "model_state_dict": {"local_shard": torch.tensor([rank])},
                "rank": rank,
                "world_size": world_size,
                "preset": {"vocab_size": 64, "n_layers": 1, "d_model": 16},
                "train_args": {"fake_quant": False},
            },
            rank_file,
        )
        rank_files.append(rank_file.name)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format": fsdp_ckpt.FSDP_LOCAL_FORMAT,
                "checkpoint_dir": str(path),
                "rank_files": rank_files,
                "world_size": world_size,
                "global_step": 7,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_detects_rank_local_fsdp_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = _write_rank_local_dir(tmp_path / "fsdp_ckpt")

    assert fsdp_ckpt.is_fsdp_rank_local_checkpoint_dir(checkpoint_dir)
    assert fsdp_ckpt.fsdp_world_size(checkpoint_dir) == 2
    assert [path.name for path in fsdp_ckpt.rank_checkpoint_files(checkpoint_dir)] == ["rank00000.pt", "rank00001.pt"]
    assert fsdp_ckpt.fsdp_rank_file(checkpoint_dir, 1).name == "rank00001.pt"


def test_inspect_cli_reports_torchrun_hints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint_dir = _write_rank_local_dir(tmp_path / "fsdp_ckpt", world_size=3)

    assert fsdp_ckpt.main(["inspect", "--checkpoint-dir", str(checkpoint_dir)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["is_fsdp_rank_local"] is True
    assert payload["world_size"] == 3
    assert "torch.distributed.run --nproc_per_node 3" in payload["torchrun_sample_loss_hint"]
    assert payload["format"] == fsdp_ckpt.FSDP_LOCAL_FORMAT


def test_sample_loss_directory_load_fails_with_actionable_torchrun_hint(tmp_path: Path) -> None:
    checkpoint_dir = _write_rank_local_dir(tmp_path / "fsdp_ckpt", world_size=2)

    with pytest.raises(RuntimeError, match="torch.distributed.run --nproc_per_node 2"):
        load_native_checkpoint(checkpoint_dir, "ledger_probe", torch.device("cpu"), dist_backend="gloo")


def test_eval_harness_hashes_fsdp_directory_by_rank_manifest(tmp_path: Path) -> None:
    checkpoint_dir = _write_rank_local_dir(tmp_path / "fsdp_ckpt", world_size=2)

    expected = fsdp_ckpt.checkpoint_fingerprint(checkpoint_dir)

    assert len(expected) == 64
    assert hash_file(str(checkpoint_dir)) == expected

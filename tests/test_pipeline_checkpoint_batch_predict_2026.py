from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnicoder.eval import pipeline_checkpoint_manifest_2026 as checkpoint_manifest


def _write_checkpoint(path: Path, *, ranks: int, manifest_world_size: int | None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {} if manifest_world_size is None else {"world_size": manifest_world_size}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / ".complete.json").write_text("{}", encoding="utf-8")
    for rank in range(ranks):
        (path / f"rank{rank:05d}.pt").write_bytes(b"")
        (path / f"rank{rank:05d}.pt.complete.json").write_text("{}", encoding="utf-8")
    return path


def test_batch_predict_manifest_world_size_accepts_four_rank_checkpoint(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=4)

    manifest = checkpoint_manifest.load_pipeline_manifest(checkpoint)

    assert manifest["world_size"] == 4
    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=0) == 4


def test_batch_predict_cli_world_size_accepts_four_rank_checkpoint_without_manifest_world_size(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=None)

    manifest = checkpoint_manifest.load_pipeline_manifest(checkpoint, expected_world_size=4)

    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=4) == 4
    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=0) == 4


def test_batch_predict_cli_world_size_can_override_stale_manifest_world_size(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=3)

    manifest = checkpoint_manifest.load_pipeline_manifest(checkpoint, expected_world_size=4)

    assert checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=4) == 4
    with pytest.raises(checkpoint_manifest.PipelineCheckpointManifestError, match="expects exactly 3 pipeline shards"):
        checkpoint_manifest.load_pipeline_manifest(checkpoint)


def test_batch_predict_rejects_explicit_world_size_rank_count_mismatch(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=3, manifest_world_size=4)

    with pytest.raises(
        checkpoint_manifest.PipelineCheckpointManifestError,
        match="expects exactly 4 pipeline shards, found 3",
    ):
        checkpoint_manifest.load_pipeline_manifest(checkpoint, expected_world_size=4)


def test_batch_predict_rejects_negative_explicit_world_size(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", ranks=4, manifest_world_size=4)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(
        checkpoint_manifest.PipelineCheckpointManifestError,
        match="explicit world size must be non-negative",
    ):
        checkpoint_manifest.resolve_expected_world_size(checkpoint, manifest, explicit_world_size=-1)


def test_pipeline_eval_parsers_accept_four_rank_world_size_args() -> None:
    pytest.importorskip("torch")
    from omnicoder.eval import pipeline_checkpoint_batch_predict_2026 as batch_predict
    from omnicoder.eval import pipeline_target_token_diagnostics_2026 as target_diagnostics
    from omnicoder.eval import pipeline_token_topk_probe_2026 as topk_probe

    batch_args = batch_predict.build_parser().parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--tasks",
            "tasks.jsonl",
            "--out",
            "predictions.jsonl",
            "--nproc-per-node",
            "4",
        ]
    )
    target_args = target_diagnostics.build_parser().parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--data",
            "data.jsonl",
            "--out",
            "target_diagnostics.json",
            "--expected-world-size",
            "4",
        ]
    )
    topk_args = topk_probe.build_parser().parse_args(
        [
            "--checkpoint",
            "ckpt",
            "--out",
            "topk.json",
            "--nproc-per-node",
            "4",
        ]
    )

    assert batch_args.nproc_per_node == 4
    assert target_args.expected_world_size == 4
    assert topk_args.expected_world_size == 4


def test_target_diagnostics_single_rank_hidden_path_does_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from omnicoder.eval import pipeline_target_token_diagnostics_2026 as target_diagnostics

    class FakeShard:
        def __call__(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.float().unsqueeze(-1)

    def fail_send(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("single-rank target diagnostics must not send to a nonexistent next rank")

    monkeypatch.setattr(target_diagnostics.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(target_diagnostics.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(target_diagnostics.dist, "send", fail_send)

    batch = torch.tensor([[1, 2, 3]], dtype=torch.long)
    hidden = target_diagnostics._pipeline_hidden(
        FakeShard(),
        batch,
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
        d_model=1,
        precision="fp32",
    )

    assert hidden is not None
    assert hidden.shape == (1, 3, 1)

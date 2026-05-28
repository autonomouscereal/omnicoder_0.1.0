from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch

from omnicoder.eval import pipeline_sample_loss_2026 as sample_loss


def test_sample_loss_chunks_overlap_to_preserve_boundary_targets() -> None:
    ids = list(range(10, 20))
    labels = [-100] * len(ids)
    labels[4] = ids[4]

    chunks = sample_loss._chunks_pair(ids, labels, seq_len=4)

    assert chunks[1][0][0] == ids[3]
    assert chunks[1][1][1] == ids[4]


def test_sample_loss_chunks_overlap_repeated_boundaries() -> None:
    ids = list(range(10, 24))
    labels = list(ids)
    labels[0] = -100

    chunks = sample_loss._chunks_pair(ids, labels, seq_len=4)

    assert [chunk[0][0] for chunk in chunks[:4]] == [10, 13, 16, 19]
    assert chunks[2][0] == [16, 17, 18, 19]
    assert chunks[2][1] == [-100, 17, 18, 19]


def test_sample_loss_defaults_to_checkpoint_loss_config(tmp_path) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        '{"train_args":{"lm_loss_chunk_tokens":7,"loss_token_stride":3,"max_loss_tokens_per_sample":5}}',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        lm_loss_chunk_tokens=None,
        loss_token_stride=None,
        max_loss_tokens_per_sample=None,
    )

    sample_loss._apply_checkpoint_loss_defaults(args)

    assert args.lm_loss_chunk_tokens == 7
    assert args.loss_token_stride == 3
    assert args.max_loss_tokens_per_sample == 5


def test_sample_loss_checkpoint_kwargs_prefers_saved_payload_shape(tmp_path) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    torch.save(
        {
            "preset": {"name": "saved_probe", "n_layers": 2, "layer_pattern": ["local", "local"]},
            "model_state_dict": {"embed.weight": torch.zeros(17, 13)},
        },
        checkpoint / "rank00000.pt",
    )

    kwargs, saved_name = sample_loss._checkpoint_model_kwargs(checkpoint, "ledger_probe")

    assert saved_name == "saved_probe"
    assert kwargs["vocab_size"] == 17
    assert kwargs["d_model"] == 13
    assert kwargs["n_layers"] == 2
    assert kwargs["layer_pattern"] == ("local", "local")
    assert kwargs["tie_embeddings"] is False

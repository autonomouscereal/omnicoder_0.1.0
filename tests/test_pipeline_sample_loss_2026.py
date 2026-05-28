from __future__ import annotations

import pytest

pytest.importorskip("torch")

from omnicoder.eval.pipeline_sample_loss_2026 import _chunks_pair


def test_sample_loss_chunks_overlap_to_preserve_boundary_targets() -> None:
    ids = list(range(10, 20))
    labels = [-100] * len(ids)
    labels[4] = ids[4]

    chunks = _chunks_pair(ids, labels, seq_len=4)

    assert chunks[1][0][0] == ids[3]
    assert chunks[1][1][1] == ids[4]


def test_sample_loss_chunks_overlap_repeated_boundaries() -> None:
    ids = list(range(10, 24))
    labels = list(ids)
    labels[0] = -100

    chunks = _chunks_pair(ids, labels, seq_len=4)

    assert [chunk[0][0] for chunk in chunks[:4]] == [10, 13, 16, 19]
    assert chunks[2][0] == [16, 17, 18, 19]
    assert chunks[2][1] == [-100, 17, 18, 19]

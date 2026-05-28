from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnicoder.eval.native_media_trainability_2026 import CORE_MODALITIES, run_native_media_trainability_probe


def test_native_media_trainability_report_covers_all_modalities() -> None:
    report = run_native_media_trainability_probe(
        steps=2,
        learning_rate=1e-4,
        min_loss_drop_ratio=-1.0,
        device="cpu",
        seed=7,
    )
    assert report["schema"] == "omnicoder.native_media_trainability_2026.v1"
    assert set(report["initial_loss"]) == set(CORE_MODALITIES)
    assert set(report["final_loss"]) == set(CORE_MODALITIES)
    assert set(report["passed_modalities"]) == set(CORE_MODALITIES)
    assert report["contract"]["continuous_media_path"].startswith("one shared")

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from omnicoder.config_2026 import get_omnicoder2026_preset
from omnicoder.inference.context_budget_2026 import estimate_budget
from omnicoder.model_contract_2026 import validate_target_contract_preset
import omnicoder.training.training_orchestration_2026 as orch


def _contract() -> dict:
    return {
        "target_profile": "omnicoder2026_20b_1m",
        "target_context_length": 1_048_576,
        "required_n_layers": 64,
        "required_d_model": 4096,
        "required_mlp_dim": 15360,
        "required_vocab_size": 330000,
        "required_mtp_heads": 2,
        "required_residual_mode": "block_attnres",
        "min_parameter_b": 18.0,
        "max_parameter_b": 23.0,
        "max_q4_weight_gib": 11.0,
        "max_native_1m_total_gib": 24.0,
        "require_q4_training_path": True,
    }


def test_target_contract_accepts_current_15360_mlp_release_shape() -> None:
    report = validate_target_contract_preset(
        "omnicoder2026_20b_1m",
        require_target_contract=True,
        contract=_contract(),
        context_ladder=[8192, 32768, 131072, 262144, 524288, 1_048_576],
        required_modalities=["text", "code", "tool", "image", "video", "audio", "music", "long_context"],
        enabled_modalities=["text", "code", "tool", "image", "video", "audio", "music", "long_context"],
        fake_quant_enabled=True,
    )

    assert report["status"] == "passed"
    assert report["checks"]["mlp_dim"] == {"actual": 15360, "expected": 15360}
    assert 22.5 < report["budget"]["params_b"] < 22.7
    assert 19.5 < report["budget"]["trunk_params_b"] < 19.7
    assert 2.9 < report["budget"]["auxiliary_params_b"] < 3.1
    assert report["budget"]["fits_24gb_native_estimate"] is True


def test_target_contract_rejects_smaller_or_legacy_target_shapes() -> None:
    with pytest.raises(ValueError, match="requires preset"):
        validate_target_contract_preset(
            "omnicoder2026_16b_1m",
            require_target_contract=True,
            contract=_contract(),
            fake_quant_enabled=True,
        )


def test_target_contract_rejects_wrong_mlp_or_missing_1m_ladder() -> None:
    bad_preset = replace(get_omnicoder2026_preset("omnicoder2026_20b_1m"), mlp_dim=16384)
    with pytest.raises(ValueError, match="mlp_dim mismatch"):
        validate_target_contract_preset(
            bad_preset,
            require_target_contract=True,
            contract=_contract(),
            context_ladder=[1_048_576],
            fake_quant_enabled=True,
        )

    with pytest.raises(ValueError, match="context ladder"):
        validate_target_contract_preset(
            "omnicoder2026_20b_1m",
            require_target_contract=True,
            contract=_contract(),
            context_ladder=[8192, 32768],
            fake_quant_enabled=True,
        )


def test_release_training_contract_report_requires_q4_path() -> None:
    cfg = {
        "model_contract": _contract(),
        "modalities": {name: {"enabled": True} for name in orch.DEFAULT_STAGE_ORDER},
        "training_plan": {
            "preset": "omnicoder2026_20b_1m",
            "context_ladder": [8192, 32768, 131072, 262144, 524288, 1_048_576],
            "required_modalities": list(orch.DEFAULT_STAGE_ORDER),
            "fake_quant": False,
        },
        "q4_recovery": {"enabled": False},
    }
    args = argparse.Namespace(preset="", context_ladder="", allow_verifier_preset=False, fake_quant=False)

    with pytest.raises(ValueError, match="q4/fake-quant"):
        orch.release_training_contract_report(cfg, args)

    cfg["training_plan"]["fake_quant"] = True
    report = orch.release_training_contract_report(cfg, args)
    assert report["status"] == "passed"
    assert report["mode"] == "target_20b_native_1m_q4"


def test_dense_profile_matches_implemented_20b_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "profiles" / "dense_omni_24gb.json").read_text(encoding="utf-8"))
    model = profile["model"]["primary_20b"]
    preset = get_omnicoder2026_preset("omnicoder2026_20b_1m")
    budget = estimate_budget("omnicoder2026_20b_1m")

    assert model["mlp_dim"] == preset.mlp_dim == 15360
    assert model["mtp_heads"] == preset.mtp_heads == 2
    assert round(float(model["estimated_params_b"]), 2) == round(budget.params_b, 2)
    assert round(float(model["q4_weight_gib_estimate"]), 2) == round(budget.weight_gib_q4, 2)
    assert round(float(model["native_1m_total_gib_estimate"]), 2) == round(budget.total_native_estimate_gib, 2)

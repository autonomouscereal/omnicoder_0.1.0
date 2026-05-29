from __future__ import annotations

from dataclasses import asdict
from typing import Any

from omnicoder.config_2026 import Omnicoder2026Preset, get_omnicoder2026_preset
from omnicoder.inference.context_budget_2026 import estimate_budget

TARGET_PRESET_2026 = "omnicoder2026_20b_1m"
TARGET_CONTEXT_TOKENS_2026 = 1_048_576
PROBE_PRESET_NAMES = {
    "probe",
    "native1m_probe",
    "ledger_probe",
    "full_ledger_probe",
    "omnicoder2026_native1m_probe",
    "omnicoder2026_full_ledger_probe",
}


def is_probe_preset(name: str) -> bool:
    return str(name or "").strip().lower().replace("-", "_") in PROBE_PRESET_NAMES


def _contract_float(contract: dict[str, Any], key: str, default: float) -> float:
    try:
        value = contract.get(key, default)
        return float(default if value in (None, "") else value)
    except Exception:
        return float(default)


def _contract_int(contract: dict[str, Any], key: str, default: int) -> int:
    try:
        value = contract.get(key, default)
        return int(default if value in (None, "") else value)
    except Exception:
        return int(default)


def _preset_from(value: str | Omnicoder2026Preset) -> Omnicoder2026Preset:
    if isinstance(value, Omnicoder2026Preset):
        return value
    return get_omnicoder2026_preset(str(value))


def validate_target_contract_preset(
    preset: str | Omnicoder2026Preset,
    *,
    require_target_contract: bool,
    allow_probe: bool = False,
    contract: dict[str, Any] | None = None,
    context_ladder: list[int] | tuple[int, ...] | None = None,
    required_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
    enabled_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
    fake_quant_enabled: bool | None = None,
) -> dict[str, Any]:
    """Fail closed when a target run drifts away from the 20B/q4/1M contract."""

    preset_obj = _preset_from(preset)
    contract = contract if isinstance(contract, dict) else {}
    if not require_target_contract:
        return {"status": "skipped", "reason": "target_contract_not_required", "preset": preset_obj.name}
    if is_probe_preset(preset_obj.name):
        if allow_probe:
            return {"status": "passed", "mode": "verifier_probe", "preset": preset_obj.name}
        raise ValueError(
            f"Refusing to train verifier preset {preset_obj.name!r} for a target-contract run. "
            f"Pass --allow_probe only for explicit validation runs, or use --preset {TARGET_PRESET_2026}."
        )

    target_profile = str(contract.get("target_profile") or TARGET_PRESET_2026)
    if preset_obj.name != target_profile:
        raise ValueError(f"target contract requires preset {target_profile!r}, got {preset_obj.name!r}")

    target_context = _contract_int(contract, "target_context_length", TARGET_CONTEXT_TOKENS_2026)
    min_params_b = _contract_float(contract, "min_parameter_b", 18.0)
    max_params_b = _contract_float(contract, "max_parameter_b", 23.0)
    max_q4_gib = _contract_float(contract, "max_q4_weight_gib", 11.0)
    max_native_gib = _contract_float(contract, "max_native_1m_total_gib", 24.0)
    min_native_gib = _contract_float(contract, "min_native_1m_total_gib", 0.0)

    if int(preset_obj.max_seq_len) < target_context:
        raise ValueError(f"target contract requires max_seq_len >= {target_context}, got {preset_obj.max_seq_len}")
    if context_ladder is not None and context_ladder:
        ladder_max = max(int(value) for value in context_ladder)
        if ladder_max < target_context:
            raise ValueError(f"target contract context ladder must reach {target_context}, got max {ladder_max}")

    checks: dict[str, Any] = {
        "n_layers": (int(preset_obj.n_layers), _contract_int(contract, "required_n_layers", 64)),
        "d_model": (int(preset_obj.d_model), _contract_int(contract, "required_d_model", 4096)),
        "mlp_dim": (int(preset_obj.mlp_dim), _contract_int(contract, "required_mlp_dim", 15360)),
        "vocab_size": (int(preset_obj.vocab_size), _contract_int(contract, "required_vocab_size", 330000)),
        "mtp_heads": (int(preset_obj.mtp_heads), _contract_int(contract, "required_mtp_heads", 2)),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"target contract {field} mismatch: expected {expected}, got {actual}")
    expected_residual = str(contract.get("required_residual_mode") or "block_attnres")
    if str(preset_obj.residual_mode) != expected_residual:
        raise ValueError(f"target contract residual_mode mismatch: expected {expected_residual!r}, got {preset_obj.residual_mode!r}")

    budget = estimate_budget(preset_obj.name, context=target_context)
    if budget.params_b < min_params_b or budget.params_b > max_params_b:
        raise ValueError(
            f"target contract parameter estimate out of range: {budget.params_b:.3f}B "
            f"not in [{min_params_b:.3f}B, {max_params_b:.3f}B]"
        )
    if budget.weight_gib_q4 > max_q4_gib:
        raise ValueError(f"target contract q4 weight estimate exceeds budget: {budget.weight_gib_q4:.3f} GiB > {max_q4_gib:.3f} GiB")
    if budget.total_native_estimate_gib < min_native_gib or budget.total_native_estimate_gib > max_native_gib:
        raise ValueError(
            f"target contract native-state estimate out of range: {budget.total_native_estimate_gib:.3f} GiB "
            f"not in [{min_native_gib:.3f}, {max_native_gib:.3f}] GiB"
        )
    if not budget.fits_24gb_native_estimate:
        raise ValueError(f"target contract native q4 estimate does not fit 24GB: {budget.total_native_estimate_gib:.3f} GiB")

    if required_modalities is not None:
        required = {str(item) for item in required_modalities}
        enabled = {str(item) for item in (enabled_modalities or [])}
        missing = sorted(required - enabled)
        if missing:
            raise ValueError(f"target contract required modalities are disabled or missing: {', '.join(missing)}")
    if bool(contract.get("require_q4_training_path")) and fake_quant_enabled is False:
        raise ValueError("target contract requires the q4/fake-quant training path to be enabled")

    return {
        "status": "passed",
        "mode": "target_20b_native_1m_q4",
        "preset": preset_obj.name,
        "target_context_length": int(target_context),
        "checks": {key: {"actual": actual, "expected": expected} for key, (actual, expected) in checks.items()},
        "residual_mode": str(preset_obj.residual_mode),
        "budget": asdict(budget),
        "context_ladder": [int(value) for value in context_ladder] if context_ladder is not None else None,
        "required_modalities": sorted(str(item) for item in required_modalities) if required_modalities is not None else None,
    }

"""
Unified constraint interface (scaffold).

Standardizes verifier calls across modalities behind a minimal API, returning
normalized margins in [0,1] where higher is better.
"""

from __future__ import annotations

from typing import Any, Dict


def text_margin(hyp: str, context: str | None = None) -> float:
    try:
        # Placeholder: lexical overlap proxy
        if not hyp:
            return 0.0
        if not context:
            return 0.5
        hs = set(hyp.lower().split())
        cs = set(context.lower().split())
        inter = len(hs & cs)
        den = max(1, min(len(hs), len(cs)))
        return max(0.0, min(1.0, inter / den))
    except Exception:
        return 0.0


def code_margin(log_str: str | None = None, tests_passed: int | None = None, tests_total: int | None = None) -> float:
    try:
        if tests_passed is not None and tests_total is not None and tests_total > 0:
            return max(0.0, min(1.0, float(tests_passed) / float(tests_total)))
        # Fallback heuristic: look for "OK" in logs
        if log_str and ("ok" in log_str.lower() or "pass" in log_str.lower()):
            return 0.7
        return 0.0
    except Exception:
        return 0.0


def image_margin(clip_z: float | None = None) -> float:
    try:
        if clip_z is None:
            return 0.0
        # Map z-score into [0,1] via logistic
        import math
        return 1.0 / (1.0 + math.exp(-clip_z))
    except Exception:
        return 0.0


def video_margin(fvd_ref_minus_pred_z: float | None = None) -> float:
    try:
        if fvd_ref_minus_pred_z is None:
            return 0.0
        import math
        return 1.0 / (1.0 + math.exp(-fvd_ref_minus_pred_z))
    except Exception:
        return 0.0


def audio_margin(wer_improvement_z: float | None = None) -> float:
    try:
        if wer_improvement_z is None:
            return 0.0
        import math
        return 1.0 / (1.0 + math.exp(-wer_improvement_z))
    except Exception:
        return 0.0


def unified_margin(weights: Dict[str, float] | None, signals: Dict[str, float]) -> float:
    """Combine modality-specific margins into a single score.

    weights keys: code|image|video|audio|text; if None, auto-average available.
    """
    try:
        keys = [k for k, v in signals.items() if isinstance(v, (int, float))]
        if not keys:
            return 0.0
        if not weights:
            return sum(float(signals[k]) for k in keys) / float(len(keys))
        total_w = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys)
        if total_w <= 0:
            return sum(float(signals[k]) for k in keys) / float(len(keys))
        return sum(max(0.0, float(weights.get(k, 0.0))) * float(signals[k]) for k in keys) / total_w
    except Exception:
        return 0.0



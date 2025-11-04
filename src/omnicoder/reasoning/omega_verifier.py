"""
Omega-Verifier: unify modality checks and compute a proof-margin.

Wraps existing constraint helpers and exposes a single compute_margin() that
maps raw signals into a normalized [0,1] margin based on per-modality weights.
"""

from __future__ import annotations

from typing import Dict, Any

try:
    from .constraints import text_margin, code_margin, image_margin, video_margin, audio_margin, unified_margin  # type: ignore
except Exception:
    # Fallback stubs
    def text_margin(hyp: str, context: str | None = None) -> float:  # type: ignore
        return 0.0
    def code_margin(log_str: str | None = None, tests_passed: int | None = None, tests_total: int | None = None) -> float:  # type: ignore
        return 0.0
    def image_margin(clip_z: float | None = None) -> float:  # type: ignore
        return 0.0
    def video_margin(fvd_ref_minus_pred_z: float | None = None) -> float:  # type: ignore
        return 0.0
    def audio_margin(wer_improvement_z: float | None = None) -> float:  # type: ignore
        return 0.0
    def unified_margin(weights: Dict[str, float] | None, signals: Dict[str, float]) -> float:  # type: ignore
        return 0.0


def compute_margin(signals: Dict[str, Any], weights: Dict[str, float] | None = None) -> float:
    """Compute unified proof-margin from raw signals.

    signals keys may include: text, code, image, video, audio; values may be raw inputs
    for the specific margin functions (e.g., code: {tests_passed, tests_total}).
    """
    try:
        s: Dict[str, float] = {}
        if 'text' in signals:
            v = signals.get('text')
            if isinstance(v, dict):
                s['text'] = float(text_margin(v.get('hyp', ''), v.get('context')))
            else:
                s['text'] = float(text_margin(str(v), None))
        if 'code' in signals:
            v = signals.get('code')
            if isinstance(v, dict):
                s['code'] = float(code_margin(v.get('log_str'), v.get('tests_passed'), v.get('tests_total')))
            else:
                s['code'] = float(code_margin(str(v), None, None))
        if 'image' in signals:
            v = signals.get('image')
            if isinstance(v, dict):
                s['image'] = float(image_margin(v.get('clip_z')))
            else:
                s['image'] = float(image_margin(None))
        if 'video' in signals:
            v = signals.get('video')
            if isinstance(v, dict):
                s['video'] = float(video_margin(v.get('fvd_ref_minus_pred_z')))
            else:
                s['video'] = float(video_margin(None))
        if 'audio' in signals:
            v = signals.get('audio')
            if isinstance(v, dict):
                s['audio'] = float(audio_margin(v.get('wer_improvement_z')))
            else:
                s['audio'] = float(audio_margin(None))
        return float(unified_margin(weights, s))
    except Exception:
        return 0.0



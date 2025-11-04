from __future__ import annotations

"""
Tool & Constraint Fabric (scaffold).

Unifies tool invocation and constraint checking for Ω-Reasoner. Exposes tiny
helpers that can be called from generation or controllers.
Also provides a minimal Perceptual Program Induction (PPI) step that attempts
to call detectors/segmenters/ASR tools (best-effort) and writes facts into an
NSS instance when available.
"""

from typing import Any, Dict, Tuple

from .tool_use import build_default_registry
from omnicoder.reasoning.constraints import unified_margin
try:
    from omnicoder.reasoning.nss import NSS
except Exception:
    NSS = None  # type: ignore


def invoke_tools_in_text(text: str) -> str:
    """Replace inline <tool:...> tags with JSON outputs."""
    reg = build_default_registry()
    repl = reg.parse_and_invoke_all(text)
    out = text
    import json as _json
    for tag, val in repl.items():
        try:
            out = out.replace(tag, _json.dumps(val))
        except Exception:
            continue
    return out


def combine_verifier_signals(signals: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    return unified_margin(weights, signals)


def analyze_inputs_for_nss(prompt: str, modes: str = "image,audio,video") -> Tuple[str, Any]:
    """Attempt to analyze referenced inputs and populate an NSS.

    Returns a tuple of (nss_context_text, nss_obj_or_None).
    This function is best-effort and safe when dependencies are missing.
    """
    modes_l = [m.strip() for m in str(modes).split(',') if m.strip()]
    if NSS is None:
        return "", None
    nss = NSS()
    try:
        reg = build_default_registry()
    except Exception:
        reg = None
    # Heuristic: look for file-like tokens in prompt and try analyzers
    tokens = prompt.split()
    paths = [t for t in tokens if any(t.lower().endswith(ext) for ext in ('.png','.jpg','.jpeg','.bmp','.gif','.mp4','.mov','.wav','.mp3','.flac','.m4a'))]
    try:
        if reg is not None:
            for p in paths[:4]:
                pl = p.lower()
                if any(pl.endswith(ext) for ext in ('.png','.jpg','.jpeg','.bmp','.gif')) and ('image' in modes_l):
                    try:
                        boxes = reg.call('yolo_detect', dict(path=p))  # type: ignore
                        # Expect boxes: [{label, x1,y1,x2,y2, score}]
                        for b in boxes or []:
                            lab = str(b.get('label','obj'))
                            nss.apply(f"ASSERT({lab}, present_in, image)")
                    except Exception:
                        pass
                if any(pl.endswith(ext) for ext in ('.mp4','.mov')) and ('video' in modes_l):
                    try:
                        tracks = reg.call('vid_track', dict(path=p))  # type: ignore
                        for t in tracks or []:
                            lab = str(t.get('label','obj'))
                            nss.apply(f"ASSERT({lab}, present_in, video)")
                    except Exception:
                        pass
                if any(pl.endswith(ext) for ext in ('.wav','.mp3','.flac','.m4a')) and ('audio' in modes_l):
                    try:
                        asr = reg.call('asr_transcribe', dict(path=p))  # type: ignore
                        if isinstance(asr, dict) and asr.get('text'):
                            txt = str(asr['text'])
                            nss.apply(f"BIND(transcript, {txt})")
                    except Exception:
                        pass
    except Exception:
        pass
    return nss.render_context(), nss



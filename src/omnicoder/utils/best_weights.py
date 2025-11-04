from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple
import json


def _all_best_meta_files(root: str | Path = "weights") -> List[Path]:
    try:
        base = Path(root)
        if not base.exists():
            return []
        # Shallow scan and one-level deep to avoid huge trees
        files: List[Path] = []
        for p in base.rglob("*_best.meta.json"):
            # Keep within weights/ only
            try:
                files.append(p)
            except Exception:
                continue
        # Sort by mtime desc so newest first
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        return files
    except Exception:
        return []


def _read_meta(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _candidate_from_meta(meta_path: Path) -> Optional[Tuple[float, str]]:
    meta = _read_meta(meta_path)
    if not isinstance(meta, dict):
        return None
    path = str(meta.get('path') or '')
    if not path:
        return None
    try:
        val = float(meta.get('best_value'))
    except Exception:
        val = float('inf')
    return (val, path)


def find_best_for(modality: str = "text", preset_hint: Optional[str] = None) -> Optional[str]:
    """Return a best checkpoint path for the given modality if one is discoverable.

    Heuristics:
    - Prefer *_best.meta.json that contains a valid 'path'.
    - If preset_hint is provided, prefer entries whose filename contains the hint.
    - Fall back to any *_best.* under weights/ if meta is missing.
    """
    # Env override wins
    env_key = {
        'text': 'OMNICODER_CKPT',
        'image': 'OMNICODER_SD_LOCAL_PATH',
        'video': 'OMNICODER_VIDEO_LOCAL_PATH',
    }.get(modality, '')
    if env_key:
        val = os.getenv(env_key, '').strip()
        if val:
            return val

    # Meta-driven selection
    metas = _all_best_meta_files('weights')
    # Prefer matches by preset hint
    if preset_hint:
        for m in metas:
            try:
                if preset_hint.lower() in m.name.lower() or preset_hint.lower() in str(m.parent.name).lower():
                    cand = _candidate_from_meta(m)
                    if cand and cand[1] and Path(cand[1]).exists():
                        return cand[1]
            except Exception:
                continue
    # Next, first valid meta
    for m in metas:
        cand = _candidate_from_meta(m)
        if cand and cand[1] and Path(cand[1]).exists():
            return cand[1]

    # Fallback: any *_best.pt in weights/
    try:
        any_best = list(Path('weights').rglob('*_best.pt'))
        any_best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if any_best:
            return str(any_best[0])
    except Exception:
        pass
    return None



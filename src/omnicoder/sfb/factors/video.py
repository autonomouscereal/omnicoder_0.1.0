from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import os

from . import FactorScore


@dataclass
class VideoSolver:
    def solve(self, factor: Any) -> FactorScore:
        aux: Dict[str, Any] = {"prefer_strings": [], "avoid_strings": []}
        # Optional FVD-based margin
        pred = os.getenv('SFB_FVD_PRED_DIR', '').strip()
        ref = os.getenv('SFB_FVD_REF_DIR', '').strip()
        z = 0.0
        if pred and ref:
            try:
                from omnicoder.eval import reward_metrics as _rm  # type: ignore
                fvd = _rm.fvd(pred, ref)
                if fvd is not None:
                    z = float(1.0 / (1.0 + max(0.0, fvd)))
            except Exception:
                z = 0.0
        # Encourage temporal phrasing if description suggests it
        try:
            meta = getattr(factor, 'meta', {})
            desc = str(meta.get('desc', '')).lower()
            if any(k in desc for k in ['before', 'after', 'then', 'first', 'next', 'finally']):
                aux["prefer_strings"] += ["then ", "first ", "next ", "finally "]
        except Exception:
            pass
        aux["video_z"] = z
        return FactorScore(name="video", score=0.05 + 0.2 * z, aux=aux)



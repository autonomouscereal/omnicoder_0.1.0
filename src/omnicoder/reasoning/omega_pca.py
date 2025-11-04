"""
Omega-PCA: Proof-Carrying Answer packer

Produces a compact JSON-serializable certificate alongside an answer text. The
certificate records inferred goals, assumptions, counterfactual checks, and
verification margins.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import json
import time


@dataclass
class Certificate:
    goals: List[Dict[str, Any]]
    assumptions: List[str]
    counterfactuals: List[str]
    verifications: Dict[str, float]
    proof_margin: float
    timestamp: float
    version: str = "omega-0.1"


def pack(answer: str, goal_belief: Dict[str, Any] | None, verif_signals: Dict[str, float] | None, assumptions: List[str] | None = None, counterfactuals: List[str] | None = None, margin: float | None = None) -> Dict[str, Any]:
    goals = []
    if isinstance(goal_belief, dict) and isinstance(goal_belief.get('hypotheses'), list):
        for h in goal_belief['hypotheses'][:5]:
            try:
                goals.append({
                    'goal': str(h.get('goal', '')),
                    'p': float(h.get('posterior', 0.0)),
                })
            except Exception:
                continue
    cert = Certificate(
        goals=goals,
        assumptions=list(assumptions or []),
        counterfactuals=list(counterfactuals or []),
        verifications={k: float(v) for k, v in (verif_signals or {}).items()},
        proof_margin=float(margin if margin is not None else 0.0),
        timestamp=time.time(),
    )
    return {
        'answer': str(answer),
        'certificate': asdict(cert),
    }


def to_json(obj: Dict[str, Any]) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({'error': 'serialization failed'})



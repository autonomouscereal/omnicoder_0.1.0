from __future__ import annotations

"""
Omega2 Proof-Carrying Answer (PCA) certificate utilities.

Collects reasoning artifacts across AGoT, Latent BFS, Symbolic Planner, and
GraphRAG, then emits a compact JSON-serializable object.

Environment knobs:
- OMNICODER_CERT_OUT=/path/to/file.jsonl  (append JSONL records)
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
import json
import os


@dataclass
class CertAGoT:
    width: int
    depth: int
    token_budget: int
    selected_id: int | None = None


@dataclass
class CertLatent:
    beam: int
    depth: int
    selected_id: int | None = None


@dataclass
class CertPlanner:
    actions: List[Dict[str, Any]]


@dataclass
class CertGraphRAG:
    triples: List[Tuple[str, str, str]]


@dataclass
class Omega2Cert:
    prompt: str
    output: str | None = None
    agot: CertAGoT | None = None
    latent: CertLatent | None = None
    planner: CertPlanner | None = None
    graphrag: CertGraphRAG | None = None
    verifier_margin: float | None = None
    acceptance_prob: float | None = None
    # Optional per-step trace entries: [{'t': int, 'entropy': float, 'chosen_id': int, ...}, ...]
    steps: Optional[List[Dict[str, Any]]] = None

    def to_json(self) -> str:
        def _coerce(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            if isinstance(obj, (list, tuple)):
                return [ _coerce(x) for x in obj ]
            if isinstance(obj, dict):
                return { k: _coerce(v) for k, v in obj.items() }
            return obj
        return json.dumps(_coerce(asdict(self)), ensure_ascii=False)


def emit(cert: Omega2Cert) -> None:
    """Append certificate JSON to a file if OMNICODER_CERT_OUT is set."""
    try:
        path = os.getenv('OMNICODER_CERT_OUT', '').strip()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(cert.to_json() + "\n")
    except Exception:
        return



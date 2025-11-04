from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any


@dataclass
class FactorScore:
    name: str
    score: float
    aux: Dict[str, Any]


class FactorSolver(Protocol):
    def solve(self, factor: Any) -> FactorScore:  # pragma: no cover - protocol
        ...


__all__ = [
    'FactorScore',
]


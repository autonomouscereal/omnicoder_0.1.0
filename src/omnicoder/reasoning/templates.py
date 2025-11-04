"""
Reasoning Template Bank (RTB) scaffold.

Implements a tiny VQ-style codebook over opaque graph embeddings. This is a
minimal placeholder with cosine similarity lookup for seeding plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import math


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) + 1e-6
    nb = math.sqrt(sum(x * x for x in b)) + 1e-6
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


@dataclass
class TemplateBank:
    codes: List[List[float]] = field(default_factory=list)
    payloads: List[str] = field(default_factory=list)

    def add(self, embedding: List[float], payload: str) -> None:
        self.codes.append(list(map(float, embedding)))
        self.payloads.append(str(payload))

    def query(self, embedding: List[float], topk: int = 1, min_sim: float = 0.0) -> List[Tuple[float, str]]:
        if not self.codes:
            return []
        sims = [(_cos(embedding, code), self.payloads[i]) for i, code in enumerate(self.codes)]
        sims.sort(key=lambda x: -x[0])
        out = []
        for s, p in sims[: max(1, topk)]:
            if s >= float(min_sim):
                out.append((float(s), p))
        return out



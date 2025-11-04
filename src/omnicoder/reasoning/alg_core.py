"""
Generalist Algorithmic Core (scaffold)

This minimal module exposes a suggest() API that detects simple algorithmic
prompts (sorting, searching, parentheses, arithmetic) and returns token-level
logit nudges to guide the first decoding step. It is intentionally tiny and
dependency-free; a future version can host a real GNN processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class AlgHint:
    bias: Dict[str, float]


class AlgorithmicCore:
    def __init__(self) -> None:
        pass

    def suggest(self, prompt: str) -> Dict[str, Dict[str, float]]:
        text = (prompt or "").lower()
        bias: Dict[str, float] = {}
        # Sorting-related
        if any(k in text for k in ["sort", "sorted", "ascending", "descending"]):
            bias.update({"[": 0.05, "]": 0.02, ",": 0.02})
        # Parentheses/balancing problems
        if any(k in text for k in ["parentheses", "brackets", "balanced"]):
            bias.update({"(": 0.05, ")": 0.05})
        # Arithmetic/program tasks
        if any(k in text for k in ["sum", "product", "factorial", "fibonacci", "prime"]):
            bias.update({":": 0.03, "def": 0.05, "return": 0.04})
        # Graph algorithms keywords
        if any(k in text for k in ["dijkstra", "bfs", "dfs", "mst", "union-find", "kruskal", "prim"]):
            bias.update({"def": 0.04, "class": 0.02, "for": 0.02})
        if not bias:
            return {"bias": {}}
        return {"bias": bias}


def build_alg_core() -> AlgorithmicCore:
    return AlgorithmicCore()



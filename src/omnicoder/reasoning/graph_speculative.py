"""
Graph-speculative branch runner (scaffold).

Runs multiple candidate branches in parallel (conceptually) and accepts the one
with the highest proof-margin per cost. This placeholder wires the interface
only and leaves heavy lifting to the main decode loop.
"""

from __future__ import annotations

from typing import Callable, List, Tuple


def select_branch(candidates: List[Tuple[float, float, int]]) -> int:
    """Select a candidate by max (margin / cost).

    candidates: list of (margin, score, cost_tokens)
    Returns index of best candidate.
    """
    if not candidates:
        return 0
    best_i = 0
    best_v = float('-inf')
    for i, (margin, _score, cost) in enumerate(candidates):
        denom = float(max(1, int(cost)))
        v = float(margin) / denom
        if v > best_v:
            best_v = v
            best_i = i
    return best_i



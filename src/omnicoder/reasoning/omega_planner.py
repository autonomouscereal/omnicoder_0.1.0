"""
Omega-Planner: Dreamer-style world-model proxy + MCTS over (state,hypothesis,plan)

This is a minimal search scaffold that can score short plan prefixes by a proxy
value and select the best branch. It integrates with the decode loop through
lightweight function calls and avoids heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple
import math


@dataclass(slots=True)
class PlanNode:
    state: Any
    action: Any
    value: float
    depth: int


def _ucb(parent_visits: int, node_visits: int, value: float, c: float = 1.2) -> float:
    # Local bindings for speed in tight loops
    if node_visits <= 0:
        return float('inf')
    return value + c * math.sqrt(math.log(max(1, parent_visits)) / node_visits)


def mcts_search(
    root_state: Any,
    expand_fn: Callable[[Any], List[Any]],
    rollout_value_fn: Callable[[Any], float],
    budget: int = 16,
    max_depth: int = 3,
) -> Tuple[Any, float]:
    """Very small MCTS: expand breadth-first, evaluate with a rollout value.

    Returns best_action and its value estimate.
    """
    try:
        # frontier holds (state, action, value, depth)
        frontier: List[PlanNode] = []
        actions = expand_fn(root_state)
        for a in actions:
            v = rollout_value_fn((root_state, a))
            frontier.append(PlanNode(state=root_state, action=a, value=v, depth=1))
        # select best by value, refine up to budget
        iters = max(1, int(budget))
        for _ in range(iters - 1):
            if not frontier:
                break
            # pick best to deepen
            node = max(frontier, key=lambda n: n.value)
            if node.depth >= int(max_depth):
                continue
            # expand one level
            for a2 in expand_fn(node.state):
                v2 = rollout_value_fn((node.state, a2))
                frontier.append(PlanNode(state=node.state, action=a2, value=0.5 * node.value + 0.5 * v2, depth=node.depth + 1))
        if not frontier:
            return None, 0.0
        best = max(frontier, key=lambda n: n.value)
        return best.action, best.value
    except Exception:
        return None, 0.0



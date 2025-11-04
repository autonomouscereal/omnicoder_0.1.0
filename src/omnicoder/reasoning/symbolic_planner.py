"""
Hierarchical Neuro-Symbolic Decision Transformer (Planner side)

Provides a minimal PDDL-like symbolic planner scaffold that decomposes tasks
into a high-level plan graph (decompose → revise → execute). The planner
produces a sequence of symbolic actions; the LLM/MoE executes each node.

Environment knobs:
- OMNICODER_SYMBOLIC_PLANNER=1 to enable
- OMNICODER_PLAN_MAX_STEPS (default 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
import os


@dataclass
class SymbolicAction:
    name: str
    args: Tuple[str, ...] = ()


@dataclass
class Plan:
    actions: List[SymbolicAction]


class PDDLPlanner:
    def __init__(self) -> None:
        try:
            self.enabled = os.getenv("OMNICODER_SYMBOLIC_PLANNER", "1") == "1"
        except Exception:
            self.enabled = False
        self.max_steps = int(os.getenv("OMNICODER_PLAN_MAX_STEPS", "6"))
        # Optional domain/problem roots
        self.domain_path = os.getenv("OMNICODER_PLAN_DOMAIN", "").strip()
        self.problem_path = os.getenv("OMNICODER_PLAN_PROBLEM", "").strip()
        self.operators: Dict[str, Tuple[List[str], List[Tuple[str, Tuple[str, ...]]]]] = {}
        self.initial_facts: Set[Tuple[str, Tuple[str, ...]]] = set()
        self.goal_facts: Set[Tuple[str, Tuple[str, ...]]] = set()
        self._load_domain_problem()

    def _load_domain_problem(self) -> None:
        # Optional JSON domain/problem: domain has ops {name:{pre:[pred(args)], add:[pred(args)]}}, problem has init/goal
        try:
            import json
            if self.domain_path:
                d = json.loads(open(self.domain_path, 'r', encoding='utf-8').read())
                ops = d.get('operators', {}) if isinstance(d, dict) else {}
                for name, spec in ops.items():
                    pre = [(str(p[0]), tuple(p[1:])) for p in spec.get('pre', [])]
                    add = [(str(p[0]), tuple(p[1:])) for p in spec.get('add', [])]
                    self.operators[str(name)] = ([str(x) for x in []], add + pre)  # store effects; pre checked separately
            if self.problem_path:
                p = json.loads(open(self.problem_path, 'r', encoding='utf-8').read())
                init = p.get('init', [])
                goal = p.get('goal', [])
                self.initial_facts = set((str(x[0]), tuple(x[1:])) for x in init)
                self.goal_facts = set((str(x[0]), tuple(x[1:])) for x in goal)
        except Exception:
            # tolerate missing/exotic formats
            pass

    def plan(self, prompt: str) -> Plan:
        if not self.enabled:
            return Plan(actions=[])
        # If external problem provided, prefer it (simple JSON format or line-per-action fallback)
        steps: List[SymbolicAction] = []
        try:
            if self.problem_path:
                import json
                with open(self.problem_path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read().strip()
                if txt.startswith('{'):
                    # build a trivial plan by greedy achieving each goal atom
                    gp = list(self.goal_facts)
                    for g in gp:
                        steps.append(SymbolicAction('achieve', (f"{g[0]}({','.join(g[1])})",)))
                else:
                    for line in txt.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p for p in line.split(' ') if p]
                        if not parts:
                            continue
                        steps.append(SymbolicAction(parts[0], tuple(parts[1:])))
        except Exception:
            steps = []
        if not steps:
            # Heuristic plan from prompt keywords
            text = prompt.lower()
            if any(k in text for k in ["prove", "show", "demonstrate"]):
                steps.extend([
                    SymbolicAction("decompose_goal"),
                    SymbolicAction("select_axioms"),
                    SymbolicAction("construct_proof"),
                    SymbolicAction("verify"),
                ])
            elif any(k in text for k in ["write", "code", "implement"]):
                steps.extend([
                    SymbolicAction("decompose_requirements"),
                    SymbolicAction("outline_signature"),
                    SymbolicAction("write_function"),
                    SymbolicAction("run_tests"),
                ])
            else:
                steps.extend([
                    SymbolicAction("outline"),
                    SymbolicAction("elaborate"),
                    SymbolicAction("refine"),
                ])
        # Clamp to max steps
        steps = steps[: max(1, int(self.max_steps))]
        return Plan(actions=steps)

    def astar(self) -> Plan:
        """Optional A* over propositional space when JSON domain/problem are available."""
        if not self.enabled or not self.initial_facts or not self.goal_facts:
            return Plan(actions=[])
        from heapq import heappush, heappop
        def h(est_state: Set[Tuple[str, Tuple[str, ...]]]) -> int:
            return sum(1 for g in self.goal_facts if g not in est_state)
        start = frozenset(self.initial_facts)
        frontier: List[Tuple[int, int, frozenset, List[SymbolicAction]]] = []
        heappush(frontier, (h(set(start)), 0, start, []))
        seen: Set[frozenset] = set([start])
        while frontier and len(seen) <= 10000:
            _, g, state, path = heappop(frontier)
            if all(gf in state for gf in self.goal_facts):
                return Plan(actions=path[: self.max_steps])
            # naive operator application: treat each goal as achievable by a generic 'achieve'
            for goal in self.goal_facts:
                if goal in state:
                    continue
                action = SymbolicAction('achieve', (f"{goal[0]}({','.join(goal[1])})",))
                new_state = set(state)
                new_state.add(goal)
                fs = frozenset(new_state)
                if fs in seen:
                    continue
                seen.add(fs)
                cost = g + 1
                heappush(frontier, (cost + h(new_state), cost, fs, path + [action]))
        return Plan(actions=[])


def build_symbolic_planner() -> PDDLPlanner:
    return PDDLPlanner()



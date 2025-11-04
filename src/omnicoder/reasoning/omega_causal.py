"""
Omega-Causal: minimal structural causal model (SCM) scaffold

Provides a tiny DSL to define variables and mechanisms, perform abductive
scoring, and compute simple counterfactual rollouts under interventions.
This is intentionally lightweight and safe to import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Variable:
    name: str
    value: Any = None
    parents: List[str] = field(default_factory=list)


@dataclass
class Mechanism:
    target: str
    func: Callable[[Dict[str, Any]], Any]


@dataclass
class SCM:
    variables: Dict[str, Variable] = field(default_factory=dict)
    mechanisms: Dict[str, Mechanism] = field(default_factory=dict)

    def add_variable(self, name: str, value: Any = None, parents: Optional[List[str]] = None) -> None:
        self.variables[name] = Variable(name=name, value=value, parents=list(parents or []))

    def add_mechanism(self, target: str, func: Callable[[Dict[str, Any]], Any]) -> None:
        self.mechanisms[target] = Mechanism(target=target, func=func)

    def forward(self) -> Dict[str, Any]:
        """Compute a forward pass in a simple topological order (parents first)."""
        # naive order: iterate until convergence or max steps
        vals = {k: v.value for k, v in self.variables.items()}
        for _ in range(max(1, len(self.variables))):
            changed = False
            for name, mech in self.mechanisms.items():
                try:
                    out = mech.func(vals)
                    if vals.get(name) != out:
                        vals[name] = out
                        changed = True
                except Exception:
                    continue
            if not changed:
                break
        for k, v in self.variables.items():
            v.value = vals.get(k, v.value)
        return vals

    def do(self, interventions: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an intervention do(X=x), recomputing descendants with patched values."""
        # copy and clamp
        vals = {k: v.value for k, v in self.variables.items()}
        vals.update(interventions)
        for _ in range(max(1, len(self.variables))):
            changed = False
            for name, mech in self.mechanisms.items():
                if name in interventions:
                    continue  # intervened
                try:
                    out = mech.func(vals)
                    if vals.get(name) != out:
                        vals[name] = out
                        changed = True
                except Exception:
                    continue
            if not changed:
                break
        return vals


def abductive_score(scm: SCM, observations: Dict[str, Any]) -> float:
    """Score how well the SCM explains observations (lower error → higher score)."""
    try:
        vals = scm.forward()
        err = 0.0
        for k, v in observations.items():
            if k in vals:
                err += 0.0 if vals[k] == v else 1.0
            else:
                err += 1.0
        return max(0.0, 1.0 - err / max(1.0, float(len(observations))))
    except Exception:
        return 0.0


def value_of_information(uncertainties: Dict[str, float], budget: int = 1) -> List[Tuple[str, float]]:
    """Select variables to observe/intervene that reduce uncertainty the most per unit cost."""
    items = list(uncertainties.items())
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[: max(1, int(budget))]


def build_minimal_scm_for_query(query: str) -> SCM:
    """Heuristic SCM template built from a text query.

    Variables: G (goal clarity), E (evidence availability), P (plan feasibility), A (answer quality)
    A ← f(G,E,P)
    """
    scm = SCM()
    scm.add_variable("G", 0.5)
    scm.add_variable("E", 0.5)
    scm.add_variable("P", 0.5)
    scm.add_variable("A", 0.0)

    def f_all(vals: Dict[str, Any]) -> float:
        g = float(vals.get("G", 0.0))
        e = float(vals.get("E", 0.0))
        p = float(vals.get("P", 0.0))
        # simple smooth AND
        return max(0.0, min(1.0, 0.33 * g + 0.33 * e + 0.34 * p))

    scm.add_mechanism("A", f_all)
    # tiny query conditioning
    q = (query or "").lower()
    if any(k in q for k in ("why", "how", "cause")):
        scm.variables["G"].value = 0.7
    if any(k in q for k in ("code", "compile", "test")):
        scm.variables["P"].value = 0.7
    if any(k in q for k in ("cite", "evidence", "source")):
        scm.variables["E"].value = 0.7
    return scm



"""
Neuro-Symbolic Scratchpad (NSS) scaffold.

Provides a typed memory with a tiny DSL for emitting and applying updates.
This is minimal and dependency-light so it can be safely imported on-device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class NSSState:
    facts: List[Tuple[str, str, str]] = field(default_factory=list)  # (head, rel, tail)
    bindings: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


class NSS:
    def __init__(self):
        self.state = NSSState()

    def apply(self, dsl: str) -> None:
        """Parse and apply a tiny subset of the DSL.

        Supported:
          ASSERT(head, rel, tail)
          BIND(var, value)
          SOLVE(expr)
        This parser is intentionally permissive and no-ops on malformed lines.
        """
        for line in dsl.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                if s.startswith("ASSERT(") and s.endswith(")"):
                    inner = s[len("ASSERT("):-1]
                    head, rel, tail = [x.strip() for x in inner.split(",", 2)]
                    self.state.facts.append((head, rel, tail))
                elif s.startswith("BIND(") and s.endswith(")"):
                    inner = s[len("BIND("):-1]
                    var, val = [x.strip() for x in inner.split(",", 1)]
                    self.state.bindings[var] = val
                elif s.startswith("SOLVE(") and s.endswith(")"):
                    inner = s[len("SOLVE("):-1]
                    self.state.constraints.append(inner)
            except Exception:
                # best-effort parsing: ignore bad lines
                continue

    def render_context(self) -> str:
        """Render a lightweight textual summary to condition generation.
        """
        parts = []
        if self.state.facts:
            facts = "; ".join([f"({h},{r},{t})" for (h, r, t) in self.state.facts[:64]])
            parts.append(f"[FACTS] {facts}")
        if self.state.bindings:
            binds = ", ".join([f"{k}={v}" for k, v in list(self.state.bindings.items())[:32]])
            parts.append(f"[BIND] {binds}")
        if self.state.constraints:
            cons = "; ".join(self.state.constraints[:32])
            parts.append(f"[CONS] {cons}")
        return "\n".join(parts)



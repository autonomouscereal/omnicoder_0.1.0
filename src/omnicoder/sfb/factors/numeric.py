from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any
import math

from . import FactorScore


@dataclass
class NumericSolver:
    def solve(self, factor: Any) -> FactorScore:
        # Attempt exact evaluation using Python/SymPy when an expression is provided
        aux: Dict[str, Any] = {"prefer_strings": ["=", "+", "-", "*", "/", "The answer is"], "avoid_strings": []}
        expr = None
        try:
            expr = str(getattr(factor, 'meta', {}).get('expr', '')).strip()
        except Exception:
            expr = None
        if not expr:
            return FactorScore(name="numeric", score=0.1, aux=aux)
        try:
            try:
                import sympy as sp  # type: ignore
                val = float(sp.N(sp.sympify(expr)))
            except Exception:
                # Fallback to Python eval in a constrained namespace
                val = float(eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt}))
            # Higher score when an exact value is produced
            aux["token_bias"] = {}
            aux["value"] = val
            return FactorScore(name="numeric", score=0.8, aux=aux)
        except Exception:
            return FactorScore(name="numeric", score=0.1, aux=aux)



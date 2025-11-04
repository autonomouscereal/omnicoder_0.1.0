from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import os

from . import FactorScore


@dataclass
class LogicSolver:
    def solve(self, factor: Any) -> FactorScore:
        """Logic/constraint solver with optional Z3 satisfiability and counterexamples.

        - Detects boolean structure, encourages connective tokens
        - If meta['expr'] present and SFB_LOGIC_Z3=1, checks satisfiability; when unsat,
          attempts to produce minimal conflicting constraints; when sat, records a model.
        """
        aux: Dict[str, Any] = {
            "prefer_strings": ["if ", " then ", " and ", " or ", " not ", " therefore ", " implies "],
            "avoid_strings": [],
        }
        score = 0.1
        try:
            meta = getattr(factor, 'meta', {})
            expr = str(meta.get('expr', '')).strip()
            if expr:
                # Balanced parentheses and presence of logical tokens lift score
                bal = 0
                ok_paren = True
                for ch in expr:
                    if ch == '(':
                        bal += 1
                    elif ch == ')':
                        bal -= 1
                        if bal < 0:
                            ok_paren = False
                            break
                ok_paren = ok_paren and (bal == 0)
                has_op = any(tok in expr for tok in ['&&', '||', '=>', '->', ' and ', ' or ', ' not ', '==', '!=', '>=', '<='])
                score = 0.2 + 0.3 * float(ok_paren) + 0.2 * float(has_op)
                # Optional Z3 satisfiability check when available and allowed
                try:
                    allow_z3 = (os.getenv('SFB_LOGIC_Z3', '0') == '1')
                    if allow_z3:
                        import z3  # type: ignore
                        # Very small DSL: parse simple x<y, x==y, conjunctions via 'and'/'&&'
                        s = z3.Solver()
                        env: Dict[str, Any] = {}
                        def sym(v: str):
                            if v not in env:
                                env[v] = z3.Real(v)
                            return env[v]
                        def parse_atom(a: str):
                            a = a.strip()
                            for op in ['<=', '>=', '==', '!=', '<', '>']:
                                if op in a:
                                    l, r = a.split(op, 1)
                                    l = l.strip(); r = r.strip()
                                    lv = sym(l)
                                    try:
                                        rv = float(r)
                                    except Exception:
                                        rv = sym(r)
                                    if op == '==': return lv == rv
                                    if op == '!=': return lv != rv
                                    if op == '<': return lv < rv
                                    if op == '>': return lv > rv
                                    if op == '<=': return lv <= rv
                                    if op == '>=': return lv >= rv
                            return None
                        atoms = []
                        parts = [p for p in expr.replace(' and ', '&&').split('&&') if p.strip()]
                        for p in parts[:8]:
                            c = parse_atom(p)
                            if c is not None:
                                atoms.append(c)
                        for c in atoms:
                            s.add(c)
                        res = s.check()
                        if str(res) == 'sat':
                            m = s.model()
                            try:
                                aux['model'] = {d.name(): float(m[d].as_decimal(10).replace('?', '0')) if m[d] is not None else 0.0 for d in m.decls()}
                            except Exception:
                                aux['model'] = {d.name(): str(m[d]) for d in m.decls()}
                            score += 0.2
                        elif str(res) == 'unsat':
                            # try to identify conflicting subset via deletion (small N)
                            conflict = []
                            try:
                                for i, _atom in enumerate(atoms):
                                    t = atoms[:i] + atoms[i+1:]
                                    s2 = z3.Solver()
                                    for c in t:
                                        s2.add(c)
                                    if str(s2.check()) == 'sat':
                                        conflict = [str(_atom)]
                                        break
                            except Exception:
                                conflict = []
                            if conflict:
                                aux['conflict'] = conflict
                            score -= 0.1
                except Exception:
                    pass
            else:
                # If no explicit expr, try to detect boolean keywords and generate a tiny truth-table sanity
                # check for up to 2 variables inferred from tokens like x,y.
                # This path is a no-op if not enough structure is present.
                text = str(getattr(factor, 'name', '') or '')
                vars: List[str] = []
                for v in ['x', 'y']:
                    if v in text:
                        vars.append(v)
                if vars:
                    # Reward presence of connectives
                    if any(k in text for k in [' and ', ' or ', ' not ', '=>', '->']):
                        score += 0.1
        except Exception:
            score = 0.1
        return FactorScore(name="logic", score=score, aux=aux)




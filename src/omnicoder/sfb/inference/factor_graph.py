from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math
import os


@dataclass
class DiscreteVariable:
    name: str
    domain: Tuple[int, ...]  # e.g., (0,1)


@dataclass
class DiscreteFactor:
    scope: Tuple[str, ...]  # tuple of variable names
    table: Dict[Tuple[int, ...], float]  # assignment -> potential (>=0)


@dataclass
class FactorGraph:
    variables: Dict[str, DiscreteVariable]
    factors: List[DiscreteFactor]


def _normalize(vec: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(0.0, float(v)) for v in vec.values())
    if s <= 0.0:
        # fallback to uniform
        n = max(1, len(vec))
        return {k: 1.0 / float(n) for k in vec.keys()}
    return {k: float(v) / float(s) for (k, v) in vec.items()}


def run_sum_product(graph: FactorGraph, iterations: int = 5) -> Dict[str, Dict[int, float]]:
    # Build neighbor maps
    var_to_factors: Dict[str, List[int]] = {vn: [] for vn in graph.variables.keys()}
    for fi, f in enumerate(graph.factors):
        for vn in f.scope:
            var_to_factors[vn].append(fi)
    # Initialize messages uniformly: m_{v->f}(x_v) and m_{f->v}(x_v)
    msg_vf: Dict[Tuple[str, int], Dict[int, float]] = {}
    msg_fv: Dict[Tuple[int, str], Dict[int, float]] = {}
    for vn, var in graph.variables.items():
        for fi in var_to_factors[vn]:
            msg_vf[(vn, fi)] = _normalize({s: 1.0 for s in var.domain})
            msg_fv[(fi, vn)] = _normalize({s: 1.0 for s in var.domain})
    # Iterate
    for _ in range(max(1, iterations)):
        # Update var->factor
        for vn, var in graph.variables.items():
            neigh = var_to_factors[vn]
            for fi in neigh:
                # product of incoming f'->v except f
                prod: Dict[int, float] = {s: 1.0 for s in var.domain}
                for fj in neigh:
                    if fj == fi:
                        continue
                    m = msg_fv[(fj, vn)]
                    for s in var.domain:
                        prod[s] *= max(1e-12, float(m.get(s, 0.0)))
                msg_vf[(vn, fi)] = _normalize(prod)
        # Update factor->var
        for fi, f in enumerate(graph.factors):
            for vn in f.scope:
                # Sum over all assignments of other vars
                var = graph.variables[vn]
                accum: Dict[int, float] = {s: 0.0 for s in var.domain}
                other = [v for v in f.scope if v != vn]
                # Enumerate assignments via recursion (small scopes expected)
                def _enum(idx: int, partial: Dict[str, int]) -> None:
                    if idx >= len(other):
                        # Compute contribution for current partial
                        assign = []
                        for vname in f.scope:
                            if vname == vn:
                                # will iterate over s
                                assign.append(None)  # placeholder
                            else:
                                assign.append(partial[vname])
                        for s in var.domain:
                            asg_tuple = []
                            k = 0
                            for vname in f.scope:
                                if vname == vn:
                                    asg_tuple.append(int(s))
                                else:
                                    asg_tuple.append(int(assign[k]))
                                    k += 1
                            tup = tuple(asg_tuple)
                            pot = float(f.table.get(tup, 0.0))
                            # multiply by incoming messages from other variables to factor
                            msg_prod = 1.0
                            for v2 in other:
                                mv = msg_vf[(v2, fi)]
                                msg_prod *= max(1e-12, float(mv.get(int(partial[v2]), 0.0)))
                            accum[int(s)] += pot * msg_prod
                        return
                    vname = other[idx]
                    dom = graph.variables[vname].domain
                    for s in dom:
                        partial[vname] = int(s)
                        _enum(idx + 1, partial)
                        del partial[vname]
                _enum(0, {})
                msg_fv[(fi, vn)] = _normalize(accum)
    # Compute marginals
    marginals: Dict[str, Dict[int, float]] = {}
    for vn, var in graph.variables.items():
        prod: Dict[int, float] = {s: 1.0 for s in var.domain}
        for fi in var_to_factors[vn]:
            m = msg_fv[(fi, vn)]
            for s in var.domain:
                prod[s] *= max(1e-12, float(m.get(s, 0.0)))
        marginals[vn] = _normalize(prod)
    return marginals


def run_max_product(graph: FactorGraph, iterations: int = 5) -> Dict[str, int]:
    """Max-product for MAP variable-wise assignment (loopy, greedy decoding)."""
    # Reuse sum-product messages but replace sum by max in factor->var
    var_to_factors: Dict[str, List[int]] = {vn: [] for vn in graph.variables.keys()}
    for fi, f in enumerate(graph.factors):
        for vn in f.scope:
            var_to_factors[vn].append(fi)
    msg_vf: Dict[Tuple[str, int], Dict[int, float]] = {}
    msg_fv: Dict[Tuple[int, str], Dict[int, float]] = {}
    for vn, var in graph.variables.items():
        for fi in var_to_factors[vn]:
            msg_vf[(vn, fi)] = {s: 1.0 for s in var.domain}
            msg_fv[(fi, vn)] = {s: 1.0 for s in var.domain}
    for _ in range(max(1, iterations)):
        for vn, var in graph.variables.items():
            neigh = var_to_factors[vn]
            for fi in neigh:
                prod: Dict[int, float] = {s: 1.0 for s in var.domain}
                for fj in neigh:
                    if fj == fi:
                        continue
                    m = msg_fv[(fj, vn)]
                    for s in var.domain:
                        prod[s] *= max(1e-12, float(m.get(s, 0.0)))
                msg_vf[(vn, fi)] = prod
        for fi, f in enumerate(graph.factors):
            for vn in f.scope:
                var = graph.variables[vn]
                best: Dict[int, float] = {s: 0.0 for s in var.domain}
                other = [v for v in f.scope if v != vn]
                def _enum(idx: int, partial: Dict[str, int]) -> None:
                    if idx >= len(other):
                        assign_fixed = []
                        for vname in f.scope:
                            if vname == vn:
                                assign_fixed.append(None)
                            else:
                                assign_fixed.append(partial[vname])
                        for s in var.domain:
                            asg = []
                            k = 0
                            for vname in f.scope:
                                if vname == vn:
                                    asg.append(int(s))
                                else:
                                    asg.append(int(assign_fixed[k])); k += 1
                            pot = float(f.table.get(tuple(asg), 0.0))
                            prod = 1.0
                            for v2 in other:
                                mv = msg_vf[(v2, fi)]
                                prod *= max(1e-12, float(mv.get(int(partial[v2]), 0.0)))
                            val = pot * prod
                            if val > best[int(s)]:
                                best[int(s)] = val
                        return
                    vname = other[idx]
                    for s in graph.variables[vname].domain:
                        partial[vname] = int(s)
                        _enum(idx + 1, partial)
                        del partial[vname]
                _enum(0, {})
                msg_fv[(fi, vn)] = best
    # Greedy decode argmax per variable
    assignment: Dict[str, int] = {}
    for vn, var in graph.variables.items():
        prod: Dict[int, float] = {s: 1.0 for s in var.domain}
        for fi in var_to_factors[vn]:
            m = msg_fv[(fi, vn)]
            for s in var.domain:
                prod[s] *= max(1e-12, float(m.get(s, 0.0)))
        best_s, best_v = None, -1.0
        for s, v in prod.items():
            if (best_s is None) or (v > best_v):
                best_s, best_v = int(s), float(v)
        assignment[vn] = int(best_s if best_s is not None else var.domain[0])
    return assignment


def build_text_semantic_graph(sfb_factors: List[Any]) -> FactorGraph:
    """Construct a tiny boolean presence graph over semantic predicates.

    - Variables: pred:<token> with domain {0,1}
    - Factors: unary potentials preferring presence for extracted tokens
    """
    tokens: Dict[str, float] = {}
    # Extract from AMR triples
    for f in sfb_factors:
        try:
            meta = getattr(f, 'meta', {}) or {}
            if str(meta.get('type', '')) == 'semantic' and str(meta.get('mode', '')) == 'amr':
                triples = meta.get('triples', [])
                if isinstance(triples, list):
                    for (src, role, tgt) in triples[:32]:
                        try:
                            if str(role).lower() in ('predicate', ':instance', 'instance'):
                                tok = str(tgt).strip().lower()
                                if tok:
                                    tokens[tok] = max(0.5, tokens.get(tok, 0.0))
                        except Exception:
                            continue
        except Exception:
            continue
    # Extract from SRL verbs
    for f in sfb_factors:
        try:
            meta = getattr(f, 'meta', {}) or {}
            if str(meta.get('type', '')) == 'semantic' and str(meta.get('mode', '')) == 'srl':
                roles = meta.get('roles', [])
                if isinstance(roles, list):
                    for r in roles[:16]:
                        v = str((r or {}).get('verb', '')).strip().lower()
                        if v:
                            tokens[v] = max(0.5, tokens.get(v, 0.0))
        except Exception:
            continue
    # Build variables and unary factors
    variables: Dict[str, DiscreteVariable] = {}
    factors: List[DiscreteFactor] = []
    base_bias = 0.6
    try:
        base_bias = float(os.getenv('SFB_SEM_PRESENCE_BIAS', '0.7'))
    except Exception:
        base_bias = 0.7
    for tok, w in list(tokens.items())[:64]:
        vname = f"pred:{tok}"
        variables[vname] = DiscreteVariable(name=vname, domain=(0, 1))
        # Prefer presence with strength from base_bias and token weight
        bias = max(0.5, min(0.95, base_bias * (0.5 + 0.5 * w)))
        table = {(0,): (1.0 - bias), (1,): bias}
        factors.append(DiscreteFactor(scope=(vname,), table=table))
    return FactorGraph(variables=variables, factors=factors)


def build_general_factor_graph(sfb_factors: List[Any]) -> FactorGraph:
    """Lift all known factor types to a small mixed discrete graph.

    - semantic tokens: pred:<token> in {0,1}
    - code/numeric/logic states: coarse latent states in {0,1} (absent,present)
    - add pairwise compatibility for simple co-occurrence (e.g., numeric with logic)
    """
    variables: Dict[str, DiscreteVariable] = {}
    factors: List[DiscreteFactor] = []
    # Semantic leaves
    sem_graph = build_text_semantic_graph(sfb_factors)
    variables.update(sem_graph.variables)
    factors.extend(sem_graph.factors)
    # Coarse latent toggles
    toggles = []
    for tname in ["code", "numeric", "logic", "vision", "audio", "video"]:
        vname = f"latent:{tname}"
        variables[vname] = DiscreteVariable(name=vname, domain=(0, 1))
        # Unary priors from presence of corresponding factors
        prior = 0.5
        for f in sfb_factors:
            meta = getattr(f, 'meta', {}) or {}
            if str(meta.get('type', '')) == tname:
                prior = max(prior, 0.7)
        factors.append(DiscreteFactor(scope=(vname,), table={(0,): (1.0 - prior), (1,): prior}))
        toggles.append(vname)
    # Pairwise compatibility between numeric and logic (they often co-occur)
    # Honor a coarse treewidth knob: if max_treewidth < 2, avoid introducing pairwise factors
    max_tw = 8
    try:
        max_tw = max(1, int(os.getenv('SFB_MAX_TREEWIDTH', '8')))
    except Exception:
        max_tw = 8
    if max_tw >= 2 and "latent:numeric" in variables and "latent:logic" in variables:
        scope = ("latent:numeric", "latent:logic")
        # prefer (1,1) a bit more
        table: Dict[Tuple[int, int], float] = {
            (0, 0): 1.0,
            (0, 1): 0.9,
            (1, 0): 0.9,
            (1, 1): 1.2,
        }
        factors.append(DiscreteFactor(scope=scope, table=table))
    # Additional pairwise: code with tests presence (proxy via code factor presence)
    if max_tw >= 2 and "latent:code" in variables and "latent:numeric" in variables:
        scope2 = ("latent:code", "latent:numeric")
        table2: Dict[Tuple[int, int], float] = {
            (0, 0): 1.0,
            (0, 1): 1.0,
            (1, 0): 0.95,
            (1, 1): 1.15,
        }
        factors.append(DiscreteFactor(scope=scope2, table=table2))
    # Simple higher-order template numeric->code->tests via an auxiliary triad if allowed
    if max_tw >= 3 and all(v in variables for v in ("latent:numeric", "latent:code", "latent:logic")):
        # Factor over (numeric, code, logic): reward (1,1,1) and penalize incompatible (1,0,1)
        scope3 = ("latent:numeric", "latent:code", "latent:logic")
        table3: Dict[Tuple[int, int, int], float] = {}
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    base = 1.0
                    if (a, b, c) == (1, 1, 1):
                        base = 1.25
                    elif (a, b, c) == (1, 0, 1):
                        base = 0.85
                    table3[(a, b, c)] = base
        factors.append(DiscreteFactor(scope=scope3, table=table3))
    # Retrieval ↔ semantic presence: if retrieval exists, slightly encourage semantic tokens
    if max_tw >= 2:
        has_retrieval = False
        for f in sfb_factors:
            meta = getattr(f, 'meta', {}) or {}
            if str(meta.get('type', '')) == 'retrieval':
                has_retrieval = True
                break
        if has_retrieval:
            for vn in list(variables.keys()):
                if vn.startswith('pred:'):
                    # Binary factor linking retrieval latent to semantic variable
                    lat = 'latent:semantic'
                    if lat not in variables:
                        variables[lat] = DiscreteVariable(name=lat, domain=(0, 1))
                        # Neutral unary prior
                        factors.append(DiscreteFactor(scope=(lat,), table={(0,): 0.5, (1,): 0.5}))
                    table4: Dict[Tuple[int, int], float] = {
                        (0, 0): 1.0,
                        (0, 1): 1.0,
                        (1, 0): 0.95,
                        (1, 1): 1.1,
                    }
                    factors.append(DiscreteFactor(scope=(lat, vn), table=table4))
    return FactorGraph(variables=variables, factors=factors)


def semantic_log_marginal_score(marginals: Dict[str, Dict[int, float]]) -> float:
    """Aggregate normalized log-marginals for presence state across variables.

    Uses average of log p(x=1) to keep it scale-invariant over variable count.
    """
    if not marginals:
        return 0.0
    acc = 0.0
    n = 0
    for vn, dist in marginals.items():
        p1 = float(dist.get(1, 0.0))
        acc += math.log(max(1e-8, p1))
        n += 1
    if n <= 0:
        return 0.0
    return float(acc / n)



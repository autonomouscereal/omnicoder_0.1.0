"""
Omega-Intent: pragmatic goal/intent inference (scaffold)

Implements a tiny RSA-style listener, a plan-recognition-as-planning proxy,
and a cooperative IRL-like prior update. Designed to be lightweight and
dependency-free; all heavy learning is deferred.

Public surface:
- infer_goals(utterance, context, k, methods, memory) -> GoalBelief

This returns a GoalBelief payload that upstream code can pass through the
pipeline (causal → planner → verifier → PCA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import math
import os


@dataclass
class GoalHypothesis:
    goal: str
    prior: float
    posterior: float
    rationale: str = ""


@dataclass
class GoalBelief:
    hypotheses: List[GoalHypothesis] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    desiderata: List[str] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict)  # e.g., temperature, alpha


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def _speaker_utility(u: str, g: str, c: Optional[str]) -> float:
    """Tiny proxy for U_speaker(u, g, c): lexical match and intent patterns."""
    try:
        u_l = (u or "").lower()
        g_l = (g or "").lower()
        score = 0.0
        # lexical overlap proxy
        us = set(u_l.split())
        gs = set(g_l.split())
        inter = len(us & gs)
        den = max(1, min(len(us), len(gs)))
        score += inter / den
        # pattern bonuses
        if any(k in u_l for k in ("why", "how", "cause", "reason")) and any(k in g_l for k in ("explain", "cause", "rationale")):
            score += 0.4
        if any(k in u_l for k in ("plan", "steps", "tool", "procedure")) and "plan" in g_l:
            score += 0.4
        if any(k in u_l for k in ("code", "compile", "test")) and "code" in g_l:
            score += 0.3
        if c:
            cs = set(c.lower().split())
            score += 0.2 * (len(us & cs) / max(1, min(len(us), len(cs))))
        return float(score)
    except Exception:
        return 0.0


def _candidate_goals_from_utterance(u: str) -> List[str]:
    u_l = (u or "").lower()
    cands = [
        "answer the question concisely",
        "explain the causes and assumptions",
        "produce a step-by-step plan",
        "write and verify code",
        "retrieve evidence and cite",
    ]
    # Add some intent-shaped variants
    if any(k in u_l for k in ("image", "video", "clip")):
        cands.append("generate or analyze image/video with grounding")
    if any(k in u_l for k in ("audio", "asr", "wer")):
        cands.append("analyze audio and back-check with ASR")
    if any(k in u_l for k in ("proof", "verify", "test")):
        cands.append("provide a proof-carrying answer with tests")
    return cands


def _plan_recognition_bias(u: str, g_list: List[str]) -> List[float]:
    """PR-as-planning proxy: prefer goals that imply fewer unexplained steps."""
    try:
        u_len = len((u or "").split())
        scores = []
        for g in g_list:
            implied_steps = 1
            if "plan" in g:
                implied_steps += 2
            if any(k in g for k in ("code", "verify")):
                implied_steps += 1
            if any(k in g for k in ("evidence", "cite")):
                implied_steps += 1
            # simple prior that favors matching the query length complexity
            target = 1 + (u_len // 12)
            diff = abs(implied_steps - target)
            scores.append(-float(diff))
        return scores
    except Exception:
        return [0.0 for _ in g_list]


def _cirl_prior_update(priors: List[float], memory: Optional[Dict[str, float]]) -> List[float]:
    """Cooperative IRL proxy: nudge priors by revealed preference weights in memory.

    memory keys may include: prefer_plan, prefer_explain, prefer_code, prefer_verify, prefer_cite
    Values are in [0,1].
    """
    if not priors:
        return priors
    prefs = memory or {}
    boosts = [1.0] * len(priors)
    for i, g in enumerate([
        "answer the question concisely",
        "explain the causes and assumptions",
        "produce a step-by-step plan",
        "write and verify code",
        "retrieve evidence and cite",
    ]):
        if "plan" in g:
            boosts[i] *= (1.0 + 0.5 * float(prefs.get("prefer_plan", 0.0)))
        if "explain" in g or "cause" in g:
            boosts[i] *= (1.0 + 0.5 * float(prefs.get("prefer_explain", 0.0)))
        if "code" in g:
            boosts[i] *= (1.0 + 0.5 * float(prefs.get("prefer_code", 0.0)))
        if "verify" in g or "proof" in g:
            boosts[i] *= (1.0 + 0.5 * float(prefs.get("prefer_verify", 0.0)))
        if "cite" in g or "evidence" in g:
            boosts[i] *= (1.0 + 0.5 * float(prefs.get("prefer_cite", 0.0)))
    updated = [max(1e-9, p * b) for p, b in zip(priors, boosts)]
    s = sum(updated)
    return [u / s for u in updated] if s > 0 else priors


def infer_goals(
    utterance: str,
    context: Optional[str] = None,
    k: int = 3,
    methods: Optional[str] = None,
    memory: Optional[Dict[str, float]] = None,
) -> GoalBelief:
    """Return a small goal posterior over latent intents.

    methods: comma list from env OMNI_GOAL_INFER or OMNICODER_GOAL_INFER; supports 'rsa','pr','cirl'.
    """
    try:
        m_env = methods or os.getenv("OMNI_GOAL_INFER", os.getenv("OMNICODER_GOAL_INFER", "rsa,pr,cirl"))
        use_rsa = "rsa" in m_env
        use_pr = "pr" in m_env
        use_cirl = "cirl" in m_env

        cand_goals = _candidate_goals_from_utterance(utterance)
        # initialize priors uniformly
        priors = [1.0 / len(cand_goals)] * len(cand_goals) if cand_goals else []

        # RSA listener: p(u|g,c) ∝ softmax(α * U_speaker)
        alpha = float(os.getenv("OMNICODER_RSA_ALPHA", os.getenv("OMNI_RSA_ALPHA", "1.0")))
        if use_rsa and cand_goals:
            utils = [alpha * _speaker_utility(utterance, g, context) for g in cand_goals]
            like = _softmax(utils)
        else:
            like = [1.0 / len(cand_goals)] * len(cand_goals) if cand_goals else []

        # PR-as-planning heuristic: bias by implied step cost alignment
        if use_pr and cand_goals:
            pr_bias = _softmax(_plan_recognition_bias(utterance, cand_goals))
        else:
            pr_bias = [1.0 / len(cand_goals)] * len(cand_goals) if cand_goals else []

        # Combine: posterior ∝ like * prior * pr_bias
        post = [max(1e-9, l * p * b) for l, p, b in zip(like, priors, pr_bias)]
        s = sum(post)
        post = [x / s for x in post] if s > 0 else priors

        # CIRL preference update on priors
        if use_cirl:
            post = _cirl_prior_update(post, memory)

        # Rank and take top-k (use nlargest to avoid full sort for small k)
        top = [i for i, _ in heapq.nlargest(max(1, int(k)), enumerate(post), key=lambda kv: kv[1])]
        out = GoalBelief()
        out.meta = {"alpha": alpha}
        for i in top:
            out.hypotheses.append(GoalHypothesis(goal=cand_goals[i], prior=1.0 / len(cand_goals), posterior=float(post[i])))
        # Very light constraints/desiderata extraction from utterance keywords
        u_l = (utterance or "").lower()
        if "concise" in u_l:
            out.desiderata.append("concise")
        if any(k in u_l for k in ("proof", "verify", "test")):
            out.constraints.append("must-verify")
        return out
    except Exception:
        return GoalBelief()



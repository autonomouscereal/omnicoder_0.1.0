"""
Omega-Reasoner controller (scaffold)

This module provides a minimal, non-invasive integration surface for a typed
Reasoning Graph (RG) that can sit alongside the existing HRM module. It exposes
lightweight hooks that default to no-ops so enabling it via env flags will not
break existing runs.

Key responsibilities (scaffolded):
- Maintain a small state for RG budgets and acceptance thresholds
- Optionally transform prompts using a Neuro-Symbolic Scratchpad (NSS)
- Provide acceptance/halting heuristics and expose knobs to the decode loop
- Route speculative graph branches (delegated to graph_speculative)

All heavy-weight functionality is intentionally deferred. This file is
structured to be safe to import even when optional deps are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import os


@dataclass
class OmegaConfig:
    rg_max_nodes: int = int(os.getenv("OMNICODER_RG_MAX_NODES", "64"))
    rg_max_depth: int = int(os.getenv("OMNICODER_RG_MAX_DEPTH", "8"))
    rg_budget_tokens: int = int(os.getenv("OMNICODER_RG_BUDGET_TOKENS", "1024"))
    rg_spec_branches: int = int(os.getenv("OMNICODER_RG_SPECULATIVE_BRANCHES", "1"))
    rg_accept_margin: float = float(os.getenv("OMNICODER_RG_ACCEPT_MARGIN", "0.0"))
    delib_heads: bool = os.getenv("OMNICODER_DELIB_HEADS", "0") == "1"
    halting: bool = os.getenv("OMNICODER_HALTING", "0") == "1"
    halting_alpha: float = float(os.getenv("OMNICODER_HALTING_ALPHA", "0.7"))
    halting_beta: float = float(os.getenv("OMNICODER_HALTING_BETA", "0.2"))
    halting_gamma: float = float(os.getenv("OMNICODER_HALTING_GAMMA", "0.1"))
    halting_theta: float = float(os.getenv("OMNICODER_HALTING_THETA", "0.85"))
    block_verify: bool = os.getenv("OMNICODER_BLOCK_VERIFY", "1") == "1"
    block_size: int = int(os.getenv("OMNICODER_BLOCK_SIZE", "4"))
    factors: str = os.getenv("OMNICODER_FACTORS", "")
    factors_iters: int = int(os.getenv("OMNICODER_FACTORS_ITERS", "1"))
    constraints: str = os.getenv("OMNICODER_CONSTRAINTS", "text")
    constraints_weight: str = os.getenv("OMNICODER_CONSTRAINT_WEIGHT", "auto")
    rt_vq_codes: int = int(os.getenv("OMNICODER_RT_VQ_CODES", "0"))
    rt_min_sim: float = float(os.getenv("OMNICODER_RT_MIN_SIM", "0.0"))
    cis_cache: bool = os.getenv("OMNICODER_CIS_CACHE", "0") == "1"
    prefix_hydrate: bool = os.getenv("OMNICODER_PREFIX_HYDRATE", "0") == "1"


@dataclass
class OmegaState:
    goals: list[str] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)


class OmegaController:
    """Lightweight Ω-Controller facade.

    Methods are intentionally conservative and avoid touching model internals.
    """

    def __init__(self, config: Optional[OmegaConfig] = None):
        self.cfg = config or OmegaConfig()
        self.state = OmegaState()

    def plan_and_transform_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Optionally transform the prompt by inserting a minimal scaffold.

        Returns (new_prompt, plan_metadata). Defaults to identity.
        """
        # In future, parse goals and seed NSS with typed facts here.
        return prompt, {}

    def acceptance_probability(self, entropy: float, kl_delta: float, verifier_margin: float) -> float:
        import math
        alpha = float(self.cfg.halting_alpha)
        beta = float(self.cfg.halting_beta)
        gamma = float(self.cfg.halting_gamma)
        # Map inputs into [0,1]-like features.
        s = alpha * (1.0 - max(0.0, min(1.0, entropy))) + beta * (1.0 - max(0.0, min(1.0, kl_delta))) + gamma * verifier_margin
        return 1.0 / (1.0 + math.exp(-s))

    def should_halt(self, p_accept: float, constraints_ok: bool) -> bool:
        return constraints_ok and (p_accept >= float(self.cfg.halting_theta))

    def suggest_decode_knobs(self) -> Dict[str, Any]:
        """Emit lightweight knobs for the autoregressive loop.

        These mirror existing CLI/env flags so the generator can respect them
        without requiring deeper surgery.
        """
        knobs: Dict[str, Any] = {
            "block_verify": bool(self.cfg.block_verify),
            "block_verify_size": int(max(1, self.cfg.block_size)),
            "speculative_branches": int(max(1, self.cfg.rg_spec_branches)),
        }
        # Optional simple tuning: enlarge block size when halting is active to amortize verifier cost
        try:
            if self.cfg.halting and knobs["block_verify_size"] < 8 and os.getenv("OMNICODER_BLOCK_VERIFY_TUNE", "0") == "1":
                knobs["block_verify_size"] = 8
        except Exception:
            pass
        return knobs

    def load_sidecars(self, root: str = "weights") -> Dict[str, Any]:
        """Load optional sidecar hints for constraints/templates/landmarks.

        Returns a dict with available hints, safe to call if files are missing.
        """
        out: Dict[str, Any] = {}
        try:
            import json
            from pathlib import Path
            p = Path(root)
            cons = p / 'constraints.json'
            if cons.exists():
                out['constraints'] = json.loads(cons.read_text(encoding='utf-8'))
            tpl = p / 'templates.json'
            if tpl.exists():
                out['templates'] = json.loads(tpl.read_text(encoding='utf-8'))
            lmk = p / 'landmarks.idx'
            if lmk.exists():
                out['landmarks_idx_path'] = str(lmk)
        except Exception:
            pass
        return out

    # Future: expose router temperature scheduler by entropy for dynamic exploration
    # and learned halting critic once trained. Kept minimal here to avoid changing
    # model internals.



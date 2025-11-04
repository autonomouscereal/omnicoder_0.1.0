from __future__ import annotations

"""
Semantic-Factoring Brain (SFB) package.

This package runs alongside the main LLM decoding path to:
 - Factorize prompts into a factor graph (FNF)
 - Run message passing / MAP to produce beliefs and factor scores
 - Optionally compile hot subgraphs to SPNs
 - Provide cross-biasing signals to the LLM logits
 - Gate speculative acceptance via a proof-margin arbiter

All components are designed to be optional and fail-safe. If SFB_ENABLE!=1,
imports or runtime errors are swallowed and the system behaves as a no-op.
"""

from .factorize import FactorizationResult, factorize_prompt
from .inference.bp import BeliefPropagation
from .compile.spn import SPNCompiler
from .fusion.cross_bias import CrossBiasFusion
from .arbiter import ProofMarginArbiter, ProofMarginInputs
from .inference.factor_graph import (
    FactorGraph,
    DiscreteVariable,
    DiscreteFactor,
    build_text_semantic_graph,
    run_sum_product,
    semantic_log_marginal_score,
)

__all__ = [
    "FactorizationResult",
    "factorize_prompt",
    "BeliefPropagation",
    "SPNCompiler",
    "CrossBiasFusion",
    "ProofMarginArbiter",
    "ProofMarginInputs",
    "FactorGraph",
    "DiscreteVariable",
    "DiscreteFactor",
    "build_text_semantic_graph",
    "run_sum_product",
    "semantic_log_marginal_score",
]



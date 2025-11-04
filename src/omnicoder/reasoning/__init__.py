"""
Reasoning package initializer.

Exposes lightweight Ω-Solver scaffolds alongside existing modules. Safe-to-import
stubs ensure missing optional deps do not break runs.
"""

from __future__ import annotations

# Re-export commonly used helpers when available
try:
    from .omega_intent import infer_goals  # type: ignore
except Exception:
    infer_goals = None  # type: ignore

try:
    from .omega_causal import (
        SCM,  # type: ignore
        build_minimal_scm_for_query,  # type: ignore
        abductive_score,  # type: ignore
        value_of_information,  # type: ignore
    )
except Exception:
    SCM = None  # type: ignore
    build_minimal_scm_for_query = None  # type: ignore
    abductive_score = None  # type: ignore
    value_of_information = None  # type: ignore

try:
    from .omega_planner import mcts_search  # type: ignore
except Exception:
    mcts_search = None  # type: ignore

try:
    from .omega_verifier import compute_margin  # type: ignore
except Exception:
    compute_margin = None  # type: ignore

try:
    from .omega_pca import pack as pack_certificate, to_json as cert_to_json  # type: ignore
except Exception:
    pack_certificate = None  # type: ignore
    cert_to_json = None  # type: ignore

# New adaptive reasoning components
try:
    from .adaptive_graph import build_agot  # type: ignore
except Exception:
    build_agot = None  # type: ignore

try:
    from .latent_bfs import build_latent_bfs  # type: ignore
except Exception:
    build_latent_bfs = None  # type: ignore

try:
    from .reflection import build_reflection  # type: ignore
except Exception:
    build_reflection = None  # type: ignore

try:
    from .symbolic_planner import build_symbolic_planner  # type: ignore
except Exception:
    build_symbolic_planner = None  # type: ignore

__all__ = [
    "infer_goals",
    "SCM",
    "build_minimal_scm_for_query",
    "abductive_score",
    "value_of_information",
    "mcts_search",
    "compute_margin",
    "pack_certificate",
    "cert_to_json",
    # new
    "build_agot",
    "build_latent_bfs",
    "build_reflection",
    "build_symbolic_planner",
]



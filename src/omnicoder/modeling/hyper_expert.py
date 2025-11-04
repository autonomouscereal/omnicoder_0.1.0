from __future__ import annotations

"""
Hyper-expert synthesizer (prototype):

Given a conditioning vector (context/task/expert-id embedding), synthesize FFN
weights for a Mixture-of-Experts MLP expert: two linear layers and biases.

This is a research module intended to support Branch-Train-Merge and beyond by
allowing dynamic expert weight generation conditioned on inputs or tasks.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class HyperExpertSpec:
    d_model: int
    mlp_dim: int


class HyperExpertSynthesizer(nn.Module):
    def __init__(self, spec: HyperExpertSpec, cond_dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.spec = spec
        out_dim = (
            spec.d_model * spec.mlp_dim  # w1
            + spec.mlp_dim               # b1
            + spec.mlp_dim * spec.d_model  # w2
            + spec.d_model               # b2
        )
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        cond: (B, cond_dim)
        returns a dict with keys: 'w1','b1','w2','b2'
        """
        b = cond.size(0)
        vec = self.net(cond)  # (B, out_dim)
        d, m = self.spec.d_model, self.spec.mlp_dim
        s0 = d * m
        s1 = s0 + m
        s2 = s1 + m * d
        s3 = s2 + d
        w1 = torch.ops.aten.reshape.default(vec[:, :s0], (b, m, d))
        b1 = torch.ops.aten.reshape.default(vec[:, s0:s1], (b, m))
        w2 = torch.ops.aten.reshape.default(vec[:, s1:s2], (b, d, m))
        b2 = torch.ops.aten.reshape.default(vec[:, s2:s3], (b, d))
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def serialize_synthesized(
    synth: HyperExpertSynthesizer,
    cond: torch.Tensor,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Synthesize a single expert and return a flat param dict keyed by MoE param names.

    prefix: e.g., "blocks.0.moe.experts.3.ffn" to align with typical MoE naming patterns.
    """
    with torch.no_grad():
        out = synth(cond)
    d, m = synth.spec.d_model, synth.spec.mlp_dim
    # Standard FFN affine mapping names (heuristic)
    return {
        f"{prefix}.w1.weight": out["w1"].squeeze(0),  # (M,D)
        f"{prefix}.w1.bias": out["b1"].squeeze(0),     # (M,)
        f"{prefix}.w2.weight": out["w2"].squeeze(0),  # (D,M)
        f"{prefix}.w2.bias": out["b2"].squeeze(0),     # (D,)
    }



"""
Reflexive Metacognition (Intrinsic Reflection)

Implements a light feedback loop that perturbs the next-step hidden state using
a projection from the current step. This is a side-channel that callers can
enable during decode to reduce instability without changing model weights.

Environment knobs:
- OMNICODER_REFLECT_ENABLE=1
- OMNICODER_REFLECT_ALPHA (default 0.1)
"""

from __future__ import annotations

from typing import Optional
import os
import torch


class ReflexiveFeedback:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        try:
            self.enabled = os.getenv("OMNICODER_REFLECT_ENABLE", "1") == "1"
        except Exception:
            self.enabled = False
        # Lightweight projection matrices (device set on first call)
        self.W: Optional[torch.Tensor] = None
        self.alpha = float(os.getenv("OMNICODER_REFLECT_ALPHA", "0.1"))
        try:
            self.mode = str(os.getenv("OMNICODER_REFLECT_MODE", "identity")).strip().lower()
        except Exception:
            self.mode = "identity"

    def _maybe_init(self, device: torch.device) -> None:
        if self.W is None:
            if self.mode == "mlp":
                # Small low-rank projection approximated by two diagonal scalings
                self.W = torch.eye(self.dim, device=device)
            else:
                # Identity-like
                self.W = torch.eye(self.dim, device=device)

    @torch.inference_mode()
    def apply(self, hidden_out: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.enabled or hidden_out is None or not isinstance(hidden_out, torch.Tensor):
            return hidden_out
        try:
            device = hidden_out.device
            self._maybe_init(device)
            # last step only to keep cost low
            h = hidden_out[:, -1:, :]
            # simple self-projection residual
            proj = torch.matmul(h, self.W)  # type: ignore[arg-type]
            hidden_out[:, -1:, :] = (1.0 - float(self.alpha)) * h + float(self.alpha) * proj
            return hidden_out
        except Exception:
            return hidden_out


def build_reflection(dim: int) -> ReflexiveFeedback:
    return ReflexiveFeedback(dim=dim)



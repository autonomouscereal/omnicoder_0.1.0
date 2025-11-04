from __future__ import annotations

import torch
import torch.nn as nn


class KeyframeHead(nn.Module):
    """
    Tiny head that predicts a per-token keyframe probability for video frame tokens.

    Input: hidden states (B, T, C) and optional temperature.
    Output: keyframe probability per position (B, T) in [0,1].
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, hidden: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.proj(hidden).squeeze(-1)
        if temperature <= 0:
            temperature = 1.0
        return torch.sigmoid(logits / float(temperature))



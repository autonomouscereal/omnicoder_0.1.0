from __future__ import annotations

import torch, torch.nn as nn, torch.nn.functional as F


class GatedMambaSSM(nn.Module):
    """
    Lightweight Mamba-like SSM block (export-disabled). Intended for full-sequence
    passes only; decode-step path should not invoke this module.
    """

    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 7):
        super().__init__()
        hidden = int(d_model * expansion)
        self.in_proj = nn.Linear(d_model, hidden, bias=False)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        h = self.in_proj(x)  # (B,T,H)
        g = torch.sigmoid(self.gate(x))
        # Depthwise 1D conv across time
        h_t = h.transpose(1, 2)  # (B,H,T)
        h_t = self.conv(h_t)
        h = h_t.transpose(1, 2)
        h = h * g
        return self.out(h)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConvSSM(nn.Module):
    """Lightweight gated 1D-conv sequence module (SSM-style surrogate).

    This block mixes information along the time axis with separable depthwise
    convolution and a gated linear unit. It has no KV cache and is intended
    to be used during full-sequence passes (training, priming). For decode-step
    streaming (use_cache=True), callers should skip this block to keep a stable
    recurrent state limited to the KV cache.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, expansion: int = 2, dropout_p: float = 0.0):
        super().__init__()
        hidden = int(d_model * expansion)
        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, hidden * 2, bias=False)  # for GLU
        self.dw_conv = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, groups=hidden, padding=kernel_size // 2)
        self.proj_out = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.norm(x)
        u, v = self.proj_in(x).chunk(2, dim=-1)  # (B,T,H)
        v = F.gelu(v)
        y = u * v  # GLU-like gating
        y = y.transpose(1, 2)  # (B,H,T)
        y = self.dw_conv(y)
        y = y.transpose(1, 2)  # (B,T,H)
        y = self.proj_out(y)
        y = self.dropout(y)
        return y



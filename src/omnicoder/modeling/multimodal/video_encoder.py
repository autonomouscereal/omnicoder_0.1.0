from __future__ import annotations

import torch
import torch.nn as nn


class VideoBackbone(nn.Module):
    """
    Lightweight video encoder that consumes a stack of frames and outputs
    a token sequence plus an optional pooled embedding.

    Input shape: (B, F, 3, H, W) where F is number of frames.
    """

    def __init__(self, d_model: int = 768, return_pooled: bool = True) -> None:
        super().__init__()
        self.return_pooled = bool(return_pooled)
        self.embed = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3)),
            nn.GELU(),
            nn.Conv3d(64, d_model, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.GELU(),
        )
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # frames: (B, F, 3, H, W)
        b, f, c, h, w = frames.shape
        x = frames.transpose(1, 2)  # (B,3,F,H,W)
        h3d = self.embed(x)  # (B, d_model, F', H', W')
        b, d, f2, h2, w2 = h3d.shape
        tokens = torch.ops.aten.reshape.default(h3d, (b, d, f2 * h2 * w2)).transpose(1, 2)  # (B, N, d)
        pooled = self.pooler(tokens.transpose(1, 2)).reshape(tokens.shape[0], tokens.shape[2], 1).squeeze(-1) if self.return_pooled else None
        return tokens, pooled

import torch, torch.nn as nn


class TemporalSSM(nn.Module):
    """Export-friendly temporal mixing for video latents.

    A small 1D depthwise convolution + GLU gate over the time dimension.
    Designed to be ONNX-friendly and device efficient.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        hidden = int(d_model * expansion)
        self.proj_in = nn.Linear(d_model, hidden * 2, bias=False)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, groups=hidden, padding=kernel_size // 2)
        self.proj_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        u, v = self.proj_in(x).chunk(2, dim=-1)
        v = torch.nn.functional.gelu(v)
        y = u * v
        y = y.transpose(1, 2)
        y = self.dw(y)
        y = y.transpose(1, 2)
        return self.proj_out(y)

class SimpleVideoEncoder(nn.Module):
    """Encode N frames by applying ViT-tiny per-frame and temporal GRU over CLS tokens."""

    def __init__(self, frame_encoder: nn.Module, d_model: int = 384, use_temporal_ssm: bool = True):
        super().__init__()
        self.frame_enc = frame_encoder
        self.temporal = nn.GRU(d_model, d_model, batch_first=True) if not use_temporal_ssm else TemporalSSM(d_model)

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, H, W)
        bsz, t, _, _, _ = frames.shape
        cls_list = []
        for i in range(t):
            tokens = self.frame_enc(frames[:, i])  # (B, Ti, C)
            cls_list.append(tokens[:, :1, :].reshape(tokens.shape[0], tokens.shape[2]))  # (B, C)
        seq = torch.stack(cls_list, dim=1)  # (B, T, C)
        if isinstance(self.temporal, nn.GRU):
            # Provide h0 explicitly for ONNX export to avoid batch-size dependent shapes
            b = seq.shape[0]
            d = seq.shape[-1]
            h0 = torch.ops.aten.new_zeros.default(seq, (1, b, d))
            out, _ = self.temporal(seq, h0)
            return out
        else:
            return self.temporal(seq)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(ch)
        self.norm2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class TinyImageRefiner(nn.Module):
    """
    Lightweight image refiner (few residual conv blocks).

    Accepts a tensor image in range [0,1], shape (B,3,H,W) and returns a refined tensor
    of the same shape. Designed to run for a few small steps (apply model N times).
    """

    def __init__(self, base_channels: int = 32, num_blocks: int = 3) -> None:
        super().__init__()
        ch = int(base_channels)
        self.in_conv = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(max(1, int(num_blocks)))])
        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

    @torch.inference_mode()
    def forward(self, img: torch.Tensor, steps: int = 1) -> torch.Tensor:
        x = img
        for _ in range(max(1, int(steps))):
            h = F.silu(self.in_conv(x))
            h = self.blocks(h)
            delta = torch.tanh(self.out_conv(h)) * 0.05
            x = torch.clamp(x + delta, 0.0, 1.0)
        return x



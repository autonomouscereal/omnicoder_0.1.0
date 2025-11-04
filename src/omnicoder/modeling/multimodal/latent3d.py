from __future__ import annotations

import torch
import torch.nn as nn


class VoxelLatentHead(nn.Module):
    """
    Project hidden tokens to a compact voxel latent grid suitable for tiny 3D rendering.

    Inputs: hidden (B, T, C)
    Returns: voxels (B, D, H, W) where D is depth slices.
    """

    def __init__(self, d_model: int, depth: int = 16, height: int = 32, width: int = 32, hidden: int = 256) -> None:
        super().__init__()
        self.depth = int(depth)
        self.height = int(height)
        self.width = int(width)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, depth * height * width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B,T,C) → take last token as scene code
        code = hidden[:, -1, :]
        vox = self.proj(code)
        return torch.ops.aten.reshape.default(vox, (vox.size(0), self.depth, self.height, self.width))


class SimpleOrthoRenderer(nn.Module):
    """
    Tiny orthographic renderer: projects voxel densities to an RGB image by learned
    per-slice color basis and alpha compositing along the depth axis.
    """

    def __init__(self, depth: int = 16, out_h: int = 256, out_w: int = 256) -> None:
        super().__init__()
        self.depth = int(depth)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.color_basis = nn.Parameter(torch.randn(depth, 3) * 0.1)

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        # vox: (B, D, H, W) in arbitrary range
        # Avoid converting symbolic shapes to Python ints inside traced/exported paths.
        # We no longer assert on depth; instead we upsample/truncate to the configured depth below.
        b, d, h, w = vox.shape
        # Resize to output with nearest/bilinear upsample
        x = vox.unsqueeze(1)  # (B,1,D,H,W)
        x = torch.nn.functional.interpolate(x, size=(self.depth, self.out_h, self.out_w), mode='trilinear', align_corners=False)
        x = x.squeeze(1)  # (B,D,H,W)
        # Alpha compositing: compute per-slice alpha via sigmoid and colors via basis
        alpha = torch.sigmoid(x)  # (B,D,H,W)
        colors = torch.tensordot(alpha, self.color_basis, dims=([1],[0]))  # (B,H,W,3)
        # Normalize to [0,1]
        rgb = torch.clamp(colors, 0.0, 1.0).permute(0,3,1,2)  # (B,3,H,W)
        return rgb



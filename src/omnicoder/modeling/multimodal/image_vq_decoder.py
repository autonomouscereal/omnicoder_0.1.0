from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class ImageVQDecoder(nn.Module):
    """
    Standalone image decoder for VQ-VAE tokens (indices).

    Inputs:
      - indices: LongTensor of shape (B, Hq, Wq)
    Outputs:
      - image: FloatTensor of shape (B, 3, H, W) in [0,1]

    The spatial output size is (Hq*patch, Wq*patch).
    """

    def __init__(self, codebook: torch.Tensor, code_dim: int, patch: int = 16) -> None:
        super().__init__()
        # aten-only dtype cast; avoid .float()
        self.register_buffer("embedding", torch.ops.aten.to.dtype(torch.ops.aten.mul.Scalar(codebook.detach(), 1.0), torch.float32, False, False))  # (K,D)
        self.code_dim = int(code_dim)
        self.patch = int(patch)
        # Lightweight decoder mirrors ImageDecoder from vqvae.py
        self.decoder = nn.Sequential(
            nn.Conv2d(self.code_dim, 128, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B,Hq,Wq)
        b, hq, wq = indices.shape
        flat = torch.ops.aten.to.dtype(torch.ops.aten.reshape.default(indices, (b, -1)), torch.long, False, False)  # (B, Hq*Wq)
        # Gather code vectors and reshape to feature map
        emb = self.embedding.index_select(0, torch.ops.aten.reshape.default(flat, (-1,)))  # (B*Hq*Wq, D)
        z_q = torch.ops.aten.reshape.default(emb, (b, hq, wq, self.code_dim)).permute(0, 3, 1, 2)
        # Decode and bound to [0,1]
        x_rec = torch.sigmoid(self.decoder(z_q))
        return x_rec

    @torch.inference_mode()
    def exportable_dummy_inputs(self, grid_hw: Tuple[int, int] = (14, 14)) -> torch.Tensor:
        hq, wq = int(grid_hw[0]), int(grid_hw[1])
        return torch.zeros(1, hq, wq, dtype=torch.long)

    @staticmethod
    def from_codebook_file(path: str | Path) -> "ImageVQDecoder":
        # Prefer safe weight-only loading when available (PyTorch >= 2.4)
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            blob = torch.load(str(path), map_location="cpu")
        codebook: torch.Tensor = torch.ops.aten.to.dtype(blob["codebook"], torch.float32, False, False)
        code_dim: int = int(blob.get("emb_dim", codebook.size(1)))
        patch: int = int(blob.get("patch", 16))
        return ImageVQDecoder(codebook=codebook, code_dim=code_dim, patch=patch)



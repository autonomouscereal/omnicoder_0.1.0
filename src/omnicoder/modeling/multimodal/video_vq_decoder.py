from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class VideoVQDecoder(nn.Module):
    """
    Minimal video decoder from a list of frame token grids (indices -> frames).

    Inputs:
      - indices_list: list of LongTensor each of shape (B, Hq, Wq)
    Outputs:
      - video: FloatTensor of shape (B, T, 3, H, W) in [0,1]
    """

    def __init__(self, codebook: torch.Tensor, code_dim: int, patch: int = 16) -> None:
        super().__init__()
        self.register_buffer("embedding", torch.ops.aten.to.dtype(torch.ops.aten.mul.Scalar(codebook.detach(), 1.0), torch.float32, False, False))
        self.code_dim = int(code_dim)
        self.patch = int(patch)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.code_dim, 128, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
        )

    def decode_frame(self, indices: torch.Tensor) -> torch.Tensor:
        b, hq, wq = indices.shape
        flat = torch.ops.aten.to.dtype(torch.ops.aten.reshape.default(indices, (b, -1)), torch.long, False, False)
        emb = self.embedding.index_select(0, torch.ops.aten.reshape.default(flat, (-1,)))
        z_q = torch.ops.aten.reshape.default(emb, (b, hq, wq, self.code_dim)).permute(0, 3, 1, 2)
        return torch.sigmoid(self.decoder(z_q))  # (B,3,H,W)

    def forward(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        frames = [self.decode_frame(idx) for idx in indices_list]
        video = torch.stack(frames, dim=1)
        return video

    @staticmethod
    def from_codebook_file(path: str | None, code_dim: int, patch: int = 16) -> "VideoVQDecoder":
        cb = torch.randn(8192, code_dim)
        if path:
            try:
                blob = torch.load(path, map_location="cpu")
                codebook = blob.get("codebook") if isinstance(blob, dict) else None
                if isinstance(codebook, torch.Tensor) and codebook.dim() == 2:
                    cb = codebook
                    code_dim = int(cb.size(1))
            except Exception:
                pass
        return VideoVQDecoder(codebook=cb, code_dim=code_dim, patch=patch)



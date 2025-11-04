from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class _Patchify(nn.Module):
    def __init__(self, patch: int = 16):
        super().__init__()
        self.patch = patch

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        # x: (B,3,H,W)
        p = self.patch
        b, c, h, w = x.shape
        assert h % p == 0 and w % p == 0
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // p) * (w // p), p * p * c)
        return x  # (B, N, 3*p*p)


class ImageVQ:
    """
    Minimal VQ encoder/decoder interface using a fixed k-means codebook on patch embeddings.

    - encode(image: np.ndarray) -> List[np.ndarray]: returns [codes] with shape (N_patches,)
    - decode(tokens: List[np.ndarray]) -> Optional[np.ndarray]: reconstructs approximation (optional)

    This lightweight module is meant as a placeholder to wire VQ token I/O without heavy deps.
    If torch is unavailable, it falls back to a no-op stub.
    """

    def __init__(self, patch: int = 16, codebook_size: int = 8192, code_dim: int = 192, codebook_path: str | None = None):
        self.patch = int(patch)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self._device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self._patchify = _Patchify(patch) if torch is not None else None
        # Randomly init a codebook; in practice, fit with k-means over training images
        self._codebook = None
        if torch is not None:
            if codebook_path and Path(codebook_path).exists():
                try:
                    blob = torch.load(codebook_path, map_location='cpu')
                    cb = blob.get('codebook') if isinstance(blob, dict) else blob
                    if isinstance(cb, torch.Tensor) and cb.dim() == 2:
                        self._codebook = nn.Parameter(cb.to(self._device), requires_grad=False)
                        self.code_dim = int(cb.size(1))
                        self.codebook_size = int(cb.size(0))
                except Exception:
                    self._codebook = nn.Parameter(torch.randn(self.codebook_size, self.code_dim) * 0.02, requires_grad=False)
            if self._codebook is None:
                self._codebook = nn.Parameter(torch.randn(self.codebook_size, self.code_dim) * 0.02, requires_grad=False)

    def _patch_embed(self, img: np.ndarray) -> Optional["torch.Tensor"]:  # type: ignore[name-defined]
        if torch is None:
            return None
        x = torch.ops.aten.to.dtype(torch.from_numpy(img).to(self._device), torch.float32, False, False)  # (H,W,3) or (3,H,W)
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        x = x / 255.0
        patches = self._patchify(x)  # (1, N, 3*p*p)
        # Simple linear projection to code_dim
        proj = patches @ torch.randn(patches.size(-1), self.code_dim, device=patches.device) * 0.02
        proj = F.normalize(proj, dim=-1)
        return proj.squeeze(0)  # (N, D)

    def encode(self, image: np.ndarray) -> List[np.ndarray]:
        if torch is None or self._codebook is None:
            # Fallback: trivial 1D token stream (downsampled grayscale)
            img = image
            if img.ndim == 3 and img.shape[-1] == 3:
                gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
            else:
                gray = img.astype(np.float32)
            small = gray[::self.patch, ::self.patch]
            return [small.flatten().astype(np.int32)]
        # Real path: nearest neighbor quantization against codebook
        emb = self._patch_embed(image)  # (N, D)
        cb = F.normalize(self._codebook, dim=-1)  # (K, D)
        # cosine distance -> argmax similarity
        sim = emb @ cb.T  # (N, K)
        ids = torch.argmax(sim, dim=-1).detach().cpu().numpy().astype(np.int32)
        return [ids]

    def decode(self, tokens: List[np.ndarray]) -> Optional[np.ndarray]:
        if torch is None or self._codebook is None:
            return None
        if not tokens:
            return None
        ids = torch.from_numpy(tokens[0].astype(np.int64)).to(self._device)  # (N,)
        code = self._codebook.index_select(0, ids)  # (N, D)
        # Inverse linear projection (pseudo): map code to pixel patches via a learned linear (random here)
        inv = code @ torch.randn(self.code_dim, 3 * self.patch * self.patch, device=code.device) * 0.02
        inv = torch.ops.aten.reshape.default(inv, (-1, 3, self.patch, self.patch))  # (N,3,p,p)
        # Reassemble into image assuming square grid
        n = inv.size(0)
        side = int(np.sqrt(n))
        if side * side != n:
            return None
        rows = []
        for r in range(side):
            from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
            row = _safe_concat([inv[r * side + c] for c in range(side)], 2)  # (3,p,side*p)
            rows.append(row)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        img = _safe_concat(rows, 1)  # (3, side*p, side*p)
        img = (img.clamp(-1, 1) * 127.5 + 127.5).byte().permute(1, 2, 0).cpu().numpy()
        return img

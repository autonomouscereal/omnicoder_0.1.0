from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class _FrameEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, emb_dim: int = 192, patch: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, emb_dim, kernel_size=patch, stride=patch)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        y = self.conv(x)  # (B,D,H/p,W/p)
        y = y.flatten(2).transpose(1, 2)  # (B,N,D)
        return F.normalize(y, dim=-1)


class VideoVQ:
    """
    Simple video VQ encoder/decoder using a codebook over frame patch embeddings.
    Encodes a BTCHW video tensor/np.ndarray into a sequence of discrete codes.
    """

    def __init__(self, patch: int = 16, codebook_size: int = 8192, code_dim: int = 192, codebook_path: str | None = None) -> None:
        self.patch = int(patch)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self._device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self._frame_enc = _FrameEncoder(emb_dim=self.code_dim, patch=self.patch) if torch is not None else None
        self._codebook = None
        if torch is not None:
            # Ensure frame encoder lives on the working device
            if self._frame_enc is not None:
                self._frame_enc = self._frame_enc.to(self._device).eval()

            if codebook_path and Path(codebook_path).exists():
                try:
                    blob = torch.load(codebook_path, map_location='cpu')
                    cb = blob.get('codebook') if isinstance(blob, dict) else blob
                    if isinstance(cb, torch.Tensor) and cb.dim() == 2:
                        self._codebook = nn.Parameter(cb.to(self._device), requires_grad=False)
                        self.code_dim = int(cb.size(1))
                        self.codebook_size = int(cb.size(0))
                except Exception:
                    self._codebook = nn.Parameter((torch.randn(self.codebook_size, self.code_dim, device=self._device) * 0.02), requires_grad=False)
            if self._codebook is None:
                self._codebook = nn.Parameter((torch.randn(self.codebook_size, self.code_dim, device=self._device) * 0.02), requires_grad=False)

    def encode(self, frames: np.ndarray) -> List[np.ndarray]:
        if torch is None or self._codebook is None:
            # Fallback: naive downsample per-frame
            if frames.ndim == 4:  # (T,H,W,C)
                frames = frames[None, ...]  # (1,T,H,W,C)
            bt, t, h, w, c = frames.shape
            step = max(1, int(self.patch))
            tokens = []
            for i in range(t):
                fr = frames[0, i]
                gray = (0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2])
                small = gray[::step, ::step]
                tokens.append(small.flatten().astype(np.int32))
            return tokens
        # torch path
        if frames.ndim == 4:  # (T,H,W,C)
            frames = frames[None, ...]
        bt, t, h, w, c = frames.shape
        # Be robust to environments where torch numpy bridge may be limited; fall back to CPU copy
        try:
            xt = torch.from_numpy(frames)
        except Exception:
            xt = torch.tensor(frames.copy())  # type: ignore[arg-type]
        x = torch.ops.aten.to.dtype(xt.to(self._device), torch.float32, False, False).permute(0, 1, 4, 2, 3)  # (B,T,C,H,W)
        x = x / 255.0
        # Encode each frame
        codes: List[np.ndarray] = []
        cb = F.normalize(self._codebook, dim=-1)
        for i in range(t):
            fi = x[:, i]
            emb = self._frame_enc(fi)  # (B,N,D)
            emb = emb.squeeze(0)
            sim = emb @ cb.t()
            ids = torch.argmax(sim, dim=-1).detach().cpu().numpy().astype(np.int32)
            codes.append(ids)
        return codes

    def decode(self, tokens: List[np.ndarray]) -> Optional[np.ndarray]:
        if torch is None or self._codebook is None or not tokens:
            return None
        # Decode frame-wise as average of codes per patch to rough pixels
        frames: List[np.ndarray] = []
        for ids_np in tokens:
            ids = torch.from_numpy(ids_np.astype(np.int64)).to(self._device)
            code = self._codebook.index_select(0, ids)
            inv = code @ torch.randn(self.code_dim, 3 * self.patch * self.patch, device=code.device) * 0.02
            inv = torch.ops.aten.reshape.default(inv, (-1, 3, self.patch, self.patch))
            n = inv.size(0)
            side = int(np.sqrt(n))
            if side * side != n:
                continue
            rows = []
            for r in range(side):
                from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
                row = _safe_concat([inv[r * side + c] for c in range(side)], 2)
                rows.append(row)
            from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
            img = _safe_concat(rows, 1).clamp(-1, 1)
            img = (img * 127.5 + 127.5).byte().permute(1, 2, 0).cpu().numpy()
            frames.append(img)
        if not frames:
            return None
        return np.stack(frames, axis=0)

# NOTE: Removed legacy stub class that shadowed the real implementation above.

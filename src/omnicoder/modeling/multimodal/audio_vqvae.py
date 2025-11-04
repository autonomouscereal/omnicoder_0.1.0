from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioVQCodebookMeta:
    codebook_size: int
    code_dim: int
    hop: int


class VectorQuantizerEMA1D(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        embed = torch.randn(self.num_embeddings, self.embedding_dim)
        embed = F.normalize(embed, dim=-1)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embed_avg", torch.ops.aten.mul.Scalar(self.embedding, 1.0))

    @torch.no_grad()
    def _ema_update(self, flat_inputs: torch.Tensor, indices: torch.Tensor) -> None:
        # VERBOSE: aten-only one_hot + dtype cast; avoid F.one_hot and .type_as
        one_hot = torch.ops.aten.one_hot.default(indices, int(self.num_embeddings))
        one_hot = torch.ops.aten.to.dtype(one_hot, flat_inputs.dtype, False, False)
        cluster_size = one_hot.sum(0)
        embed_sum = one_hot.t() @ flat_inputs
        self.cluster_size.data = (self.cluster_size.data * self.decay + cluster_size * (1.0 - self.decay))
        self.embed_avg.data = (self.embed_avg.data * self.decay + embed_sum * (1.0 - self.decay))
        n = torch.ops.aten.mul.Scalar(self.quant.cluster_size, 1.0)
        cluster_norm = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        embed_norm = self.embed_avg / cluster_norm.unsqueeze(1)
        embed_norm = F.normalize(embed_norm, dim=-1)
        self.embedding.copy_(embed_norm)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, Tq)
        b, d, tq = z.shape
        flat = z.permute(0, 2, 1).reshape(b * tq, d)
        ez = F.normalize(flat, dim=-1)
        e = F.normalize(self.embedding, dim=-1)
        sim = ez @ e.t()
        idx = sim.argmax(dim=-1)
        z_q = torch.ops.aten.reshape.default(self.embedding.index_select(0, idx), (b, tq, d)).permute(0, 2, 1)
        if self.training:
            self._ema_update(flat.detach(), idx.detach())
        z_q_st = z + (z_q - z).detach()
        return z_q_st, torch.ops.aten.reshape.default(idx, (b, tq)), z_q


class AudioEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, 8, 4, 2),  # /4
            nn.GELU(),
            nn.Conv1d(base, base, 8, 4, 2),   # /4
            nn.GELU(),
            nn.Conv1d(base, out_dim, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioDecoder(nn.Module):
    def __init__(self, in_dim: int = 128, base: int = 128, out_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, base, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose1d(base, base, 8, 4, 2),
            nn.GELU(),
            nn.ConvTranspose1d(base, out_ch, 8, 4, 2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AudioVQVAE(nn.Module):
    """
    1D VQ‑VAE for audio waveforms at normalized range [-1,1].
    Downsamples by 16x (two stride‑4 convs), vector‑quantizes at the latent rate, then decodes.
    """

    def __init__(self, codebook_size: int = 2048, code_dim: int = 128, decay: float = 0.99) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.encoder = AudioEncoder(out_dim=self.code_dim)
        self.decoder = AudioDecoder(in_dim=self.code_dim)
        self.quant = VectorQuantizerEMA1D(self.codebook_size, self.code_dim, decay=decay)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,1,T) in [-1,1]
        z = self.encoder(x)
        z_q_st, idx, z_q = self.quant(z)
        xr = self.decoder(z_q_st)
        rec = F.mse_loss(xr, x)
        com = F.mse_loss(z.detach(), z_q)
        with torch.no_grad():
            n = torch.ops.aten.mul.Scalar(self.quant.cluster_size, 1.0)
            n = n / (n.sum() + 1e-6)
            ppx = torch.exp(-torch.sum(n * torch.log(n + 1e-6)))
        return rec, com, ppx, xr, idx

    @torch.inference_mode()
    def encode_numpy(self, wav: np.ndarray) -> np.ndarray:
        self.eval()
        if wav.ndim == 1:
            wav = wav[None, :]
        t = torch.ops.aten.to.dtype(torch.from_numpy(wav), torch.float32, False, False).clamp(-1.0, 1.0)
        x = t.unsqueeze(0)  # (1,1,T)
        rec, com, ppx, xr, idx = self.forward(x)
        return idx.squeeze(0).detach().cpu().numpy().astype(np.int32)

    @torch.inference_mode()
    def decode_numpy(self, idx: np.ndarray) -> np.ndarray:
        self.eval()
        tq = int(idx.size)
        ids = torch.from_numpy(idx.astype(np.int64))
        ids = torch.ops.aten.reshape.default(ids, (1, tq))
        z_q = self.quant.embedding.index_select(0, torch.ops.aten.reshape.default(ids, (-1,)))
        z_q = torch.ops.aten.reshape.default(z_q, (1, tq, self.code_dim)).permute(0, 2, 1)
        xr = self.decoder(z_q)
        wav = xr.squeeze(0).squeeze(0).clamp(-1, 1).detach().cpu().numpy()
        return wav

    @torch.inference_mode()
    def export_codebook(self, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "codebook": self.quant.embedding.detach().cpu(),
            "emb_dim": int(self.code_dim),
            "hop": 16,  # two stride‑4 convs
        }
        _safe_save(blob, out_path)



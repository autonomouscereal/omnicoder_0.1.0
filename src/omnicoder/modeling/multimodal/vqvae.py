import math
from typing import Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
    """EMA Vector Quantizer (codebook with exponential moving averages).

    - code_dim: embedding dimension of codes
    - num_codes: size of the codebook
    - decay: EMA decay
    - eps: numerical stability
    """

    def __init__(self, code_dim: int, num_codes: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.code_dim = int(code_dim)
        self.num_codes = int(num_codes)
        self.decay = float(decay)
        self.eps = float(eps)

        embed = torch.randn(num_codes, code_dim)
        embed = F.normalize(embed, dim=1)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", torch.ops.aten.mul.Scalar(embed, 1.0))

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, codes: torch.Tensor) -> None:
        # VERBOSE: Replace F.one_hot + .type with aten.one_hot + aten.to.dtype to keep aten-only targets
        onehot = torch.ops.aten.one_hot.default(codes, int(self.num_codes))
        onehot = torch.ops.aten.to.dtype(onehot, flat.dtype, False, False)
        cluster_size = onehot.sum(0)
        embed_sum = onehot.t() @ flat

        # Avoid autograd version bumps during training: operate on .data
        self.cluster_size.data = (self.cluster_size.data * self.decay + cluster_size * (1 - self.decay))
        self.embed_avg.data = (self.embed_avg.data * self.decay + embed_sum * (1 - self.decay))

        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))
        self.embed.copy_(F.normalize(self.embed, dim=1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize z (B, C, H, W) to nearest code vectors.
        Returns (z_q, codes) where z_q has same shape as z and codes are (B, H, W) ints.
        """
        B, C, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        # distances to codebook embeddings
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x.e
        x2 = (flat * flat).sum(1, keepdim=True)
        e2 = (self.embed * self.embed).sum(1, keepdim=True).t()
        logits = x2 + e2 - 2 * (flat @ self.embed.t())
        codes = torch.argmin(logits, dim=1)
        z_q = torch.ops.aten.reshape.default(self.embed[codes], (B, H, W, C)).permute(0, 3, 1, 2)
        # EMA updates (train only)
        if self.training:
            self._ema_update(flat.detach(), codes.detach())
        return z_q, torch.ops.aten.reshape.default(codes, (B, H, W))


class ImageEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, 4, 2, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 4, 2, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageDecoder(nn.Module):
    def __init__(self, out_ch: int = 3, dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(),
            nn.ConvTranspose2d(dim, out_ch, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ImageVQVAE(nn.Module):
    """Compact VQ-VAE for images (for codebook training and export).

    - codebook_size: number of discrete codes
    - code_dim: latent channel dimension
    """

    def __init__(self, codebook_size: int = 8192, code_dim: int = 192, ema_decay: float = 0.99):
        super().__init__()
        self.encoder = ImageEncoder(dim=code_dim)
        self.quant = EMAQuantizer(code_dim=code_dim, num_codes=codebook_size, decay=ema_decay)
        self.decoder = ImageDecoder(dim=code_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, codes = self.quant(z)
        x_rec = self.decoder(z_q)
        return x_rec, codes

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, codes = self.quant(z)
        return codes

    @torch.no_grad()
    def decode_indices(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (B, H, W) -> (B, C, H, W)
        emb = self.quant.embed[torch.ops.aten.reshape.default(codes, (-1,))]
        emb = torch.ops.aten.reshape.default(emb, (codes.size(0), codes.size(1), codes.size(2), -1))
        z_q = emb.permute(0, 3, 1, 2)
        return self.decoder(z_q)

    @torch.no_grad()
    def save_codebook(self, path: str) -> None:
        _safe_save({"embed": self.quant.embed.cpu(), "num_codes": self.quant.num_codes, "code_dim": self.quant.code_dim}, path)

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQCodebookMeta:
    codebook_size: int
    code_dim: int
    patch: int


class VectorQuantizerEMA(nn.Module):
    """
    Exponential moving average vector quantizer.

    - codebook: (K, D) embedding vectors
    - inputs: (B, D, H, W) continuous latents
    - outputs: quantized latents (B, D, H, W), code indices (B, H, W)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
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
        # flat_inputs: (N, D), indices: (N,)
        # VERBOSE: aten-only one_hot; avoid .type_as to keep dtype-only cast via aten.to.dtype
        one_hot = torch.ops.aten.one_hot.default(indices, int(self.num_embeddings))
        one_hot = torch.ops.aten.to.dtype(one_hot, flat_inputs.dtype, False, False)
        cluster_size = one_hot.sum(0)
        embed_sum = one_hot.t() @ flat_inputs

        self.cluster_size.data = (self.cluster_size.data * self.decay + cluster_size * (1.0 - self.decay))
        self.embed_avg.data = (self.embed_avg.data * self.decay + embed_sum * (1.0 - self.decay))

        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps)
            / (n + self.num_embeddings * self.eps)
            * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        embed_normalized = F.normalize(embed_normalized, dim=-1)
        self.embedding.copy_(embed_normalized)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, H, W)
        b, d, h, w = z.shape
        flat_z = z.permute(0, 2, 3, 1).reshape(b * h * w, d)  # (N, D)
        # distances via cosine similarity (normalize)
        ez = F.normalize(flat_z, dim=-1)
        e = F.normalize(self.embedding, dim=-1)
        sim = ez @ e.t()  # (N, K)
        idx = torch.argmax(sim, dim=-1)  # (N,)
        z_q = torch.ops.aten.reshape.default(self.embedding.index_select(0, idx), (b, h, w, d)).permute(0, 3, 1, 2)

        if self.training:
            self._ema_update(flat_z.detach(), idx.detach())

        # commitment loss (straight-through estimator)
        z_q_st = z + (z_q - z).detach()
        return z_q_st, torch.ops.aten.reshape.default(idx, (b, h, w)), z_q


class ImageEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 128, out_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(base, base, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(base, out_dim, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageDecoder(nn.Module):
    def __init__(self, in_dim: int = 192, base: int = 128, out_ch: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, base, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(base, base, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(base, out_ch, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ImageVQVAE(nn.Module):
    """
    Patch-wise VQ-VAE for images (per 4x downsample, i.e., patch=4*4=16 effective area).
    Encoder downsamples by 4, vector-quantizes latent features per spatial location, then decodes.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        code_dim: int = 192,
        patch: int = 16,
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.patch = int(patch)
        self.encoder = ImageEncoder(out_dim=self.code_dim)
        self.decoder = ImageDecoder(in_dim=self.code_dim)
        self.quant = VectorQuantizerEMA(self.codebook_size, self.code_dim, decay=decay)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,3,H,W) in [0,1]
        z = self.encoder(x)
        z_q_st, idx, z_q = self.quant(z)
        x_rec = torch.sigmoid(self.decoder(z_q_st))
        # losses
        rec_loss = F.mse_loss(x_rec, x)
        # commitment: encourage encoder outputs to commit to code vectors
        commit_loss = F.mse_loss(z.detach(), z_q)
        with torch.no_grad():
            # perplexity (code usage quality)
            n_codes = torch.ops.aten.mul.Scalar(self.quant.cluster_size, 1.0)
            n_codes = n_codes / (n_codes.sum() + 1e-6)
            perplexity = torch.exp(-torch.sum(n_codes * torch.log(n_codes + 1e-6)))
        return rec_loss, commit_loss, perplexity, x_rec, idx

    @torch.inference_mode()
    def encode_numpy(self, img: np.ndarray) -> np.ndarray:
        # img: (H,W,3) uint8 or float32 [0,1]
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(img)
        if x.dtype != torch.float32:
            x = torch.ops.aten.to.dtype(x, torch.float32, False, False)
        if x.max() > 1.0:
            x = x / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        _rec, _com, _ppx, _xrec, idx = self.forward(x)
        return idx.squeeze(0).detach().cpu().numpy().astype(np.int32)

    @torch.inference_mode()
    def decode_numpy(self, idx: np.ndarray, grid_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        self.eval()
        device = next(self.parameters()).device
        # idx: (Hq,Wq) code indices
        if idx.ndim == 1 and grid_shape is not None:
            hq, wq = grid_shape
            idx = idx.reshape(hq, wq)
        if idx.ndim != 2:
            return None
        hq, wq = int(idx.shape[0]), int(idx.shape[1])
        idx_t = torch.from_numpy(idx.astype(np.int64)).to(device)
        z_q = self.quant.embedding.index_select(0, torch.ops.aten.reshape.default(idx_t, (-1,)))
        z_q = torch.ops.aten.reshape.default(z_q, (hq, wq, self.code_dim))
        z_q = z_q.permute(2, 0, 1).unsqueeze(0)
        x_rec = torch.sigmoid(self.decoder(z_q))
        img = (x_rec.squeeze(0).permute(1, 2, 0) * 255.0).clamp(0, 255).byte().cpu().numpy()
        return img

    @torch.inference_mode()
    def export_codebook(self, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "codebook": self.quant.embedding.detach().cpu(),
            "emb_dim": int(self.code_dim),
            "patch": int(self.patch),
        }
        _safe_save(blob, out_path)

    @torch.inference_mode()
    def save_codebook(self, out_path: str) -> None:
        """Alias for backward compatibility with training scripts."""
        self.export_codebook(out_path)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to discrete code indices shaped (B, T).

        Returns a flattened sequence of codes to match downstream trainer expectations.
        """
        self.eval()
        z = self.encoder(x)
        _, idx, _ = self.quant(z)
        # idx: (B, Hq, Wq) -> (B, T)
        b, h, w = idx.shape
        return idx.reshape(b, h * w)



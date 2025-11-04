from __future__ import annotations

import torch
import torch.nn as nn


class TextEmbedder(nn.Module):
    """
    Tiny learnable text embedder that maps token ids to a pooled embedding.
    This mirrors the training-side helper and is used at inference to produce
    a fixed-dimension text vector compatible with `PreAligner`.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(int(vocab_size), int(embed_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(ids)  # (B, T, D)
        return e.mean(dim=1)  # (B, D)


class HiddenToImageCond(nn.Module):
    """
    Lightweight aligner mapping LLM hidden states (d_model) to image-condition vectors.

    This can be extended to cross-attention or FiLM conditioning. For now, we pool
    the last token and apply a small MLP.
    """

    def __init__(self, d_model: int, cond_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        # Optional FiLM-like conditioning (scale/shift) to modulate U-Net blocks when supported
        self.film_scale = nn.Linear(cond_dim, cond_dim)
        self.film_shift = nn.Linear(cond_dim, cond_dim)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden: (B, T, C) or (B, C)
        if hidden.dim() == 3:
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden
        cond = self.proj(pooled)
        scale = self.film_scale(cond)
        shift = self.film_shift(cond)
        return cond, scale, shift


class ContinuousLatentHead(nn.Module):
    """
    Optional continuous latent head for non-text modalities (e.g., image/audio).

    Produces D-dimensional continuous tokens per position; to be consumed by an
    external decoder (e.g., flow-matching or diffusion decoder).
    """

    def __init__(self, d_model: int, latent_dim: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B,T,C) -> (B,T,D_latent)
        return self.proj(hidden)


class ConceptLatentHead(nn.Module):
    """
    Projects transformer hidden states into a shared, normalized concept embedding space.

    This head is intended to be aligned (via InfoNCE/triplet losses) with modality
    embeddings produced by `PreAligner` to create a unified latent space across
    experts and modalities.
    """

    def __init__(self, d_model: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B,T,C) or (B,C) → (B,E)
        if hidden.dim() == 3:
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden
        z = self.net(pooled)
        return nn.functional.normalize(z, dim=-1)


class PreAligner(nn.Module):
    """
    Lightweight pre-alignment encoders that map modality-specific features
    (text/image/audio/video) into a shared embedding space for improved
    routing and cross-modal coherence.

    Each modality has a small projection head; the heads can be trained with
    a contrastive objective (e.g., InfoNCE) prior to multimodal fusion.
    """

    def __init__(self, embed_dim: int = 256, text_dim: int = 512, image_dim: int = 768,
                 audio_dim: int = 512, video_dim: int = 768) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(text_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.image_proj = nn.Sequential(nn.Linear(image_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.video_proj = nn.Sequential(nn.Linear(video_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B,T,C) or (B,C)
        if tokens.dim() == 3:
            return tokens.mean(dim=1)
        return tokens

    def forward(self, *,
                text: torch.Tensor | None = None,
                image: torch.Tensor | None = None,
                audio: torch.Tensor | None = None,
                video: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if text is not None:
            out["text"] = nn.functional.normalize(self.text_proj(self._pool(text)), dim=-1)
        if image is not None:
            out["image"] = nn.functional.normalize(self.image_proj(self._pool(image)), dim=-1)
        if audio is not None:
            out["audio"] = nn.functional.normalize(self.audio_proj(self._pool(audio)), dim=-1)
        if video is not None:
            out["video"] = nn.functional.normalize(self.video_proj(self._pool(video)), dim=-1)
        return out

    @staticmethod
    def info_nce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Compute a symmetric InfoNCE loss between two normalized batches a and b.
        a, b: (B, D) normalized.
        """
        # Ensure dtype match to avoid bf16/float matmul issues under CPU autocast
        if a.dtype != b.dtype:
            # Prefer float32 for numerical stability
            a = a.float()
            b = b.float()
        # Scale by temperature via aten to avoid Python max()/division
        _zero = torch.ops.aten.mul.Scalar(a, 0.0)
        _t = torch.ops.aten.add.Scalar(torch.ops.aten.sum.default(_zero), float(temperature))
        _t = torch.ops.aten.clamp_min.default(_t, 1e-6)
        logits = torch.ops.aten.div.Tensor((a @ b.t()), _t)
        # Avoid .size: derive batch via a reduction
        _basev = torch.ops.aten.sum.dim_IntList(a, [1], False)  # (B,)
        labels = torch.ops.aten.ones_like.default(_basev, dtype=torch.long)
        labels = torch.ops.aten.cumsum.default(labels, 0)
        labels = torch.ops.aten.sub.Tensor(labels, torch.ops.aten.new_ones.default(labels, labels.shape, dtype=labels.dtype))
        loss_a = nn.functional.cross_entropy(logits, labels)
        loss_b = nn.functional.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)

    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
        """
        Margin triplet loss for normalized embeddings. Encourages anchor to be
        closer to positive than to negative by a margin.
        """
        # Cosine distances
        d_pos = 1.0 - (anchor * positive).sum(dim=-1)
        d_neg = 1.0 - (anchor * negative).sum(dim=-1)
        return torch.clamp(d_pos - d_neg + float(margin), min=0.0).mean()


class CrossModalVerifier(nn.Module):
    """
    A tiny verifier that scores alignment between two modality embeddings.
    Inputs should be normalized embeddings from `PreAligner`.
    Returns a scalar score in [0, 1] via sigmoid on cosine similarity.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: (B, D) normalized
        sim = (a * b).sum(dim=-1, keepdim=True)
        return torch.sigmoid(sim)

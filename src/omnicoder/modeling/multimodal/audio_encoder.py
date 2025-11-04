from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioBackbone(nn.Module):
    """
    Lightweight audio encoder producing token sequence and pooled embedding.

    Prefers torchaudio MelSpectrogram when available; otherwise, computes a
    simple log-magnitude spectrogram via torch.stft. Features are passed
    through a small ConvNet → Transformer pooling to yield embeddings.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64, d_model: int = 512, return_pooled: bool = True) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.return_pooled = bool(return_pooled)
        self.n_mels = int(n_mels)
        self.d_model = int(d_model)

        # Frontend mel or spectrogram projection
        self._use_torchaudio = False
        try:
            import torchaudio  # type: ignore
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels, n_fft=400, hop_length=160
            )
            self._use_torchaudio = True
        except Exception:
            self.melspec = None

        # Conv projection to d_model
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(64, self.d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
        )
        # Pooler to get a single embedding
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def _log_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute a log-magnitude spectrogram or mel-spectrogram.
        wav: (B, T)
        returns: (B, 1, F, T')
        """
        if self._use_torchaudio and self.melspec is not None:
            with torch.no_grad():
                feat = self.melspec(wav)  # (B, n_mels, frames)
        else:
            # Plain spectrogram fallback
            win = 400
            hop = 160
            spec = torch.stft(wav, n_fft=win, hop_length=hop, return_complex=True)
            feat = spec.abs()  # (B, F, frames)
            # Reduce to n_mels bands by average pooling along freq axis
            if feat.size(1) > self.n_mels:
                k = feat.size(1) // self.n_mels
                feat = F.avg_pool1d(feat, kernel_size=k, stride=k)
        feat = torch.log(feat + 1e-5)
        return feat.unsqueeze(1)  # (B,1,F,T')

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # wav: (B, T)
        spec = self._log_spec(wav)  # (B,1,F,T')
        h = self.conv(spec)  # (B, C=d_model, F', T'')
        b, c, f, t = h.shape
        tokens = torch.ops.aten.reshape.default(h, (b, c, f * t)).transpose(1, 2)  # (B, N, C)
        pooled = None
        if self.return_pooled:
            pooled = self.pooler(tokens.transpose(1, 2)).squeeze(-1)  # (B, C)
        return tokens, pooled



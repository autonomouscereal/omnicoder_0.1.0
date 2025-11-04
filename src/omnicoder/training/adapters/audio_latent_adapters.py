from __future__ import annotations

"""
Audio latent adapters for reconstruction training.

- EnCodecAdapter: uses the EnCodec model to produce codec latents from audio
- MelAdapter: produces mel features and projects to a latent vector
- ONNXAudioEncoderAdapter: loads an ONNX encoder callable if available
"""

from typing import Optional
import warnings

# Suppress librosa's pkg_resources deprecation warning (setuptools>=81)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.intervals")

import torch


class EnCodecAdapter:
    def __init__(self, sr: int = 32000, device: Optional[str] = None):
        try:
            from encodec import EncodecModel  # type: ignore
            from encodec.utils import convert_audio  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("EnCodec is required for EnCodecAdapter. Install with extras [audio].") from e
        self._encodec = EncodecModel.encodec_model_24khz() if sr <= 24000 else EncodecModel.encodec_model_48khz()
        self._encodec.set_target_bandwidth(6.0)
        self.sr = sr
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._convert = convert_audio
        self._encodec.to(self.device)
        self._encodec.eval()

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (T,) in [-1,1] -> latent vector (D) pooled across codebooks/time."""
        wav = wav.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,T)
        wav = self._convert(wav, self.sr, self._encodec.sample_rate, self._encodec.channels)
        encoded_frames = self._encodec.encode(wav)
        # encoded_frames: list per chunk, each is list of codebooks -> (B, K, T_codes)
        # Pool across time and codebooks by averaging
        codes = []
        for chunk in encoded_frames:
            # tensor shape (B, n_q, T_codes)
            cb = torch.cat([c for c in chunk], dim=1)
            codes.append(cb.float().mean(dim=2))  # (B, n_q)
        z = torch.cat(codes, dim=1).squeeze(0)  # (n_q_total,)
        return z


class MelAdapter:
    def __init__(self, n_mels: int = 80, out_dim: int = 16):
        # Use librosa if available; otherwise fall back to a simple mel-like projection on short inputs
        self._has_librosa = False
        try:
            import librosa  # type: ignore
            _ = librosa.__version__
            self._has_librosa = True
        except Exception:
            self._has_librosa = False
        self.n_mels = n_mels
        self.out_dim = out_dim

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        # If input already looks like a mel-spectrogram (n_mels, T), pool directly
        if wav.dim() == 2 and int(wav.size(0)) == int(self.n_mels):
            m_t = wav.float().detach().cpu()
            pooled = m_t.mean(dim=1)  # (n_mels,)
            w = torch.randn((self.n_mels, self.out_dim), dtype=pooled.dtype)
            return pooled @ w
        y = wav.detach().cpu().numpy()
        m = None
        if self._has_librosa:
            try:
                import librosa  # type: ignore
                n_fft = min(2048, max(64, int(2 ** (int((len(y) + 1).bit_length())))))
                hop = max(32, min(512, n_fft // 4))
                m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=n_fft, hop_length=hop).astype('float32')
            except Exception:
                m = None
        if m is None:
            # Fallback: build a tiny mel-like filterbank and apply to a toy STFT magnitude
            import numpy as _np
            T = max(16, y.shape[0])
            win = min(64, T)
            step = max(16, win // 2)
            mags = []
            for i in range(0, T - win + 1, step):
                seg = y[i:i+win]
                spec = _np.fft.rfft(seg * _np.hanning(win)).astype(_np.complex64)
                mags.append(_np.abs(spec))
            if not mags:
                mags = [_np.abs(_np.fft.rfft(y * _np.hanning(T)))]
            M = _np.stack(mags, axis=1)  # (F, frames)
            F = M.shape[0]
            fb = _np.linspace(0, 1, F, dtype=_np.float32)
            fb = _np.stack([_np.power(fb, 0.5 + k * 0.5) for k in range(self.n_mels)], axis=0)  # (n_mels, F)
            fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-6)
            m = (fb @ M).astype('float32')  # (n_mels, frames)
        m_t = torch.from_numpy(m)
        # Ensure 2D (n_mels, frames); pool over time to (n_mels,)
        if m_t.dim() == 1:
            m_t = m_t.unsqueeze(1)
        pooled = m_t.mean(dim=1)  # (n_mels,)
        # Random projection to out_dim for latent vector with correct shapes
        w = torch.randn((self.n_mels, self.out_dim), dtype=pooled.dtype)
        z = pooled @ w  # (out_dim,)
        return z


class ONNXAudioEncoderAdapter:
    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("onnxruntime is required for ONNXAudioEncoderAdapter.") from e
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.detach().cpu().numpy().astype('float32')
        if x.ndim == 1:
            x = x[None, None, :]
        outs = self.sess.run(None, {self.sess.get_inputs()[0].name: x})
        z = torch.from_numpy(outs[0])
        if z.dim() > 1:
            z = z.mean(dim=tuple(range(1, z.dim())))
        return z



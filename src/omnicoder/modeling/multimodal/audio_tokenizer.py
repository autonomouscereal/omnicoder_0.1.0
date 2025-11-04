from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class AudioTokenizer:
    """
    EnCodec/DAC-compatible audio tokenizer wrapper.

    Tries to load `facebook/encodec` via the `encodec` Python package. If unavailable,
    falls back to a trivial passthrough stub.
    """

    def __init__(self, sample_rate: int = 32000, model_name: str = "encodec_24khz") -> None:
        # Allow environment override to lock EnCodec small for mobile
        try:
            import os as _os
            env_name = _os.getenv("OMNICODER_AUDIO_TOKENIZER", "").strip()
            if env_name:
                model_name = env_name
        except Exception:
            pass
        self.sample_rate = int(sample_rate)
        self.model_name = model_name
        self._backend = None
        self._model = None
        self._load_backend()

    def _load_backend(self) -> None:
        if self._backend is not None:
            return
        # Allow explicit small-tokenizer selection
        try:
            import os as _os
            if _os.getenv("OMNICODER_AUDIO_TOKENIZER", "").strip().lower() == "small":
                self._backend = "small"
                return
        except Exception:
            pass
        # Try EnCodec reference implementation (prefer small configs)
        try:
            from encodec import EncodecModel  # type: ignore
            from encodec.utils import convert_audio  # type: ignore
            self._encodec_convert = convert_audio
            m = EncodecModel.encodec_model_24khz() if "24" in self.model_name else EncodecModel.encodec_model_48khz()
            m.set_target_bandwidth(6.0)  # small bandwidth for mobile
            self._model = m.eval()
            self._backend = "encodec"
            return
        except Exception:
            pass
        # Best-effort: try Descript Audio Codec (DAC) when available
        try:
            # Lazy import to avoid hard dependency
            from dac import DAC  # type: ignore
            target = "24khz" if self.sample_rate <= 24000 else "44khz"
            self._model = DAC.load(target)
            self._backend = "dac"
            return
        except Exception:
            pass
        # Fallback small tokenizer (mu-law-like integer codes)
        self._backend = "small"

    @staticmethod
    def _to_tensor(wave: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        assert torch is not None, "PyTorch required for audio tokenization"
        if wave.ndim == 1:
            wave = wave[None, :]
        t = torch.from_numpy(wave).float()  # (C, T)
        return t

    def encode(self, audio_wave: np.ndarray) -> List[np.ndarray]:
        """
        Returns a list of codebooks, each an int32 numpy array shaped (T_codes,).
        """
        if self._backend == "encodec":
            assert torch is not None
            with torch.inference_mode():
                wav_t = self._to_tensor(audio_wave)
                wav_t = self._encodec_convert(wav_t, self.sample_rate, self._model.sample_rate, self._model.channels)
                # EnCodec expects input of shape (B, C, T). Our helper returns (C, T).
                # Expand batch dimension to 1 using pure torch ops to satisfy the backend.
                if wav_t.dim() == 2:
                    wav_t = wav_t.unsqueeze(0)
                frames = self._model.encode(wav_t)
                # frames: list of lists of code tensors per frame, shape (C, T_frame)
                # Flatten along time
                codes = [torch.cat([c[0] for c in frames], dim=-1).cpu().numpy().astype(np.int32) for c in zip(*frames)]
                return list(codes)
        if self._backend == "dac":  # best-effort API usage; falls back if unavailable
            try:
                assert torch is not None
                wav_t = self._to_tensor(audio_wave)
                # Attempt simple resample via torchaudio if needed
                try:
                    import torchaudio  # type: ignore
                    model_sr = getattr(self._model, "sample_rate", self.sample_rate)
                    if int(model_sr) != int(self.sample_rate):
                        wav_t = torchaudio.functional.resample(wav_t, self.sample_rate, int(model_sr))
                except Exception:
                    pass
                # Many DAC builds expose encode() returning (latents, codes, scale)
                out = self._model.encode(wav_t.unsqueeze(0))  # type: ignore[attr-defined]
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    codes = out[1]
                    if isinstance(codes, (list, tuple)):
                        return [c.squeeze().cpu().numpy().astype(np.int32) for c in codes]
                # Fallback: quantize latents if codes are not available
                if isinstance(out, (list, tuple)) and len(out) >= 1:
                    lat = out[0]
                    lat_np = lat.squeeze().detach().cpu().numpy()
                    # Simple uniform quantization to 8-bit as a tiny tokenizer
                    lat_q = np.clip((lat_np - lat_np.min()) / (lat_np.ptp() + 1e-6), 0, 1)
                    lat_q = (lat_q * 255.0).astype(np.int32)
                    return [lat_q.reshape(-1)]
            except Exception:
                # Fall through to small tokenizer
                pass
        # Small tokenizer: 8-bit mu-law-ish codes (device-friendly)
        x = np.clip(audio_wave, -1.0, 1.0)
        x = ((x + 1.0) * 127.5).astype(np.int32)
        return [x]

    def decode(self, codes: List[np.ndarray]) -> Optional[np.ndarray]:
        if self._backend == "encodec":
            assert torch is not None
            with torch.inference_mode():
                # Rebuild frames from concatenated codes by splitting into uniform chunks is non-trivial
                # For simplicity, decode as a single chunk using model.decode([(codes_per_codebook,)])
                code_tensors = [torch.from_numpy(c.astype(np.int32))[None, None, :] for c in codes]
                wav = self._model.decode([(code_tensors,)])  # type: ignore[arg-type]
                out = wav[0].cpu().numpy()
                return out
        if self._backend == "dac":
            try:
                assert torch is not None
                # Some DAC builds expose decode() from codes directly
                code_tensors = [torch.from_numpy(c.astype(np.int32))[None, None, :] for c in codes]
                wav = self._model.decode([(code_tensors,)])  # type: ignore[attr-defined]
                return wav[0].cpu().numpy()
            except Exception:
                pass
        # Stub: mu-law-ish dequantization
        if not codes:
            return None
        x = codes[0].astype(np.float32)
        x = (x / 127.5) - 1.0
        return x

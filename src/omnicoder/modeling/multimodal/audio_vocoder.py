from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


class HiFiGANVocoder:
    """
    Lightweight wrapper to run a HiFi-GAN vocoder.

    Backends (auto-detected):
      - coqui_hifigan: Uses Coqui TTS vocoder model loader (recommended for simplicity)
      - onnx: Loads an ONNX Runtime HiFi-GAN generator if you provide a model path

    Usage:
      voc = HiFiGANVocoder(backend='auto', device='cpu')
      wav = voc.vocode(mel)  # mel: (n_mels, T) numpy float32 in log-mel domain expected by backend
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "cpu",
        onnx_model_path: Optional[str] = None,
        coqui_vocoder_name: str = "vocoder_models/en/ljspeech/hifigan_v2",
    ) -> None:
        # Allow environment overrides to lock mobile-friendly backends
        try:
            import os as _os
            env_backend = _os.getenv("OMNICODER_VOCODER_BACKEND", "").strip()
            if env_backend:
                backend = env_backend
            env_path = _os.getenv("OMNICODER_VOCODER_MODEL", "").strip()
            if env_path:
                onnx_model_path = env_path
        except Exception:
            pass
        self.device = device
        self.backend: Optional[str] = None
        self._model = None
        self.onnx_model_path = onnx_model_path
        self.coqui_vocoder_name = coqui_vocoder_name
        self._ensure_backend(backend)

    def _ensure_backend(self, preferred: str = "auto") -> None:
        if self.backend is not None:
            return
        # Try Coqui TTS vocoder
        if preferred in ("auto", "coqui_hifigan"):
            try:
                from TTS.api import TTS  # type: ignore

                # Coqui will download on first use if internet is allowed, or you can
                # replace with a local path like: self.coqui_vocoder_name = "./weights/audio/coqui_hifigan"
                voc = TTS(self.coqui_vocoder_name)
                # The TTS API expects mel tensors with shape (B, n_mels, T)
                self._model = voc
                self.backend = "coqui_hifigan"
                return
            except Exception:
                pass
        # Try ONNX Runtime backend
        if preferred in ("auto", "onnx") and self.onnx_model_path:
            try:
                import onnxruntime as ort  # type: ignore

                sess_opt = ort.SessionOptions()
                self._ort = ort.InferenceSession(self.onnx_model_path, sess_opt, providers=["CPUExecutionProvider"])  # noqa: E501
                self.backend = "onnx"
                return
            except Exception:
                pass
        # As a last resort, try importing a local PyTorch HiFi-GAN generator
        if preferred in ("auto", "torch") and torch is not None:
            try:
                # Expect a TorchScripted generator at onnx_model_path (misnamed var but reused)
                if self.onnx_model_path and Path(self.onnx_model_path).suffix in {".pt", ".pth", ".ts"}:
                    model = torch.jit.load(self.onnx_model_path, map_location=self.device)  # type: ignore[arg-type]
                    model.eval()
                    self._model = model
                    self.backend = "torchscript"
                    return
            except Exception:
                pass
        # If nothing worked, leave backend None
        self.backend = None

    def is_ready(self) -> bool:
        return self.backend is not None and self._model is not None

    @staticmethod
    def _to_tensor(x: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        assert torch is not None
        t = torch.from_numpy(x.astype(np.float32))
        return t

    def vocode(self, mel: np.ndarray, out_path: Optional[str] = None, sample_rate: int = 22050) -> Optional[np.ndarray]:
        """
        Convert a mel-spectrogram to waveform using the configured backend.
        - mel: (n_mels, T) float32 numpy array. Some backends expect log-mel in a specific scale.
        Returns waveform as float32 numpy array of shape (T,), or saves to out_path if supported by backend.
        """
        if not self.is_ready():
            return None
        if self.backend == "coqui_hifigan":
            # Coqui provides a simple API to vocode from mel to wav
            try:
                # The TTS API uses mel in shape (B, n_mels, T)
                mel_b = mel[None, :, :]
                wav = self._model.vocode(mel_b)
                # Some versions return path if out_path is provided; we standardize to array
                if isinstance(wav, (str, Path)):
                    try:
                        import soundfile as sf  # type: ignore
                        data, sr = sf.read(str(wav))
                        return data.astype(np.float32)
                    except Exception:
                        return None
                if isinstance(wav, list):
                    wav = wav[0]
                # Ensure numpy float32 mono
                if hasattr(wav, "cpu"):
                    wav = wav.cpu().numpy()
                wav = np.asarray(wav, dtype=np.float32)
                if wav.ndim > 1:
                    wav = wav.mean(axis=0)
                if out_path:
                    try:
                        import soundfile as sf  # type: ignore
                        p = Path(out_path)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(p), wav, samplerate=sample_rate)
                    except Exception:
                        pass
                return wav
            except Exception:
                return None
        if self.backend == "onnx":
            try:
                # ONNX graph should accept named input, e.g., "mel" -> (1, n_mels, T)
                mel_b = mel[None, :, :].astype(np.float32)
                inputs = {self._ort.get_inputs()[0].name: mel_b}
                out = self._ort.run(None, inputs)[0]  # (1, T) or (1, 1, T)
                wav = out.squeeze().astype(np.float32)
                if out_path:
                    try:
                        import soundfile as sf  # type: ignore
                        p = Path(out_path)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(p), wav, samplerate=sample_rate)
                    except Exception:
                        pass
                return wav
            except Exception:
                return None
        if self.backend == "torchscript":
            try:
                assert torch is not None
                with torch.inference_mode():
                    mel_t = self._to_tensor(mel[None, :, :])  # (1, n_mels, T)
                    mel_t = mel_t.to(self.device)
                    wav_t = self._model(mel_t)
                    wav = wav_t.squeeze().detach().cpu().numpy().astype(np.float32)
                if out_path:
                    try:
                        import soundfile as sf  # type: ignore
                        p = Path(out_path)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(p), wav, samplerate=sample_rate)
                    except Exception:
                        pass
                return wav
            except Exception:
                return None
        return None



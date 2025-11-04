from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ASRAdapter:
    """On-device ASR adapter with multiple backends.

    Tries faster-whisper (preferred), then whisper. Accepts either a path to an
    audio file or a waveform array.
    """

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._backend = None
        self._model = None

    def _ensure_backend(self) -> bool:
        if self._backend is not None:
            return True
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = WhisperModel(self.model_size, device="cpu")
            self._backend = "faster_whisper"
            return True
        except Exception:
            pass
        # Fallback to whisper
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model(self.model_size)
            self._backend = "whisper"
            return True
        except Exception:
            return False

    def transcribe(self, audio: str | np.ndarray) -> Optional[str]:
        if not self._ensure_backend():
            return None
        if isinstance(audio, str):
            path = Path(audio)
            if self._backend == "faster_whisper":
                segments, _ = self._model.transcribe(str(path))
                return " ".join([s.text for s in segments])
            else:
                import whisper  # type: ignore
                result = self._model.transcribe(str(path))
                return result.get("text", "")
        else:
            # Save to temp wav for simplicity
            import tempfile, soundfile as sf  # type: ignore
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, 16000)
                return self.transcribe(f.name)

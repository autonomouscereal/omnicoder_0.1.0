from __future__ import annotations

from pathlib import Path
from typing import Optional


class TTSAdapter:
    """On-device TTS adapter.

    Tries Coqui TTS, falling back to pyttsx3 (system TTS). Returns path to an
    output WAV if successful.
    """

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", piper_model: str | None = None):
        self.model_name = model_name
        self.piper_model = piper_model  # path to a Piper .onnx model (if using Piper CLI)
        self._backend = None
        self._model = None

    def _ensure_backend(self) -> bool:
        if self._backend is not None:
            return True
        try:
            from TTS.api import TTS  # type: ignore
            self._model = TTS(self.model_name)
            self._backend = "coqui"
            return True
        except Exception:
            pass
        # Try Piper CLI if available and a model was provided
        try:
            import os
            import shutil
            if self.piper_model is None:
                self.piper_model = os.environ.get("PIPER_MODEL", None)
            if self.piper_model and shutil.which("piper") is not None:
                self._backend = "piper"
                return True
        except Exception:
            pass
        try:
            import pyttsx3  # type: ignore
            self._model = pyttsx3.init()
            self._backend = "pyttsx3"
            return True
        except Exception:
            return False

    def tts(self, text: str, out_path: str = "weights/tts_out.wav", speaker_id: Optional[str] = None) -> Optional[Path]:
        if not self._ensure_backend():
            return None
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self._backend == "coqui":
            # Coqui supports many models; tacotron2-DDC + HiFi-GAN vocoder is a compact default
            try:
                self._model.tts_to_file(text=text, file_path=str(p), speaker=speaker_id)
                return p
            except Exception:
                return None
        elif self._backend == "piper":
            # Invoke Piper CLI (fast, on-device ONNX TTS)
            import subprocess
            cmd = [
                "piper",
                "--model",
                str(self.piper_model),
                "--output_file",
                str(p),
            ]
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True)
                return p
            except Exception:
                return None
        elif self._backend == "pyttsx3":
            try:
                self._model.save_to_file(text, str(p))
                self._model.runAndWait()
                return p
            except Exception:
                return None
        return None

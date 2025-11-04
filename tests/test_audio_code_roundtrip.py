from __future__ import annotations

import numpy as np

from omnicoder.modeling.multimodal.audio_tokenizer import AudioTokenizer
from omnicoder.modeling.multimodal.vocab_map import map_audio_tokens
from omnicoder.config import MultiModalConfig


def test_encode_map_basic():
    # Synthetic 1s sine wave at 16kHz
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    tok = AudioTokenizer(sample_rate=sr)
    codes = tok.encode(wave)
    assert isinstance(codes, list) and len(codes) >= 1
    mmc = MultiModalConfig()
    mapped = map_audio_tokens(codes[0].tolist(), mmc)
    assert len(mapped) == codes[0].shape[0]
    assert mapped[0] >= mmc.audio_vocab_start



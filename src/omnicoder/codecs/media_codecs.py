from __future__ import annotations

import hashlib
from pathlib import Path

from omnicoder.tokenization.ledger_codec import LedgerCodec, LedgerPacket


class HashRangeCodec:
    """Deterministic placeholder edge codec until Cosmos/ACE/LTX codecs are wired.

    This is not the final media tokenizer. It gives the data factory and trunk a
    stable ledger contract while real visual/audio/music tokenizers are trained
    or wrapped.
    """

    def __init__(self, range_name: str, codec_name: str):
        self.range_name = range_name
        self.codec_name = codec_name
        self.ledger_codec = LedgerCodec()
        self.ranges = self.ledger_codec.ledger.as_config_ranges()

    def encode_file(self, path: str, token_count: int = 256) -> LedgerPacket:
        p = Path(path)
        data = p.read_bytes()
        digest = hashlib.blake2b(data, digest_size=32).digest()
        lo, hi = self.ranges[self.range_name]
        span = hi - lo
        ids = tuple(lo + digest[i % len(digest)] % span for i in range(max(1, int(token_count))))
        return LedgerPacket(ids, self.range_name, self.codec_name, {"path": str(p), "bytes": len(data)})


class VisionLedgerCodec(HashRangeCodec):
    def __init__(self):
        super().__init__("vision_semantic", "cosmos_visual_placeholder_v1")


class VideoLedgerCodec(HashRangeCodec):
    def __init__(self):
        super().__init__("vision_residual", "cosmos_ltx_video_placeholder_v1")


class SpeechLedgerCodec(HashRangeCodec):
    def __init__(self):
        super().__init__("speech_tts", "speech_rvq_placeholder_v1")


class AudioMusicLedgerCodec(HashRangeCodec):
    def __init__(self):
        super().__init__("audio_music", "ace_step_music_placeholder_v1")

from __future__ import annotations

from omnicoder.tokenization.ledger_codec import LedgerCodec, LedgerPacket


class TextLedgerCodec:
    def __init__(self):
        self.codec = LedgerCodec()

    def encode(self, text: str) -> LedgerPacket:
        return self.codec.encode_text_bytes(text)

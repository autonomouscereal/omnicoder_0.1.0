from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER, OmniLedger2026


@dataclass(frozen=True)
class LedgerPacket:
    token_ids: tuple[int, ...]
    modality: str
    codec: str
    metadata: dict


class LedgerCodec:
    def __init__(self, ledger: OmniLedger2026 | None = None):
        self.ledger = ledger or DEFAULT_LEDGER

    def encode_text_bytes(self, text: str) -> LedgerPacket:
        lo, hi = self.ledger.as_config_ranges()["text"]
        span = hi - lo
        ids = tuple(lo + (b % span) for b in text.encode("utf-8", errors="replace"))
        return LedgerPacket(ids or (lo,), "text", "byte_fallback_v1", {"chars": len(text)})

    def wrap_existing_ids(self, ids: Iterable[int], modality: str, codec: str, metadata: dict | None = None) -> LedgerPacket:
        cleaned = tuple(int(x) for x in ids)
        for tid in cleaned:
            self.ledger.lookup(tid)
        return LedgerPacket(cleaned, modality, codec, metadata or {})

    def control(self, name_hash: int, count: int = 1) -> LedgerPacket:
        lo, hi = self.ledger.as_config_ranges()["control"]
        span = hi - lo
        ids = tuple(lo + ((int(name_hash) + i) % span) for i in range(max(1, int(count))))
        return LedgerPacket(ids, "control", "ledger_control_v1", {})

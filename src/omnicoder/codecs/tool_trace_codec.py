from __future__ import annotations

import hashlib
import json

from omnicoder.tokenization.ledger_codec import LedgerCodec, LedgerPacket


class ToolTraceLedgerCodec:
    def __init__(self):
        self.ledger_codec = LedgerCodec()
        self.lo, self.hi = self.ledger_codec.ledger.as_config_ranges()["tool_agent"]

    def encode_event(self, event: dict, token_count: int = 64) -> LedgerPacket:
        payload = json.dumps(event, sort_keys=True, ensure_ascii=False).encode("utf-8", errors="replace")
        digest = hashlib.blake2b(payload, digest_size=32).digest()
        span = self.hi - self.lo
        ids = tuple(self.lo + digest[i % len(digest)] % span for i in range(max(1, int(token_count))))
        return LedgerPacket(ids, "tool_agent", "tool_trace_hash_v1", {"keys": sorted(event.keys())})

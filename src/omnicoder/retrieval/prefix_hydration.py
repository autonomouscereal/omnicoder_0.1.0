"""
Prefix hydration utility (scaffold).

Computes a simple semantic fingerprint of a prompt to lookup prior session
landmarks/KV caches and hydrate decode state when sufficiently similar.
"""

from __future__ import annotations

import hashlib
from typing import List


def semantic_fingerprint(text: str, num_buckets: int = 256) -> List[int]:
    """Hash words into fixed buckets as a cheap fingerprint.

    Returns a list of small integers suitable for indexing caches.
    """
    toks = [t for t in text.lower().split() if t]
    out = [0 for _ in range(max(1, int(num_buckets)))]
    for w in toks:
        h = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
        out[h % len(out)] += 1
    return out



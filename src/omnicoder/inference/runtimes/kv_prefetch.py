from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Deque
from collections import deque


@dataclass
class PagingConfig:
    n_layers: int
    heads: int
    page_len: int
    max_pages_ram: int
    prefetch_ahead: int = 1


class LRUKVPager:
    """
    Simulate an LRU-paged KV cache with a simple prefetch predictor.

    Pages are identified by (layer_index, page_index). We assume a decode pattern
    that advances one position per token and maps positions to pages via
    page_index = position // page_len.
    """

    def __init__(self, cfg: PagingConfig) -> None:
        self.cfg = cfg
        self.cache: Dict[Tuple[int, int], int] = {}
        self.lru: Deque[Tuple[int, int]] = deque()
        self.hits: int = 0
        self.misses: int = 0
        self.prefetches: int = 0
        self.prefetch_hits: int = 0
        self.stalls: int = 0

    def _touch(self, key: Tuple[int, int]) -> None:
        # Move key to the back (most recently used)
        try:
            self.lru.remove(key)
        except ValueError:
            pass
        self.lru.append(key)
        self.cache[key] = 1
        # Evict if over capacity
        while len(self.lru) > self.cfg.max_pages_ram:
            old = self.lru.popleft()
            self.cache.pop(old, None)

    def _prefetch_keys(self, next_pos: int) -> List[Tuple[int, int]]:
        # Predict next pages: contiguous lookahead for each layer
        start_page = next_pos // self.cfg.page_len
        keys: List[Tuple[int, int]] = []
        for l in range(self.cfg.n_layers):
            for k in range(1, self.cfg.prefetch_ahead + 1):
                keys.append((l, start_page + k))
        return keys

    def access(self, pos: int) -> None:
        # Access current page for each layer
        cur_page = pos // self.cfg.page_len
        keys = [(l, cur_page) for l in range(self.cfg.n_layers)]
        # Prefetch before access
        for key in self._prefetch_keys(pos):
            if key not in self.cache:
                self.prefetches += 1
            else:
                self.prefetch_hits += 1
            self._touch(key)
        # Access keys
        for key in keys:
            if key in self.cache:
                self.hits += 1
            else:
                self.misses += 1
                # Count this positional access as a stall
                self.stalls += 1
            self._touch(key)

    def stats(self) -> Dict[str, float]:
        total = max(1, self.hits + self.misses)
        return {
            "hit_rate": float(self.hits) / float(total),
            "miss_rate": float(self.misses) / float(total),
            "prefetches": float(self.prefetches),
            "prefetch_hits": float(self.prefetch_hits),
            "stall_ratio": float(self.stalls) / float(total),
        }



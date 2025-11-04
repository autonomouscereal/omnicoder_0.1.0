from __future__ import annotations

import collections
from typing import Callable, Dict, Optional

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
import threading
import queue


class ExpertPager:
    """
    Lightweight LRU pager for MoE experts. Experts are identified by integer
    indices [0..E-1]. A factory callable is registered for each expert; when an
    expert is requested, the pager returns a resident module or constructs it via
    the factory and inserts it into the cache, evicting the least recently used
    resident if capacity is exceeded.

    Stats counters are maintained for hits, misses, and evictions.
    """

    def __init__(self, capacity: int = 8, device: Optional[torch.device] = None,
                 state_dir: Optional[str] = None, persist_on_evict: bool = True) -> None:
        self.capacity = max(1, int(capacity))
        self._factories: Dict[int, Callable[[], nn.Module]] = {}
        self._resident: Dict[int, nn.Module] = {}
        self._lru = collections.OrderedDict()  # key order tracks recency
        self.device = device
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.state_dir = state_dir
        self.persist_on_evict = bool(persist_on_evict)
        # Optional async prefetch worker
        self._async_enabled = True
        self._q: "queue.Queue[int]" = queue.Queue(maxsize=1024)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def register(self, idx: int, factory: Callable[[], nn.Module]) -> None:
        self._factories[int(idx)] = factory

    def _touch(self, idx: int) -> None:
        if idx in self._lru:
            self._lru.move_to_end(idx, last=True)
        else:
            self._lru[idx] = True

    def get(self, idx: int) -> nn.Module:
        i = int(idx)
        if i in self._resident:
            self.hits += 1
            self._touch(i)
            return self._resident[i]
        self.misses += 1
        # Construct or load and insert
        mod: nn.Module
        loaded = False
        if self.state_dir is not None:
            try:
                import os
                from pathlib import Path
                p = Path(self.state_dir) / f"expert_{i}.pt"
                if p.exists():
                    mod = torch.load(str(p), map_location=self.device or 'cpu')  # type: ignore[assignment]
                    loaded = True
                else:
                    loaded = False
            except Exception:
                loaded = False
        if not loaded:
            factory = self._factories.get(i, None)
            if factory is None:
                raise KeyError(f"No factory registered for expert {i}")
            mod = factory()
        if self.device is not None:
            try:
                mod.to(self.device)
            except Exception:
                pass
        # Evict if over capacity
        if len(self._resident) >= self.capacity:
            # Pop least-recently used
            try:
                evict_idx, _ = self._lru.popitem(last=False)
                if evict_idx in self._resident:
                    self.evictions += 1
                    try:
                        # Persist evicted module if requested, then free memory
                        if self.persist_on_evict and self.state_dir is not None:
                            import os
                            from pathlib import Path
                            Path(self.state_dir).mkdir(parents=True, exist_ok=True)
                            # NOTE: Use robust save to avoid inline_container writer errors when evicting
                            # expert modules under parallel workloads. This preserves cache correctness
                            # while avoiding flaky IO failures.
                            _safe_save(self._resident[evict_idx], str(Path(self.state_dir) / f"expert_{evict_idx}.pt"))
                        # Best-effort: move to CPU to free device memory, then drop reference
                        self._resident[evict_idx].to('cpu')  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        del self._resident[evict_idx]
                    except Exception:
                        self._resident.pop(evict_idx, None)
            except Exception:
                pass
        self._resident[i] = mod
        self._touch(i)
        return mod

    def prefetch(self, indices: list[int]) -> None:
        # Best-effort synchronous prefetch
        for idx in indices:
            if int(idx) not in self._resident and int(idx) in self._factories:
                try:
                    _ = self.get(int(idx))
                except Exception:
                    continue

    def request_prefetch(self, indices: list[int]) -> None:
        """
        Queue indices for background prefetch. Starts a worker lazily on first call.
        """
        if not self._async_enabled:
            self.prefetch(indices)
            return
        if self._worker is None or (self._worker and not self._worker.is_alive()):
            self._start_worker()
        for idx in indices:
            try:
                self._q.put_nowait(int(idx))
            except Exception:
                break

    def _start_worker(self) -> None:
        def _run() -> None:
            while not self._stop.is_set():
                try:
                    idx = self._q.get(timeout=0.1)
                except Exception:
                    continue
                try:
                    if int(idx) not in self._resident:
                        _ = self.get(int(idx))
                except Exception:
                    pass
                finally:
                    try:
                        self._q.task_done()
                    except Exception:
                        pass
        self._stop.clear()
        self._worker = threading.Thread(target=_run, name="ExpertPagerPrefetch", daemon=True)
        try:
            self._worker.start()
        except Exception:
            self._async_enabled = False

    def warm_hint(self, probs: torch.Tensor, topk: int = 2) -> None:
        """
        Provide a warm-start hint to prefetch likely-to-be-used experts.
        probs: (B, T, E) routing probabilities or logits.
        """
        try:
            if probs.dim() == 3:
                if probs.dtype.is_floating_point:
                    # treat as logits/probs; take topk along experts
                    _, idx = torch.topk(probs, k=max(1, int(topk)), dim=-1)
                else:
                    idx = probs.long().unsqueeze(-1)
                uniq = torch.unique(idx.reshape(-1)).tolist()
                self.prefetch([int(u) for u in uniq])
        except Exception:
            return

    def stats(self) -> dict:
        return {
            'capacity': int(self.capacity),
            'resident': int(len(self._resident)),
            'hits': int(self.hits),
            'misses': int(self.misses),
            'evictions': int(self.evictions),
            'state_dir': str(self.state_dir) if self.state_dir is not None else None,
            'persist_on_evict': bool(self.persist_on_evict),
            'async_prefetch': bool(self._async_enabled),
        }



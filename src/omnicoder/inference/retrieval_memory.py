from __future__ import annotations

"""
External retrieval memory with simple ANN over embeddings and optional LRU/TTL.

This module provides a lightweight, device-friendly external memory that can be
queried by text and written with (hidden, next_token) pairs under a learned
write-policy. It uses a NumPy-backed store to avoid heavy dependencies and
persists across runs. For large corpora and product-quantized indices, prefer
the offline PQ builder tools (`tools/pq_build.py`).
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class ExternalRetrievalMemory:
    def __init__(self, root: str, dim: int, capacity: int = 50000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.capacity = int(capacity)
        # Memory files
        self.vec_path = self.root / "mem_vecs.npy"
        self.meta_path = self.root / "mem_meta.jsonl"
        # Load existing
        if self.vec_path.exists():
            try:
                self.vecs = np.load(self.vec_path)
                if self.vecs.ndim != 2 or self.vecs.shape[1] != self.dim:
                    self.vecs = np.zeros((0, self.dim), dtype=np.float32)
            except Exception:
                self.vecs = np.zeros((0, self.dim), dtype=np.float32)
        else:
            self.vecs = np.zeros((0, self.dim), dtype=np.float32)
        # metadata lines (text, token, timestamp)
        self.metas: List[dict] = []
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.metas.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                pass

    def size(self) -> int:
        return int(self.vecs.shape[0])

    def add(self, vec: np.ndarray, text: str = "", token: int | None = None) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(1, self.dim)
        if self.vecs.shape[0] >= self.capacity:
            # LRU eviction: drop oldest 1/32th to make room
            drop = max(1, self.capacity // 32)
            self.vecs = self.vecs[drop:, :]
            self.metas = self.metas[drop:]
        self.vecs = np.concatenate([self.vecs, v], axis=0)
        self.metas.append({"text": str(text), "token": int(token) if token is not None else None})
        try:
            np.save(self.vec_path, self.vecs)
        except Exception:
            pass
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                for m in self.metas:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float, str]]:
        if self.vecs.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, self.dim)
        # Cosine similarity
        V = self.vecs
        # Normalize
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)
        sims = (Vn @ qn.T).reshape(-1)
        idx = np.argsort(-sims)[: max(1, int(k))]
        out: List[Tuple[int, float, str]] = []
        for i in idx.tolist():
            meta = self.metas[i] if i < len(self.metas) else {"text": "", "token": None}
            out.append((int(i), float(sims[i]), str(meta.get("text", ""))))
        return out

    def search_text(self, text: str, k: int = 5) -> List[Tuple[int, float, str]]:
        """Embed a text by averaging a toy token embedding to avoid heavy deps, then search.

        Note: For real deployments, replace with a tiny text encoder; this is a light placeholder.
        """
        if not text:
            return []
        # Simple hashing-based embedding
        rng = np.random.RandomState(abs(hash(text)) % (2**31 - 1))
        vec = rng.randn(self.dim).astype(np.float32)
        return self.search(vec, k=k)


class CascadingMemoryController:
    """
    Tiered long-context orchestrator combining:
      - A rolling token tail window for local coherence
      - Optional compressed summaries (placeholder string summaries here)
      - External retrieval via ExternalRetrievalMemory

    This controller stays outside the decode hot path. Call update_tokens() once
    per user-provided text chunk; use bundle_for_query() to fetch a small set of
    strings to prepend/bias generation.
    """

    def __init__(self, window: int = 2048, retriever: ExternalRetrievalMemory | None = None) -> None:
        self.window = int(max(1, window))
        self.retriever = retriever
        self._tail: list[int] = []
        self._summaries: list[str] = []

    def update_tokens(self, token_ids: list[int]) -> None:
        self._tail.extend(int(t) for t in token_ids)
        if len(self._tail) > self.window:
            overflow = self._tail[:-self.window]
            self._tail = self._tail[-self.window :]
            # Placeholder: record a compact summary marker
            self._summaries.append(f"sum:{len(overflow)}")

    def bundle_for_query(self, query_text: str, k: int = 3) -> list[str]:
        out: list[str] = []
        if self._summaries:
            out.append(" ".join(self._summaries[-min(3, len(self._summaries)):]))
        if self.retriever is not None and query_text:
            try:
                hits = self.retriever.search_text(query_text, k=max(1, int(k)))
                out.extend([h[2] for h in hits])
            except Exception:
                pass
        return out



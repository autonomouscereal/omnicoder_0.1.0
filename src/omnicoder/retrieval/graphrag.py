"""
GraphRAG: Neuro-symbolic graph retrieval with KG triple overlays (K-BERT style).

This module exposes a tiny interface to retrieve triples (h,r,t) given a query
string and to produce token-level overlays that can be injected as a soft bias
to MoE/SFB.

Environment knobs:
- OMNICODER_GRAPHRAG_ENABLE=1
- OMNICODER_GRAPHRAG_ROOT=path/to/kg_index
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import math


@dataclass
class KGTriple:
    head: str
    relation: str
    tail: str


class GraphRAG:
    def __init__(self, root: str | None = None) -> None:
        try:
            self.enabled = os.getenv("OMNICODER_GRAPHRAG_ENABLE", "1") == "1"
        except Exception:
            self.enabled = False
        self.root = root or os.getenv("OMNICODER_GRAPHRAG_ROOT", "")
        try:
            self.max_edges = int(os.getenv("OMNICODER_GRAPHRAG_MAX_EDGES", "1024"))
        except Exception:
            self.max_edges = 1024
        # Optional vector index files (loaded lazily to avoid blocking first token)
        self._vec_index = None
        self._vec_map: Optional[List[str]] = None
        self._faiss = None
        self._faiss_path: Optional[str] = None
        self._bg_loading = False
        # Optional torch-accelerated index on CPU (no copy when using from_numpy)
        self._vec_index_t = None  # type: ignore[var-annotated]
        try:
            # Lazy, non-blocking index load by default
            if self.root and os.getenv("OMNICODER_GRAPHRAG_BG_LOAD", "1") == "1":
                import threading  # local import
                def _bg() -> None:
                    try:
                        self._maybe_load_vectors()
                    except Exception:
                        pass
                    finally:
                        self._bg_loading = False
                self._bg_loading = True
                threading.Thread(target=_bg, daemon=True).start()
            # If background load disabled, still avoid blocking: defer to first retrieve()
        except Exception:
            pass

    def _maybe_load_vectors(self) -> None:
        try:
            if not self.root:
                return
            from pathlib import Path
            import numpy as np
            try:
                import torch  # type: ignore
                _TORCH_OK = True
            except Exception:
                torch = None  # type: ignore
                _TORCH_OK = False
            rootp = Path(self.root)
            vecp = rootp / 'embeddings.npy'
            idsp = rootp / 'ids.txt'
            if vecp.exists() and idsp.exists():
                # Use memory-mapped read to reduce resident memory for large KGs
                self._vec_index = np.load(str(vecp), mmap_mode='r')
                self._vec_map = [line.strip() for line in open(idsp, 'r', encoding='utf-8', errors='ignore') if line.strip()]
                # Optional: FAISS index for cosine similarity (use inner product on normalized)
                try:
                    import faiss  # type: ignore
                    xb = np.array(self._vec_index, dtype='float32', copy=False)
                    # normalize
                    norms = (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8)
                    xb = xb / norms
                    # Try loading a persisted index first
                    self._faiss_path = str((rootp / 'faiss.index').as_posix())
                    if os.path.exists(self._faiss_path):
                        try:
                            index = faiss.read_index(self._faiss_path)
                            self._faiss = index
                            return
                        except Exception:
                            self._faiss = None
                    # IVFPQ if configured, else FlatIP
                    ivf_nlist = int(os.getenv('OMNICODER_FAISS_NLIST', '0') or '0')
                    pq_m = int(os.getenv('OMNICODER_FAISS_PQ_M', '0') or '0')
                    if ivf_nlist > 0 and pq_m > 0:
                        quantizer = faiss.IndexFlatIP(xb.shape[1])
                        index = faiss.IndexIVFPQ(quantizer, xb.shape[1], ivf_nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
                        index.train(xb)
                        index.add(xb)
                    else:
                        index = faiss.IndexFlatIP(xb.shape[1])
                        index.add(xb)
                    # Persist index for reuse (best-effort)
                    try:
                        faiss.write_index(index, self._faiss_path)
                    except Exception:
                        pass
                    self._faiss = index
                except Exception:
                    self._faiss = None
                # Build a torch view for CPU-accelerated cosine sim when FAISS is absent
                try:
                    if _TORCH_OK and self._faiss is None:
                        xb = np.array(self._vec_index, dtype='float32', copy=True)
                        # from_numpy now gets a writable copy, avoiding undefined behavior warnings
                        xt = torch.from_numpy(xb)
                        # Normalize along dim=1 for cosine via inner product
                        self._vec_index_t = xt / (xt.norm(dim=1, keepdim=True) + 1e-8)
                except Exception:
                    self._vec_index_t = None
        except Exception:
            self._vec_index = None
            self._vec_map = None

    def retrieve(self, query: str, k: int = 8) -> List[KGTriple]:
        if not self.enabled or not self.root:
            return []
        triples: List[KGTriple] = []
        try:
            import time
            import json
            from pathlib import Path
            # If vectors not loaded and not loading, attempt a quick, best-effort lazy load
            if self._vec_index is None and self._vec_map is None and (not self._bg_loading):
                try:
                    self._maybe_load_vectors()
                except Exception:
                    pass
            # Try vector search first with fuzzy-matched seed entities
            if self._vec_index is not None and self._vec_map is not None:
                try:
                    import numpy as np  # type: ignore
                    try:
                        import torch  # type: ignore
                        _TORCH_OK = True
                    except Exception:
                        torch = None  # type: ignore
                        _TORCH_OK = False
                    seed_idx: List[int] = []
                    try:
                        from rapidfuzz import fuzz, process  # type: ignore
                        # Extract top seed matches against id strings
                        cand = process.extract(query, self._vec_map, scorer=fuzz.partial_ratio, limit=min(32, len(self._vec_map)))
                        for name, score, idx in cand:
                            if score >= 60:
                                seed_idx.append(int(idx))
                    except Exception:
                        # fallback: substring scan
                        ql = query.lower()
                        for i, name in enumerate(self._vec_map):
                            if name and name.lower() in ql:
                                seed_idx.append(i)
                                if len(seed_idx) >= 8:
                                    break
                    if seed_idx:
                        seeds = self._vec_index[np.array(seed_idx, dtype=np.int64)]
                        qv = seeds.mean(axis=0).astype('float32')
                        # normalize
                        qv = qv / (np.linalg.norm(qv) + 1e-8)
                        if self._faiss is not None:
                            import faiss  # type: ignore
                            try:
                                nprobe = int(os.getenv('OMNICODER_FAISS_NPROBE', '10'))
                                if hasattr(self._faiss, 'nprobe'):
                                    self._faiss.nprobe = max(1, nprobe)
                            except Exception:
                                pass
                            D, I = self._faiss.search(qv.reshape(1, -1), max(1, k))
                            idxs = [int(i) for i in I[0] if int(i) >= 0]
                        else:
                            # Prefer torch cosine search when available for speed
                            if _TORCH_OK and (self._vec_index_t is not None):
                                q = torch.from_numpy(qv.copy())
                                q = q / (q.norm() + 1e-8)
                                sims_t = torch.matmul(self._vec_index_t, q)
                                topk = torch.topk(sims_t, k=max(1, k), largest=True)
                                idxs = [int(i) for i in topk.indices.tolist()]
                            else:
                                xb = self._vec_index.astype('float32')
                                xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8)
                                sims = np.dot(xb, qv)
                                idxs = list(np.argsort(-sims)[: max(1, k)].astype(int))
                        names = [self._vec_map[i] for i in idxs if i < len(self._vec_map)]
                        # Expand into triples using edges when possible; else relate neighbors pairwise
                        rootp = Path(self.root)
                        edges_path = rootp / 'edges.jsonl'
                        if edges_path.exists():
                            try:
                                # Fast pass: emit edges whose endpoints are in names
                                name_set = set(n.lower() for n in names)
                                # Bound scan time to avoid blocking
                                t0 = time.perf_counter(); time_budget = float(os.getenv('OMNICODER_GRAPHRAG_RETRIEVE_MS', '25')) / 1000.0
                                for line in open(edges_path, 'r', encoding='utf-8', errors='ignore'):
                                    if len(triples) >= max(1, k):
                                        break
                                    if (time.perf_counter() - t0) > time_budget:
                                        break
                                    if not line.strip():
                                        continue
                                    rec = json.loads(line)
                                    h = str(rec.get('h', ''))
                                    r = str(rec.get('r', ''))
                                    t = str(rec.get('t', ''))
                                    if h and t and (h.lower() in name_set or t.lower() in name_set):
                                        triples.append(KGTriple(head=h, relation=r or 'rel', tail=t))
                            except Exception:
                                pass
                        if not triples:
                            for i, ni in enumerate(names[: max(1, k)]):
                                nj = names[(i + 1) % len(names)] if names else ni
                                triples.append(KGTriple(head=ni, relation='related_to', tail=nj))
                except Exception:
                    pass
            # Expect a simple index: one JSONL per entity with neighbors, or a single edges.jsonl
            rootp = Path(self.root)
            edges_path = rootp / 'edges.jsonl'
            if edges_path.exists():
                # naive substring match over edges with a strict time budget
                ql = query.lower()
                t0 = time.perf_counter(); time_budget = float(os.getenv('OMNICODER_GRAPHRAG_RETRIEVE_MS', '25')) / 1000.0
                for line in open(edges_path, 'r', encoding='utf-8', errors='ignore'):
                    if not line.strip():
                        if (time.perf_counter() - t0) > time_budget:
                            break
                        continue
                    try:
                        rec = json.loads(line)
                        h = str(rec.get('h', ''))
                        r = str(rec.get('r', ''))
                        t = str(rec.get('t', ''))
                        if (h and h.lower() in ql) or (t and t.lower() in ql):
                            triples.append(KGTriple(head=h, relation=r or 'rel', tail=t))
                            if len(triples) >= max(1, k):
                                break
                    except Exception:
                        continue
            else:
                # Fallback: parse entities by split and fabricate local relations
                toks = [t for t in query.replace('\n', ' ').split(' ') if t]
                for i in range(min(k, max(0, len(toks) - 1))):
                    h = toks[i]
                    t = toks[i + 1]
                    triples.append(KGTriple(head=h, relation="co_occurs_with", tail=t))
        except Exception:
            triples = []
        return triples

    def to_overlay_text(self, triples: List[KGTriple]) -> str:
        if not triples:
            return ""
        lines = [f"({x.head}) -[{x.relation}]-> ({x.tail})" for x in triples]
        return "\n".join(lines)


def collect_bias_ids(triples: List[KGTriple], encode_fn, max_terms: int = 8, tail_tokens: int = 3) -> List[int]:
    """Utility: map triples to a small set of token ids for biasing.

    - Uses up to max_terms triples
    - For each string (head, relation, tail), encodes and keeps last tail_tokens ids
    """
    ids_to_bias: List[int] = []
    try:
        for t in list(triples)[: max(1, int(max_terms))]:
            try:
                for s in (getattr(t, 'head', ''), getattr(t, 'relation', ''), getattr(t, 'tail', '')):
                    s = str(s).strip()
                    if not s:
                        continue
                    tids = encode_fn(s)
                    if isinstance(tids, (list, tuple)):
                        ids_to_bias.extend(int(x) for x in list(tids)[-max(1, int(tail_tokens)):])
            except Exception:
                continue
    except Exception:
        ids_to_bias = []
    # deduplicate while preserving order
    try:
        ids_unique = list(dict.fromkeys([int(x) for x in ids_to_bias if isinstance(x, int)]))
    except Exception:
        ids_unique = []
    return ids_unique


def build_graphrag() -> GraphRAG:
    return GraphRAG()



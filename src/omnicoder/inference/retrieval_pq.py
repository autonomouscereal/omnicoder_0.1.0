from __future__ import annotations

"""
Product-Quantized (PQ) ANN retriever with mmap-backed index for low-RAM devices.

This module builds a lightweight PQ index over fixed-length text chunks (hashed
TF-IDF bag-of-words or simple encoder features) and answers ANN queries with
asymmetric distance computation. The index is stored on disk (np.memmap) and
memory-mapped at query time to minimize RAM usage on mobile/edge.

Design goals:
 - No heavyweight dependencies; pure NumPy. (Optional FAISS can be added.)
 - Simple training: k-means per subspace via Lloyd iterations with few iters.
 - Mmap arrays for codebooks and codes; O(1) memory overhead on load.

Usage:
  # Build (once)
  pq = PqRetriever.build_from_text_folder("./docs", chunk_size=600, stride=520,
                                          m=16, ks=256, out_dir="weights/pq")
  # Load and query
  pq2 = PqRetriever("weights/pq")
  hits = pq2.search("explain SIMD on ARM", k=3)

This code is intentionally simple and can be replaced by a vendor ANN if desired.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


def _read_texts(root: str, max_chars: int = 4000) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith((".txt", ".md")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                s = open(p, "r", encoding="utf-8", errors="ignore").read()
                if max_chars > 0 and len(s) > max_chars:
                    s = s[:max_chars]
                rows.append((p, s))
            except Exception:
                pass
    return rows


def _chunk(text: str, size: int, stride: int) -> List[str]:
    if size <= 0:
        return [text]
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        out.append(text[i : min(n, i + size)])
        i += max(1, stride)
    return out


def _tfidf_embed(chunks: List[str], vocab: Optional[dict] = None, max_vocab: int = 50000) -> Tuple[np.ndarray, dict]:
    # Tokenize on whitespace; lowercase; simple hashing vocab if not provided.
    if vocab is None:
        vocab = {}
        for s in chunks:
            for tok in s.lower().split():
                if tok not in vocab:
                    if len(vocab) >= max_vocab:
                        continue
                    vocab[tok] = len(vocab)
    V = len(vocab)
    # Document frequencies
    df = np.zeros(V, dtype=np.int32)
    docs = []
    for s in chunks:
        idxs = [vocab[t] for t in set(s.lower().split()) if t in vocab]
        for j in idxs:
            df[j] += 1
        docs.append([vocab[t] for t in s.lower().split() if t in vocab])
    N = max(1, len(chunks))
    idf = np.log((N + 1) / (df + 1)) + 1.0
    # Sparse bag -> dense TF-IDF with L2 norm
    X = np.zeros((len(chunks), V), dtype=np.float32)
    for i, ids in enumerate(docs):
        if not ids:
            continue
        counts = {}
        for j in ids:
            counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            X[i, j] = float(c)
    X = X * idf[None, :]
    # L2 normalize
    X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    X = X / X_norm
    return X.astype(np.float32), vocab


def _kmeans(x: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    # init centers by random samples; allow replacement when n < k so we always have k centers
    idx = rng.choice(n, size=int(k), replace=(n < k))
    c = x[idx].copy()
    # Lloyd iterations
    for _ in range(max(1, iters)):
        # assignments
        # cosine distance -> normalize and argmax
        x_n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
        c_n = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-6)
        sim = x_n @ c_n.T
        a = np.argmax(sim, axis=1)
        # update
        for j in range(c.shape[0]):
            pts = x[a == j]
            if pts.shape[0] > 0:
                c[j] = pts.mean(axis=0)
    # Ensure exactly k centers (pad by repeating if needed)
    if c.shape[0] < k:
        reps = k - c.shape[0]
        extra_idx = rng.choice(c.shape[0], size=reps, replace=True)
        c = np.concatenate([c, c[extra_idx]], axis=0)
    return c.astype(np.float32), a.astype(np.int32)


@dataclass
class PqMeta:
    dim: int
    m: int
    ks: int
    vocab_path: str
    codes_path: str
    codebooks_path: str
    offsets_path: str


class PqRetriever:
    def __init__(self, pq_dir: str) -> None:
        self.root = Path(pq_dir)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.dim = int(meta["dim"])  # original dim
        self.m = int(meta["m"])     # subspaces
        self.ks = int(meta["ks"])   # codes per subspace
        self.vocab = json.loads(open(self.root / meta["vocab_path"], "r", encoding="utf-8").read())
        self.codebooks = np.load(self.root / meta["codebooks_path"], mmap_mode="r")  # (m, ks, dsub)
        self.codes = np.load(self.root / meta["codes_path"], mmap_mode="r")          # (N, m) uint8
        self.offsets = json.loads(open(self.root / meta["offsets_path"], "r", encoding="utf-8").read())

    @staticmethod
    def build_from_text_folder(root: str, chunk_size: int, stride: int, m: int, ks: int, out_dir: str) -> "PqRetriever":
        rows = _read_texts(root)
        chunks: List[str] = []
        offsets: List[Tuple[str, int]] = []
        for fp, txt in rows:
            cs = _chunk(txt, chunk_size, stride)
            for i, s in enumerate(cs):
                chunks.append(s)
                offsets.append((fp, i))
        X, vocab = _tfidf_embed(chunks)
        n, d = X.shape
        # Make dim divisible by m by zero-padding feature columns when needed
        if d % m != 0:
            pad = m - (d % m)
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
            d = X.shape[1]
        dsub = d // m
        codebooks = np.zeros((m, ks, dsub), dtype=np.float32)
        codes = np.zeros((n, m), dtype=np.uint8)
        for i in range(m):
            xsub = X[:, i * dsub : (i + 1) * dsub]
            c, a = _kmeans(xsub, ks, iters=8, seed=1234 + i)
            codebooks[i] = c
            codes[:, i] = a.astype(np.uint8)
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        np.save(outp / "codebooks.npy", codebooks)
        np.save(outp / "codes.npy", codes)
        # Back-compat: emit a simple index artifact expected by older tooling/tests
        try:
            np.save(outp / "pq.index.npy", codes)
            Path(outp / "pq.index").write_bytes(b"")
        except Exception:
            pass
        (outp / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (outp / "offsets.json").write_text(json.dumps(offsets), encoding="utf-8")
        meta = {
            "dim": int(d), "m": int(m), "ks": int(ks),
            "vocab_path": "vocab.json",
            "codes_path": "codes.npy",
            "codebooks_path": "codebooks.npy",
            "offsets_path": "offsets.json",
        }
        (outp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return PqRetriever(str(outp))

    # Back-compat wrapper for tools.pq_build legacy CLI
    @staticmethod
    def build_from_folder(root: str, out_dir: str, dim: int, nlist: int, m: int, ks: int, max_docs: int | None = None) -> "PqRetriever":
        # The simple TF-IDF builder ignores dim/nlist/max_docs; keep signature to avoid breaking older scripts
        chunk_size = 600
        stride = 520
        return PqRetriever.build_from_text_folder(root, chunk_size=chunk_size, stride=stride, m=m, ks=ks, out_dir=out_dir)

    @staticmethod
    def build_from_embeddings(emb: np.ndarray, offsets: list[tuple[str, int]], m: int, ks: int, out_dir: str) -> "PqRetriever":
        """Build PQ index from precomputed embeddings (float32), shape (N,D). Writes mmap sidecars and meta.

        offsets: list of (source_path, chunk_index) for each row in emb.
        """
        assert emb.ndim == 2 and emb.dtype == np.float32
        n, d = int(emb.shape[0]), int(emb.shape[1])
        # Make dim divisible by m by zero-padding feature columns when needed
        if d % m != 0:
            pad = m - (d % m)
            emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant")
            d = int(emb.shape[1])
        dsub = d // m
        codebooks = np.zeros((m, ks, dsub), dtype=np.float32)
        codes = np.zeros((n, m), dtype=np.uint8)
        for i in range(m):
            xsub = emb[:, i * dsub : (i + 1) * dsub]
            c, a = _kmeans(xsub, ks, iters=12, seed=4321 + i)
            codebooks[i] = c
            codes[:, i] = a.astype(np.uint8)
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        np.save(outp / "codebooks.npy", codebooks)
        np.save(outp / "codes.npy", codes)
        # Back-compat: emit a simple index artifact expected by older tooling/tests
        try:
            np.save(outp / "pq.index.npy", codes)
            Path(outp / "pq.index").write_bytes(b"")
        except Exception:
            pass
        # write offsets
        (outp / "offsets.json").write_text(json.dumps([(str(p), int(i)) for (p, i) in offsets]), encoding="utf-8")
        # write a dummy vocab since embeddings are external
        (outp / "vocab.json").write_text(json.dumps({}), encoding="utf-8")
        meta = {
            "dim": int(d), "m": int(m), "ks": int(ks),
            "vocab_path": "vocab.json",
            "codes_path": "codes.npy",
            "codebooks_path": "codebooks.npy",
            "offsets_path": "offsets.json",
        }
        (outp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return PqRetriever(str(outp))

    def write_budget_profile(self, out_path: str, group_bytes: int = 4096, ram_budget_bytes: int = 64 * 1024 * 1024) -> None:
        """Write a simple sidecar JSON with scan batch size recommendations given a RAM budget.

        group_bytes: approximate bytes per scan group (implementation-specific).
        ram_budget_bytes: target RAM budget for codes and temporaries.
        """
        bytes_per_row = int(self.codes.shape[1])  # 1 byte per subcode
        batch = max(64, ram_budget_bytes // max(1, bytes_per_row))
        prof = {
            "bytes_per_row": bytes_per_row,
            "recommended_batch": int(batch),
            "ram_budget_bytes": int(ram_budget_bytes),
            "group_bytes": int(group_bytes),
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(prof), encoding="utf-8")

    def _embed_query(self, text: str) -> np.ndarray:
        # Same TF-IDF as build (no IDF recompute on query; approximate)
        toks = [t for t in text.lower().split() if t in self.vocab]
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for t in toks:
            v[self.vocab[t]] += 1.0
        nrm = np.linalg.norm(v) + 1e-6
        v = (v / nrm).astype(np.float32)
        # Pad to match PQ subspace layout if build padded features
        try:
            m = int(self.m)
            dsub = int(self.codebooks.shape[2])
            target = m * dsub
            if v.shape[0] < target:
                v = np.pad(v, (0, target - v.shape[0]), mode="constant")
            elif v.shape[0] > target:
                v = v[:target]
        except Exception:
            pass
        return v

    def search(self, query: str, k: int = 3, partition_size: int = 0, budget_bytes: int = 0) -> List[Tuple[int, float, str]]:
        q = self._embed_query(query)
        d = q.shape[0]
        dsub = d // self.m
        # Precompute per-subspace centroid similarities
        sims = []
        for i in range(self.m):
            cb = self.codebooks[i]  # (ks, dsub)
            qsub = q[i * dsub : (i + 1) * dsub]
            qn = qsub / (np.linalg.norm(qsub) + 1e-6)
            c_n = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-6)
            sims.append(qn[None, :] @ c_n.T)  # (1, ks)
        sims = [s.reshape(-1) for s in sims]
        # Asymmetric distance: sum of centroid sims per code index
        # Compute scores in batches to avoid large memory
        N = self.codes.shape[0]
        batch = max(1024, 1)
        if int(budget_bytes) > 0:
            # Estimate bytes per row for codes and adjust batch accordingly
            bytes_per_row = int(self.codes.shape[1])  # uint8 per subcode
            est = max(1, int(budget_bytes) // max(1, bytes_per_row))
            batch = max(64, min(batch, est))
        best: List[Tuple[int, float]] = []
        # Optional partitioning to reduce peak memory
        part = max(int(partition_size), 0)
        step = batch if part == 0 else min(batch, part)
        for start in range(0, N, step):
            end = min(N, start + batch)
            codes_b = self.codes[start:end]  # (B, m)
            score = np.zeros(codes_b.shape[0], dtype=np.float32)
            for i in range(self.m):
                score += sims[i][codes_b[:, i]]
            # Top-k in this batch
            idx = np.argpartition(-score, kth=min(k, score.shape[0]-1))[:k]
            for j in idx:
                best.append((start + int(j), float(score[j])))
        # Global top-k
        best.sort(key=lambda x: -x[1])
        best = best[:k]
        hits: List[Tuple[int, float, str]] = []
        for row, sc in best:
            fp, chunk_idx = self.offsets[row]
            hits.append((row, sc, f"{fp}#chunk{chunk_idx}"))
        return hits

    def scan_budget_profile(self, ram_budget_bytes: int = 64 * 1024 * 1024) -> dict:
        bytes_per_row = int(self.codes.shape[1])
        batch = max(64, ram_budget_bytes // max(1, bytes_per_row))
        return {"bytes_per_row": bytes_per_row, "recommended_batch": int(batch), "ram_budget_bytes": int(ram_budget_bytes)}

# The file previously contained a duplicate module (with a misplaced future import)
# implementing alternative PQ paths. It has been removed for correctness and clarity.




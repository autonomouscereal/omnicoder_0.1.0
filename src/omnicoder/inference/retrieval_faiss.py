from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    out: List[str] = []
    buff: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buff.append(ch)
        else:
            if buff:
                tok = ''.join(buff)
                if len(tok) > 2:
                    out.append(tok)
                buff = []
    if buff:
        tok = ''.join(buff)
        if len(tok) > 2:
            out.append(tok)
    return out


@dataclass
class RetrievedChunk:
    doc_id: int
    score: float
    text: str


class FaissRetriever:
    """On-device ANN retriever using FAISS with a hashed TF-IDF embedding.

    - No external encoder required; uses hashing trick to map tokens to a fixed dim.
    - Index: `faiss.IndexFlatIP` over L2-normalized vectors (cosine similarity).
    - Input: folder of `.txt` or a JSONL with `{text: ...}` lines; chunks long files.
    """

    def __init__(self, path: str, dim: int = 4096, chunk_size: int = 512, stride: int = 448):
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "faiss is required for FaissRetriever (install with extras 'bench' or faiss-cpu)."
            ) from e

        self.faiss = __import__('faiss')  # lazy handle
        self.dim = int(dim)
        self.docs: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.df: dict[str, int] = {}
        self.num_docs: int = 0
        self._build_corpus(path, chunk_size, stride)
        # Build vectors and FAISS index
        import numpy as np

        vecs = []
        for toks in self.doc_tokens:
            v = self._tfidf_vec(toks)
            # L2 normalize for cosine via inner product
            n = np.linalg.norm(v) + 1e-12
            vecs.append(v / n)
        xb = np.stack(vecs).astype('float32') if vecs else np.zeros((0, self.dim), dtype='float32')
        self.index = self.faiss.IndexFlatIP(self.dim)
        if xb.shape[0] > 0:
            self.index.add(xb)

    def _build_corpus(self, path: str, chunk_size: int, stride: int) -> None:
        import json
        p = Path(path)
        if p.is_dir():
            for fp in p.rglob('*.txt'):
                try:
                    text = fp.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                for i in range(0, len(text), stride):
                    chunk = text[i:i+chunk_size]
                    if chunk.strip():
                        self._add_doc(chunk)
        else:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            ex = json.loads(line)
                            txt = ex.get('text', '')
                            if txt:
                                self._add_doc(txt)
                        except Exception:
                            continue
            except Exception:
                text = p.read_text(encoding='utf-8', errors='ignore')
                self._add_doc(text)

    def _add_doc(self, text: str) -> None:
        toks = _tokenize(text)
        self.docs.append(text)
        self.doc_tokens.append(toks)
        self.num_docs += 1
        seen = set(toks)
        for t in seen:
            self.df[t] = self.df.get(t, 0) + 1

    def _hash(self, token: str) -> int:
        # Simple deterministic hash to [0, dim)
        return (hash(token) % self.dim + self.dim) % self.dim

    def _tfidf_vec(self, toks: List[str]):
        import numpy as np
        v = np.zeros((self.dim,), dtype='float32')
        if not toks:
            return v
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            df = self.df.get(t, 0)
            if df <= 0:
                continue
            # Smoothed IDF; use log to dampen very frequent tokens
            idf = math.log((self.num_docs + 1.0) / (df + 1.0)) + 1.0
            w = (c / len(toks)) * float(idf)
            v[self._hash(t)] += float(w)
        return v

    def search(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        import numpy as np
        qv = self._tfidf_vec(_tokenize(query))
        qn = np.linalg.norm(qv) + 1e-12
        qv = (qv / qn).astype('float32')[None, :]
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(qv, max(1, k))
        out: List[RetrievedChunk] = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            out.append(RetrievedChunk(doc_id=int(i), score=float(s), text=self.docs[int(i)]))
        return out



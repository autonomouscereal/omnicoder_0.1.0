from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    # Very light tokenizer: lowercase, split on whitespace/punct
    # Keeps alphanumerics; drops very short tokens
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


class LocalRetriever:
    """Simple TF-IDF retriever for offline/local corpora.

    - Indexes .txt files under a folder or a single .jsonl with a `text` field per line
    - Computes sparse TF-IDF vectors using Python dicts (no extra deps)
    - Returns top-k passages for a query string
    """

    def __init__(self, path: str, chunk_size: int = 512, stride: int = 448):
        self.docs: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.df: dict[str, int] = {}
        self.num_docs: int = 0
        self._build_index(path, chunk_size, stride)

    def _add_doc(self, text: str) -> None:
        tokens = _tokenize(text)
        self.docs.append(text)
        self.doc_tokens.append(tokens)
        self.num_docs += 1
        seen: set[str] = set(tokens)
        for t in seen:
            self.df[t] = self.df.get(t, 0) + 1

    def _build_index(self, path: str, chunk_size: int, stride: int) -> None:
        p = Path(path)
        if p.is_dir():
            for fp in p.rglob('*.txt'):
                try:
                    text = fp.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                # chunk long files
                for i in range(0, len(text), stride):
                    chunk = text[i:i+chunk_size]
                    if chunk.strip():
                        self._add_doc(chunk)
        else:
            # try jsonl with {text: ...}
            try:
                import json
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
                # fallback: treat as plain text
                text = p.read_text(encoding='utf-8', errors='ignore')
                self._add_doc(text)

    def _tfidf(self, tokens: List[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        vec: dict[str, float] = {}
        for t, c in counts.items():
            df = self.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log((self.num_docs + 1) / (df + 1)) + 1.0
            vec[t] = (c / len(tokens)) * idf
        return vec

    def search(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        q_tokens = _tokenize(query)
        q_vec = self._tfidf(q_tokens)
        scores: List[Tuple[float, int]] = []
        for i, doc_toks in enumerate(self.doc_tokens):
            d_vec = self._tfidf(doc_toks)
            # dot product
            s = 0.0
            for t, w in q_vec.items():
                s += w * d_vec.get(t, 0.0)
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        out: List[RetrievedChunk] = []
        for s, i in scores[:k]:
            out.append(RetrievedChunk(doc_id=i, score=s, text=self.docs[i]))
        return out



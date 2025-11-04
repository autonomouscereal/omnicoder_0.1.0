from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    # Very small tokenizer: split on non-alnum, keep short words
    out: List[str] = []
    buf: List[str] = []
    for ch in text:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tok = "".join(buf)
                if 2 <= len(tok) <= 32:
                    out.append(tok)
                buf = []
    if buf:
        tok = "".join(buf)
        if 2 <= len(tok) <= 32:
            out.append(tok)
    return out[:512]


def _hash_feature(tok: str, num_buckets: int) -> int:
    # Simple FNV-1a-like hash for stability across runs
    h = 2166136261
    for ch in tok:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % max(1, num_buckets))


@dataclass
class GoalPriorHead:
    """Tiny hashed linear model mapping prompt bag-of-words to goal priors.

    - Uses a fixed number of feature buckets for compactness
    - Stores per-goal weight vectors; inference is a sigmoid over sum of weights
    - Trained via simple SGD on a small JSONL with {prompt, goals:{name:0/1}}
    """

    goals: List[str]
    num_buckets: int = 1024
    lr: float = 0.1
    weights: Dict[str, List[float]] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {g: [0.0 for _ in range(self.num_buckets)] for g in self.goals}

    def featurize(self, prompt: str) -> List[int]:
        toks = _tokenize(prompt)
        return [_hash_feature(t, self.num_buckets) for t in toks]

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = 2.718281828 ** (-x)
            return 1.0 / (1.0 + z)
        z = 2.718281828 ** (x)
        return z / (1.0 + z)

    def predict(self, prompt: str) -> Dict[str, float]:
        feats = self.featurize(prompt)
        priors: Dict[str, float] = {}
        if self.weights is None:
            return {g: 0.0 for g in self.goals}
        for g in self.goals:
            w = self.weights.get(g) or [0.0 for _ in range(self.num_buckets)]
            s = 0.0
            for idx in feats[:1024]:
                s += w[int(idx)]
            priors[g] = float(self._sigmoid(s))
        return priors

    def train_from_jsonl(self, jsonl_path: str, epochs: int = 3) -> None:
        """Train on JSONL with rows: {"prompt":"...", "goals":{"code":1, "vqa":0, ...}}"""
        if self.weights is None:
            self.__post_init__()
        rows: List[Tuple[str, Dict[str, int]]] = []
        with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    pr = str(obj.get('prompt', ''))
                    gl = obj.get('goals', {}) or {}
                    labs: Dict[str, int] = {}
                    for g in self.goals:
                        v = gl.get(g)
                        labs[g] = int(v) if isinstance(v, (int, float)) else 0
                    rows.append((pr, labs))
                except Exception:
                    continue
        for _ in range(max(1, int(epochs))):
            for prompt, labs in rows:
                feats = self.featurize(prompt)
                for g in self.goals:
                    y = float(labs.get(g, 0))
                    w = self.weights[g]
                    s = 0.0
                    for idx in feats[:1024]:
                        s += w[int(idx)]
                    p = self._sigmoid(s)
                    grad = (y - p) * self.lr
                    for idx in feats[:1024]:
                        j = int(idx)
                        w[j] = float(w[j] + grad)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'goals': self.goals,
                'num_buckets': self.num_buckets,
                'weights': self.weights,
            }, f)

    @staticmethod
    def load(path: str) -> GoalPriorHead:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        head = GoalPriorHead(goals=list(obj.get('goals', [])), num_buckets=int(obj.get('num_buckets', 1024)))
        head.weights = {k: list(v) for (k, v) in (obj.get('weights') or {}).items()}
        return head



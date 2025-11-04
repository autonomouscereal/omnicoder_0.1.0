from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from typing import Any

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_OK = False


class KNNCache:
    """Compact kNN-LM cache over hidden states.

    Stores (hidden_state, next_token_id) pairs and supports nearest-neighbor
    queries to produce a token distribution that can be blended with the model
    distribution at decode time.

    If faiss is available, uses an IndexFlatIP/L2 backend; otherwise falls back
    to a numpy implementation.
    """

    def __init__(self, dim: int, use_cosine: bool = True, use_faiss: bool = True):
        self.dim = int(dim)
        self.use_cosine = bool(use_cosine)
        self._xs: Optional[np.ndarray] = None  # (N, D)
        self._ys: Optional[np.ndarray] = None  # (N,)
        self._faiss = None
        self._index = None
        self._age: Optional[np.ndarray] = None  # (N,) usage counts for aging out
        # Optional torch backend (device-accelerated)
        self._xs_t = None  # type: ignore[var-annotated]
        self._ys_t = None  # type: ignore[var-annotated]
        self._age_t = None  # type: ignore[var-annotated]
        if use_faiss:
            try:
                import faiss  # type: ignore
                self._faiss = faiss
                if self.use_cosine:
                    # Cosine via inner product with normalized vectors
                    self._index = faiss.IndexFlatIP(self.dim)
                else:
                    self._index = faiss.IndexFlatL2(self.dim)
            except Exception:
                self._faiss = None
                self._index = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.use_cosine:
            return x
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def add(self, h: np.ndarray, token_id: int) -> None:
        assert h.ndim == 1 and h.shape[0] == self.dim
        x = h.astype(np.float32, copy=False)[None, :]
        if self.use_cosine:
            x = self._normalize(x)
        if self._xs is None:
            self._xs = x.copy()
            self._ys = np.array([int(token_id)], dtype=np.int64)
            self._age = np.array([1], dtype=np.int64)
        else:
            self._xs = np.concatenate([self._xs, x], axis=0)
            self._ys = np.concatenate([self._ys, np.array([int(token_id)], dtype=np.int64)], axis=0)
            self._age = np.concatenate([self._age, np.array([1], dtype=np.int64)], axis=0) if self._age is not None else np.ones((self._xs.shape[0],), dtype=np.int64)
        # Sync FAISS index lazily
        if self._index is not None:
            self._index.add(x)

    def add_batch(self, hs: np.ndarray, token_ids: np.ndarray) -> None:
        assert hs.ndim == 2 and hs.shape[1] == self.dim
        assert token_ids.ndim == 1 and token_ids.shape[0] == hs.shape[0]
        x = hs.astype(np.float32, copy=False)
        if self.use_cosine:
            x = self._normalize(x)
        if self._xs is None:
            self._xs = x.copy()
            self._ys = token_ids.astype(np.int64, copy=False)
            self._age = np.ones((self._xs.shape[0],), dtype=np.int64)
        else:
            self._xs = np.concatenate([self._xs, x], axis=0)
            self._ys = np.concatenate([self._ys, token_ids.astype(np.int64, copy=False)], axis=0)
            if self._age is not None:
                self._age = np.concatenate([self._age, np.ones((x.shape[0],), dtype=np.int64)], axis=0)
        if self._index is not None:
            self._index.add(x)

    # --- Torch-accelerated backend (optional) ---
    def _normalize_t(self, x: 'torch.Tensor') -> 'torch.Tensor':  # type: ignore[name-defined]
        if (not self.use_cosine) or (not _TORCH_OK):
            return x
        return F.normalize(x, dim=-1)

    def add_torch(self, h: 'torch.Tensor', token_id: int) -> None:  # type: ignore[name-defined]
        if not _TORCH_OK:
            # Fallback: move to CPU numpy path
            self.add(h.detach().cpu().numpy().astype(np.float32), int(token_id))
            return
        assert h.dim() == 1 and int(h.numel()) == self.dim
        x = h.detach().to(dtype=torch.float32).reshape(1, -1)
        if self.use_cosine:
            x = self._normalize_t(x)
        if self._xs_t is None:
            self._xs_t = x
            self._ys_t = torch.tensor([int(token_id)], dtype=torch.long, device=x.device)
            self._age_t = torch.ones((1,), dtype=torch.long, device=x.device)
        else:
            self._xs_t = torch.cat([self._xs_t, x], dim=0)
            self._ys_t = torch.cat([self._ys_t, torch.tensor([int(token_id)], dtype=torch.long, device=x.device)], dim=0)
            self._age_t = torch.cat([self._age_t, torch.ones((1,), dtype=torch.long, device=x.device)], dim=0)

    def size_torch(self) -> int:
        return 0 if (self._xs_t is None) else int(self._xs_t.size(0))

    def query_torch(self, h: 'torch.Tensor', k: int = 16, vocab_size: int = 32000) -> 'torch.Tensor':  # type: ignore[name-defined]
        if (not _TORCH_OK) or (self._xs_t is None) or (self.size_torch() == 0):
            # Return zeros on device of h
            return (h.detach().new_zeros((vocab_size,), dtype=torch.float32))
        x = h.detach().to(dtype=torch.float32).reshape(1, -1)
        if self.use_cosine:
            x = self._normalize_t(x)
        k_eff = min(int(k), self.size_torch())
        # Similarity (cosine via inner product on normalized reps)
        sims = torch.matmul(x, self._xs_t.t()).squeeze(0)  # (N,)
        # topk indices (no full sort)
        topk_vals, topk_idx = torch.topk(sims, k=k_eff, largest=True)
        # Softmax over sims to get weights
        w = torch.softmax(topk_vals - topk_vals.max(), dim=0).to(dtype=torch.float32)
        # Simple aging boost
        if self._age_t is not None:
            ages = self._age_t[topk_idx].to(dtype=torch.float32)
            ages = (ages - ages.min()) / (ages.max() - ages.min() + 1e-6)
            w = w * (0.9 + 0.1 * ages)
            w = w / (w.sum() + 1e-8)
        # Update ages
        if self._age_t is not None:
            self._age_t[topk_idx] = torch.clamp(self._age_t[topk_idx] + 1, max=torch.iinfo(self._age_t.dtype).max)  # type: ignore[arg-type]
        tokens = self._ys_t[topk_idx]
        probs = torch.zeros((vocab_size,), dtype=torch.float32, device=x.device)
        # Accumulate via index_add (guard ids in range)
        mask = (tokens >= 0) & (tokens < vocab_size)
        if mask.any():
            probs.index_add_(0, tokens[mask], w[mask])
            s = probs.sum()
            if s > 0:
                probs = probs / s
        return probs

    def size(self) -> int:
        return 0 if self._xs is None else int(self._xs.shape[0])

    def query(self, h: np.ndarray, k: int = 16, vocab_size: int = 32000) -> np.ndarray:
        """Return a probability distribution over tokens from kNN.

        h: shape (D,)
        returns: probs shape (V,)
        """
        if self._xs is None or self.size() == 0:
            return np.zeros((vocab_size,), dtype=np.float32)
        x = h.astype(np.float32, copy=False)[None, :]
        if self.use_cosine:
            x = self._normalize(x)
        k_eff = min(int(k), self.size())
        if self._index is not None:
            D, I = self._index.search(x, k_eff)
            idxs = I[0]
            sims = D[0]
        else:
            # brute force
            if self.use_cosine:
                sims = (x @ self._xs.T)[0]
                idxs = np.argpartition(-sims, k_eff - 1)[:k_eff]
                sims = sims[idxs]
            else:
                dif = self._xs - x
                d2 = np.sum(dif * dif, axis=1)
                idxs = np.argpartition(d2, k_eff - 1)[:k_eff]
                sims = -d2[idxs]
        # Softmax over sims to get weights, then accumulate by token id
        sims = sims - np.max(sims)
        w = np.exp(sims).astype(np.float32)
        if w.sum() <= 0:
            return np.zeros((vocab_size,), dtype=np.float32)
        w = w / w.sum()
        # Simple aging boost: prioritize fresher entries slightly
        if self._age is not None:
            try:
                ages = self._age[idxs].astype(np.float32)
                ages = (ages - ages.min()) / (ages.max() - ages.min() + 1e-6)
                w = w * (0.9 + 0.1 * ages)
                w = w / (w.sum() + 1e-8)
            except Exception:
                pass
        # Update usage ages for retrieved indices (favor frequently used entries)
        try:
            if self._age is not None:
                self._age[idxs] = np.minimum(self._age[idxs] + 1, np.iinfo(self._age.dtype).max)
        except Exception:
            pass
        tokens = self._ys[idxs]
        probs = np.zeros((vocab_size,), dtype=np.float32)
        # accumulate
        for ti, wi in zip(tokens, w):
            if 0 <= int(ti) < vocab_size:
                probs[int(ti)] += float(wi)
        # normalize again to be safe
        s = probs.sum()
        if s > 0:
            probs = probs / s
        return probs

    def prune(self, max_items: int = 4096) -> None:
        """Keep at most max_items most recent items; rebuild FAISS index if present."""
        if self._xs is None:
            return
        if self._xs.shape[0] <= max(1, int(max_items)):
            return
        n_keep = int(max_items)
        xs = self._xs[-n_keep:]
        ys = self._ys[-n_keep:]
        age = self._age[-n_keep:] if self._age is not None else None
        self._xs, self._ys, self._age = xs, ys, age
        if self._index is not None and self._faiss is not None:
            try:
                if self.use_cosine:
                    self._index = self._faiss.IndexFlatIP(self.dim)
                else:
                    self._index = self._faiss.IndexFlatL2(self.dim)
                self._index.add(self._normalize(xs) if self.use_cosine else xs)
            except Exception:
                self._index = None
        # Torch backend pruning
        if self._xs_t is not None:
            if self._xs_t.size(0) > n_keep:
                self._xs_t = self._xs_t[-n_keep:]
                if self._ys_t is not None:
                    self._ys_t = self._ys_t[-n_keep:]
                if self._age_t is not None:
                    self._age_t = self._age_t[-n_keep:]

    def save(self, path: str) -> None:
        try:
            import numpy as _np
            if self._xs is None or self._ys is None:
                return
            _np.savez_compressed(path, xs=self._xs, ys=self._ys, age=(self._age if self._age is not None else _np.array([], dtype=_np.int64)))
        except Exception:
            return

    def load(self, path: str) -> bool:
        try:
            import numpy as _np
            import os as _os
            if (not _os.path.exists(path)):
                return False
            blob = _np.load(path)
            xs = blob.get('xs')
            ys = blob.get('ys')
            age = blob.get('age')
            if xs is None or ys is None:
                return False
            self._xs = xs.astype('float32', copy=False)
            self._ys = ys.astype('int64', copy=False)
            self._age = age.astype('int64', copy=False) if age is not None else None
            # Rebuild FAISS index if available
            if self._index is not None and self._faiss is not None:
                try:
                    if self.use_cosine:
                        self._index = self._faiss.IndexFlatIP(self.dim)
                        self._index.add(self._normalize(self._xs))
                    else:
                        self._index = self._faiss.IndexFlatL2(self.dim)
                        self._index.add(self._xs)
                except Exception:
                    self._index = None
            return True
        except Exception:
            return False



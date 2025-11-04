from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.training.simple_tokenizer import TextTokenizer


class TextFolderDataset(Dataset):
    def __init__(self, folder: str, tokenizer: TextTokenizer, seq_len: int = 256):
        # Be defensive: avoid scanning massive/hidden dirs (e.g., .venv, site-packages)
        # and tolerate OSErrors on platforms with restricted dirs. Fall back to an
        # empty list (the __getitem__ path synthesizes a trivial sample).
        self.paths = []
        try:
            root = Path(folder)
            if root.is_file() and root.suffix.lower() == ".txt":
                self.paths = [root]
            elif root.is_dir():
                for p in root.rglob("*.txt"):
                    parts = {part.lower() for part in p.parts}
                    # Skip typical large/hidden or irrelevant dirs
                    if any(seg.startswith('.') for seg in p.parts):
                        continue
                    if any(skip in parts for skip in {".venv", "venv", "site-packages", "dist-packages", "__pycache__", ".git"}):
                        continue
                    # Additional common large dirs in editable installs
                    sp = str(p).lower()
                    if ("/site-packages/" in sp) or ("/dist-packages/" in sp):
                        continue
                    self.paths.append(p)
        except Exception:
            self.paths = []
        self.tok = tokenizer
        self.seq_len = seq_len
        # Augmentation knobs
        import os
        self.noise_prob = float(os.environ.get('OMNICODER_TEXT_NOISE_P', '0.0'))
        self.noise_drop_prob = float(os.environ.get('OMNICODER_TEXT_NOISE_DROP_P', '0.0'))
        self.subliminal_prob = float(os.environ.get('OMNICODER_TEXT_SUBLIMINAL_P', '0.0'))
        self.subliminal_file = os.environ.get('OMNICODER_TEXT_SUBLIMINAL_FILE', '').strip()
        self._subliminal_pool: list[str] = []
        if self.subliminal_file:
            try:
                from pathlib import Path as _Path
                p = _Path(self.subliminal_file)
                if p.exists():
                    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
                        s = line.strip()
                        if s:
                            self._subliminal_pool.append(s)
            except Exception:
                self._subliminal_pool = []

    def __len__(self):
        return max(1, len(self.paths))

    def __getitem__(self, idx):
        if not self.paths:
            ids = self.tok.encode("hello world")
        else:
            text = self.paths[idx].read_text(encoding='utf-8', errors='ignore')
            # Optional text noise and subliminal mixing
            try:
                import random as _rand
                if self.noise_prob > 0.0 and _rand.random() < self.noise_prob:
                    # Token-level noise: randomly mutate a few token ids
                    base_ids = self.tok.encode(text)
                    n = max(1, int(0.01 * len(base_ids)))
                    for _ in range(n):
                        if not base_ids:
                            break
                        j = _rand.randrange(0, len(base_ids))
                        # Small jitter around the token id
                        base_ids[j] = max(0, base_ids[j] + _rand.choice([-2, -1, 1, 2]))
                    ids = base_ids
                else:
                    ids = self.tok.encode(text)
                # Drop spans
                if self.noise_drop_prob > 0.0 and _rand.random() < self.noise_drop_prob and len(ids) > 8:
                    a = _rand.randrange(0, len(ids) - 4)
                    b = min(len(ids), a + _rand.randrange(1, max(2, len(ids)//16)))
                    ids = ids[:a] + ids[b:]
                # Subliminal insertions: append a small unrelated sentence occasionally
                if self._subliminal_pool and self.subliminal_prob > 0.0 and _rand.random() < self.subliminal_prob:
                    extra = _rand.choice(self._subliminal_pool)
                    ids_extra = self.tok.encode("\n" + extra)
                    ids = ids + ids_extra
            except Exception:
                ids = self.tok.encode(text)
        ids = ids[: self.seq_len]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, labels


class DataModule:
    def __init__(self, train_folder: str, val_folder: Optional[str] = None, seq_len: int = 256, batch_size: int = 4):
        self.train_folder = train_folder
        self.val_folder = val_folder or train_folder
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = TextTokenizer()

    def train_loader(self) -> DataLoader:
        ds = TextFolderDataset(self.train_folder, self.tokenizer, self.seq_len)
        nw = recommend_num_workers()
        try:
            import torch as _torch
            pin = bool(_torch.cuda.is_available())
        except Exception:
            pin = False
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': nw,
            'pin_memory': pin,
            'persistent_workers': bool(nw and nw > 0),
        }
        # Prefetch only applies when workers > 0; choose a small multiple of workers
        if kwargs['persistent_workers']:
            try:
                kwargs['prefetch_factor'] = max(2, min(8, int(nw) * 2))
            except Exception:
                pass
        return DataLoader(ds, **kwargs)

    def val_loader(self) -> DataLoader:
        ds = TextFolderDataset(self.val_folder, self.tokenizer, self.seq_len)
        nw = recommend_num_workers()
        try:
            import torch as _torch
            pin = bool(_torch.cuda.is_available())
        except Exception:
            pin = False
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': nw,
            'pin_memory': pin,
            'persistent_workers': bool(nw and nw > 0),
        }
        if kwargs['persistent_workers']:
            try:
                kwargs['prefetch_factor'] = max(2, min(8, int(nw) * 2))
            except Exception:
                pass
        return DataLoader(ds, **kwargs)

    # Optional teacher-mark loader for retrieval write-policy supervision.
    # Expects JSONL lines: {"text": str, "write_marks": [indices]}
    def teacher_marks_loader(self, path: str | None = None) -> DataLoader | None:
        if path is None:
            return None
        try:
            import json
            from torch.utils.data import Dataset
            class _MarksDataset(Dataset):
                def __init__(self, p: str, tok: TextTokenizer, seq_len: int):
                    self.items = []
                    # Read robustly: skip blank/comment lines; tolerate per-line JSON errors
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if not s or s.startswith('#'):
                                continue
                            try:
                                obj = json.loads(s)
                                if isinstance(obj, dict) and 'text' in obj:
                                    self.items.append(obj)
                            except Exception:
                                # skip malformed line
                                continue
                    self.tok = tok
                    self.seq_len = seq_len
                def __len__(self):
                    return len(self.items)
                def __getitem__(self, idx):
                    it = self.items[idx]
                    text = it.get('text','')
                    marks = it.get('write_marks', [])
                    ids = self.tok.encode(text)[: self.seq_len]
                    if len(ids) < self.seq_len:
                        ids = ids + [0] * (self.seq_len - len(ids))
                    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
                    # Build a dense mark vector aligned to input length
                    mv = torch.zeros_like(input_ids, dtype=torch.float32)
                    for t in marks:
                        try:
                            ti = int(t)
                            if 0 <= ti < mv.numel():
                                mv[ti] = 1.0
                        except Exception:
                            continue
                    return {'input_ids': input_ids, 'write_marks': mv}
            ds = _MarksDataset(path, self.tokenizer, self.seq_len)
            if len(ds) == 0:
                return None
            nw = recommend_num_workers()
            try:
                import torch as _torch
                pin = bool(_torch.cuda.is_available())
            except Exception:
                pin = False
            kwargs = {
                'batch_size': self.batch_size,
                'shuffle': True,
                'num_workers': nw,
                'pin_memory': pin,
                'persistent_workers': bool(nw and nw > 0),
            }
            if kwargs['persistent_workers']:
                try:
                    kwargs['prefetch_factor'] = max(2, min(8, int(nw) * 2))
                except Exception:
                    pass
            return DataLoader(ds, **kwargs)
        except Exception:
            return None

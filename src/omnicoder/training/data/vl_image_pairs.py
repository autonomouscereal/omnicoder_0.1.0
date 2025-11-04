from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_image(path: str, size: Tuple[int, int] | None = None) -> torch.Tensor:
    """Load an image file and return a CHW float tensor in [-1, 1]."""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size[0], size[1]), Image.BICUBIC)
    arr = np.asarray(img).astype('float32') / 255.0
    arr = (arr * 2.0) - 1.0
    # HWC -> CHW
    chw = np.transpose(arr, (0, 1, 2)) if arr.shape[-1] == 1 else np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(chw)


class ImagePairsDataset(Dataset):
    """
    Minimal dataset for supervised image latent reconstruction.

    Supports JSONL with fields {"image": "/path/to/img.jpg", "text": "prompt"}
    or a directory of images (prompts default to empty strings).
    """

    def __init__(self, source: str, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.image_size = image_size
        p = Path(source)
        self.samples: List[Tuple[str, str]] = []
        if p.is_file() and p.suffix.lower() in ('.jsonl', '.json'):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        img = str(rec.get('image', ''))
                        txt = str(rec.get('text', ''))
                        if img:
                            self.samples.append((img, txt))
                    except Exception:
                        continue
        elif p.is_dir():
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                for f in p.rglob(ext):
                    self.samples.append((str(f), ''))
        else:
            raise ValueError(f"Unsupported source: {source}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, txt = self.samples[idx]
        x = _load_image(img_path, self.image_size)
        return x, txt



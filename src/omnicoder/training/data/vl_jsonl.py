from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer


class VLJSONLDataset(Dataset):
    """
    Vision-Language dataset based on a JSONL file with records:
      {"image": "/path/to/img.jpg", "text": "caption or question..."}

    Returns fused feature sequences and masked labels for language-only loss.
    """

    def __init__(
        self,
        jsonl_path: str,
        d_model: int,
        vision_dim: int = 384,
        image_size: Tuple[int, int] = (224, 224),
        max_frames: int = 0,
    ) -> None:
        self.path = Path(jsonl_path)
        self.records: List[dict] = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.size = (int(image_size[0]), int(image_size[1]))
        self.tokenizer = get_text_tokenizer(prefer_hf=True)
        self.composer = MultimodalComposer(d_model=d_model, vision_dim=vision_dim)
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        from PIL import Image  # type: ignore

        img = Image.open(image_path).convert("RGB").resize(self.size)
        arr = np.array(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
        return t

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path: str = str(rec["image"])  # type: ignore
        text: str = str(rec["text"])      # type: ignore

        image = self._load_image_tensor(img_path).unsqueeze(0)  # (1,3,H,W)
        input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)  # (1, T)

        # Build fused features: [IMG_BOS, proj(img_tokens)..., IMG_EOS, text_emb...]
        # Note: composer handles embedding through model later; here we just prepare inputs
        return image, input_ids


class VLDataModule:
    def __init__(
        self,
        jsonl_path: str,
        d_model: int,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.ds = VLJSONLDataset(jsonl_path=jsonl_path, d_model=d_model, image_size=image_size)
        self.batch_size = int(batch_size)

    def loader(self) -> DataLoader:
        import sys as _sys
        nw = 0 if _sys.platform.startswith("win") else recommend_num_workers()
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=nw, collate_fn=self._collate)

    @staticmethod
    def _collate(batch):
        images = torch.stack([b[0].squeeze(0) for b in batch], dim=0)  # (B,3,H,W)
        # pad text ids to max len
        id_seqs: List[torch.Tensor] = [b[1].squeeze(0) for b in batch]
        max_len = max(t.size(0) for t in id_seqs)
        pad_id = 0
        padded = []
        for t in id_seqs:
            if t.size(0) < max_len:
                t = torch.cat([t, torch.full((max_len - t.size(0),), pad_id, dtype=torch.long)], dim=0)
            padded.append(t)
        input_ids = torch.stack(padded, dim=0)  # (B, T)
        return images, input_ids



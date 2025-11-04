from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from omnicoder.modeling.multimodal.video_vq import VideoVQ
from omnicoder.training.simple_tokenizer import get_text_tokenizer


class VideoVLVQDataset(Dataset):
    """
    JSONL with records: {"video": "/path/to.mp4", "text": "caption"}
    Emits (vq_token_ids, text_ids) where vq_token_ids is a 1D LongTensor of concatenated
    per-frame VQ tokens with frame separators.
    """

    def __init__(self, jsonl_path: str, vq_codebook: str, patch: int = 16):
        self.recs = [json.loads(l) for l in open(jsonl_path, 'r', encoding='utf-8', errors='ignore') if l.strip()]
        self.tok = get_text_tokenizer(prefer_hf=True)
        self.vq = VideoVQ(patch=patch, codebook_path=vq_codebook)

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2  # type: ignore
        r = self.recs[idx]
        path = str(Path(r['video']))
        cap = cv2.VideoCapture(path)
        frames: List[np.ndarray] = []
        i = 0
        while i < 16:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(fr)
            i += 1
        cap.release()
        if not frames:
            frames = [np.zeros((224,224,3), dtype=np.uint8) for _ in range(4)]
        arr = np.stack(frames, axis=0)
        codes = self.vq.encode(arr)  # list of arrays per frame
        # Concatenate with frame separators (codebook_size id used as separator)
        sep_id = self.vq.codebook_size
        merged: List[int] = []
        for i, c in enumerate(codes):
            merged.extend([int(x) for x in c.tolist()])
            if i < len(codes) - 1:
                merged.append(sep_id)
        vq_ids = torch.tensor(merged, dtype=torch.long)
        text_ids = torch.tensor(self.tok.encode(r.get('text', '')), dtype=torch.long)
        return vq_ids, text_ids



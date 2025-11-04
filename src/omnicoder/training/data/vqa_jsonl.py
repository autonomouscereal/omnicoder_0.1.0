from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.training.simple_tokenizer import get_text_tokenizer


class VQAJSONLDataset(Dataset):
    """
    VQA-style dataset based on a JSONL file with records:
      {"image": "/path/to/img.jpg", "question": "...", "answer": "..."}

    Returns (image_tensor, input_ids) where input_ids encodes a formatted
    text sequence like "Q: ...\nA:" and cross-entropy loss is computed over
    the answer segment as next-token prediction.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_size: Tuple[int, int] = (224, 224),
        qa_template: str = "Q: {q}\nA:",
    ) -> None:
        self.path = Path(jsonl_path)
        self.records: List[dict] = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.size = (int(image_size[0]), int(image_size[1]))
        self.template = qa_template
        self.tokenizer = get_text_tokenizer(prefer_hf=True)

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
        # Be tolerant to VL jsonl that uses {image, text} instead of VQA's {image, question, answer}
        img_path: str = str(rec["image"])  # type: ignore
        question_val = rec.get("question", rec.get("text", ""))
        question: str = str(question_val)
        answer: str = str(rec.get("answer", ""))

        image = self._load_image_tensor(img_path)  # (3,H,W)
        prompt = self.template.format(q=question)
        # Pack as: [prompt tokens][answer tokens]
        ids_prompt = self.tokenizer.encode(prompt)
        ids_answer = self.tokenizer.encode(answer)
        ids = ids_prompt + ids_answer
        input_ids = torch.tensor(ids, dtype=torch.long)
        # Return also the split index for computing loss on answer only (optional)
        split_idx = len(ids_prompt)
        return image, input_ids, split_idx


class VQADataModule:
    def __init__(self, jsonl_path: str, batch_size: int = 1, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.ds = VQAJSONLDataset(jsonl_path=jsonl_path, image_size=image_size)
        self.batch_size = int(batch_size)

    def loader(self) -> DataLoader:
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=recommend_num_workers(), collate_fn=self._collate)

    @staticmethod
    def _collate(batch):
        images = torch.stack([b[0] for b in batch], dim=0)  # (B,3,H,W)
        id_seqs: List[torch.Tensor] = [b[1] for b in batch]
        splits: List[int] = [int(b[2]) for b in batch]
        max_len = max(t.size(0) for t in id_seqs)
        pad_id = 0
        padded = []
        for t in id_seqs:
            if t.size(0) < max_len:
                t = torch.cat([t, torch.full((max_len - t.size(0),), pad_id, dtype=torch.long)], dim=0)
            padded.append(t)
        input_ids = torch.stack(padded, dim=0)
        return images, input_ids, torch.tensor(splits, dtype=torch.long)



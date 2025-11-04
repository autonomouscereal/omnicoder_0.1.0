from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.training.simple_tokenizer import get_text_tokenizer


class VideoJSONLDataset(Dataset):
    """
    Video-Language dataset based on a JSONL file with records like:
      {"video": "/path/to/video_or_frames_dir", "text": "caption or question..."}

    The `video` field can be:
      - a directory containing frame images (.png/.jpg), or
      - a single video file (optional; requires OpenCV to decode)
    """

    def __init__(
        self,
        jsonl_path: str,
        image_size: Tuple[int, int] = (224, 224),
        max_frames: int = 16,
    ) -> None:
        self.path = Path(jsonl_path)
        self.records: List[dict] = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.size = (int(image_size[0]), int(image_size[1]))
        self.max_frames = int(max_frames)
        self.tokenizer = get_text_tokenizer(prefer_hf=True)

    def __len__(self) -> int:
        return len(self.records)

    def _load_frames_from_dir(self, d: Path) -> np.ndarray:
        files = sorted([f for f in d.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        arrs: List[np.ndarray] = []
        for f in files:
            try:
                from PIL import Image  # type: ignore
                img = Image.open(f).convert("RGB").resize(self.size)
                arrs.append(np.array(img))
            except Exception:
                continue
        if not arrs:
            return np.zeros((1, self.size[1], self.size[0], 3), dtype=np.uint8)
        # Subsample uniformly to max_frames
        t = len(arrs)
        if t > self.max_frames:
            idx = np.linspace(0, t - 1, num=self.max_frames).round().astype(int)
            arrs = [arrs[i] for i in idx]
        return np.stack(arrs, axis=0)

    def _load_frames_from_video(self, p: Path) -> np.ndarray:
        try:
            import cv2  # type: ignore
        except Exception:
            # Fallback: return a single black frame if OpenCV not available
            return np.zeros((1, self.size[1], self.size[0], 3), dtype=np.uint8)
        cap = cv2.VideoCapture(str(p))
        frames: List[np.ndarray] = []
        ok = True
        while ok:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        if not frames:
            return np.zeros((1, self.size[1], self.size[0], 3), dtype=np.uint8)
        t = len(frames)
        if t > self.max_frames:
            idx = np.linspace(0, t - 1, num=self.max_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vpath = Path(str(rec["video"]))  # type: ignore
        text: str = str(rec["text"])     # type: ignore
        if vpath.is_dir():
            frames = self._load_frames_from_dir(vpath)
        else:
            frames = self._load_frames_from_video(vpath)
        # Normalize to float32 0..1 and to tensor (T,3,H,W)
        frames = frames.astype("float32") / 255.0
        video_btchw = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
        input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
        return video_btchw, input_ids


class VideoDataModule:
    def __init__(self, jsonl_path: str, batch_size: int = 1, image_size: Tuple[int, int] = (224, 224), max_frames: int = 16) -> None:
        self.ds = VideoJSONLDataset(jsonl_path=jsonl_path, image_size=image_size, max_frames=max_frames)
        self.batch_size = int(batch_size)

    def loader(self) -> DataLoader:
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=recommend_num_workers(), collate_fn=self._collate)

    @staticmethod
    def _collate(batch):
        videos = torch.cat([b[0] for b in batch], dim=0)  # (B, T, 3, H, W)
        seqs: List[torch.Tensor] = [b[1].squeeze(0) for b in batch]
        max_len = max(t.size(0) for t in seqs)
        pad_id = 0
        padded = []
        for t in seqs:
            if t.size(0) < max_len:
                t = torch.cat([t, torch.full((max_len - t.size(0),), pad_id, dtype=torch.long)], dim=0)
            padded.append(t)
        input_ids = torch.stack(padded, dim=0)
        return videos, input_ids



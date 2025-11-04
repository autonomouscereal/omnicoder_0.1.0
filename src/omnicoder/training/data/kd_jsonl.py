from __future__ import annotations

"""
KD JSONL dataset for distillation with optional rationales and router targets.

JSONL record schema (per line):
{
  "text": "training text ...",            # required
  "rationale": "optional rationale ...", # optional, appended before text
  "router_targets": {                      # optional, expert routing targets
     "num_layers": 12,
     "num_experts": 8,
     "layer_means": [ [float]*E ]*L        # per-layer mean probs over experts
  }
}
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from omnicoder.training.simple_tokenizer import get_text_tokenizer


class KDJSONLDataset(Dataset):
    def __init__(self, jsonl: str, seq_len: int = 512, include_rationale: bool = True) -> None:
        self.path = Path(jsonl)
        self.seq_len = int(seq_len)
        self.include_rationale = bool(include_rationale)
        self.records: List[dict] = []
        for line in self.path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                if 'text' in ex and isinstance(ex['text'], str):
                    self.records.append(ex)
            except Exception:
                continue
        self.tok = get_text_tokenizer(prefer_hf=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        ex = self.records[idx]
        text = ex.get('text', '')
        rat = ex.get('rationale', '')
        if self.include_rationale and rat:
            text = str(rat) + "\n" + str(text)
        ids = self.tok.encode(text)
        ids = ids[: self.seq_len]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        # Router targets: per-layer mean probs
        rt = ex.get('router_targets', None)
        return input_ids, labels, rt


def collate_kd(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    rts = [b[2] for b in batch]
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), rts


def kd_loader(jsonl: str, seq_len: int = 512, batch_size: int = 2, include_rationale: bool = True) -> DataLoader:
    ds = KDJSONLDataset(jsonl, seq_len=seq_len, include_rationale=include_rationale)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True, collate_fn=collate_kd)



from __future__ import annotations

"""
Train a small user-preference expert using LoRA on recent user text, and build a
small local retriever index for personalized RAG.

Inputs:
  - --data: folder of .txt files or a JSONL with {text: ...}
  - --steps: training steps (small; intended to be run frequently)

Outputs:
  - LoRA adapter weights (pt)
  - Simple TF-IDF index (saved as JSON with texts) for local retrieval
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import TextTokenizer
from omnicoder.utils.resources import recommend_num_workers


class _TextLines(Dataset):
    def __init__(self, path: str, seq_len: int = 256) -> None:
        self.items: List[str] = []
        p = Path(path)
        if p.is_dir():
            for q in p.rglob("*.txt"):
                try:
                    s = q.read_text(encoding="utf-8", errors="ignore").strip()
                    if s:
                        self.items.append(s)
                except Exception:
                    continue
        else:
            try:
                if p.suffix.lower() == ".jsonl":
                    with p.open("r", encoding="utf-8") as f:
                        for line in f:
                            t = line.strip()
                            if not t:
                                continue
                            try:
                                obj = json.loads(t)
                                if isinstance(obj, dict) and "text" in obj:
                                    self.items.append(str(obj["text"]))
                            except Exception:
                                # treat as plain text
                                self.items.append(t)
                else:
                    s = p.read_text(encoding="utf-8", errors="ignore").strip()
                    if s:
                        self.items.append(s)
            except Exception:
                self.items = []
        self.tok = TextTokenizer()
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return max(1, len(self.items))

    def __getitem__(self, idx: int):
        text = self.items[idx % len(self.items)] if self.items else "hello world"
        ids = self.tok.encode(text)[: self.seq_len]
        if len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def _inject_lora_linear(module: nn.Module, r: int, alpha: int, dropout: float) -> int:
    replaced = 0
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and child.out_features >= 64:
            parent = module
            path = name.split(".")
            for p in path[:-1]:
                parent = getattr(parent, p)
            leaf_name = path[-1]
            base = getattr(parent, leaf_name)

            class LoRALinear(nn.Module):
                def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
                    super().__init__()
                    self.base = base
                    self.r = r
                    self.dropout = nn.Dropout(dropout)
                    self.scaling = alpha / max(r, 1)
                    self.A = nn.Linear(base.in_features, r, bias=False)
                    self.B = nn.Linear(r, base.out_features, bias=False)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.base(x) + self.B(self.A(self.dropout(x))) * self.scaling

            lora = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, leaf_name, lora)
            replaced += 1
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a small preference expert (LoRA) and build a local retriever index")
    ap.add_argument("--data", type=str, default=os.getenv("OMNICODER_PREF_DATA", "examples/preferences.jsonl"))
    ap.add_argument("--seq_len", type=int, default=int(os.getenv("OMNICODER_PREF_SEQ_LEN", "256")))
    ap.add_argument("--steps", type=int, default=int(os.getenv("OMNICODER_PREF_STEPS", "200")))
    ap.add_argument("--batch", type=int, default=int(os.getenv("OMNICODER_PREF_BATCH", "2")))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_PREF_DEVICE", ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.add_argument("--preset", type=str, default=os.getenv("OMNICODER_PREF_PRESET", os.getenv("OMNICODER_STUDENT_PRESET", "mobile_2gb")))
    ap.add_argument("--lora_r", type=int, default=int(os.getenv("OMNICODER_PREF_LORA_R", "8")))
    ap.add_argument("--lora_alpha", type=int, default=int(os.getenv("OMNICODER_PREF_LORA_ALPHA", "16")))
    ap.add_argument("--lora_dropout", type=float, default=float(os.getenv("OMNICODER_PREF_LORA_DROPOUT", "0.05")))
    ap.add_argument("--out", type=str, default=os.getenv("OMNICODER_PREF_OUT", "weights/pref_expert_lora.pt"))
    ap.add_argument("--rag_out", type=str, default=os.getenv("OMNICODER_PREF_RAG_OUT", "weights/pref_index.json"))
    args = ap.parse_args()

    # Data
    ds = _TextLines(args.data, seq_len=args.seq_len)
    if len(ds) == 0:
        print("[pref] no data; nothing to train")
        return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=recommend_num_workers())

    # Model
    from omnicoder.inference.generate import build_mobile_model_by_name
    model = build_mobile_model_by_name(args.preset)
    model.to(args.device)
    model.train()
    _ = _inject_lora_linear(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Train quick CE
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    steps = int(args.steps)
    it = iter(dl)
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(args.device)
        y = y.to(args.device)
        opt.zero_grad(set_to_none=True)
        out = model(x, past_kv=None, use_cache=False)
        logits = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"[pref] step {step+1}/{steps} loss={float(loss.item()):.4f}")

    # Save LoRA adapter tensors
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _safe_save(model.state_dict(), args.out)
    print(f"[pref] saved LoRA expert to {args.out}")

    # Build a tiny TF-IDF index as a JSON for on-device lookup (no heavy deps)
    try:
        items = ds.items
        vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        docs: list[list[str]] = []
        for s in items:
            toks = [t for t in s.lower().split() if t]
            docs.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        for t in df.keys():
            if t not in vocab:
                vocab[t] = len(vocab)
        # Persist as raw texts for simplicity; runtime can re-tokenize
        Path(args.rag_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rag_out, "w", encoding="utf-8") as f:
            json.dump({"texts": items[:10000]}, f)
        # Also emit JSONL lines with {"text": ...}
        try:
            p_jsonl = Path(args.rag_out).with_suffix('.jsonl')
            with p_jsonl.open('w', encoding='utf-8') as fj:
                for t in items[:10000]:
                    fj.write(json.dumps({"text": t}) + "\n")
        except Exception:
            pass
        print(f"[pref] wrote RAG index to {args.rag_out}")
    except Exception as e:
        print(f"[pref] building RAG index failed: {e}")


if __name__ == "__main__":
    main()



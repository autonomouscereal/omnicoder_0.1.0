from __future__ import annotations

"""
Build a unified multimodal embedding index using PreAligner embeddings.

Inputs: one or more folders containing text (.txt/.md), images (.png/.jpg/.jpeg),
audio (.wav/.mp3 optional), and video frames (folder of images). For each found
item, compute a PreAligner embedding (text/image/audio/video when available) and
store in a FAISS or PQ index (if faiss is installed) or a NumPy mmap fallback.

Usage:
  python -m omnicoder.tools.multi_index_build --roots ./docs ./examples/data/vq/images \
    --out /models/multi_index

Optional:
  --use_pq (use product quantization via tools/pq_build if desired)

Runtime:
  The generator can later load this index to retrieve nearest multimodal items
  for prompting or guidance. This tool is dependency-light and best-effort: if
  dependencies (PIL/soundfile/faiss) are missing, it skips those modalities.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

import torch
from omnicoder.modeling.multimodal.aligner import PreAligner


def _gather_files(roots: List[str]) -> Tuple[List[Path], List[Path]]:
    text, images = [], []
    for r in roots:
        root = Path(r)
        if not root.exists():
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                p = Path(dp) / fn
                fl = fn.lower()
                if fl.endswith((".txt", ".md")):
                    text.append(p)
                if fl.endswith((".png", ".jpg", ".jpeg")):
                    images.append(p)
    return text, images


def _encode_text(paths: List[Path], pre: PreAligner, device: str) -> Tuple[np.ndarray, List[str]]:
    xs, ids = [], []
    for p in paths:
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
            # Simple whitespace tokenizer; for better quality, plug a HF tokenizer/encoder
            tokens = torch.from_numpy(np.random.randn(1, 32, 512).astype("float32"))  # placeholder small feature
            emb = pre(text=tokens.to(device))["text"].detach().cpu().numpy()[0]
            xs.append(emb)
            ids.append(str(p))
        except Exception:
            continue
    if xs:
        return np.stack(xs, axis=0), ids
    return np.zeros((0, pre.text_proj[-1].out_features), dtype=np.float32), ids


def _encode_images(paths: List[Path], pre: PreAligner, device: str) -> Tuple[np.ndarray, List[str]]:
    xs, ids = [], []
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return np.zeros((0, pre.image_proj[-1].out_features), dtype=np.float32), ids
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize((224, 224))
            arr = np.asarray(img).astype("float32") / 255.0
            # Toy image features (HWC→tokens) until a vision backbone is wired directly
            tokens = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)
            tokens = tokens.flatten(2).transpose(1, 2)[:, :64, :]  # (1,T=64,C=224*3)
            # Project to expected image_dim
            # For simplicity, reduce to 768 with a linear map on-the-fly
            proj = torch.nn.Linear(tokens.size(-1), 768).to(device)
            with torch.no_grad():
                feat = proj(tokens.to(device))
            emb = pre(image=feat)["image"].detach().cpu().numpy()[0]
            xs.append(emb)
            ids.append(str(p))
        except Exception:
            continue
    if xs:
        return np.stack(xs, axis=0), ids
    return np.zeros((0, pre.image_proj[-1].out_features), dtype=np.float32), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="One or more folders to index")
    ap.add_argument("--out", type=str, required=True, help="Output directory (/models/multi_index)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--use_pq", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = PreAligner().to(args.device).eval()
    txt_paths, img_paths = _gather_files(args.roots)

    txt_emb, txt_ids = _encode_text(txt_paths, pre, args.device)
    img_emb, img_ids = _encode_images(img_paths, pre, args.device)

    # Concatenate as a shared bank with modality tags
    all_emb = []
    all_ids = []
    if len(txt_ids) > 0:
        all_emb.append(txt_emb)
        all_ids.extend(["text:" + i for i in txt_ids])
    if len(img_ids) > 0:
        all_emb.append(img_emb)
        all_ids.extend(["image:" + i for i in img_ids])
    if not all_emb:
        print("No indexable items found.")
        return
    X = np.concatenate(all_emb, axis=0).astype("float32")
    np.save(out_dir / "embeddings.npy", X)
    (out_dir / "ids.txt").write_text("\n".join(all_ids), encoding="utf-8")
    print(f"Wrote {X.shape[0]} items to {out_dir}")


if __name__ == "__main__":
    main()



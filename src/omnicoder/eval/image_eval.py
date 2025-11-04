from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def _load_pairs_from_jsonl(jsonl_path: str) -> List[Tuple[Path, str]]:
    pairs: List[Tuple[Path, str]] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except Exception:
            continue
        img = ex.get("file") or ex.get("image")
        prompt = ex.get("prompt") or ex.get("caption") or ""
        if img and prompt:
            pairs.append((Path(img), str(prompt)))
    return pairs


def _compute_clip_scores(pairs: List[Tuple[Path, str]]) -> float:
    try:
        import torch  # type: ignore
        import clip  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        print("[clipscore] Please install: pip install git+https://github.com/openai/CLIP.git pillow torch")
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    sims: List[float] = []
    for img_path, prompt in pairs:
        try:
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            txt = clip.tokenize([prompt]).to(device)
            with torch.no_grad():
                imf = model.encode_image(img)
                tf = model.encode_text(txt)
                imf = imf / (imf.norm(dim=-1, keepdim=True) + 1e-6)
                tf = tf / (tf.norm(dim=-1, keepdim=True) + 1e-6)
                sim = (imf @ tf.T).squeeze().item()
            sims.append(float(sim))
        except Exception:
            continue
    if not sims:
        return 0.0
    # Convert cosine similarity (-1..1) to a 0..100-like score (optional scaling)
    avg = sum(sims) / len(sims)
    return float(avg * 100.0)


def _compute_fid(pred_dir: str, ref_dir: str) -> float:
    try:
        from cleanfid import fid  # type: ignore
    except Exception:
        print("[fid] Please install: pip install clean-fid")
        return -1.0
    try:
        score = fid.compute_fid(pred_dir, ref_dir, mode="clean")
        return float(score)
    except Exception as e:
        print(f"[fid] Failed: {e}")
        return -1.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Image evaluation: CLIPScore and FID")
    ap.add_argument("--mode", choices=["clip", "fid"], required=True)
    ap.add_argument("--jsonl", type=str, default="", help="For clip mode: JSONL with {file, prompt}")
    ap.add_argument("--pred_dir", type=str, default="", help="For fid mode: directory of generated images")
    ap.add_argument("--ref_dir", type=str, default="", help="For fid mode: directory of reference images")
    args = ap.parse_args()

    if args.mode == "clip":
        if not args.jsonl:
            print("--jsonl is required for clip mode")
            return
        pairs = _load_pairs_from_jsonl(args.jsonl)
        score = _compute_clip_scores(pairs)
        print(f"CLIPScore(avg*100): {score:.2f}")
        return

    # FID
    if not args.pred_dir or not args.ref_dir:
        print("--pred_dir and --ref_dir are required for fid mode")
        return
    score = _compute_fid(args.pred_dir, args.ref_dir)
    if score >= 0:
        print(f"FID(clean): {score:.3f}")
    else:
        print("FID not computed (missing dependency or error)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_video_dir(path: str, num_frames: int = 16, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    import os
    import cv2  # type: ignore
    vids: List[np.ndarray] = []
    for fp in sorted(os.listdir(path)):
        if not fp.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        full = str(Path(path) / fp)
        cap = cv2.VideoCapture(full)
        frames: List[np.ndarray] = []
        ok = True
        while ok:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
            frames.append(fr)
        cap.release()
        if not frames:
            continue
        # Uniformly sample to num_frames
        t = len(frames)
        if t >= num_frames:
            idx = np.linspace(0, t - 1, num_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        else:
            # loop-pad
            while len(frames) < num_frames:
                frames += frames
            frames = frames[:num_frames]
        arr = np.stack(frames, axis=0)  # (T,H,W,C)
        vids.append(arr)
    if not vids:
        return np.zeros((0, num_frames, size[1], size[0], 3), dtype=np.uint8)
    return np.stack(vids, axis=0)


def _compute_fvd(pred_dir: str, ref_dir: str, num_frames: int = 16) -> float:
    try:
        import torch  # type: ignore
        from pytorch_fvd import FVD  # type: ignore
    except Exception:
        print("[fvd] Please install: pip install pytorch-fvd")
        return -1.0

    preds_np = _load_video_dir(pred_dir, num_frames=num_frames, size=(224, 224))  # (N,T,H,W,C)
    refs_np = _load_video_dir(ref_dir, num_frames=num_frames, size=(224, 224))
    if preds_np.shape[0] == 0 or refs_np.shape[0] == 0:
        print("[fvd] No videos found or could not decode.")
        return -1.0
    # Convert to torch: (N,T,C,H,W), float in [0,1]
    preds_t = torch.from_numpy(preds_np).permute(0, 1, 4, 2, 3).float() / 255.0
    refs_t = torch.from_numpy(refs_np).permute(0, 1, 4, 2, 3).float() / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds_t = preds_t.to(device)
    refs_t = refs_t.to(device)
    model = FVD(device=device)
    with torch.no_grad():
        score = model(refs_t, preds_t).item()
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser(description="Video evaluation: FVD (pytorch-fvd)")
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--ref_dir", type=str, required=True)
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args()

    score = _compute_fvd(args.pred_dir, args.ref_dir, num_frames=int(args.frames))
    if score >= 0:
        print(f"FVD: {score:.3f}")
    else:
        print("FVD not computed (missing dependency or no videos).")


if __name__ == "__main__":
    main()

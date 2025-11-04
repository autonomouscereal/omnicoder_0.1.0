"""
Train a simple VQ codebook for video frames by sampling frame patch embeddings.

This aligns with `modeling/multimodal/video_vq.py` (code_dim, patch) and produces
`weights/video_vq_codebook.pt` that can be loaded by `VideoVQ(codebook_path=...)`.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn


class FramePatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, emb_dim: int = 192, patch: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, emb_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y.flatten(2).transpose(1, 2)  # (B, N, D)


def kmeans_fit(embs: torch.Tensor, k: int, iters: int = 50) -> torch.Tensor:
    device = embs.device
    n, d = embs.shape
    idx = torch.randperm(n, device=device)[:k]
    centers = embs[idx].clone()
    for _ in range(iters):
        sims = embs @ centers.t()
        assign = sims.argmax(dim=1)
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                centers[j] = embs[mask].mean(dim=0)
    centers = nn.functional.normalize(centers, dim=-1)
    return centers


def _read_video_frames(path: str, max_frames: int, resize: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for video VQ training.") from e
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    c = 0
    while c < max_frames and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize > 0:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        c += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames, axis=0)  # (T,H,W,3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a video VQ codebook (frame patch embeddings)")
    ap.add_argument('--videos', type=str, required=True, help='Text file with one video path per line')
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--emb_dim', type=int, default=192)
    ap.add_argument('--codebook_size', type=int, default=8192)
    ap.add_argument('--frames_per_video', type=int, default=16)
    ap.add_argument('--samples', type=int, default=100000, help='Total patches to sample for kmeans')
    ap.add_argument('--out', type=str, default='weights/video_vq_codebook.pt')
    args = ap.parse_args()

    video_list = [l.strip() for l in Path(args.videos).read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed = FramePatchEmbed(emb_dim=args.emb_dim, patch=args.patch).to(device).eval()

    patches: List[torch.Tensor] = []
    total = 0
    with torch.inference_mode():
        for vp in video_list:
            try:
                frames = _read_video_frames(vp, max_frames=int(args.frames_per_video), resize=int(args.resize))
            except Exception:
                continue
            x = torch.from_numpy(frames).float().to(device) / 255.0
            x = x.permute(0, 3, 1, 2)  # (T,3,H,W)
            pe = embed(x)  # (T, N, D)
            tn, n, d = pe.shape
            pe = nn.functional.normalize(pe.reshape(tn * n, d), dim=-1)
            patches.append(pe.cpu())
            total += pe.size(0)
            if total >= args.samples:
                break
    if not patches:
        raise SystemExit("No frames collected or zero patches extracted.")
    embs = torch.cat(patches, dim=0)[: args.samples].to(device)
    centers = kmeans_fit(embs, k=int(args.codebook_size), iters=50).cpu()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    payload = {'codebook': centers, 'emb_dim': int(args.emb_dim), 'patch': int(args.patch)}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, payload, meta={'train_args': {'samples': int(args.samples)}})
    else:
        _safe_save(payload, args.out)
        final = args.out
    # Best by simple inertia proxy (mean distance to nearest center), lower is better
    try:
        if callable(maybe_save_best):
            with torch.inference_mode():
                ce = centers.to(device)
                sims = embs @ ce.t()
                assign = sims.argmax(dim=1)
                d = 1.0 - sims.gather(1, assign.unsqueeze(1)).squeeze(1)
                inertia = float(d.mean().item())
                maybe_save_best(args.out, nn.Identity(), 'vq_inertia', inertia, higher_is_better=False)
    except Exception:
        pass
    print(f"Saved video VQ codebook to {args.out} with shape {tuple(centers.shape)}")


if __name__ == '__main__':
    main()

"""
Train a video VQ codebook: sample frame patch embeddings across a folder of videos
and fit a k-means codebook. Saves a torch tensor usable by `VideoVQ`.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from omnicoder.utils.resources import recommend_num_workers


class VideoFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 224, step: int = 8) -> None:
        self.paths: List[Path] = [p for p in Path(root).rglob('*') if p.suffix.lower() in ('.mp4', '.mov', '.avi')]
        self.image_size = int(image_size)
        self.step = int(step)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import cv2  # type: ignore
        path = str(self.paths[idx])
        cap = cv2.VideoCapture(path)
        frames: List[np.ndarray] = []
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if (i % self.step) == 0:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                fr = cv2.resize(fr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                frames.append(fr)
            i += 1
        cap.release()
        if not frames:
            # produce a blank
            fr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frames = [fr]
        arr = np.stack(frames, axis=0).astype('float32') / 255.0
        return torch.from_numpy(arr).permute(0,3,1,2)  # (T,3,H,W)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, emb_dim: int = 192, patch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, emb_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y.flatten(2).transpose(1, 2)


def kmeans_fit(embs: torch.Tensor, k: int, iters: int = 50) -> torch.Tensor:
    device = embs.device
    n, d = embs.shape
    idx = torch.randperm(n, device=device)[:k]
    centers = embs[idx].clone()
    for _ in range(iters):
        sims = embs @ centers.t()
        assign = sims.argmax(dim=1)
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                centers[j] = embs[mask].mean(dim=0)
    centers = nn.functional.normalize(centers, dim=-1)
    return centers


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a video VQ codebook from frame patches")
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--emb_dim', type=int, default=192)
    ap.add_argument('--codebook_size', type=int, default=8192)
    ap.add_argument('--samples', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--out', type=str, default='weights/video_vq_codebook.pt')
    args = ap.parse_args()

    ds = VideoFolderDataset(args.data, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=recommend_num_workers())
    embed = PatchEmbed(emb_dim=args.emb_dim, patch=args.patch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed.to(device).eval()

    patches: List[torch.Tensor] = []
    total = 0
    with torch.inference_mode():
        for vids in dl:
            # vids: (B,T,3,H,W) -> flatten frames
            B, T, C, H, W = vids.shape
            x = vids.to(device).reshape(B * T, C, H, W)
            pe = embed(x)
            N, D = pe.shape[1], pe.shape[2]
            pe = nn.functional.normalize(pe.reshape(B * T * N, D), dim=-1)
            patches.append(pe.cpu())
            total += pe.size(0)
            if total >= args.samples:
                break
    if not patches:
        raise SystemExit("No frames found or zero patches extracted.")
    embs = torch.cat(patches, dim=0)[: args.samples]
    embs = embs.to(device)
    centers = kmeans_fit(embs, k=int(args.codebook_size), iters=50).cpu()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {'codebook': centers, 'emb_dim': int(args.emb_dim), 'patch': int(args.patch)}
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar, maybe_save_best  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
        maybe_save_best = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, payload, meta={'train_args': {'samples': int(args.samples)}})
    else:
        _safe_save(payload, args.out)
        final = args.out
    try:
        if callable(maybe_save_best):
            ce = centers.to(device)
            sims = embs @ ce.t()
            assign = sims.argmax(dim=1)
            d = 1.0 - sims.gather(1, assign.unsqueeze(1)).squeeze(1)
            inertia = float(d.mean().item())
            maybe_save_best(args.out, nn.Identity(), 'vq_inertia', inertia, higher_is_better=False)
    except Exception:
        pass
    print(f"Saved video VQ codebook to {args.out} with shape {tuple(centers.shape)}")


if __name__ == '__main__':
    main()



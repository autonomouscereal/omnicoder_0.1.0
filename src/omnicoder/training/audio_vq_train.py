from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf  # type: ignore
import torch

from omnicoder.modeling.multimodal.audio_vqvae import AudioVQVAE


def _list_wavs(root: str) -> List[str]:
    paths: List[str] = []
    p = Path(root)
    if p.is_file():
        return [str(p)]
    for ext in (".wav", ".flac", ".ogg"):
        for x in p.rglob(f"*{ext}"):
            paths.append(str(x))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an audio VQ‑VAE codebook and export learned embeddings")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--segment", type=int, default=32768, help="Random segment length in samples")
    ap.add_argument("--codebook_size", type=int, default=2048)
    ap.add_argument("--code_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="weights/audio_vq_codebook.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioVQVAE(codebook_size=int(args.codebook_size), code_dim=int(args.code_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    files = _list_wavs(args.data)
    if not files:
        raise SystemExit("No audio files found")

    step = 0
    model.train()
    rng = np.random.default_rng(1234)
    while step < int(args.steps):
        batch = []
        for _ in range(int(args.batch)):
            fn = files[rng.integers(0, len(files))]
            wav, sr = sf.read(fn)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            # Normalize to [-1,1]
            wav = wav.astype(np.float32)
            if np.max(np.abs(wav)) > 1e-6:
                wav = wav / np.max(np.abs(wav))
            if wav.size < args.segment:
                pad = np.zeros((args.segment - wav.size,), dtype=np.float32)
                wav = np.concatenate([wav, pad], axis=0)
            start = int(rng.integers(0, max(1, wav.size - args.segment)))
            seg = wav[start : start + args.segment]
            batch.append(seg)
        x = torch.from_numpy(np.stack(batch, axis=0)).to(device).unsqueeze(1)  # (B,1,T)
        rec, com, ppx, xr, idx = model(x)
        loss = rec + 0.25 * com
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            print(f"step {step}/{args.steps} rec={rec.item():.4f} com={com.item():.4f} ppx={float(ppx):.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.export_codebook(args.out)  # type: ignore[attr-defined]
    print(f"Saved audio VQ‑VAE codebook to {args.out} (K={args.codebook_size}, D={args.code_dim})")


if __name__ == "__main__":
    main()



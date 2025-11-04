from __future__ import annotations

"""
One-button tiny VQ-VAE quickstart: generate assets and train tiny image/audio/video codebooks.

Usage:
  python -m omnicoder.tools.tiny_vq_quickstart --steps_img 50 --steps_audio 200 --samples_video 2048
"""

import argparse
import os
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print(" ", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(f"[run] failed: {e}")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiny VQ-VAE assets + trainers (CPU-friendly)")
    ap.add_argument("--base", type=str, default="examples/data/vq", help="Assets base directory")
    ap.add_argument("--out", type=str, default="weights", help="Weights output directory")
    ap.add_argument("--steps_img", type=int, default=50)
    ap.add_argument("--steps_audio", type=int, default=200)
    ap.add_argument("--seg_audio", type=int, default=8000)
    ap.add_argument("--samples_video", type=int, default=2048)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--codebook_size", type=int, default=128)
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate tiny assets
    print("[1] Generating tiny assets ...")
    rc = _run([
        "python", "-m", "omnicoder.tools.make_tiny_vq_assets"
    ])
    if rc != 0:
        print("[warn] make_tiny_vq_assets returned non-zero; continuing")

    # Prepare toy video list if mp4 exists
    vid_mp4 = base / "video" / "toy.mp4"
    vid_list = base / "video" / "toylist.txt"
    if vid_mp4.exists():
        vid_list.parent.mkdir(parents=True, exist_ok=True)
        vid_list.write_text(str(vid_mp4) + "\n", encoding="utf-8")

    # 2) Train tiny image VQ-VAE
    print("[2] Training tiny image VQ-VAE codebook ...")
    img_root = base / "images"
    img_out = out_dir / "image_vq_tiny.pt"
    _run([
        "python", "-m", "omnicoder.training.vq_train",
        "--data", str(img_root),
        "--steps", str(int(args.steps_img)),
        "--batch", "4",
        "--emb_dim", str(int(args.emb_dim)),
        "--codebook_size", str(int(args.codebook_size)),
        "--out", str(img_out),
    ])

    # 3) Train tiny audio VQ-VAE
    print("[3] Training tiny audio VQ-VAE codebook ...")
    aud_root = base / "audio"
    aud_out = out_dir / "audio_vq_tiny.pt"
    _run([
        "python", "-m", "omnicoder.training.audio_vq_train",
        "--data", str(aud_root),
        "--steps", str(int(args.steps_audio)),
        "--batch", "2",
        "--segment", str(int(args.seg_audio)),
        "--codebook_size", str(int(args.codebook_size)),
        "--code_dim", str(int(args.emb_dim)),
        "--out", str(aud_out),
    ])

    # 4) Train tiny video VQ codebook (k-means over frame patches)
    print("[4] Training tiny video VQ codebook ...")
    vid_out = out_dir / "video_vq_tiny.pt"
    if vid_list.exists():
        _run([
            "python", "-m", "omnicoder.training.video_vq_train",
            "--videos", str(vid_list),
            "--resize", "64",
            "--patch", "16",
            "--emb_dim", str(int(args.emb_dim)),
            "--codebook_size", str(int(args.codebook_size)),
            "--frames_per_video", "8",
            "--samples", str(int(args.samples_video)),
            "--out", str(vid_out),
        ])
    else:
        print(f"[warn] {vid_list} not found; skipping tiny video VQ.")

    print("[DONE] Tiny VQ-VAE quickstart completed.")


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Generate tiny assets for VQ-VAE smoke training:
 - Images: small colored and gradient PNGs
 - Audio: short sine wave WAV
 - Video: short synthetic MP4 if OpenCV is available (optional)

Outputs under examples/data/vq/{images,audio,video}
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _make_images(root: Path, size: Tuple[int, int] = (64, 64)) -> None:
    from PIL import Image  # type: ignore
    _ensure_dir(root)
    w, h = int(size[0]), int(size[1])
    # Solid colors
    for name, color in [("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))]:
        img = Image.new("RGB", (w, h), color)
        img.save(root / f"{name}.png")
    # Gradient
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(grad, (h, 1))
    grad_rgb = np.stack([arr, np.flipud(arr), arr.T[:h, :]], axis=-1)
    Image.fromarray(grad_rgb.astype(np.uint8), mode="RGB").save(root / "gradient.png")


def _make_audio(root: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    _ensure_dir(root)
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    # 440 Hz sine
    x = 0.2 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    try:
        import soundfile as sf  # type: ignore
        sf.write(str(root / "sine.wav"), x, sr)
    except Exception:
        # Fallback to wave module
        import wave, struct
        with wave.open(str(root / "sine.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            for v in x:
                w.writeframes(struct.pack('<h', int(np.clip(v, -1.0, 1.0) * 32767)))


def _make_video(root: Path, frames: int = 16, size: Tuple[int, int] = (64, 64)) -> None:
    _ensure_dir(root)
    try:
        import cv2  # type: ignore
    except Exception:
        # Create a folder of frame PNGs if cv2 absent
        from PIL import Image  # type: ignore
        for i in range(frames):
            arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            arr[:, :, 0] = (i * 15) % 255
            arr[:, :, 1] = (i * 7) % 255
            arr[:, :, 2] = (i * 3) % 255
            Image.fromarray(arr).save(root / f"frame_{i:03d}.png")
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path = str(root / "toy.mp4")
    vw = cv2.VideoWriter(path, fourcc, 8.0, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 15) % 255
        frame[:, :, 1] = (i * 7) % 255
        frame[:, :, 2] = (i * 3) % 255
        vw.write(frame)
    vw.release()


def main() -> None:
    base = Path("examples/data/vq")
    img_dir = base / "images"
    aud_dir = base / "audio"
    vid_dir = base / "video"
    _make_images(img_dir)
    _make_audio(aud_dir)
    _make_video(vid_dir)
    print(f"Wrote tiny assets under {base}")


if __name__ == "__main__":
    main()



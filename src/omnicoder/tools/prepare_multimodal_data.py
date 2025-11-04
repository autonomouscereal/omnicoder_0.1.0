from __future__ import annotations

"""
prepare_multimodal_data: Best-effort tiny dataset bootstrapper to enable fully
multimodal end-to-end training on a fresh workspace. It creates or copies small
sample data for:

- text: data/text/*.txt
- code: examples/code_eval/examples.jsonl (already present)
- vision-language (VL): examples/vl_auto.jsonl (already present)
- ASR: data/asr/wavs/*.wav (from examples) + transcripts.jsonl
- TTS: data/tts/texts.txt
- FVD canary: weights/sidecars/fvd_pred/*.mp4 and weights/sidecars/fvd_ref/*.mp4

It is idempotent and skips work if targets already exist.
"""

import os
from pathlib import Path
import json
from typing import Tuple


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        _ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")


def _copy_if_missing(src: Path, dst: Path) -> None:
    try:
        if src.exists() and not dst.exists():
            _ensure_dir(dst.parent)
            dst.write_bytes(src.read_bytes())
    except Exception:
        pass


def _make_dummy_video(path: Path, frames: int = 8, size: Tuple[int, int] = (160, 120)) -> None:
    """Create a tiny MP4 by writing colored frames via OpenCV. Falls back to image sequence if opencv missing."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        _ensure_dir(path.parent)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, 8.0, size)
        for i in range(max(1, frames)):
            # simple gradient frame
            x = np.linspace(0, 255, size[0], dtype=np.uint8)
            y = np.linspace(0, 255, size[1], dtype=np.uint8)
            xx, yy = np.meshgrid(x, y)
            frame = np.stack([xx, yy, ((i * 32) % 255) * np.ones_like(xx)], axis=-1)
            vw.write(frame)
        vw.release()
    except Exception:
        # Fallback: write a sequence of PNGs under a folder with the same stem
        try:
            import numpy as np  # type: ignore
            import imageio.v2 as iio  # type: ignore
            seq_dir = path.with_suffix("")
            _ensure_dir(seq_dir)
            for i in range(max(1, frames)):
                x = np.linspace(0, 255, size[0], dtype=np.uint8)
                y = np.linspace(0, 255, size[1], dtype=np.uint8)
                xx, yy = np.meshgrid(x, y)
                img = np.stack([xx, yy, ((i * 32) % 255) * np.ones_like(xx)], axis=-1)
                iio.imwrite(str(seq_dir / f"frame_{i:03d}.png"), img)
        except Exception:
            pass


def prepare_all(out_root: str = "weights") -> dict:
    root = Path(out_root)
    _ensure_dir(root)

    # 1) Text corpus (tiny)
    text_dir = Path("data/text")
    _ensure_dir(text_dir)
    if not any(text_dir.rglob("*.txt")):
        # Use bundled multimodal prompt as seed
        mm_path = Path("examples/prompts/multimodal.txt")
        content = mm_path.read_text(encoding="utf-8") if mm_path.exists() else (
            "OmniCoder: a small synthetic corpus for bootstrap.\n" * 128
        )
        _write_text_if_missing(text_dir / "bootstrap.txt", content)

    # 2) Code eval data: already under examples/code_eval
    code_jsonl = Path("examples/code_eval/examples.jsonl")
    _ensure_dir(code_jsonl.parent)
    if not code_jsonl.exists():
        _write_text_if_missing(code_jsonl, json.dumps({"prompt": "def add(a,b):", "target": "return a+b"}) + "\n")

    # 3) VL JSONL (image-text pairs)
    vl_jsonl = Path("examples/vl_auto.jsonl")
    if not vl_jsonl.exists():
        # Create a minimal file with a couple of image-text pairs from examples images
        imgs_dir = Path("examples/data/vq/images")
        entries = []
        for name in ["red.png", "green.png", "blue.png", "gradient.png"]:
            p = imgs_dir / name
            if p.exists():
                entries.append({"image": str(p.as_posix()), "text": name.replace(".png", " square")})
        _ensure_dir(vl_jsonl.parent)
        vl_jsonl.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    # 4) ASR: copy sine.wav and create transcripts
    asr_dir = Path("data/asr/wavs")
    _ensure_dir(asr_dir)
    src_wav = Path("examples/data/vq/audio/sine.wav")
    if src_wav.exists():
        _copy_if_missing(src_wav, asr_dir / "sine.wav")
    trans = Path("data/asr/transcripts.jsonl")
    if not trans.exists():
        _write_text_if_missing(trans, json.dumps({"path": str((asr_dir / "sine.wav").as_posix()), "text": "sine wave"}) + "\n")

    # 5) TTS: simple sentences list
    tts_dir = Path("data/tts")
    _ensure_dir(tts_dir)
    _write_text_if_missing(tts_dir / "texts.txt", "Hello world.\nOmniCoder says hi.\n")

    # 6) FVD canary: create tiny videos for pred/ref
    sidecars = root / "sidecars"
    pred_dir = sidecars / "fvd_pred"
    ref_dir = sidecars / "fvd_ref"
    _ensure_dir(pred_dir); _ensure_dir(ref_dir)
    pred_mp4 = pred_dir / "dummy_pred.mp4"
    ref_mp4 = ref_dir / "dummy_ref.mp4"
    if not pred_mp4.exists():
        _make_dummy_video(pred_mp4)
    if not ref_mp4.exists():
        _make_dummy_video(ref_mp4)

    # Normalize all paths to POSIX style so consumers do not embed container-specific prefixes
    def _pp(p: Path) -> str:
        try:
            return str(p.as_posix())
        except Exception:
            return str(p)
    return {
        "text_dir": _pp(text_dir),
        "code_jsonl": _pp(code_jsonl),
        "vl_jsonl": _pp(vl_jsonl),
        "asr_dir": _pp(asr_dir.parent),
        "tts_dir": _pp(tts_dir),
        "fvd_pred_dir": _pp(pred_dir),
        "fvd_ref_dir": _pp(ref_dir),
        "media_dir": _pp(root / "media"),
    }


def prepare_media_gallery(root: str = "weights") -> str:
    """Create a small media folder with a few images and wavs for cycle-consistency."""
    media = Path(root) / "media"
    _ensure_dir(media)
    # Images
    for name in ["red.png", "green.png", "blue.png", "gradient.png"]:
        src = Path("examples/data/vq/images") / name
        if src.exists():
            _copy_if_missing(src, media / name)
    # Audio
    wav_src = Path("examples/data/vq/audio/sine.wav")
    if wav_src.exists():
        _copy_if_missing(wav_src, media / "sine.wav")
    return str(media)


def main() -> None:
    out_root = os.getenv("OMNICODER_OUT_ROOT", "weights")
    info = prepare_all(out_root)
    media_dir = prepare_media_gallery(out_root)
    info["media_dir"] = media_dir
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()



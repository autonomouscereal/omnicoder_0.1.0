from __future__ import annotations

"""
SFB setup utility: prepares minimal sidecar datasets for multimodal/metrics gating and
exports/defaults env variables so the SFB parallel stack works out-of-the-box.

Creates (if missing):
- examples/sidecars/clip_pairs.jsonl      (points to sample images)
- examples/sidecars/asr_pairs.jsonl       (ASR WER JSONL referencing example audio)
- examples/sidecars/code_tasks.jsonl      (PAL-style simple code eval)
- examples/sidecars/fad_pred/, fad_ref/   (audio copies for FAD)
- examples/sidecars/fvd_pred/, fvd_ref/   (tiny mp4 videos for FVD; best-effort if cv2 available)

Sets env vars when not present:
- SFB_ENABLE=1
- SFB_FACTORIZER=amr,srl
- SFB_BLOCK_VERIFY=1, SFB_BLOCK_VERIFY_SIZE=4
- SFB_CLIP_JSONL, SFB_ASR_JSONL, SFB_CODE_TASKS_JSONL
- SFB_FAD_PRED_DIR, SFB_FAD_REF_DIR
- SFB_FVD_PRED_DIR, SFB_FVD_REF_DIR
"""

import os
from pathlib import Path
from typing import Tuple


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_if_absent(path: Path, content: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')


def _prepare_clip_jsonl(root: Path) -> Path:
    j = root / 'clip_pairs.jsonl'
    content = (
        '{"file": "examples/data/vq/images/blue.png", "prompt": "a blue square"}\n'
        '{"file": "examples/data/vq/images/green.png", "prompt": "a green square"}\n'
        '{"file": "examples/data/vq/images/gradient.png", "prompt": "a gradient image"}\n'
    )
    _write_if_absent(j, content)
    return j


def _prepare_asr_jsonl(root: Path) -> Path:
    j = root / 'asr_pairs.jsonl'
    content = (
        '{"file": "examples/data/vq/audio/sine.wav", "ref": "sine wave"}\n'
        '{"file": "examples/data/vq/audio/sine.wav", "ref": "a sine tone"}\n'
    )
    _write_if_absent(j, content)
    return j


def _prepare_code_jsonl(root: Path) -> Path:
    j = root / 'code_tasks.jsonl'
    content = (
        '{"candidates": ["def add(a,b):\\n    return a+b\\n"], "tests": "assert add(2,2)==4"}\n'
        '{"candidates": ["def mul(a,b):\\n    return a*b\\n"], "tests": "assert mul(3,4)==12"}\n'
    )
    _write_if_absent(j, content)
    return j


def _prepare_fad_dirs(root: Path) -> Tuple[Path, Path]:
    pred = root / 'fad_pred'
    ref = root / 'fad_ref'
    _ensure_dir(pred); _ensure_dir(ref)
    src = Path('examples/data/vq/audio/sine.wav')
    try:
        if src.exists():
            # Copy (or link) to both dirs
            import shutil
            shutil.copy2(src, pred / 'sine_pred.wav')
            shutil.copy2(src, ref / 'sine_ref.wav')
    except Exception:
        pass
    return pred, ref


def _make_mp4(path: Path, color: Tuple[int, int, int]) -> bool:
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
        w, h = 224, 224
        fps = 8
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for _ in range(16):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = color
            vw.write(frame)
        vw.release()
        return True
    except Exception:
        return False


def _prepare_fvd_dirs(root: Path) -> Tuple[Path, Path]:
    pred = root / 'fvd_pred'
    ref = root / 'fvd_ref'
    _ensure_dir(pred); _ensure_dir(ref)
    ok1 = _make_mp4(pred / 'pred.mp4', (0, 0, 255))
    ok2 = _make_mp4(ref / 'ref.mp4', (0, 255, 0))
    if not (ok1 and ok2):
        # Best-effort: leave empty; caller will degrade to zeros
        pass
    return pred, ref


def setup_sidecars_and_env() -> None:
    root = Path('examples/sidecars')
    _ensure_dir(root)
    clip_j = _prepare_clip_jsonl(root)
    asr_j = _prepare_asr_jsonl(root)
    code_j = _prepare_code_jsonl(root)
    fad_pred, fad_ref = _prepare_fad_dirs(root)
    fvd_pred, fvd_ref = _prepare_fvd_dirs(root)
    # Set defaults only if missing to respect user overrides
    os.environ.setdefault('SFB_ENABLE', '1')
    os.environ.setdefault('SFB_FACTORIZER', 'amr,srl')
    os.environ.setdefault('SFB_BLOCK_VERIFY', '1')
    os.environ.setdefault('SFB_BLOCK_VERIFY_SIZE', '4')
    os.environ.setdefault('SFB_CLIP_JSONL', str(clip_j))
    os.environ.setdefault('SFB_ASR_JSONL', str(asr_j))
    os.environ.setdefault('SFB_CODE_TASKS_JSONL', str(code_j))
    os.environ.setdefault('SFB_FAD_PRED_DIR', str(fad_pred))
    os.environ.setdefault('SFB_FAD_REF_DIR', str(fad_ref))
    os.environ.setdefault('SFB_FVD_PRED_DIR', str(fvd_pred))
    os.environ.setdefault('SFB_FVD_REF_DIR', str(fvd_ref))


def main() -> None:
    setup_sidecars_and_env()
    print("[sfb_setup] Sidecars ready and env defaults set.")


if __name__ == '__main__':
    main()



from __future__ import annotations

"""
Unified data engine for scalable ingestion, filtering, and light synthesis across modalities.

Supports inputs:
- Text: a folder or .txt file
- Vision-Language (VL): JSONL with fields {image, text}
- Video: directory with video files; emits JSONL with {video, prompt?}
- Audio: directory with wav files; emits JSONL with {audio, prompt?}

Features:
- Length/size filters (token length for text; min image size; max frames; audio duration)
- Deduplication by stable content hashing
- Light synthesis options (e.g., negative VL pairs via random mismatch)
- Shuffling and output manifests

Usage:
    python -m omnicoder.training.data.engine --text_dir data/text --vl_jsonl data/vl.jsonl \
        --video_dir data/videos --audio_dir data/audio --out_jsonl weights/manifest.jsonl \
        --min_tokens 8 --max_tokens 2048 --dedup --shuffle --vl_negatives 0.1
"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _stable_hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _stable_hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _tokenize_len(text: str) -> int:
    # Lightweight proxy for token count; counts whitespace-separated tokens
    return max(0, len(text.strip().split()))


@dataclass
class EngineConfig:
    min_tokens: int = 0
    max_tokens: int = 4096
    min_image_px: int = 64
    max_video_frames: int = 512
    min_audio_sec: float = 0.25
    max_audio_sec: float = 60.0
    dedup: bool = True
    shuffle: bool = True
    vl_negatives: float = 0.0  # probability to synthesize a negative VL pair
    max_samples_per_source: int = 0  # 0 => unlimited


class DataEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self._seen: set[str] = set()

    def _maybe_keep(self, rec: dict) -> bool:
        # Filters based on modality
        if 'text' in rec and isinstance(rec.get('text'), str):
            n = _tokenize_len(rec['text'])
            if n < int(self.cfg.min_tokens) or n > int(self.cfg.max_tokens):
                return False
        if 'image' in rec:
            try:
                from PIL import Image  # type: ignore
                p = Path(rec['image'])
                if not p.exists():
                    return False
                with Image.open(p) as im:
                    w, h = im.size
                if min(w, h) < int(self.cfg.min_image_px):
                    return False
            except Exception:
                # Keep record if we cannot verify size
                pass
        if 'video' in rec:
            # Optional: estimate frames via OpenCV
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(str(rec['video']))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frames > 0 and frames > int(self.cfg.max_video_frames):
                    return False
            except Exception:
                pass
        if 'audio' in rec:
            try:
                import soundfile as sf  # type: ignore
                import numpy as np  # type: ignore
                data, sr = sf.read(rec['audio'])
                dur = float(len(data) / max(sr, 1))
                if dur < float(self.cfg.min_audio_sec) or dur > float(self.cfg.max_audio_sec):
                    return False
            except Exception:
                pass
        return True

    def _dedup_key(self, rec: dict) -> Optional[str]:
        # Build a stable key by hashing primary resource(s)
        try:
            if 'image' in rec:
                return f"img:{_stable_hash_file(Path(rec['image']))}"
            if 'video' in rec:
                return f"vid:{_stable_hash_file(Path(rec['video']))}"
            if 'audio' in rec:
                return f"aud:{_stable_hash_file(Path(rec['audio']))}"
            # For text-only, hash normalized text
            if 'text' in rec:
                return f"txt:{_stable_hash_bytes(rec['text'].strip().lower().encode('utf-8'))}"
        except Exception:
            return None
        return None

    def ingest_text_dir(self, folder: str, tag: str = 'text') -> List[dict]:
        out: List[dict] = []
        root = Path(folder)
        paths: List[Path] = []
        if root.is_file() and root.suffix.lower() == '.txt':
            paths = [root]
        elif root.is_dir():
            paths = sorted(root.rglob('*.txt'))
        limit = int(self.cfg.max_samples_per_source) if int(self.cfg.max_samples_per_source) > 0 else None
        for i, p in enumerate(paths):
            if limit is not None and i >= limit:
                break
            try:
                txt = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            rec = {'modality': 'text', 'path': str(p), 'text': txt}
            if not self._maybe_keep(rec):
                continue
            if self.cfg.dedup:
                k = self._dedup_key(rec)
                if k and k in self._seen:
                    continue
                if k:
                    self._seen.add(k)
            out.append(rec)
        return out

    def ingest_vl_jsonl(self, jsonl_path: str) -> List[dict]:
        out: List[dict] = []
        try:
            lines = Path(jsonl_path).read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            return out
        limit = int(self.cfg.max_samples_per_source) if int(self.cfg.max_samples_per_source) > 0 else None
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            if limit is not None and i >= limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict) or 'image' not in obj or 'text' not in obj:
                continue
            rec = {'modality': 'vl', 'image': obj['image'], 'text': obj['text']}
            if not self._maybe_keep(rec):
                continue
            if self.cfg.dedup:
                k = self._dedup_key(rec)
                if k and k in self._seen:
                    continue
                if k:
                    self._seen.add(k)
            out.append(rec)
        # Optionally synthesize negatives by mismatching texts/images
        if float(self.cfg.vl_negatives) > 0.0 and len(out) >= 2:
            num_neg = int(len(out) * float(self.cfg.vl_negatives))
            for _ in range(num_neg):
                a, b = random.sample(out, 2)
                neg = {'modality': 'vl', 'image': a['image'], 'text': b['text'], 'negative': True}
                out.append(neg)
        return out

    def ingest_video_dir(self, folder: str) -> List[dict]:
        exts = {'.mp4', '.mov', '.avi', '.mkv'}
        out: List[dict] = []
        root = Path(folder)
        paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in exts]
        limit = int(self.cfg.max_samples_per_source) if int(self.cfg.max_samples_per_source) > 0 else None
        for i, p in enumerate(paths):
            if limit is not None and i >= limit:
                break
            rec = {'modality': 'video', 'video': str(p)}
            if not self._maybe_keep(rec):
                continue
            if self.cfg.dedup:
                k = self._dedup_key(rec)
                if k and k in self._seen:
                    continue
                if k:
                    self._seen.add(k)
            out.append(rec)
        return out

    def ingest_audio_dir(self, folder: str) -> List[dict]:
        exts = {'.wav', '.flac'}
        out: List[dict] = []
        root = Path(folder)
        paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in exts]
        limit = int(self.cfg.max_samples_per_source) if int(self.cfg.max_samples_per_source) > 0 else None
        for i, p in enumerate(paths):
            if limit is not None and i >= limit:
                break
            rec = {'modality': 'audio', 'audio': str(p)}
            if not self._maybe_keep(rec):
                continue
            if self.cfg.dedup:
                k = self._dedup_key(rec)
                if k and k in self._seen:
                    continue
                if k:
                    self._seen.add(k)
            out.append(rec)
        return out

    def write_manifest(self, records: List[dict], out_jsonl: str) -> None:
        if bool(self.cfg.shuffle):
            random.shuffle(records)
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, 'w', encoding='utf-8') as f:
            for rec in records:
                try:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                except Exception:
                    continue


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified data engine: ingest/filter/synthesize across modalities")
    ap.add_argument('--text_dir', type=str, default='')
    ap.add_argument('--vl_jsonl', type=str, default='')
    ap.add_argument('--video_dir', type=str, default='')
    ap.add_argument('--audio_dir', type=str, default='')
    ap.add_argument('--out_jsonl', type=str, required=True)
    # Filters and options
    ap.add_argument('--min_tokens', type=int, default=0)
    ap.add_argument('--max_tokens', type=int, default=4096)
    ap.add_argument('--min_image_px', type=int, default=64)
    ap.add_argument('--max_video_frames', type=int, default=512)
    ap.add_argument('--min_audio_sec', type=float, default=0.25)
    ap.add_argument('--max_audio_sec', type=float, default=60.0)
    ap.add_argument('--dedup', action='store_true')
    ap.add_argument('--no_dedup', action='store_true')
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--no_shuffle', action='store_true')
    ap.add_argument('--vl_negatives', type=float, default=0.0)
    ap.add_argument('--max_samples_per_source', type=int, default=0)
    args = ap.parse_args()

    cfg = EngineConfig(
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_image_px=int(args.min_image_px),
        max_video_frames=int(args.max_video_frames),
        min_audio_sec=float(args.min_audio_sec),
        max_audio_sec=float(args.max_audio_sec),
        dedup=(False if args.no_dedup else (True if args.dedup else True)),
        shuffle=(False if args.no_shuffle else (True if args.shuffle else True)),
        vl_negatives=float(args.vl_negatives),
        max_samples_per_source=int(args.max_samples_per_source),
    )
    eng = DataEngine(cfg)

    records: List[dict] = []
    if args.text_dir:
        records += eng.ingest_text_dir(args.text_dir)
    if args.vl_jsonl and Path(args.vl_jsonl).exists():
        records += eng.ingest_vl_jsonl(args.vl_jsonl)
    if args.video_dir and Path(args.video_dir).exists():
        records += eng.ingest_video_dir(args.video_dir)
    if args.audio_dir and Path(args.audio_dir).exists():
        records += eng.ingest_audio_dir(args.audio_dir)

    eng.write_manifest(records, args.out_jsonl)
    print(json.dumps({"status": "ok", "num_records": len(records), "out": args.out_jsonl}))


if __name__ == '__main__':
    main()



from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Any

from omnicoder.config import MultiModalConfig


@dataclass
class VocabSidecar:
    text_range: List[int]
    image_range: List[int]
    video_range: List[int]
    audio_range: List[int]

    @staticmethod
    def from_config(cfg: MultiModalConfig) -> "VocabSidecar":
        text_start, text_end = 0, 32000 - 1
        img_start = int(cfg.image_vocab_start)
        img_end = img_start + int(cfg.image_codebook_size) - 1
        vid_start = int(cfg.video_vocab_start)
        vid_end = vid_start + int(cfg.video_codebook_size) - 1
        aud_start = int(cfg.audio_vocab_start)
        aud_end = aud_start + int(cfg.audio_codebook_size) - 1
        return VocabSidecar(
            text_range=[text_start, text_end],
            image_range=[img_start, img_end],
            video_range=[vid_start, vid_end],
            audio_range=[aud_start, aud_end],
        )

    def integrity_checks(self) -> None:
        # Ensure non-overlapping monotonically increasing ranges
        ranges = [
            ("text", tuple(self.text_range)),
            ("image", tuple(self.image_range)),
            ("video", tuple(self.video_range)),
            ("audio", tuple(self.audio_range)),
        ]
        # start<=end and no overlaps
        def _ok(r: tuple[int, int]) -> bool:
            return int(r[0]) <= int(r[1])

        for name, r in ranges:
            if not _ok(r):
                raise ValueError(f"Invalid range for {name}: {r}")
        # sort by start and ensure no overlaps between adjacent ranges
        sorted_ranges = sorted(ranges, key=lambda kv: kv[1][0])
        for (name0, a), (name1, b) in zip(sorted_ranges, sorted_ranges[1:]):
            a0, a1 = tuple(a)
            b0, _b1 = tuple(b)
            if int(a1) >= int(b0):
                raise ValueError(f"Overlapping ranges: {(name0, a)} vs {(name1, b)}")

    def to_json(self) -> str:
        return json.dumps({
            "text": self.text_range,
            "image": self.image_range,
            "video": self.video_range,
            "audio": self.audio_range,
        }, indent=2)

    def save(self, path: str | Path) -> None:
        self.integrity_checks()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "VocabSidecar":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        # Support both compact ranges and explicit starts/sizes
        if all(k in data for k in ("text", "image", "video", "audio")):
            side = VocabSidecar(
                text_range=[int(data["text"][0]), int(data["text"][1])],
                image_range=[int(data["image"][0]), int(data["image"][1])],
                video_range=[int(data["video"][0]), int(data["video"][1])],
                audio_range=[int(data["audio"][0]), int(data["audio"][1])],
            )
        else:
            # Fallback to layout-style json (text_size, *_start, *_size)
            text_size = int(data.get("text_size", 32000))
            image_start = int(data.get("image_start", text_size))
            image_size = int(data.get("image_size", 8192))
            video_start = int(data.get("video_start", image_start + image_size))
            video_size = int(data.get("video_size", 8192))
            audio_start = int(data.get("audio_start", video_start + video_size))
            audio_size = int(data.get("audio_size", 2048))
            side = VocabSidecar(
                text_range=[0, text_size - 1],
                image_range=[image_start, image_start + image_size - 1],
                video_range=[video_start, video_start + video_size - 1],
                audio_range=[audio_start, audio_start + audio_size - 1],
            )
        side.integrity_checks()
        return side

    def as_layout(self) -> "VocabLayout":
        # Convert ranges to a VocabLayout with starts and sizes
        text_size = int(self.text_range[1]) - int(self.text_range[0]) + 1
        image_start = int(self.image_range[0])
        image_size = int(self.image_range[1]) - image_start + 1
        video_start = int(self.video_range[0])
        video_size = int(self.video_range[1]) - video_start + 1
        audio_start = int(self.audio_range[0])
        audio_size = int(self.audio_range[1]) - audio_start + 1
        return VocabLayout(
            text_size=text_size,
            image_start=image_start,
            image_size=image_size,
            video_start=video_start,
            video_size=video_size,
            audio_start=audio_start,
            audio_size=audio_size,
        )

@dataclass
class VocabLayout:
    text_size: int = 32000
    image_start: int = 32000
    image_size: int = 8192
    video_start: int = 32000 + 8192
    video_size: int = 8192
    audio_start: int = 32000 + 8192 + 8192
    audio_size: int = 2048

    def validate(self) -> None:
        assert self.text_size > 0
        assert self.image_start >= self.text_size
        assert self.video_start >= self.image_start + self.image_size
        assert self.audio_start >= self.video_start + self.video_size


def _starts_from(arg: Any, kind: str) -> int:
    if isinstance(arg, MultiModalConfig):
        if kind == 'image':
            return int(arg.image_vocab_start)
        if kind == 'video':
            return int(arg.video_vocab_start)
        if kind == 'audio':
            return int(arg.audio_vocab_start)
    # Default or VocabLayout
    layout = arg if isinstance(arg, VocabLayout) else VocabLayout()
    layout.validate()
    if kind == 'image':
        return int(layout.image_start)
    if kind == 'video':
        return int(layout.video_start)
    if kind == 'audio':
        return int(layout.audio_start)
    return 0


def map_image_tokens(image_codes: List[int], layout_or_cfg: Any = None) -> List[int]:
    start = _starts_from(layout_or_cfg, 'image')
    return [start + int(c) for c in image_codes]


def map_video_tokens(video_codes: List[int], layout_or_cfg: Any = None) -> List[int]:
    start = _starts_from(layout_or_cfg, 'video')
    return [start + int(c) for c in video_codes]


def map_audio_tokens(audio_codes: List[int], layout_or_cfg: Any = None) -> List[int]:
    start = _starts_from(layout_or_cfg, 'audio')
    return [start + int(c) for c in audio_codes]

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class ModalityId(IntEnum):
    text = 0
    control = 1
    vision_semantic = 2
    vision_residual = 3
    speech_tts = 4
    audio_music = 5
    music_control = 6
    time_space = 7
    tool_agent = 8
    flow = 9


@dataclass(frozen=True)
class TokenRange:
    name: str
    begin: int
    end: int
    modality: ModalityId
    generation: str
    description: str

    def contains(self, token_id: int) -> bool:
        return self.begin <= int(token_id) < self.end

    @property
    def size(self) -> int:
        return self.end - self.begin


class OmniLedger2026:
    """Single typed token ledger for the Omnicoder 2026 one-trunk model.

    Edge codecs may be modality-specific, but the trunk receives only these
    integer IDs. There are no learned modality adapters inside the trunk.
    """

    def __init__(self, ranges: Iterable[TokenRange] | None = None):
        self.ranges = tuple(ranges or default_ranges())
        self._validate()

    @property
    def vocab_size(self) -> int:
        return max(r.end for r in self.ranges)

    def lookup(self, token_id: int) -> TokenRange:
        tid = int(token_id)
        for tr in self.ranges:
            if tr.contains(tid):
                return tr
        raise ValueError(f"token id {tid} is outside Omnicoder2026 ledger")

    def modality_id(self, token_id: int) -> int:
        return int(self.lookup(token_id).modality)

    def as_config_ranges(self) -> dict[str, tuple[int, int]]:
        return {r.name: (int(r.begin), int(r.end)) for r in self.ranges}

    def as_metadata(self) -> dict:
        return {
            "schema": "omnicoder2026_token_ledger_v1",
            "vocab_size": int(self.vocab_size),
            "ranges": [
                {
                    "name": r.name,
                    "begin": int(r.begin),
                    "end": int(r.end),
                    "modality_id": int(r.modality),
                    "generation": r.generation,
                    "description": r.description,
                }
                for r in self.ranges
            ],
            "trunk_rule": "all modalities enter as token ids through shared embeddings and shared output heads",
            "edge_rule": "raw media is encoded/decoded outside the trunk by codecs, not in-trunk adapters",
        }

    def _validate(self) -> None:
        ordered = sorted(self.ranges, key=lambda r: r.begin)
        for prev, cur in zip(ordered, ordered[1:]):
            if prev.end > cur.begin:
                raise ValueError(f"overlapping token ranges: {prev.name} and {cur.name}")
        for r in ordered:
            if r.begin < 0 or r.end <= r.begin:
                raise ValueError(f"invalid token range: {r}")


def default_ranges() -> tuple[TokenRange, ...]:
    return (
        TokenRange("text", 0, 128_000, ModalityId.text, "autoregressive", "Qwen/ChatML-style text and code tokens"),
        TokenRange("control", 128_000, 132_096, ModalityId.control, "autoregressive", "system, task, style, routing, safety, and document boundary tokens"),
        TokenRange("vision_semantic", 132_096, 197_632, ModalityId.vision_semantic, "autoregressive_or_flow", "low-rate visual semantic tokens"),
        TokenRange("vision_residual", 197_632, 214_016, ModalityId.vision_residual, "masked_discrete_flow", "visual reconstruction/detail residual tokens"),
        TokenRange("speech_tts", 214_016, 279_552, ModalityId.speech_tts, "autoregressive_or_flow", "speech semantic, prosody, speaker, and TTS codec tokens"),
        TokenRange("audio_music", 279_552, 312_320, ModalityId.audio_music, "autoregressive_or_flow", "general audio, sound effect, and music codec tokens"),
        TokenRange("music_control", 312_320, 320_512, ModalityId.music_control, "autoregressive", "beat, key, tempo, stem, instrument, and arrangement tokens"),
        TokenRange("time_space", 320_512, 324_608, ModalityId.time_space, "autoregressive", "time, frame, pixel-grid, spectrogram, and sequence-alignment tokens"),
        TokenRange("tool_agent", 324_608, 328_704, ModalityId.tool_agent, "autoregressive", "tool call, memory, verifier, terminal, and agent action tokens"),
        TokenRange("flow", 328_704, 330_000, ModalityId.flow, "masked_discrete_flow", "mask, denoise step, edit span, and refinement-control tokens"),
    )


DEFAULT_LEDGER = OmniLedger2026()

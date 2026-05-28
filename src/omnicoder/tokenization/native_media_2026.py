from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

NativeMediaKind = Literal["image", "video", "audio", "music", "tts", "ocr"]

MEDIA_TYPE_IDS: dict[str, int] = {
    "image": 1,
    "video": 2,
    "audio": 3,
    "music": 4,
    "tts": 5,
    "ocr": 6,
}


@dataclass(frozen=True)
class NativeMediaPacket:
    """Fixed-preprocess continuous media packet for the shared trunk bridge."""

    kind: NativeMediaKind
    features: torch.Tensor
    type_ids: torch.Tensor
    positions: torch.Tensor
    feature_dim: int


def _fit_dim(features: torch.Tensor, feature_dim: int) -> torch.Tensor:
    if features.shape[-1] == int(feature_dim):
        return features
    if features.shape[-1] > int(feature_dim):
        return features[..., : int(feature_dim)]
    return F.pad(features, (0, int(feature_dim) - features.shape[-1]))


def _packet(kind: NativeMediaKind, features: torch.Tensor, positions: torch.Tensor, feature_dim: int) -> NativeMediaPacket:
    features = _fit_dim(features.float(), int(feature_dim))
    type_id = MEDIA_TYPE_IDS[kind]
    type_ids = torch.full(features.shape[:2], int(type_id), dtype=torch.long, device=features.device)
    return NativeMediaPacket(kind=kind, features=features, type_ids=type_ids, positions=positions.float(), feature_dim=int(feature_dim))


def image_to_native_patches(image: torch.Tensor, *, patch: int = 32, feature_dim: int = 3072) -> NativeMediaPacket:
    """Patchify RGB-like images directly into continuous trunk features.

    Input shape is ``[B,C,H,W]``. The returned feature channel defaults to
    ``3*32*32``, mirroring the SenseNova-style direct RGB patch target.
    """

    if image.dim() != 4:
        raise ValueError(f"image tensor must be [B,C,H,W], got {tuple(image.shape)}")
    b, c, h, w = image.shape
    pad_h = (patch - (h % patch)) % patch
    pad_w = (patch - (w % patch)) % patch
    if pad_h or pad_w:
        image = F.pad(image.float(), (0, pad_w, 0, pad_h))
    else:
        image = image.float()
    patches = F.unfold(image, kernel_size=(patch, patch), stride=(patch, patch)).transpose(1, 2)
    grid_h = image.shape[-2] // patch
    grid_w = image.shape[-1] // patch
    ys, xs = torch.meshgrid(torch.arange(grid_h, device=image.device), torch.arange(grid_w, device=image.device), indexing="ij")
    positions = torch.stack(
        (
            torch.zeros_like(xs, dtype=torch.float32),
            ys.float() / max(1, grid_h - 1),
            xs.float() / max(1, grid_w - 1),
            torch.zeros_like(xs, dtype=torch.float32),
        ),
        dim=-1,
    ).view(1, -1, 4).expand(b, -1, -1)
    return _packet("image", patches, positions, feature_dim)


def video_to_native_patches(video: torch.Tensor, *, patch: int = 32, feature_dim: int = 3072) -> NativeMediaPacket:
    """Patchify video frames into one shared continuous token stream.

    Input shape is ``[B,T,C,H,W]``.
    """

    if video.dim() != 5:
        raise ValueError(f"video tensor must be [B,T,C,H,W], got {tuple(video.shape)}")
    b, t, c, h, w = video.shape
    frames = video.reshape(b * t, c, h, w)
    image_packet = image_to_native_patches(frames, patch=patch, feature_dim=feature_dim)
    patches_per_frame = image_packet.features.shape[1]
    features = image_packet.features.reshape(b, t * patches_per_frame, -1)
    frame_pos = torch.arange(t, device=video.device, dtype=torch.float32).view(1, t, 1, 1) / max(1, t - 1)
    spatial = image_packet.positions.reshape(b, t, patches_per_frame, 4)
    positions = spatial.clone()
    positions[..., 0:1] = frame_pos
    return _packet("video", features, positions.view(b, t * patches_per_frame, 4), feature_dim)


def waveform_to_native_segments(
    waveform: torch.Tensor,
    *,
    kind: NativeMediaKind = "audio",
    segment: int = 3072,
    stride: int | None = None,
    feature_dim: int = 3072,
) -> NativeMediaPacket:
    """Segment raw waveform/music/TTS tensors without an audio codec.

    Input shape is ``[B,S]`` or ``[B,C,S]``. Channels are flattened into each
    fixed segment, then padded/truncated to the shared feature dimension.
    """

    if kind not in {"audio", "music", "tts"}:
        raise ValueError(f"waveform segments support audio/music/tts, got {kind!r}")
    if waveform.dim() == 2:
        waveform = waveform[:, None, :]
    if waveform.dim() != 3:
        raise ValueError(f"waveform tensor must be [B,S] or [B,C,S], got {tuple(waveform.shape)}")
    b, c, s = waveform.shape
    step = int(stride or segment)
    pad = (segment - (s % step)) % step
    if pad:
        waveform = F.pad(waveform.float(), (0, pad))
    else:
        waveform = waveform.float()
    chunks = waveform.unfold(dimension=-1, size=int(segment), step=step).transpose(1, 2).reshape(b, -1, c * int(segment))
    count = chunks.shape[1]
    time = torch.arange(count, device=waveform.device, dtype=torch.float32).view(1, count, 1) / max(1, count - 1)
    positions = torch.cat((time, torch.zeros((b, count, 3), device=waveform.device)), dim=-1)
    return _packet(kind, chunks, positions, feature_dim)


def ocr_image_to_native_patches(image: torch.Tensor, *, patch: int = 32, feature_dim: int = 3072) -> NativeMediaPacket:
    packet = image_to_native_patches(image, patch=patch, feature_dim=feature_dim)
    return _packet("ocr", packet.features, packet.positions, feature_dim)

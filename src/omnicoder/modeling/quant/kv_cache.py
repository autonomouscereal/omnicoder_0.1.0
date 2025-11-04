from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch


KvQuantScheme = Literal["none", "u8", "nf4"]


@dataclass
class KvQuantMeta:
    """Metadata for quantized KV tensors.

    For per-group quantization along the last dimension (latent dim), we store
    scale and zero/mean per group. Shapes follow broadcasting rules to enable
    efficient dequantization:

    - For u8: scale and zero of shape (B, H, T, G), where G = ceil(D / group_size)
    - For nf4: scale and mean of shape (B, H, T, G)
    """

    scheme: KvQuantScheme
    group_size: int
    # scale and offset tensors; interpretation depends on scheme
    k_scale: torch.Tensor | None = None
    k_offset: torch.Tensor | None = None  # zero for u8, mean for nf4
    v_scale: torch.Tensor | None = None
    v_offset: torch.Tensor | None = None

    def to(self, device: torch.device) -> "KvQuantMeta":
        if self.k_scale is not None:
            self.k_scale = self.k_scale.to(device)
        if self.k_offset is not None:
            self.k_offset = self.k_offset.to(device)
        if self.v_scale is not None:
            self.v_scale = self.v_scale.to(device)
        if self.v_offset is not None:
            self.v_offset = self.v_offset.to(device)
        return self


@dataclass
class PagedKvCache:
    """Blockwise paged KV cache for long-context streaming.

    Stores latent K/V in fixed-size pages per layer, enabling block reuse and
    host pinning for memory-constrained devices. This is a CPU-side structure
    used to simulate/export paged KV semantics for mobile runtimes.
    """

    page_len: int
    max_pages: int
    # list of pages: each page is (k_lat, v_lat) with shape (B,H,page_len,DL)
    pages: list[tuple[torch.Tensor, torch.Tensor]]

    def __init__(self, page_len: int = 256, max_pages: int = 128):
        self.page_len = int(page_len)
        self.max_pages = int(max_pages)
        self.pages = []

    def append_tokens(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> None:
        """Append new tokens into pages; split across pages as needed."""
        assert k_lat.shape == v_lat.shape
        B, H, T, DL = k_lat.shape
        offset = 0
        while offset < T:
            remain = T - offset
            take = min(self.page_len, remain)
            chunk_k = torch.ops.aten.slice.Tensor(k_lat, 2, int(offset), int(offset + take), 1)
            chunk_v = torch.ops.aten.slice.Tensor(v_lat, 2, int(offset), int(offset + take), 1)
            self.pages.append((chunk_k.cpu(), chunk_v.cpu()))
            if len(self.pages) > self.max_pages:
                # drop oldest
                self.pages.pop(0)
            offset += take

    def materialize_tail(self, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated tail of size up to `window` tokens in (B,H,T,DL)."""
        if not self.pages:
            raise ValueError("PagedKvCache is empty")
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        total = 0
        for k, v in reversed(self.pages):
            t = k.size(2)
            if total + t <= window:
                ks.append(k)
                vs.append(v)
                total += t
            else:
                need = window - total
                if need > 0:
                    ks.append(k[:, :, -need:, :])
                    vs.append(v[:, :, -need:, :])
                    total += need
                break
        ks.reverse(); vs.reverse()
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        k_cat = _safe_concat(ks, 2)
        v_cat = _safe_concat(vs, 2)
        return k_cat, v_cat


# NormalFloat4 (NF4) codebook approximating bitsandbytes
_NF4_CODEBOOK = torch.tensor(
    [
        -1.078125, -0.8515625, -0.671875, -0.5234375,
        -0.39453125, -0.27734375, -0.1640625, -0.0546875,
         0.0546875,  0.1640625,  0.27734375,  0.39453125,
         0.5234375,  0.671875,   0.8515625,   1.078125,
    ], dtype=torch.float32
)


def _compute_group_params(x: torch.Tensor, group_size: int, scheme: KvQuantScheme) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group quantization parameters along last dim.

    Returns (scale, offset), where for u8 offset is zero-point, for nf4 offset is mean.
    Shapes: (B, H, T, G)
    """
    B, H, T, D = x.shape
    G = (D + group_size - 1) // group_size
    pad = G * group_size - D
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad), value=0.0)
    xg = torch.ops.aten.reshape.default(x, (B, H, T, G, group_size))
    if scheme == "u8":
        xmax = xg.amax(dim=-1)
        xmin = xg.amin(dim=-1)
        scale = (xmax - xmin).clamp(min=1e-8) / 255.0
        zero = (-xmin / scale).round().clamp(0, 255)
        return scale, zero
    elif scheme == "nf4":
        mean = xg.mean(dim=-1)
        var = ((xg - torch.ops.aten.reshape.default(mean, (mean.shape[0], mean.shape[1], mean.shape[2], 1))) ** 2).mean(dim=-1).clamp(min=1e-8)
        std = var.sqrt()
        # We map x -> (x - mean) / std, then quantize to nearest codebook value
        return std, mean
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")


@torch.no_grad()
def quantize_kv(
    k_lat: torch.Tensor,
    v_lat: torch.Tensor,
    scheme: KvQuantScheme = "u8",
    group_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, KvQuantMeta]:
    """Quantize latent K/V per (B,H,T,group) along last dim.

    Returns quantized tensors (uint8 for u8, uint8 code indices for nf4) and metadata.
    """
    assert k_lat.shape == v_lat.shape and k_lat.dim() == 4, "Expect (B,H,T,D)"
    device = k_lat.device
    B, H, T, D = k_lat.shape
    G = (D + group_size - 1) // group_size
    pad = G * group_size - D
    if pad > 0:
        k_lat = torch.nn.functional.pad(k_lat, (0, pad), value=0.0)
        v_lat = torch.nn.functional.pad(v_lat, (0, pad), value=0.0)
    if scheme == "u8":
        k_scale, k_zero = _compute_group_params(k_lat, group_size, scheme)
        v_scale, v_zero = _compute_group_params(v_lat, group_size, scheme)
        # Broadcast to group_size and quantize
        # FIX: avoid expand in compiled regions; build tiled tensors via repeat_interleave
        k_scale_b = torch.repeat_interleave(torch.ops.aten.reshape.default(k_scale, (k_scale.shape[0], k_scale.shape[1], k_scale.shape[2], 1)), repeats=int(group_size), dim=-1)
        v_scale_b = torch.ops.aten.mul.Scalar(k_scale_b, 1.0); v_scale_b = v_scale_b * 0 + 1; v_scale_b = torch.ops.aten.reshape.default(v_scale, (v_scale.shape[0], v_scale.shape[1], v_scale.shape[2], 1)) * v_scale_b
        k_zero_b = torch.repeat_interleave(torch.ops.aten.reshape.default(k_zero, (k_zero.shape[0], k_zero.shape[1], k_zero.shape[2], 1)), repeats=int(group_size), dim=-1)
        v_zero_b = torch.repeat_interleave(torch.ops.aten.reshape.default(v_zero, (v_zero.shape[0], v_zero.shape[1], v_zero.shape[2], 1)), repeats=int(group_size), dim=-1)
        kq = (torch.ops.aten.reshape.default(k_lat, (B, H, T, G, group_size)) / k_scale_b + k_zero_b).round().clamp(0, 255).to(torch.uint8)
        vq = (torch.ops.aten.reshape.default(v_lat, (B, H, T, G, group_size)) / v_scale_b + v_zero_b).round().clamp(0, 255).to(torch.uint8)
        kq = torch.ops.aten.reshape.default(kq, (B, H, T, G * group_size))
        vq = torch.ops.aten.reshape.default(vq, (B, H, T, G * group_size))
        if pad > 0:
            kq = kq[..., :D]
            vq = vq[..., :D]
        meta = KvQuantMeta("u8", group_size, k_scale.to(device), k_zero.to(device), v_scale.to(device), v_zero.to(device))
        return kq, vq, meta
    elif scheme == "nf4":
        codebook = _NF4_CODEBOOK.to(device)
        k_std, k_mean = _compute_group_params(k_lat, group_size, scheme)
        v_std, v_mean = _compute_group_params(v_lat, group_size, scheme)
        def _q(x, mean, std):
            xg = torch.ops.aten.reshape.default(x, (B, H, T, G, group_size))
            zn = ((xg - torch.ops.aten.reshape.default(mean, (mean.shape[0], mean.shape[1], mean.shape[2], 1))) / torch.ops.aten.reshape.default(std, (std.shape[0], std.shape[1], std.shape[2], 1))).clamp(-2.0, 2.0)
            # Find nearest codebook index
            # (B,H,T,G,group) vs (C) -> distances across codebook entries
            d = (torch.ops.aten.reshape.default(zn, (B, H, T, G, group_size, 1)) - torch.ops.aten.reshape.default(codebook, (1, 1, 1, 1, 1, -1))) ** 2
            idx = torch.argmin(d, dim=-1).to(torch.uint8)
            return torch.ops.aten.reshape.default(idx, (B, H, T, G * group_size))
        kq = _q(k_lat, k_mean, k_std)
        vq = _q(v_lat, v_mean, v_std)
        if pad > 0:
            kq = kq[..., :D]
            vq = vq[..., :D]
        meta = KvQuantMeta("nf4", group_size, k_std.to(device), k_mean.to(device), v_std.to(device), v_mean.to(device))
        return kq, vq, meta
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")


@torch.no_grad()
def dequantize_kv(
    kq: torch.Tensor,
    vq: torch.Tensor,
    meta: KvQuantMeta,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize KV according to provided metadata.

    Returns float32 tensors of shape (B,H,T,D).
    """
    assert kq.shape == vq.shape and kq.dim() == 4
    B, H, T, D = kq.shape
    G = (D + meta.group_size - 1) // meta.group_size
    pad = G * meta.group_size - D
    if pad > 0:
        # Pad quantized arrays so that view(..., G, group) is valid
        pad_fn = torch.nn.functional.pad
        kq = pad_fn(kq, (0, pad), value=0)
        vq = pad_fn(vq, (0, pad), value=0)
    kqv = torch.ops.aten.reshape.default(kq, (B, H, T, G, meta.group_size))
    vqv = torch.ops.aten.reshape.default(vq, (B, H, T, G, meta.group_size))
    if meta.scheme == "u8":
        assert meta.k_scale is not None and meta.k_offset is not None
        assert meta.v_scale is not None and meta.v_offset is not None
        # Local bindings for faster attribute access in tight path
        k_off = torch.ops.aten.reshape.default(meta.k_offset, (meta.k_offset.shape[0], meta.k_offset.shape[1], 1))
        k_scl = torch.ops.aten.reshape.default(meta.k_scale, (meta.k_scale.shape[0], meta.k_scale.shape[1], 1))
        v_off = torch.ops.aten.reshape.default(meta.v_offset, (meta.v_offset.shape[0], meta.v_offset.shape[1], 1))
        v_scl = torch.ops.aten.reshape.default(meta.v_scale, (meta.v_scale.shape[0], meta.v_scale.shape[1], 1))
        k = (kqv.to(torch.float32) - k_off) * k_scl
        v = (vqv.to(torch.float32) - v_off) * v_scl
    elif meta.scheme == "nf4":
        codebook = _NF4_CODEBOOK.to(kq.device)
        assert meta.k_scale is not None and meta.k_offset is not None
        assert meta.v_scale is not None and meta.v_offset is not None
        k_s = torch.ops.aten.reshape.default(meta.k_scale, (meta.k_scale.shape[0], meta.k_scale.shape[1], 1)); k_m = torch.ops.aten.reshape.default(meta.k_offset, (meta.k_offset.shape[0], meta.k_offset.shape[1], 1))
        v_s = torch.ops.aten.reshape.default(meta.v_scale, (meta.v_scale.shape[0], meta.v_scale.shape[1], 1)); v_m = torch.ops.aten.reshape.default(meta.v_offset, (meta.v_offset.shape[0], meta.v_offset.shape[1], 1))
        cb = codebook.to(torch.float32)
        # aten-only dtype casts for indices to avoid method-style .long()
        k_idx = torch.ops.aten.to.dtype(kqv, torch.long, False, False)
        v_idx = torch.ops.aten.to.dtype(vqv, torch.long, False, False)
        k = cb[k_idx] * k_s + k_m
        v = cb[v_idx] * v_s + v_m
    else:
        raise ValueError(f"Unsupported scheme: {meta.scheme}")
    k = torch.ops.aten.reshape.default(k, (B, H, T, G * meta.group_size))
    v = torch.ops.aten.reshape.default(v, (B, H, T, G * meta.group_size))
    if pad > 0:
        k = k[..., :D]
        v = v[..., :D]
    return k, v


def save_meta_json(meta: KvQuantMeta, path: str | Path) -> None:
    data = {
        "scheme": meta.scheme,
        "group_size": int(meta.group_size),
        # Store shapes only; numeric tensors are large. Calibration tools save stats separately when needed.
        "k_scale_shape": list(meta.k_scale.shape) if meta.k_scale is not None else None,
        "k_offset_shape": list(meta.k_offset.shape) if meta.k_offset is not None else None,
        "v_scale_shape": list(meta.v_scale.shape) if meta.v_scale is not None else None,
        "v_offset_shape": list(meta.v_offset.shape) if meta.v_offset is not None else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))




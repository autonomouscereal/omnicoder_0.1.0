from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVSyncModule(nn.Module):
    """
    Audio–Visual synchronization module.

    Projects per-frame audio features and per-frame video latent features into a
    common space, performs cross-attention in both directions, and produces:
      - fused video features conditioned on audio (for optional downstream use)
      - an alignment score in [0, 1] indicating lip-sync/temporal agreement

    Expected shapes:
      - audio_seq: (B, T_a, d_audio)
      - video_seq: (B, T_v, d_video)

    When T_a != T_v, features are time-aligned by simple linear interpolation to
    the shorter sequence length for alignment scoring. Cross-attention operates
    on the provided lengths without enforcing equality.
    """

    def __init__(self, d_audio: int, d_video: int, d_model: int = 512, num_heads: int = 4) -> None:
        super().__init__()
        dm = int(max(32, d_model))
        self.d_model = dm
        self.audio_proj = nn.Linear(int(d_audio), dm, bias=False)
        self.video_proj = nn.Linear(int(d_video), dm, bias=False)
        # Cross-attention: video queries attend to audio keys/values, and vice versa
        self.av_attn = nn.MultiheadAttention(dm, num_heads=max(1, int(num_heads)), batch_first=True)
        self.va_attn = nn.MultiheadAttention(dm, num_heads=max(1, int(num_heads)), batch_first=True)
        # Output projection for fused video features
        self.fuse_out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, dm))

    def forward(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # audio_seq: (B, T_a, d_audio); video_seq: (B, T_v, d_video)
        if audio_seq.dim() != 3 or video_seq.dim() != 3:
            raise RuntimeError("AVSyncModule expects (B,T,C) tensors for audio and video inputs")

        a = self.audio_proj(audio_seq)  # (B,T_a,D)
        v = self.video_proj(video_seq)  # (B,T_v,D)

        # Cross-attention
        # Video queries attend to audio for audio-conditioned video features
        v_fused, _ = self.av_attn(query=v, key=a, value=a, need_weights=False)
        v_fused = self.fuse_out(v_fused)

        # Audio queries attend to video (not used by caller currently but helps alignment)
        a_att, _ = self.va_attn(query=a, key=v, value=v, need_weights=False)

        # Alignment score based on pooled cosine similarity between the two attended streams
        # Export-safe length alignment: crop both to Tmin via aten-only arithmetic; avoid Python ints and F.interpolate sizes
        _sha = torch.ops.aten._shape_as_tensor(a_att)
        _Ta = torch.ops.aten.select.int(_sha, 0, 1)  # scalar tensor
        _shv = torch.ops.aten._shape_as_tensor(v_fused)
        _Tv = torch.ops.aten.select.int(_shv, 0, 1)  # scalar tensor
        _Ta_i = torch.ops.aten.to.dtype(_Ta, a_att.dtype, False, False)
        _Tv_i = torch.ops.aten.to.dtype(_Tv, a_att.dtype, False, False)
        _Tmin = torch.ops.aten.minimum.default(_Ta_i, _Tv_i)
        # Build start=0 and end=Tmin as scalars via aten (cast to long for indices)
        _zero = torch.ops.aten.mul.Scalar(_Ta, 0)
        _Tmin_l = torch.ops.aten.to.dtype(_Tmin, torch.long, False, False)
        # Crop only when needed by composing with arithmetic masks on indices
        # Slice a_att: (B,Ta,D) -> (B,Tmin,D)
        a_att = torch.ops.aten.slice.Tensor(a_att, 1, _zero, _Tmin_l, 1)
        v_fused = torch.ops.aten.slice.Tensor(v_fused, 1, _zero, _Tmin_l, 1)

        # Pooled embeddings
        a_pool = F.normalize(torch.ops.aten.mean.dim_IntList(a_att, [1], False), dim=-1)  # (B,D)
        v_pool = F.normalize(torch.ops.aten.mean.dim_IntList(v_fused, [1], False), dim=-1)  # (B,D)
        cos = (a_pool * v_pool).sum(dim=-1, keepdim=True)  # (B,1), in [-1,1]
        align = 0.5 * (cos + 1.0)  # map to [0,1]

        # Return fused video features and the alignment score
        return v_fused, align



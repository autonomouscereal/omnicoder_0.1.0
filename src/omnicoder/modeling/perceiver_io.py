from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverIOTower(nn.Module):

    def __init__(self, d_io: int = 128, d_latent: int = 256, n_latents: int = 64, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.d_io = int(d_io)
        self.d_latent = int(d_latent)
        self.n_latents = int(n_latents)
        self.n_layers = int(n_layers)
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.d_latent) * 0.02)
        # Cross-attn projections
        self.q_proj = nn.Linear(self.d_latent, self.d_latent, bias=False)
        self.k_proj = nn.Linear(self.d_io, self.d_latent, bias=False)
        self.v_proj = nn.Linear(self.d_io, self.d_latent, bias=False)
        # Latent self-attn + MLP stacks
        self.sa_layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(self.d_latent),
                'qkv': nn.Linear(self.d_latent, self.d_latent * 3, bias=False),
                'proj': nn.Linear(self.d_latent, self.d_latent, bias=False),
                'ln2': nn.LayerNorm(self.d_latent),
                'mlp': nn.Sequential(
                    nn.Linear(self.d_latent, self.d_latent * 4),
                    nn.GELU(),
                    nn.Linear(self.d_latent * 4, self.d_latent),
                ),
            }) for _ in range(self.n_layers)
        ])
        # IO heads
        self.to_io = nn.Linear(self.d_latent, self.d_io, bias=False)

    def forward(self, io_feats: torch.Tensor) -> torch.Tensor:
        # io_feats: (B, T_io, D_io) of raw bytes/pixels/audio patches pre-projected
        b, t, _ = io_feats.shape
        # FIX: avoid .expand in compiled regions; materialize by repeat_interleave to keep scheduler happy
        lat = torch.repeat_interleave(self.latents.unsqueeze(0), repeats=int(b), dim=0)
        # Cross-attend IO -> latents
        q = self.q_proj(lat)
        k = self.k_proj(io_feats)
        v = self.v_proj(io_feats)
        attn = torch.softmax((q @ k.transpose(1, 2)) / max(1e-6, float(self.d_latent) ** 0.5), dim=-1)
        lat = lat + attn @ v
        # Latent SA + MLP
        for blk in self.sa_layers:
            ln1 = blk['ln1'](lat)
            qkv = blk['qkv'](ln1)
            # FIX: avoid torch.chunk builtin target under Inductor; split via narrow
            d = int(qkv.size(-1) // 3)
            q = torch.narrow(qkv, -1, 0, d)
            k = torch.narrow(qkv, -1, d, d)
            v = torch.narrow(qkv, -1, 2 * d, d)
            a = torch.softmax((q @ k.transpose(1, 2)) / max(1e-6, float(self.d_latent) ** 0.5), dim=-1)
            lat = lat + blk['proj'](a @ v)
            lat = lat + blk['mlp'](blk['ln2'](lat))
        # Emit IO-shaped features by broadcasting latents and simple pooling
        # Materialize broadcast to (B,T,DL) without .expand
        pooled = torch.repeat_interleave(lat.mean(dim=1, keepdim=True), repeats=int(t), dim=1)
        return self.to_io(pooled)



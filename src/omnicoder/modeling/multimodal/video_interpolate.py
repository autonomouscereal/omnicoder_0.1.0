from __future__ import annotations

import torch
import torch.nn as nn


class LatentInterpolator(nn.Module):
    """
    ORT-friendly latent interpolator: given keyframe latents z0, z1, predicts
    intermediate latents at M evenly spaced timesteps.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, hidden), nn.GELU(), nn.Linear(hidden, latent_dim)
        )

    def forward(self, z0: torch.Tensor, z1: torch.Tensor, num_steps: int) -> torch.Tensor:
        # z0, z1: (B, T, D) keyframe-aligned; returns (B, num_steps, T, D)
        b, t, d = z0.shape
        taus = torch.linspace(0, 1, steps=int(num_steps), device=z0.device)
        taus = torch.ops.aten.reshape.default(taus, (1, -1, 1, 1))
        z0e = torch.ops.aten.reshape.default(z0, (z0.shape[0], 1, z0.shape[1], z0.shape[2]))
        z0e = torch.repeat_interleave(z0e, repeats=int(num_steps), dim=1)
        z1e = torch.ops.aten.reshape.default(z1, (z1.shape[0], 1, z1.shape[1], z1.shape[2]))
        z1e = torch.repeat_interleave(z1e, repeats=int(num_steps), dim=1)
        taue = taus.expand_as(z0e[..., :1])
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        inp = _safe_concat([z0e, z1e, taue], -1)
        out = self.mlp(inp)
        return out



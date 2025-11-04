from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLatentRefiner(nn.Module):
	"""
	Lightweight latent refiner for continuous heads (image/audio).

	- Operates on latent vectors (B, D) or sequences (B, T, D)
	- Two-layer MLP with LayerNorm and residual gating
	- Optional temporal 1D conv when a time dimension is present
	"""

	def __init__(self, latent_dim: int, hidden_mult: int = 2, use_temporal: bool = False, kernel_size: int = 3) -> None:
		super().__init__()
		hidden = int(max(1, hidden_mult) * latent_dim)
		self.norm = nn.LayerNorm(latent_dim)
		self.ff1 = nn.Linear(latent_dim, hidden, bias=False)
		self.ff2 = nn.Linear(hidden, latent_dim, bias=False)
		self.gate = nn.Parameter(torch.tensor(1.0))
		self.use_temporal = bool(use_temporal)
		if self.use_temporal:
			self.tconv = nn.Conv1d(latent_dim, latent_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=latent_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B,D) or (B,T,D)
		if x.dim() == 2:
			h = self.ff2(F.gelu(self.ff1(self.norm(x))))
			return x + self.gate * h
		elif x.dim() == 3:
			b, t, d = x.shape
			x2 = x.reshape(b * t, d)
			h = self.ff2(F.gelu(self.ff1(self.norm(x2))))
			y = x2 + self.gate * h
			y = y.reshape(b, t, d)
			if self.use_temporal:
				# Depthwise temporal filtering on channels
				y_t = y.transpose(1, 2)  # (B,D,T)
				y_t = self.tconv(y_t)
				y = y_t.transpose(1, 2)
			return y
		else:
			raise ValueError("Expected input of shape (B,D) or (B,T,D)")

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLatentRefinerVector(nn.Module):
    """
    Tiny MLP refiner for continuous latents (vector form). Input/Output: (B, D)
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.SiLU()]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLatentRefinerSeq(nn.Module):
    """
    Sequence-aware refiner sharing the same formulation; kept for explicitness.
    """

    def __init__(self, latent_dim: int, hidden_mult: int = 2, use_temporal: bool = False, kernel_size: int = 3) -> None:
        super().__init__()
        hidden = int(max(1, hidden_mult) * latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.ff1 = nn.Linear(latent_dim, hidden, bias=False)
        self.ff2 = nn.Linear(hidden, latent_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.use_temporal = bool(use_temporal)
        if self.use_temporal:
            self.tconv = nn.Conv1d(latent_dim, latent_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected input of shape (B,T,D)")
        b, t, d = x.shape
        x2 = x.reshape(b * t, d)
        h = self.ff2(F.gelu(self.ff1(self.norm(x2))))
        y = x2 + self.gate * h
        y = y.reshape(b, t, d)
        if self.use_temporal:
            y_t = y.transpose(1, 2)
            y_t = self.tconv(y_t)
            y = y_t.transpose(1, 2)
        return y



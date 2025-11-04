from __future__ import annotations

import torch


def sample_sigma(batch_size: int, device: torch.device, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    """Sample noise scale sigma from a log-uniform distribution (EDM-style)."""
    u = torch.rand(batch_size, device=device)
    log_sigma = torch.log(torch.tensor(sigma_min, device=device)) * (1 - u) + torch.log(torch.tensor(sigma_max, device=device)) * u
    return torch.exp(log_sigma)


def add_noise(clean: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise: x_t = x0 + sigma * eps. Returns (x_t, eps)."""
    eps = torch.randn_like(clean)
    return clean + torch.ops.aten.reshape.default(sigma, (-1, *([1] * (clean.dim() - 1)))) * eps, eps


def loss_weight(sigma: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Return per-sample loss weights; default inverse proportional to sigma^p."""
    return (1.0 / (sigma.clamp_min(1e-6) ** p)).detach()


# VP (variance preserving) cosine schedule helpers
def vp_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    s = 0.008
    return torch.cos((t + s) / (1 + s) * torch.pi / 2).pow(2)


def sample_vp_params(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(batch_size, device=device)
    a_bar = vp_alpha_bar(t)
    alpha = a_bar.sqrt()
    sigma = (1.0 - a_bar).clamp_min(1e-6).sqrt()
    return alpha, sigma


# VE (variance exploding) – reuse log-uniform sigma
def sample_ve_sigma(batch_size: int, device: torch.device, sigma_min: float = 0.01, sigma_max: float = 50.0) -> torch.Tensor:
    return sample_sigma(batch_size, device, sigma_min, sigma_max)



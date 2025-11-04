import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2d(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(ch_in, ch_out, k, padding=p)
        self.norm = nn.GroupNorm(num_groups=max(1, ch_out // 8), num_channels=ch_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ImageLatentRefiner2D(nn.Module):
    """
    Tiny 2D refiner network for image latents.
    Intended to run for 4–8 steps (or once) to improve latent quality.
    Input/Output: (B, C, H, W) latent space tensors.
    """

    def __init__(self, latent_channels: int = 4, hidden: int = 64, depth: int = 4):
        super().__init__()
        ch = latent_channels
        layers: list[nn.Module] = [ConvBlock2d(ch, hidden)]
        for _ in range(max(1, depth - 2)):
            layers += [ConvBlock2d(hidden, hidden)]
        layers += [nn.Conv2d(hidden, ch, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual refinement
        return x + self.net(x)


class ConvBlock1d(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, k: int = 5):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv1d(ch_in, ch_out, k, padding=p)
        self.norm = nn.GroupNorm(num_groups=max(1, ch_out // 8), num_channels=ch_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AudioLatentRefiner1D(nn.Module):
    """
    Tiny 1D refiner for audio latents (e.g., mel or EnCodec latent streams).
    Input/Output: (B, C, T) latent time series.
    """

    def __init__(self, latent_channels: int = 16, hidden: int = 64, depth: int = 4):
        super().__init__()
        ch = latent_channels
        layers: list[nn.Module] = [ConvBlock1d(ch, hidden)]
        for _ in range(max(1, depth - 2)):
            layers += [ConvBlock1d(hidden, hidden)]
        layers += [nn.Conv1d(hidden, ch, kernel_size=5, padding=2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VideoTemporalSSMRefiner(nn.Module):
    """
    Lightweight temporal refiner over sequences of frame latents.
    Treat frames as (B, T, C, H, W); we pool spatially and apply a small gated 1D conv over T,
    then up-project and add back as a residual to each frame latent.
    """

    def __init__(self, latent_channels: int = 4, hidden: int = 64, kernel: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.Sequential(
            nn.Conv1d(latent_channels, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.SiLU(),
            nn.Conv1d(hidden, latent_channels, kernel_size=kernel, padding=kernel // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x_ = x.reshape(b * t, c, h, w)
        g = self.pool(x_).reshape(b, t, c).transpose(1, 2)  # (B, C, T)
        delta = self.temporal(g).transpose(1, 2).reshape(b, t, c, 1, 1)
        return x + delta.expand_as(x)



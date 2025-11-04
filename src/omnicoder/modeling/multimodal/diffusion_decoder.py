from __future__ import annotations

from typing import Optional, Tuple

import torch


class DiffusionDecoder:
    """
    Light wrapper that turns LLM hidden states into diffusion conditioning and calls a
    pluggable U-Net/VAE pipeline (Stable Diffusion-like) for images or a video U-Net.

    This remains backend-agnostic: you can plug in diffusers, onnxruntime, Core ML, or
    ExecuTorch U-Net graphs. The LLM should pass the final hidden state (last token or
    pooled) via `conditioning`.
    """

    def __init__(
        self,
        kind: str = "image",
        scheduler: str = "ddim",
        backend: str = "none",
    ) -> None:
        self.kind = kind
        self.scheduler = scheduler
        self.backend = backend  # e.g., 'diffusers', 'onnx', 'coreml', 'execu'
        self._pipe = None

    def load_backend(self, **kwargs) -> None:
        """Optional: lazily load heavy pipelines on demand."""
        self._pipe = kwargs.get("pipe", None)

    @torch.inference_mode()
    def decode(
        self,
        conditioning: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        steps: int = 20,
        size: Tuple[int, int] = (512, 512),
    ):
        # conditioning: (B, T, C) or (B, C). We pool to a single vector per sample.
        if conditioning.dim() == 3:
            cond = conditioning[:, -1, :]
        else:
            cond = conditioning

        if self.backend == "diffusers" and self._pipe is not None:
            # Example (pseudo): self._pipe(image_embeds=cond, num_inference_steps=steps, height=size[1], width=size[0])
            return self._pipe(cond, steps, size)
        if self.backend in {"onnx", "coreml", "execu"} and self._pipe is not None:
            return self._pipe(cond, steps, size)
        return None

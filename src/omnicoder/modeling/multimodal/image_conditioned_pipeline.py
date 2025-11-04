from __future__ import annotations

"""
Conditioned Stable Diffusion wrapper (diffusers backend)

Allows passing an extra conditioning vector (from `HiddenToImageCond`) by
appending it as an additional cross-attention token to the text encoder output.

This approach requires no model surgery and works with SD 1.x pipelines:
 - We build `prompt_embeds` and `negative_prompt_embeds` manually
 - We concatenate a single token per sample: cond token for positive, zeros for negative
 - Then we call the underlying StableDiffusionPipeline with `prompt_embeds=...`

Note: FiLM-style scale/shift outputs from `HiddenToImageCond` are not consumed here
to keep compatibility with stock UNet2DConditionModel. If you need FiLM, implement a
custom UNet with AdaLN or register hooks in a custom pipeline.
"""

from typing import List, Optional, Tuple

import torch


class ConditionedSDPipeline:
    def __init__(self, pipe) -> None:
        # `pipe` is a diffusers StableDiffusionPipeline (already moved to device)
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.device = next(self.text_encoder.parameters()).device

    @torch.inference_mode()
    def _encode_text(self, prompts: List[str], dtype: torch.dtype) -> torch.Tensor:
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(self.device)
        # text_encoder returns (last_hidden_state, pooled)
        out = self.text_encoder(input_ids)[0].to(dtype)
        return out  # (B, seq, C)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        conditioning: Optional[torch.Tensor],  # (B,C) or (B,T,C)
        steps: int,
        size: Tuple[int, int],
        guidance_scale: float = 7.5,
    ):
        bsz = 1
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # Encode text
        pos = self._encode_text([prompt], dtype=dtype)
        neg = self._encode_text([""], dtype=dtype)

        # Optional: append one cond token per sample to cross-attn context
        if conditioning is not None:
            if conditioning.dim() == 3:
                cond = conditioning[:, -1, :]
            else:
                cond = conditioning
            cond = cond.to(self.device).to(dtype).unsqueeze(1)  # (B,1,C)
            # Ensure channel dim matches text hidden size
            if cond.shape[-1] != pos.shape[-1]:
                # Project with a small linear layer on the fly to match dims
                proj = torch.nn.Linear(cond.shape[-1], pos.shape[-1], bias=False).to(self.device).to(dtype)
                with torch.no_grad():
                    cond = proj(cond)
            from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
            pos = _safe_cat(pos, cond, 1)
            from omnicoder.utils.torchutils import zeros_like_shape as _zeros_like_shape  # type: ignore
            z = _zeros_like_shape(cond, (cond.shape[0], cond.shape[1]))
            from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
            neg = _safe_cat(neg, z, 1)

        images = self.pipe(
            prompt_embeds=pos,
            negative_prompt_embeds=neg,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            height=int(size[1]),
            width=int(size[0]),
        ).images
        return images[0]



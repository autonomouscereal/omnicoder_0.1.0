from __future__ import annotations

"""
ONNX Runtime Stable Diffusion callable for ImageGenPipeline backend="onnx".

Requires an Optimum-exported SD pipeline directory containing:
 - text_encoder.onnx
 - unet.onnx
 - vae_decoder.onnx

Implements a basic DDIM denoising loop with classifier-free guidance.
This runner uses transformers' CLIPTokenizer and diffusers' DDIMScheduler
for scheduling only (lightweight). UNet/VAE run with ONNX Runtime.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


class ORTSDCallable:
    def __init__(self, pipeline_dir: str, provider: str = "CPUExecutionProvider", provider_options: dict | None = None, guidance_scale: float = 7.5) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        self.root = Path(pipeline_dir)
        self.provider = provider
        self.guidance_scale = float(guidance_scale)

        # Sessions
        provs = [provider]
        prov_opts = [provider_options] if provider_options else None
        # Robustly resolve ONNX file paths from common Optimum export layouts
        def _first_existing(paths: list[Path]) -> Path:
            for p in paths:
                if p.exists():
                    return p
            # If none exist, return the first candidate to surface a clear error downstream
            return paths[0]
        enc_path = _first_existing([
            self.root / "text_encoder.onnx",
            self.root / "text_encoder" / "model.onnx",
            self.root / "text_encoder" / "model_fp16.onnx",
        ])
        unet_path = _first_existing([
            self.root / "unet.onnx",
            self.root / "unet" / "model.onnx",
            self.root / "unet" / "model_fp16.onnx",
        ])
        vae_path = _first_existing([
            self.root / "vae_decoder.onnx",
            self.root / "vae_decoder" / "model.onnx",
            self.root / "vae_decoder" / "decoder_model.onnx",
            self.root / "vae_decoder" / "model_fp16.onnx",
        ])
        self.enc = ort.InferenceSession(str(enc_path), providers=provs, provider_options=prov_opts)  # type: ignore[arg-type]
        self.unet = ort.InferenceSession(str(unet_path), providers=provs, provider_options=prov_opts)  # type: ignore[arg-type]
        self.vae = ort.InferenceSession(str(vae_path), providers=provs, provider_options=prov_opts)  # type: ignore[arg-type]

        # Tokenizer and scheduler (python-side only)
        try:
            from transformers import CLIPTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers required for CLIPTokenizer") from e
        try:
            from diffusers import DDIMScheduler  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("diffusers required for scheduler") from e

        # SD 1.5 uses openai/clip-vit-large-patch14; prefer local cache when offline
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=False)
        except Exception:
            # Fallback to local cache only to avoid network hard-fail paths
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        # Initialize a default DDIM scheduler (params approximating SD v1.x)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
        )

        # Cache io names for speed
        self._enc_in = self.enc.get_inputs()[0].name
        self._enc_out = self.enc.get_outputs()[0].name

        unet_inputs = self.unet.get_inputs()
        self._unet_in_sample = unet_inputs[0].name
        self._unet_in_t = unet_inputs[1].name
        self._unet_in_ctx = unet_inputs[2].name
        self._unet_out = self.unet.get_outputs()[0].name

        self._vae_in = self.vae.get_inputs()[0].name
        self._vae_out = self.vae.get_outputs()[0].name

        self.latent_scale = 0.18215

    def _encode(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        out = self.enc.run([self._enc_out], {self._enc_in: tokens})[0]
        return out.astype(np.float32)

    def __call__(self, prompt: str, steps: int, size: Tuple[int, int]):
        bsz = 1
        height, width = size[1], size[0]
        # Prepare text conditioning (uncond + cond)
        text_ctx = self._encode(["", prompt])  # (2, seq, dim)

        # Scheduler
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps

        # Initialize latent
        latent = np.random.randn(bsz, 4, height // 8, width // 8).astype(np.float32)
        latent = latent * self.scheduler.init_noise_sigma

        # Iterate denoising
        for t in timesteps:
            # Duplicate latents for classifier-free guidance
            latent_in = np.concatenate([latent, latent], axis=0).astype(np.float32)
            # UNet expects scaled latents
            # Convert scalar timestep to shape (2,)
            t_arr = np.array([t, t], dtype=np.float32)
            eps = self.unet.run(
                [self._unet_out],
                {
                    self._unet_in_sample: latent_in,
                    self._unet_in_t: t_arr,
                    self._unet_in_ctx: text_ctx,
                },
            )[0]
            # Split uncond/cond
            eps_uncond, eps_text = np.split(eps, 2, axis=0)
            eps_guided = eps_uncond + self.guidance_scale * (eps_text - eps_uncond)
            # DDIM step
            latent = self.scheduler.step(eps_guided, t, latent).prev_sample.astype(np.float32)

        # Decode
        latents = (latent / self.latent_scale).astype(np.float32)
        img = self.vae.run([self._vae_out], {self._vae_in: latents})[0][0]  # (3,H,W) in -1..1
        img = np.clip((img / 2.0 + 0.5), 0.0, 1.0)

        # To PIL
        try:
            from PIL import Image  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Pillow required for saving images") from e
        hwc = np.transpose((img * 255.0).astype(np.uint8), (1, 2, 0))
        return Image.fromarray(hwc)



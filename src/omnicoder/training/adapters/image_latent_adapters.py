from __future__ import annotations

"""
Image latent adapters for reconstruction training.

- DiffusersAdapter: builds a VAE encoder (e.g., from Stable Diffusion) and encodes images to latents
- ONNXAdapter: uses an ORT callable directory if it exposes an encoder; otherwise raises
"""

from typing import Optional

import torch


class DiffusersAdapter:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        try:
            from diffusers import AutoencoderKL  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Diffusers is required for DiffusersAdapter. Install with extras [gen].") from e
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae.to(self.device)
        self.vae.eval()

    @torch.inference_mode()
    def encode(self, img_chw: torch.Tensor) -> torch.Tensor:
        """CHW in [-1,1] -> latent vector (D). We pool spatial dims of the VAE latent."""
        x = img_chw.unsqueeze(0).to(self.device)  # (1,C,H,W)
        posterior = self.vae.encode(x)
        if hasattr(posterior, "latent_dist"):
            z = posterior.latent_dist.sample()
        else:
            z = posterior.sample()
        z = z.mean(dim=(0, 2, 3))  # (C_lat)
        return z


class ONNXAdapter:
    def __init__(self, onnx_dir: str, device: Optional[str] = None):
        """Adapter for ONNX SD exports; prefers a direct VAE encoder session when available.

        If a full ORT Stable Diffusion callable is available, we will use it only if it
        exposes an encode() method; otherwise we fall back to loading `vae_encoder.onnx`
        directly and pooling the latent.
        """
        self._onnx_dir = onnx_dir
        try:
            from omnicoder.inference.runtimes.onnx_image_decode import ORTSDCallable  # type: ignore
            self._call = ORTSDCallable(onnx_dir)
        except Exception:
            self._call = None  # type: ignore
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Lazy ORT VAE encoder session
        self._vae_sess = None

    @torch.inference_mode()
    def encode(self, img_chw: torch.Tensor) -> torch.Tensor:
        # Fast-path: ORT callable exposes encode
        if self._call is not None and hasattr(self._call, "encode"):
            z = self._call.encode(img_chw)
            return z if isinstance(z, torch.Tensor) else torch.tensor(z)
        # Fallback: load vae_encoder.onnx and run it directly
        try:
            import onnxruntime as ort  # type: ignore
            from pathlib import Path
            onnx_dir = Path(self._onnx_dir)
            enc_path = onnx_dir / "vae_encoder.onnx"
            if self._vae_sess is None:
                if not enc_path.exists():
                    raise RuntimeError(f"vae_encoder.onnx not found in {onnx_dir}")
                providers = ["CUDAExecutionProvider"] if (torch.cuda.is_available() and self._device.startswith("cuda")) else ["CPUExecutionProvider"]
                self._vae_sess = ort.InferenceSession(str(enc_path), providers=providers)
                # Cache IO names
                outs = self._vae_sess.get_outputs()
                ins = self._vae_sess.get_inputs()
                self._enc_in = ins[0].name
                self._enc_out = outs[0].name
            # Preprocess: CHW float32 in [-1,1] to NCHW
            x = img_chw.to(torch.float32).clamp(-1.0, 1.0).unsqueeze(0).cpu().numpy()
            out = self._vae_sess.run([self._enc_out], {self._enc_in: x})[0]  # (1,C,H',W')
            z = torch.from_numpy(out).squeeze(0)  # (C_lat,H',W')
            # Pool spatial to a vector
            z_vec = z.mean(dim=(1, 2)) if z.ndim == 3 else z
            return z_vec
        except Exception as e:
            raise NotImplementedError("ONNXAdapter.encode requires encode() on ORT callable or a vae_encoder.onnx.") from e


class DiffusersFlowAdapter:
    """
    Produces diffusion-style targets on VAE latents: predict noise epsilon for z_t.

    We pool VAE latents spatially to a vector and project to a desired latent_dim
    to align with the model's continuous head. This is a simplified flow objective.
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", latent_dim: int = 16, device: Optional[str] = None):
        try:
            from diffusers import AutoencoderKL  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Diffusers is required for DiffusersFlowAdapter. Install with extras [gen].") from e
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae.to(self.device).eval()
        self.latent_dim = int(latent_dim)
        with torch.no_grad():
            # Initialize lightweight random projection placeholders. These are not trained.
            # Vector projection (from pooled channel dim -> latent_dim)
            self.proj = None  # will be lazily created once we see the VAE latent channel count
            # Patch projection (from channel dim -> latent_dim) applied per spatial location
            self.patch_proj = None  # lazily created

    @torch.inference_mode()
    def encode_flow(self, img_chw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns diffusion-style targets on VAE latents.

        Output shapes:
          - eps_targets: (P, D) where P = H'*W' flattened VAE latent patches
          - z0_targets:  (P, D) matching the same layout

        Callers that prefer vector targets may average across P.
        """
        x = img_chw.unsqueeze(0).to(self.device)
        posterior = self.vae.encode(x)
        if hasattr(posterior, "latent_dist"):
            z0 = posterior.latent_dist.sample()  # (1, C, H', W')
        else:
            z0 = posterior.sample()  # (1, C, H', W')
        z0 = z0.squeeze(0)  # (C, H', W')
        C = int(z0.size(0))
        H = int(z0.size(1))
        W = int(z0.size(2))
        # Lazily create projection matrices on first use, sized by channel dim
        if self.proj is None:
            self.proj = torch.randn(C, self.latent_dim, device=self.device) * 0.05
        if self.patch_proj is None:
            self.patch_proj = torch.randn(C, self.latent_dim, device=self.device) * 0.05
        # Vector (global pooled) projection path retained for backwards compatibility
        z0_vec = z0.mean(dim=(1, 2))  # (C,)
        z0_vec = z0_vec @ self.proj  # (D)
        # Patch projection: flatten spatial then apply linear projection per location
        z0_flat = torch.ops.aten.reshape.default(z0, (C, H * W)).transpose(0, 1)  # (P, C)
        z0_patches = z0_flat @ self.patch_proj  # (P, D)
        # Sample epsilon in latent space for flow objective (per-patch)
        eps = torch.randn_like(z0_patches)
        # Return per-patch targets; callers can pool if needed
        return eps, z0_patches

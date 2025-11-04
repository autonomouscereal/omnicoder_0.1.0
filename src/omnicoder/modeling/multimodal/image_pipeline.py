from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    # Optional mini-CLIP style verifier
    from .aligner import PreAligner, CrossModalVerifier, TextEmbedder  # type: ignore
except Exception:
    PreAligner = None  # type: ignore
    CrossModalVerifier = None  # type: ignore
    TextEmbedder = None  # type: ignore
try:
    # Optional lightweight vision backbone for embedding images
    from .vision_encoder import VisionBackbone  # type: ignore
except Exception:
    VisionBackbone = None  # type: ignore

class ImageGenPipeline:
    """
    Minimal image generation wrapper.

    - If `backend='diffusers'`, expects a preinstalled local Stable Diffusion
      pipeline (weights available locally). Provide either `hf_id` (HuggingFace
      model id) or `local_path` (directory with pipeline files).
    - Other backends can be added (ONNX/Core ML/ExecuTorch) by exposing a
      callable with the same interface as `_pipe`.
    """

    def __init__(
        self,
        backend: str = "diffusers",
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        hf_id: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> None:
        # Allow environment overrides to lock lightweight U-Net pipelines or ONNX callable
        try:
            import os as _os
            env_backend = _os.getenv("OMNICODER_IMAGE_BACKEND", "").strip()
            if env_backend:
                backend = env_backend
            env_ref = _os.getenv("OMNICODER_SD_MODEL", "").strip()
            if env_ref and not hf_id and not local_path:
                hf_id = env_ref
        except Exception:
            pass
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.hf_id = hf_id
        self.local_path = local_path
        self.hf_id = hf_id
        self._pipe = None

    def _load_diffusers(self) -> bool:
        try:
            from diffusers import StableDiffusionPipeline
        except Exception:
            return False
        model_ref = self.local_path if self.local_path else (self.hf_id or "")
        if not model_ref:
            return False
        # Use dtype kwarg (torch_dtype is deprecated)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_ref,
            dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        self._pipe = pipe
        return True

    def _load_diffusers_onnx(self) -> bool:
        try:
            from diffusers import OnnxStableDiffusionPipeline  # type: ignore
        except Exception:
            return False
        model_ref = self.local_path if self.local_path else (self.hf_id or "")
        if not model_ref:
            return False
        try:
            pipe = OnnxStableDiffusionPipeline.from_pretrained(model_ref, provider="CPUExecutionProvider")
        except Exception:
            return False
        self._pipe = pipe
        return True

    def ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        if self.backend == "diffusers":
            return self._load_diffusers()
        if self.backend in {"onnx", "diffusers_onnx"}:
            return self._load_diffusers_onnx()
        return False

    def load_backend(self, pipe) -> bool:
        """
        Inject a callable backend (e.g., ORT Stable Diffusion runner) with the
        signature: (prompt: str, steps: int, size: (w,h)) -> PIL.Image or path.
        """
        self._pipe = pipe
        return True

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        conditioning: Optional[torch.Tensor] = None,
        steps: int = 20,
        size: Tuple[int, int] = (512, 512),
        out_path: Optional[str] = None,
        refiner_steps: int = 0,
        # Cross-modal verifier gating (mini-CLIP style; off by default)
        cm_verifier: bool = False,
        cm_threshold: float = 0.6,
        text_embed: Optional[torch.Tensor] = None,
    ) -> Optional[Path]:
        if not self.ensure_loaded():
            return None
        if self.backend in {"diffusers", "diffusers_onnx"}:
            # Optionally produce multiple candidates and select by cross-modal verifier
            import os as _os
            try:
                n_cand = int(_os.getenv("OMNICODER_IMAGE_NCAND", "1"))
            except Exception:
                n_cand = 1
            images = []
            for _ in range(max(1, n_cand)):
                # If a conditioning tensor is provided, use our conditioned wrapper
                if conditioning is not None:
                    try:
                        from .image_conditioned_pipeline import ConditionedSDPipeline
                        cond_pipe = ConditionedSDPipeline(self._pipe)
                        img_i = cond_pipe.generate(prompt, conditioning, steps, size)
                    except Exception:
                        img_i = self._pipe(
                            prompt=prompt,
                            num_inference_steps=int(steps),
                            height=int(size[1]),
                            width=int(size[0]),
                        ).images[0]
                else:
                    img_i = self._pipe(
                        prompt=prompt,
                        num_inference_steps=int(steps),
                        height=int(size[1]),
                        width=int(size[0]),
                    ).images[0]
                if refiner_steps and isinstance(img_i, type(getattr(img_i, 'copy', None))):
                    try:
                        import numpy as _np
                        import PIL.Image as _PIL
                        from .refiner import TinyImageRefiner
                        t = torch.ops.aten.to.dtype(torch.from_numpy(_np.array(img_i)), torch.float32, False, False).permute(2,0,1) / 255.0
                        t = t.unsqueeze(0)
                        ref = TinyImageRefiner()
                        t2 = ref(t, steps=int(refiner_steps))[0]
                        img_np = (t2.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                        img_i = _PIL.fromarray(img_np)
                    except Exception:
                        pass
                images.append(img_i)
            # Choose best candidate when requested
            image = images[0]
            if bool(cm_verifier) and text_embed is not None and CrossModalVerifier is not None and len(images) > 1:
                try:
                    best_s = float('-inf'); best_img = images[0]
                    for im in images:
                        s = _cm_verifier_score_from_image(im, text_embed)
                        if float(s) > float(best_s):
                            best_s = float(s); best_img = im
                    image = best_img
                except Exception:
                    image = images[0]
            if out_path:
                p = Path(out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                image.save(p)
                return p
            # Optional cross-modal rejection when no output path is requested
            if bool(cm_verifier) and text_embed is not None and CrossModalVerifier is not None:
                try:
                    score = _cm_verifier_score_from_image(image, text_embed)
                    if float(score) < float(cm_threshold):
                        return None
                except Exception:
                    pass
            return None
        if self.backend in {"onnx", "coreml", "execu"}:
            # Optional multi-candidate selection via CLIP-like scoring (default off)
            import os as _os
            try:
                n_cand = int(_os.getenv("OMNICODER_IMAGE_NCAND", "1"))
            except Exception:
                n_cand = 1
            selector = (_os.getenv("OMNICODER_IMAGE_SELECT", "none").strip().lower())
            cand_images = []
            for _ in range(max(1, n_cand)):
                result = self._pipe(prompt=prompt, steps=int(steps), size=(int(size[0]), int(size[1])))  # type: ignore
                cand_images.append(result)
            chosen = None
            if selector in {"clip", "openclip"} and cand_images:
                # Try open-clip first, then transformers CLIP
                try:
                    from PIL import Image as _PILImage  # type: ignore
                    import numpy as _np
                    import torch as _torch
                    # open-clip
                    try:
                        import open_clip  # type: ignore
                        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
                        tokenizer = open_clip.get_tokenizer("ViT-B-32")
                        txt = tokenizer([prompt])
                        with _torch.no_grad():
                            txt_feat = model.encode_text(txt).float()
                            txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-6)
                        scores = []
                        for img in cand_images:
                            try:
                                if isinstance(img, _PILImage.Image):
                                    ten = preprocess(img).unsqueeze(0)
                                else:
                                    # If path or array, try to load to PIL
                                    ten = preprocess(_PILImage.open(str(img)))
                                    ten = ten.unsqueeze(0)
                                with _torch.no_grad():
                                    img_feat = model.encode_image(ten).float()
                                    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
                                    s = (img_feat * txt_feat).sum().item()
                                scores.append(s)
                            except Exception:
                                scores.append(float("-inf"))
                        if any(_s > float("-inf") for _s in scores):
                            idx = max(enumerate(scores), key=lambda kv: kv[1])[0]
                            chosen = cand_images[int(idx)]
                    except Exception:
                        # transformers CLIP fallback
                        try:
                            from transformers import CLIPProcessor, CLIPModel  # type: ignore
                            import torch as _torch2
                            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                            inputs = proc(text=[prompt]*len(cand_images), images=cand_images, return_tensors="pt", padding=True)
                            with _torch2.no_grad():
                                out = model(**inputs)
                                logits = out.logits_per_image.squeeze(-1) if out.logits_per_image.dim()>1 else out.logits_per_image
                            idx = int(_torch2.argmax(logits).item())
                            chosen = cand_images[idx]
                        except Exception:
                            chosen = cand_images[0]
                except Exception:
                    chosen = cand_images[0]
            else:
                chosen = cand_images[0] if cand_images else None

            if out_path and chosen is not None:
                # If result is a PIL image, save; if it's a path, just return
                try:
                    from PIL import Image  # type: ignore
                    if isinstance(chosen, Image.Image):  # type: ignore[attr-defined]
                        # Optional cross-modal verifier-based rejection before saving
                        if bool(cm_verifier) and text_embed is not None and CrossModalVerifier is not None:
                            try:
                                score = _cm_verifier_score_from_image(chosen, text_embed)
                                if float(score) < float(cm_threshold):
                                    return None
                            except Exception:
                                pass
                        p = Path(out_path)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        chosen.save(p)
                        return p
                except Exception:
                    pass
                try:
                    p = Path(str(chosen))
                    return p
                except Exception:
                    return None
            return None
        return None


def _cm_verifier_score_from_image(image, text_embed: torch.Tensor) -> float:
    """
    Compute a simple cross-modal score using the `CrossModalVerifier`. If unavailable,
    return 1.0 to keep the pipeline permissive by default.
    """
    if CrossModalVerifier is None:
        return 1.0
    try:
        # Prefer a real image embedding via VisionBackbone + PreAligner when available
        if VisionBackbone is not None and PreAligner is not None:
            import numpy as _np
            from PIL import Image as _PILImage  # type: ignore
            if isinstance(image, _PILImage.Image):
                arr = _np.array(image).astype(_np.float32) / 255.0
            else:
                arr = _np.array(_PILImage.open(str(image))).astype(_np.float32) / 255.0
            img = torch.from_numpy(arr).float().permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
            import os as _os
            vb = VisionBackbone(backend=_os.getenv("OMNICODER_VISION_BACKEND", "dinov3"), d_model=768, return_pooled=True).eval()
            with torch.no_grad():
                _, pooled = vb(img)
                if pooled is None:
                    pooled = vb(img)[0].mean(dim=1)
                ed = int(text_embed.size(-1))
                pal = PreAligner(embed_dim=ed, text_dim=ed, image_dim=768).eval()
                emb = pal(image=pooled)
                img_vec = emb.get("image", None)
            if img_vec is None:
                return 1.0
        else:
            # Fallback proxy: average RGB channels and tile to D
            import numpy as _np
            from PIL import Image as _PILImage  # type: ignore
            if isinstance(image, _PILImage.Image):
                arr = _np.array(image).astype(_np.float32) / 255.0
            else:
                arr = _np.array(_PILImage.open(str(image))).astype(_np.float32) / 255.0
            per_ch = arr.mean(axis=(0, 1))  # (3,)
            D = int(text_embed.size(-1))
            base = torch.from_numpy(per_ch).float()
            tiled = base.repeat(int(D // max(1, base.numel())) + 1)[:D]
            img_vec = torch.nn.functional.normalize(tiled, dim=0).unsqueeze(0)
        txt_vec = torch.nn.functional.normalize(text_embed, dim=-1)
        if txt_vec.dim() == 1:
            txt_vec = txt_vec.unsqueeze(0)
        cmv = CrossModalVerifier().eval()
        with torch.no_grad():
            s = cmv(txt_vec, img_vec)
        return float(s.squeeze().mean().item())
    except Exception:
        return 1.0


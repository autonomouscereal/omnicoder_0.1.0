from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

from .vision_encoder import VisionBackbone
from .audio_encoder import AudioBackbone as WavAudioBackbone
from .grounding import GroundingHead, SegmentationHead
from .latent3d import VoxelLatentHead, SimpleOrthoRenderer
from .video_heads import KeyframeHead
from .av_sync import AVSyncModule
from .vqvae import ImageVQVAE  # image VQ-VAE (codes)
from .video_vq import VideoVQ  # lightweight video VQ tokens
from .audio_vqvae import AudioVQVAE  # audio VQ-VAE (codes)
from .asr import ASRAdapter
from .tts import TTSAdapter
from .image_pipeline import ImageGenPipeline
from .aligner import PreAligner, HiddenToImageCond, ConceptLatentHead, ContinuousLatentHead
from ..gaussian.renderer import GaussianSplatRenderer3D, GaussianSplatRenderer2D  # type: ignore


class ModalityProjector(nn.Module):
    """Project vision/video feature dimension to the LLM's d_model."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in)
        return self.proj(x)


@dataclass
class FusionTokens:
    """
    Legacy container kept for backward-compat docs; actual learnable parameters are
    registered directly on the module (self.img_bos, etc.) so that .to(device)/.cuda()
    move them with the composer. Keeping this dataclass avoids breaking imports.
    """
    img_bos: Any | None = None
    img_eos: Any | None = None
    vid_bos: Any | None = None
    vid_eos: Any | None = None
    aud_bos: Any | None = None
    aud_eos: Any | None = None


class MelProjector(nn.Module):
    """Project mel features (B, T, M) to model dim with LayerNorm (aten-only ops)."""
    def __init__(self, in_mels: int = 80, out_dim: int = 384) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_mels), int(out_dim), bias=False)
        self.norm = nn.LayerNorm(int(out_dim))

    def forward(self, mel_btm: torch.Tensor) -> torch.Tensor:
        y = self.proj(mel_btm)
        return self.norm(y)


class MultimodalComposer(nn.Module):
    """
    Compose image/video embeddings with text token embeddings into a single sequence
    of features suitable for feeding directly into the LLM forward (which accepts
    already-embedded features of shape (B, T, C)).
    """

    def __init__(self, d_model: int, vision_dim: int, image_codebook_size: int = 8192, video_codebook_size: int = 8192, audio_codebook_size: int = 4096) -> None:
        super().__init__()
        self.projector = ModalityProjector(in_dim=vision_dim, out_dim=d_model)
        # Learned special tokens
        # Register learned delimiter tokens DIRECTLY on the module so device moves
        # propagate correctly. Previously these lived inside a plain dataclass
        # which prevented nn.Module from discovering/moving them, causing device
        # mismatches during fusion (CPU tokens vs CUDA embeddings) at cat().
        self.img_bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.img_eos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.vid_bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.vid_eos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.aud_bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.aud_eos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Keep a legacy placeholder for external references; unused by forward
        self.tokens = FusionTokens()
        self.vision = VisionBackbone(backend="auto", d_model=vision_dim, return_pooled=False)
        # lightweight video encoder built from the same vision backbone (CLS per frame)
        from .video_encoder import SimpleVideoEncoder
        self.video_enc = SimpleVideoEncoder(frame_encoder=self.vision, d_model=vision_dim)
        # Real audio backbones:
        # - WavAudioBackbone: raw PCM -> tokens/pooled
        # - MelProjector: precomputed mels (B,T,M) -> tokens via linear+norm
        self.audio_wav = WavAudioBackbone(sample_rate=16000, n_mels=80, d_model=vision_dim, return_pooled=False)
        self.audio_mel = MelProjector(in_mels=80, out_dim=vision_dim)
        # VQ token embeddings (indices -> vision_dim features)
        self.img_code_embed = nn.Embedding(int(image_codebook_size), int(vision_dim))
        self.vid_code_embed = nn.Embedding(int(video_codebook_size), int(vision_dim))
        self.aud_code_embed = nn.Embedding(int(audio_codebook_size), int(vision_dim))
        # Optional downstream multimodal heads (grounding/segmentation/3D latent)
        self.ground = GroundingHead(d_model=vision_dim, num_anchors=64)
        self.segment = SegmentationHead(d_model=vision_dim, hidden=max(128, vision_dim // 2))
        self.latent3d = VoxelLatentHead(d_model=vision_dim, depth=16, height=32, width=32, hidden=max(128, vision_dim // 2))
        self.renderer3d = SimpleOrthoRenderer(depth=16, out_h=224, out_w=224)
        # Video analysis heads
        self.video_keyframe = KeyframeHead(d_model=vision_dim)
        # Audio–Visual synchronization module (projects to common space and aligns)
        self.avsync = AVSyncModule(d_audio=vision_dim, d_video=vision_dim, d_model=vision_dim, num_heads=4)
        # Optional VQ encoders (best-effort init; kept lightweight)
        try:
            self.img_vq = ImageVQVAE(codebook_size=int(image_codebook_size), code_dim=vision_dim)
        except Exception:
            self.img_vq = None  # type: ignore[attr-defined]
        try:
            self.vid_vq = VideoVQ()
        except Exception:
            self.vid_vq = None  # type: ignore[attr-defined]
        try:
            self.aud_vq = AudioVQVAE(codebook_size=int(audio_codebook_size))
        except Exception:
            self.aud_vq = None  # type: ignore[attr-defined]
        # ASR/TTS/Image pipelines (optional adapters)
        self.asr = ASRAdapter(model_size="small")
        self.tts = TTSAdapter()
        self.img_gen = ImageGenPipeline()
        # Pre-alignment and conditioning heads
        self.prealign = PreAligner(
            embed_dim=256,
            text_dim=int(d_model),
            image_dim=int(vision_dim),
            audio_dim=int(vision_dim),
            video_dim=int(vision_dim),
        )
        self.hidden_to_img_cond = HiddenToImageCond(d_model=int(d_model), cond_dim=int(vision_dim), hidden_dim=max(int(vision_dim) * 2, 256))
        self.concept_head = ConceptLatentHead(d_model=int(d_model), embed_dim=256)
        self.continuous_latent = ContinuousLatentHead(d_model=int(d_model), latent_dim=16)
        # Gaussian renderers for 3D/2D primitives -> images (shared vision latent)
        self.gs3d = GaussianSplatRenderer3D(kernel=11)
        self.gs2d = GaussianSplatRenderer2D(kernel=11)

    def _repeat_tokens_for_batch(self, token: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Materialize (1,1,C) learned token across batch to (B,1,C) with aten.repeat_interleave
        # No device moves here; composer must be moved at setup time. This avoids broadcast views and respects compile/CG.
        return torch.ops.aten.repeat_interleave.self_int(token, int(batch_size), 0)

    def fuse_text_image(
        self,
        model_with_embed: nn.Module,
        input_ids: torch.Tensor,
        image_bchw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a fused features tensor (B, T_total, d_model):
          [IMG_BOS, proj(img_tokens)..., IMG_EOS, text_emb...]
        """
        img_tokens, _ = self.vision(image_bchw)
        bsz = image_bchw.shape[0]
        segs = [
            self._repeat_tokens_for_batch(self.img_bos, bsz),
            self.projector(img_tokens),
            self._repeat_tokens_for_batch(self.img_eos, bsz),
        ]
        text_emb = model_with_embed.embed(input_ids)
        segs.append(text_emb)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    def vision_heads(
        self,
        tokens_btC: torch.Tensor,
        text_emb_bC: Optional[torch.Tensor] = None,
        do_ground: bool = True,
        do_segment: bool = False,
        do_latent3d: bool = False,
    ) -> dict:
        out = {}
        if do_ground and text_emb_bC is not None:
            try:
                boxes, scores, t_scores = self.ground(tokens_btC, text_emb_bC)
                out.update({"boxes": boxes, "obj": scores, "text_scores": t_scores})
            except Exception:
                pass
        if do_segment:
            try:
                # caller should supply num_patches; infer via sqrt heuristic when tokens is a grid
                t = tokens_btC.shape[1]
                num_patches = int(torch.ops.aten.clamp_max.default(torch.ops.aten.sqrt.default(torch.tensor(t, dtype=torch.float32)), t))
                mask_lo = self.segment(tokens_btC, num_patches=num_patches)
                out.update({"mask_lo": mask_lo})
            except Exception:
                pass
        if do_latent3d:
            try:
                vox = self.latent3d(tokens_btC)
                rgb = self.renderer3d(vox)
                out.update({"vox": vox, "rgb": rgb})
            except Exception:
                pass
        return out

    def video_heads(self, video_tokens_btC: torch.Tensor, temperature: float = 1.0) -> dict:
        out = {}
        try:
            kf = self.video_keyframe(video_tokens_btC, temperature=float(temperature))
            out["keyframe_prob"] = kf
        except Exception:
            pass
        return out

    def av_sync(self, audio_tokens_bTC: torch.Tensor, video_tokens_bTC: torch.Tensor) -> dict:
        out = {}
        try:
            v_fused, align = self.avsync(audio_tokens_bTC, video_tokens_bTC)
            out.update({"video_fused": v_fused, "av_align": align})
        except Exception:
            pass
        return out

    # ----- OCR / VQ encode bridges (optional; not hot path) -----
    @torch.inference_mode()
    def encode_image_to_codes(self, image_bchw: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if getattr(self, 'img_vq', None) is None:
                return None
            # ImageVQVAE.encode expects (B,3,H,W) float in [0,1] and returns (B,T)
            codes_bt = self.img_vq.encode(image_bchw.clamp(0, 1))  # type: ignore[attr-defined]
            return codes_bt
        except Exception:
            return None

    @torch.inference_mode()
    def encode_video_to_codes(self, video_btchw: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if getattr(self, 'vid_vq', None) is None:
                return None
            # VideoVQ encodes per-frame patches to a 1D index stream using numpy/torch
            # Accept only CPU tensors here; caller can .cpu() before calling if needed
            v = video_btchw.detach().cpu().numpy()
            codes_list = self.vid_vq.encode(v)  # type: ignore[attr-defined]
            # Expect a single list of indices; convert to torch (B=1 assumed for simplicity)
            import numpy as _np
            if isinstance(codes_list, list) and len(codes_list) >= 1:
                arr = _np.asarray(codes_list[0], dtype=_np.int64)
                return torch.from_numpy(arr.reshape(1, -1))
            return None
        except Exception:
            return None

    @torch.inference_mode()
    def encode_audio_to_codes(self, audio_wav_bt: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if getattr(self, 'aud_vq', None) is None:
                return None
            x = audio_wav_bt.unsqueeze(1) if audio_wav_bt.dim() == 2 else audio_wav_bt
            # AudioVQVAE returns idx: (B, Tq)
            _rec, _com, _ppx, _xr, idx = self.aud_vq(x)  # type: ignore[attr-defined]
            return idx
        except Exception:
            return None

    # ----- Orchestration helpers (optional) -----
    @torch.inference_mode()
    def asr_to_text(self, audio_path_or_wav) -> Optional[str]:
        try:
            return self.asr.transcribe(audio_path_or_wav)
        except Exception:
            return None

    @torch.inference_mode()
    def tts_from_text(self, text: str, out_path: str = "weights/tts_out.wav") -> Optional[str]:
        try:
            p = self.tts.tts(text, out_path)
            return str(p) if p is not None else None
        except Exception:
            return None

    @torch.inference_mode()
    def generate_image(self, prompt: str, steps: int = 20, size: tuple[int, int] = (512, 512), out_path: Optional[str] = None) -> Optional[str]:
        try:
            p = self.img_gen.generate(prompt, steps=int(steps), size=(int(size[0]), int(size[1])), out_path=out_path)
            return str(p) if p is not None else None
        except Exception:
            return None

    def fuse_text_video(
        self,
        self_model: nn.Module,
        input_ids: torch.Tensor,
        video_btchw: torch.Tensor,
        max_frames: int = 16,
    ) -> torch.Tensor:
        bsz = video_btchw.shape[0]
        # Tracer-safe crop to first K frames without Tensor->Python conversions
        _F = video_btchw.shape[1]
        _k = torch.ops.aten.add.Scalar(torch.ops.aten.mul.Scalar(_F, 0), int(max_frames))
        _end = torch.ops.aten.minimum.default(_F, _k)
        video_btchw = torch.ops.aten.slice.Tensor(video_btchw, 1, 0, _end, 1)
        vid_tokens, _ = self.video_enc(video_btchw)
        segs = [
            self._repeat_tokens_for_batch(self.vid_bos, bsz),
            self.projector(vid_tokens),
            self._repeat_tokens_for_batch(self.vid_eos, bsz),
        ]
        text_emb = self_model.embed(input_ids)
        segs.append(text_emb)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    def fuse_text_image_codes(
        self,
        model_with_embed: nn.Module,
        input_ids: torch.Tensor,
        image_codes_bt: torch.Tensor,
    ) -> torch.Tensor:
        bsz = image_codes_bt.shape[0]
        tok = self.img_code_embed(image_codes_bt.long())
        segs = [
            self._repeat_tokens_for_batch(self.img_bos, bsz),
            self.projector(tok),
            self._repeat_tokens_for_batch(self.img_eos, bsz),
        ]
        text_emb = model_with_embed.embed(input_ids)
        segs.append(text_emb)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    def fuse_text_video_codes(
        self,
        model_with_embed: nn.Module,
        input_ids: torch.Tensor,
        video_codes_bt: torch.Tensor,
    ) -> torch.Tensor:
        bsz = video_codes_bt.shape[0]
        tok = self.vid_code_embed(video_codes_bt.long())
        segs = [
            self._repeat_tokens_for_batch(self.vid_bos, bsz),
            self.projector(tok),
            self._repeat_tokens_for_batch(self.vid_eos, bsz),
        ]
        text_emb = model_with_embed.embed(input_ids)
        segs.append(text_emb)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    def fuse_text_audio_codes(
        self,
        model_with_embed: nn.Module,
        input_ids: torch.Tensor,
        audio_codes_bt: torch.Tensor,
    ) -> torch.Tensor:
        bsz = audio_codes_bt.shape[0]
        tok = self.aud_code_embed(audio_codes_bt.long())
        segs = [
            self._repeat_tokens_for_batch(self.aud_bos, bsz),
            self.projector(tok),
            self._repeat_tokens_for_batch(self.aud_eos, bsz),
        ]
        text_emb = model_with_embed.embed(input_ids)
        segs.append(text_emb)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    @torch.inference_mode()
    def map_codes_to_vocab(self, codes_bt: torch.Tensor, offset: int) -> torch.Tensor:
        # Return codes + offset, shape preserved
        off = torch.ops.aten.add.Scalar(codes_bt, float(offset))
        return off.long()

    @torch.inference_mode()
    def decode_image_from_codes(self, image_codes_bt: torch.Tensor, grid_shape: tuple[int, int] | None = None) -> Optional[torch.Tensor]:
        try:
            if getattr(self, 'img_vq', None) is None:
                return None
            # Support B=1 for now; extend as needed
            idx = image_codes_bt.detach().cpu().numpy().astype('int32')
            if idx.ndim == 2 and idx.shape[0] == 1:
                from numpy import sqrt as _sqrt
                if grid_shape is None:
                    n = int(idx.shape[1])
                    s = int(_sqrt(n))
                    grid_shape = (s, s)
                img = self.img_vq.decode_indices(torch.from_numpy(idx[0]).long())  # type: ignore[attr-defined]
                return img
            return None
        except Exception:
            return None

    @torch.inference_mode()
    def decode_audio_from_codes(self, audio_codes_bt: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if getattr(self, 'aud_vq', None) is None:
                return None
            # aud_vq expects (B, Tq)
            idx = audio_codes_bt.long()
            # Reconstruct via embedding lookup + decoder
            z_q = self.aud_vq.quant.embedding.index_select(0, torch.ops.aten.reshape.default(idx, (-1,)))  # type: ignore[attr-defined]
            z_q = torch.ops.aten.reshape.default(z_q, (idx.size(0), idx.size(1), self.aud_vq.code_dim)).permute(0, 2, 1)  # type: ignore[attr-defined]
            xr = self.aud_vq.decoder(z_q)  # type: ignore[attr-defined]
            return xr
        except Exception:
            return None

    def build_router_conditioning(self) -> dict:
        # Return last pooled features and pre-aligned embeddings for router conditioning
        out: dict = {}
        try:
            pools = getattr(self, 'last_pools', {})
            if isinstance(pools, dict):
                out.update(pools)
                # Compute pre-aligned embeddings (best effort)
                pe: dict = {}
                try:
                    pe = self.prealign(
                        text=pools.get('text', None),
                        image=pools.get('image', None),
                        audio=pools.get('audio', None),
                        video=pools.get('video', None),
                    )
                except Exception:
                    pe = {}
                if isinstance(pe, dict):
                    out['prealign'] = pe
        except Exception:
            pass
        return out

    def conditioning_from_hidden(self, hidden_btC: torch.Tensor) -> dict:
        # Map hidden states to image conditioning vectors + FiLM params and concept/continuous latents
        out: dict = {}
        try:
            cond, scale, shift = self.hidden_to_img_cond(hidden_btC)
            out['image_cond'] = cond
            out['film_scale'] = scale
            out['film_shift'] = shift
        except Exception:
            pass
        try:
            out['concept'] = self.concept_head(hidden_btC)
        except Exception:
            pass
        try:
            out['continuous_latent'] = self.continuous_latent(hidden_btC)
        except Exception:
            pass
        return out

    def fuse_text_audio(
        self,
        model_with_embed: nn.Module,
        input_ids: torch.Tensor,
        audio_bmt: torch.Tensor,
    ) -> torch.Tensor:
        # Similar construction for audio
        aud_tokens = self.audio_mel(audio_bmt)
        bsz = audio_bmt.shape[0]
        segs = [
            self._repeat_tokens_for_batch(self.aud_bos, bsz),
            self.projector(aud_tokens),
            self._repeat_tokens_for_batch(self.aud_eos, bsz),
        ]
        text_emb = model_with_embed.embed(input_ids)
        segs.append(text_emb)
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    def fuse_all(
        self,
        model_with_embed: nn.Module,
        *,
        input_ids: Optional[torch.Tensor] = None,
        image_bchw: Optional[torch.Tensor] = None,
        video_btchw: Optional[torch.Tensor] = None,
        audio_bmt: Optional[torch.Tensor] = None,
        audio_wav_bt: Optional[torch.Tensor] = None,
        # Optional discrete VQ tokens (indices)
        image_codes_bt: Optional[torch.Tensor] = None,
        video_codes_bt: Optional[torch.Tensor] = None,
        audio_codes_bt: Optional[torch.Tensor] = None,
        # Optional Gaussian primitives for on-the-fly rendering
        gs3d_pos_bnh3: Optional[torch.Tensor] = None,
        gs3d_cov_bnh33: Optional[torch.Tensor] = None,
        gs3d_cov_diag_bnh3: Optional[torch.Tensor] = None,
        gs3d_rgb_bnh3: Optional[torch.Tensor] = None,
        gs3d_opa_bnh1: Optional[torch.Tensor] = None,
        gs3d_K_b33: Optional[torch.Tensor] = None,
        gs3d_R_b33: Optional[torch.Tensor] = None,
        gs3d_t_b3: Optional[torch.Tensor] = None,
        gs2d_mean_bng2: Optional[torch.Tensor] = None,
        gs2d_cov_diag_bng2: Optional[torch.Tensor] = None,
        gs2d_rgb_bng3: Optional[torch.Tensor] = None,
        gs2d_opa_bng1: Optional[torch.Tensor] = None,
        max_frames: int = 16,
    ) -> torch.Tensor:
        """
        Unified fusion for any subset of modalities. Order:
          [IMG] [VID] [AUD] [TEXT]
        Returns (B, T_total, d_model).
        """
        device = next(model_with_embed.parameters()).device
        segs: list[torch.Tensor] = []
        pools: dict[str, torch.Tensor] = {}
        bsz = None
        if image_bchw is not None:
            if image_bchw.device != device:
                raise RuntimeError("fuse_all: image must be on model device")
            if image_bchw.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise RuntimeError("fuse_all: image dtype must be float16/bfloat16/float32")
        if video_btchw is not None:
            if video_btchw.device != device:
                raise RuntimeError("fuse_all: video must be on model device")
            if video_btchw.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise RuntimeError("fuse_all: video dtype must be float16/bfloat16/float32")
        if audio_bmt is not None:
            if audio_bmt.device != device:
                raise RuntimeError("fuse_all: audio must be on model device")
            if audio_bmt.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise RuntimeError("fuse_all: audio dtype must be float16/bfloat16/float32")
        if image_bchw is not None:
            vis_tokens, _ = self.vision(image_bchw)
            segs.append(self._repeat_tokens_for_batch(self.img_bos, image_bchw.shape[0]))
            segs.append(self.projector(vis_tokens))
            segs.append(self._repeat_tokens_for_batch(self.img_eos, image_bchw.shape[0]))
            pools['image'] = torch.ops.aten.mean.dim(vis_tokens, [1, 2], False)
        # Gaussian 3D rendering path (to shared latent via vision backbone)
        if (gs3d_pos_bnh3 is not None) and (gs3d_rgb_bnh3 is not None) and (gs3d_opa_bnh1 is not None) and (gs3d_K_b33 is not None) and (gs3d_R_b33 is not None) and (gs3d_t_b3 is not None):
            B = gs3d_pos_bnh3.shape[0]
            H = int(image_bchw.shape[2]) if image_bchw is not None else 224
            W = int(image_bchw.shape[3]) if image_bchw is not None else 224
            rgb3d, depth3d = self.gs3d(gs3d_pos_bnh3, gs3d_cov_bnh33, gs3d_cov_diag_bnh3, gs3d_rgb_bnh3, gs3d_opa_bnh1, gs3d_K_b33, gs3d_R_b33, gs3d_t_b3, (H, W))
            vis_tokens3d, _ = self.vision(rgb3d)
            segs.append(self._repeat_tokens_for_batch(self.img_bos, B))
            segs.append(self.projector(vis_tokens3d))
            segs.append(self._repeat_tokens_for_batch(self.img_eos, B))
            pools['image'] = torch.ops.aten.mean.dim(vis_tokens3d, [1, 2], False)
        # Gaussian 2D rendering path
        if (gs2d_mean_bng2 is not None) and (gs2d_cov_diag_bng2 is not None) and (gs2d_rgb_bng3 is not None) and (gs2d_opa_bng1 is not None):
            B = gs2d_mean_bng2.shape[0]
            H = int(image_bchw.shape[2]) if image_bchw is not None else 224
            W = int(image_bchw.shape[3]) if image_bchw is not None else 224
            rgb2d = self.gs2d(gs2d_mean_bng2, gs2d_cov_diag_bng2, gs2d_rgb_bng3, gs2d_opa_bng1, (H, W))
            vis_tokens2d, _ = self.vision(rgb2d)
            segs.append(self._repeat_tokens_for_batch(self.img_bos, B))
            segs.append(self.projector(vis_tokens2d))
            segs.append(self._repeat_tokens_for_batch(self.img_eos, B))
            pools['image'] = torch.ops.aten.mean.dim(vis_tokens2d, [1, 2], False)
        if video_btchw is not None:
            # Only take up to max_frames by uniform subsampling to avoid large T
            bsz = video_btchw.shape[0]
            frames = video_btchw.shape[1]
            if frames > int(max_frames):
                step = max(1, frames // int(max_frames))
                video_btchw = video_btchw[:, ::step][:, :int(max_frames)]
            vid_tokens, _ = self.video_enc(video_btchw)
            segs.append(self._repeat_tokens_for_batch(self.vid_bos, bsz))
            segs.append(self.projector(vid_tokens))
            segs.append(self._repeat_tokens_for_batch(self.vid_eos, bsz))
            pools['video'] = torch.ops.aten.mean.dim(vid_tokens, [1, 2], False)
        if audio_bmt is not None:
            # Interpret provided audio as mel (B,T,M) and project
            aud_tokens = self.audio_mel(audio_bmt)
            bsz = audio_bmt.shape[0]
            segs.append(self._repeat_tokens_for_batch(self.aud_bos, bsz))
            segs.append(self.projector(aud_tokens))
            segs.append(self._repeat_tokens_for_batch(self.aud_eos, bsz))
            pools['audio'] = torch.ops.aten.mean.dim(aud_tokens, [1], False)
        if audio_wav_bt is not None:
            # Raw PCM (B,T) → tokens via real audio backbone
            wav_tokens, _wav_pooled = self.audio_wav(audio_wav_bt)
            bsz = audio_wav_bt.shape[0]
            segs.append(self._repeat_tokens_for_batch(self.aud_bos, bsz))
            segs.append(self.projector(wav_tokens))
            segs.append(self._repeat_tokens_for_batch(self.aud_eos, bsz))
            pools['audio_wav'] = torch.ops.aten.mean.dim(wav_tokens, [1, 2], False)
        # Optional: fuse pre-tokenized discrete codes via embedding layers
        if image_codes_bt is not None:
            # image_codes_bt: (B, T_codes) int64
            tok = self.img_code_embed(image_codes_bt.long())
            bsz = tok.shape[0]
            segs.append(self._repeat_tokens_for_batch(self.img_bos, bsz))
            segs.append(self.projector(tok))
            segs.append(self._repeat_tokens_for_batch(self.img_eos, bsz))
            pools['image_codes'] = torch.ops.aten.mean.dim(tok, [1, 2], False)
        if video_codes_bt is not None:
            tok = self.vid_code_embed(video_codes_bt.long())
            bsz = tok.shape[0]
            segs.append(self._repeat_tokens_for_batch(self.vid_bos, bsz))
            segs.append(self.projector(tok))
            segs.append(self._repeat_tokens_for_batch(self.vid_eos, bsz))
            pools['video_codes'] = torch.ops.aten.mean.dim(tok, [1, 2], False)
        if audio_codes_bt is not None:
            tok = self.aud_code_embed(audio_codes_bt.long())
            bsz = tok.shape[0]
            segs.append(self._repeat_tokens_for_batch(self.aud_bos, bsz))
            segs.append(self.projector(tok))
            segs.append(self._repeat_tokens_for_batch(self.aud_eos, bsz))
            pools['audio_codes'] = torch.ops.aten.mean.dim(tok, [1, 2], False)
        if input_ids is not None:
            text_emb = model_with_embed.embed(input_ids)
            segs.append(text_emb)
            pools['text'] = torch.ops.aten.mean.dim(text_emb, [1], False)
        # Persist last pools for modality-aware routing
        try:
            self.last_pools = pools  # type: ignore[attr-defined]
        except Exception:
            pass
        from omnicoder.utils.torchutils import safe_concat as _safe_concat  # type: ignore
        return _safe_concat(segs, 1)

    def compose(
        self,
        model_with_embed: nn.Module,
        *,
        input_ids: Optional[torch.Tensor] = None,
        image_bchw: Optional[torch.Tensor] = None,
        video_btchw: Optional[torch.Tensor] = None,
        audio_bmt: Optional[torch.Tensor] = None,
        audio_wav_bt: Optional[torch.Tensor] = None,
        image_codes_bt: Optional[torch.Tensor] = None,
        video_codes_bt: Optional[torch.Tensor] = None,
        audio_codes_bt: Optional[torch.Tensor] = None,
        gs3d_pos_bnh3: Optional[torch.Tensor] = None,
        gs3d_cov_bnh33: Optional[torch.Tensor] = None,
        gs3d_cov_diag_bnh3: Optional[torch.Tensor] = None,
        gs3d_rgb_bnh3: Optional[torch.Tensor] = None,
        gs3d_opa_bnh1: Optional[torch.Tensor] = None,
        gs3d_K_b33: Optional[torch.Tensor] = None,
        gs3d_R_b33: Optional[torch.Tensor] = None,
        gs3d_t_b3: Optional[torch.Tensor] = None,
        gs2d_mean_bng2: Optional[torch.Tensor] = None,
        gs2d_cov_diag_bng2: Optional[torch.Tensor] = None,
        gs2d_rgb_bng3: Optional[torch.Tensor] = None,
        gs2d_opa_bng1: Optional[torch.Tensor] = None,
        max_frames: int = 16,
        compute_heads: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        fused = self.fuse_all(
            model_with_embed=model_with_embed,
            input_ids=input_ids,
            image_bchw=image_bchw,
            video_btchw=video_btchw,
            audio_bmt=audio_bmt,
            audio_wav_bt=audio_wav_bt,
            image_codes_bt=image_codes_bt,
            video_codes_bt=video_codes_bt,
            audio_codes_bt=audio_codes_bt,
            gs3d_pos_bnh3=gs3d_pos_bnh3,
            gs3d_cov_bnh33=gs3d_cov_bnh33,
            gs3d_cov_diag_bnh3=gs3d_cov_diag_bnh3,
            gs3d_rgb_bnh3=gs3d_rgb_bnh3,
            gs3d_opa_bnh1=gs3d_opa_bnh1,
            gs3d_K_b33=gs3d_K_b33,
            gs3d_R_b33=gs3d_R_b33,
            gs3d_t_b3=gs3d_t_b3,
            gs2d_mean_bng2=gs2d_mean_bng2,
            gs2d_cov_diag_bng2=gs2d_cov_diag_bng2,
            gs2d_rgb_bng3=gs2d_rgb_bng3,
            gs2d_opa_bng1=gs2d_opa_bng1,
            max_frames=max_frames,
        )
        side: dict = {}
        if bool(compute_heads):
            try:
                pools = getattr(self, 'last_pools', {})
                if isinstance(pools, dict):
                    side['pools'] = pools
                    # Pre-aligned embeddings
                    try:
                        pe = self.prealign(
                            text=pools.get('text', None),
                            image=pools.get('image', None),
                            audio=pools.get('audio', None),
                            video=pools.get('video', None),
                        )
                        side['prealign'] = pe
                    except Exception:
                        pass
            except Exception:
                pass
        return fused, side



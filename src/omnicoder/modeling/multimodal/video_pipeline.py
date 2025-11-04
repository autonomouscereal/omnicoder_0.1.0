from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
try:
    from omnicoder.modeling.multimodal.aligner import CrossModalVerifier  # type: ignore
except Exception:
    CrossModalVerifier = None  # type: ignore
import torch


class VideoGenPipeline:
    """
    Minimal text-to-video wrapper with diffusers backends if available.

    Tries, in order:
      - TextToVideoSDPipeline (older SD video)
      - StableVideoDiffusionPipeline (image-to-video)

    If only image-to-video is available, this pipeline first generates a seed image
    via an image backend (optional) then animates it.
    """

    def __init__(
        self,
        backend: str = "diffusers",
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        hf_id: Optional[str] = None,
        local_path: Optional[str] = None,
        use_temporal_ssm: bool = True,
        temporal_ssm_dim: int = 384,
        temporal_ssm_kernel: int = 5,
        temporal_ssm_expansion: int = 2,
    ) -> None:
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.hf_id = hf_id
        self.local_path = local_path
        self._pipe = None
        self._mode = None  # 't2v' or 'i2v'
        self._temporal = None
        # Keep a small tail of previously generated frames to enable default
        # segment chaining across calls (long-form continuation)
        self._last_frames: List[np.ndarray] = []
        if use_temporal_ssm:
            try:
                from omnicoder.export.onnx_export_temporal import TemporalSSM  # type: ignore
                self._temporal = TemporalSSM(d_model=int(temporal_ssm_dim), kernel_size=int(temporal_ssm_kernel), expansion=int(temporal_ssm_expansion))
            except Exception:
                self._temporal = None

    def _load_diffusers(self) -> bool:
        try:
            from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline  # type: ignore
        except Exception:
            return False
        # Prefer a lightweight, distilled image-to-video model when none provided
        default_lite = "stabilityai/stable-video-diffusion-img2vid"
        model_ref = self.local_path if self.local_path else (self.hf_id or default_lite)
        if not model_ref:
            return False
        # Try T2V first
        try:
            pipe = TextToVideoSDPipeline.from_pretrained(model_ref, dtype=self.dtype)
            pipe = pipe.to(self.device)
            self._pipe = pipe
            self._mode = "t2v"
            return True
        except Exception:
            pass
        # Fallback to image-to-video
        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(model_ref, dtype=self.dtype)
            pipe = pipe.to(self.device)
            self._pipe = pipe
            self._mode = "i2v"
            return True
        except Exception:
            return False

    def ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        if self.backend == "diffusers":
            return self._load_diffusers()
        return False

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        steps: int = 25,
        size: Tuple[int, int] = (512, 320),  # width, height
        num_frames: int = 24,
        out_path: Optional[str] = None,
        seed_image: Optional[str] = None,
        temporal_filter: bool = True,
        temporal_alpha: float = 0.7,
        temporal_passes: int = 1,
        keyframe_cadence: int = 0,
        interp_strength: float = 0.5,
        onnx_video_dir: Optional[str] = None,
        onnx_provider: str = "CPUExecutionProvider",
        onnx_provider_profile: Optional[str] = None,
        cm_verifier: bool = False,
        cm_threshold: float = 0.6,
        text_embed: Optional[torch.Tensor] = None,
        continue_from: Optional[List[np.ndarray]] = None,
    ) -> Optional[Path]:
        if not self.ensure_loaded():
            return None
        # Default to chaining from the last segment when continue_from is not provided
        frames: List[np.ndarray] = list(continue_from) if isinstance(continue_from, list) else list(self._last_frames)
        if onnx_video_dir:
            # ORT i2v callable path
            try:
                import json, os
                from omnicoder.inference.runtimes.onnx_video_decode import ORTI2VCallable
                provider = onnx_provider
                provider_options = None
                if onnx_provider_profile and Path(onnx_provider_profile).exists():
                    try:
                        data = json.loads(Path(onnx_provider_profile).read_text(encoding='utf-8'))
                        provider = str(data.get('provider', provider))
                        provider_options = data.get('provider_options', None)
                    except Exception:
                        pass
                # If no seed image given, synthesize a simple gray frame
                if seed_image is None and len(frames) == 0:
                    h, w = size[1], size[0]
                    seed = np.full((1, 3, h, w), 0.5, dtype=np.float32)
                else:
                    if seed_image is None:
                        return None
                    from PIL import Image  # type: ignore
                    img = Image.open(seed_image).convert('RGB').resize((size[0], size[1]))
                    seed = np.array(img).astype(np.float32) / 255.0
                    seed = seed.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)
                ort_i2v = ORTI2VCallable(onnx_video_dir, provider=provider, provider_options=provider_options)
                frames = ort_i2v.generate(seed, num_frames)
                frames = [f for f in frames]
            except Exception:
                return None
        elif self._mode == "t2v":
            result = self._pipe(prompt=prompt, num_inference_steps=int(steps), num_frames=num_frames)
            frames = [np.array(f) for f in result.frames[0]]  # list of PIL images
        elif self._mode == "i2v":
            if seed_image is None:
                return None
            try:
                from PIL import Image  # type: ignore
                img = Image.open(seed_image).convert('RGB').resize((size[0], size[1]))
            except Exception:
                return None
            result = self._pipe(image=img, num_inference_steps=int(steps))
            frames = [np.array(f) for f in result.frames[0]]
        else:
            return None

        # Optional keyframe + interpolation pipeline (simple RGB-space interpolation)
        if int(keyframe_cadence) > 0 and len(frames) > 2 and int(num_frames) > len(frames):
            try:
                k = max(1, int(keyframe_cadence))
                keyframes = [frames[i] for i in range(0, len(frames), k)]
                if keyframes[-1] is not frames[-1]:
                    keyframes.append(frames[-1])
                # Interpolate linearly between keyframes to reach target length
                total = int(num_frames)
                segs = len(keyframes) - 1
                per = max(1, (total // segs))
                new_frames: List[np.ndarray] = []
                for s in range(segs):
                    a = keyframes[s].astype(np.float32)
                    b = keyframes[s+1].astype(np.float32)
                    steps = per if s < segs - 1 else (total - len(new_frames))
                    # Precompute easing weights for this segment
                    if steps <= 1:
                        tau_vals = [0.0]
                    else:
                        tau_vals = [t / float(steps - 1) for t in range(steps)]
                    tau_vals = [(1.0 - float(interp_strength)) * t + float(interp_strength) * (t * t * (3 - 2 * t)) for t in tau_vals]
                    for tau in tau_vals:
                        f = (1.0 - tau) * a + tau * b
                        new_frames.append(np.clip(f, 0, 255).astype(np.uint8))
                if len(new_frames) >= 2:
                    frames = new_frames[:total]
            except Exception:
                pass

        # Optional learned temporal SSM smoothing on RGB sequences (very lightweight proxy)
        if self._temporal is not None and len(frames) >= 2:
            try:
                # Convert frames to (T,C,H,W) float32 in [0,1]
                tchw = torch.ops.aten.to.dtype(torch.from_numpy(np.stack(frames, axis=0)).permute(0,3,1,2), torch.float32, False, False) / 255.0
                # Global average pool to features, run temporal SSM, and blend back as a simple per-frame gain (proxy)
                B = 1; T, C, H, W = tchw.shape
                feats = torch.ops.aten.reshape.default(tchw, (T, C, -1)).mean(dim=-1).unsqueeze(0)  # (1,T,C)
                feats_out = self._temporal(feats)  # (1,T,C)
                gains = torch.sigmoid(feats_out.squeeze(0)).unsqueeze(-1).unsqueeze(-1)  # (T,C,1,1)
                tchw = torch.clamp(tchw * gains, 0.0, 1.0)
                frames = [ ( (tchw[i].permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8) ) for i in range(T) ]
            except Exception:
                pass

        # Optional cross-modal verifier-based rejection (mini-CLIP style proxy)
        if bool(cm_verifier) and text_embed is not None and CrossModalVerifier is not None and len(frames) > 0:
            try:
                score = cm_verifier_score_from_frames(frames, text_embed)
                if float(score) < float(cm_threshold):
                    return None
            except Exception:
                pass

        # Temporal consistency post-filter (optical-flow guided blending)
        if temporal_filter and len(frames) >= 2:
            frames = self._apply_temporal_consistency(frames, alpha=float(temporal_alpha), passes=int(max(1, temporal_passes)))

        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                import cv2  # type: ignore
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cv2.VideoWriter(str(p), fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
                for fr in frames:
                    bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                    vw.write(bgr)
                vw.release()
                # Write a tiny sidecar with generation metadata
                try:
                    import json
                    side = {
                        "backend": self.backend,
                        "mode": self._mode,
                        "steps": int(steps),
                        "num_frames": int(num_frames),
                        "temporal_filter": bool(temporal_filter),
                        "temporal_alpha": float(temporal_alpha),
                        "temporal_passes": int(temporal_passes),
                        "keyframe_cadence": int(keyframe_cadence),
                        "interp_strength": float(interp_strength),
                        "size": [int(size[0]), int(size[1])],
                        "onnx_video_dir": str(onnx_video_dir) if onnx_video_dir else None,
                    }
                    Path(str(p) + ".json").write_text(json.dumps(side, indent=2), encoding='utf-8')
                except Exception:
                    pass
                # Remember a short tail for default continuation next call
                try:
                    self._last_frames = frames[-8:]
                except Exception:
                    self._last_frames = []
                return p
            except Exception:
                return None
        # Update chaining state even when not writing a file
        try:
            self._last_frames = frames[-8:]
        except Exception:
            self._last_frames = []
        return None

    def _apply_temporal_consistency(self, frames: List[np.ndarray], alpha: float = 0.7, passes: int = 1) -> List[np.ndarray]:
        """Reduce flicker using optical-flow warping + color normalization.

        - alpha: blend weight between current frame and warped previous frame
        - passes: apply multiple stabilization passes
        """
        try:
            import cv2  # type: ignore
        except Exception:
            return frames
        def _match_color(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
            # simple mean/std match in LAB space
            try:
                src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
                ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)
                for c in range(3):
                    s = src_lab[..., c]
                    r = ref_lab[..., c]
                    s_mean, s_std = float(s.mean()), float(s.std() + 1e-6)
                    r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
                    s = (s - s_mean) * (r_std / s_std) + r_mean
                    src_lab[..., c] = np.clip(s, 0, 255).astype(np.uint8)
                return cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
            except Exception:
                return src

        def _stabilize_once(frames_in: List[np.ndarray]) -> List[np.ndarray]:
            out: List[np.ndarray] = [frames_in[0]]
            # Use DIS optical flow (fast) when available; fallback to Farneback
            try:
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            except Exception:
                dis = None
            for i in range(1, len(frames_in)):
                prev = frames_in[i - 1]
                cur = frames_in[i]
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
                cur_gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
                if dis is not None:
                    flow = dis.calc(prev_gray, cur_gray, None)
                else:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                h, w = cur_gray.shape
                grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (grid_x + flow[..., 0]).astype(np.float32)
                map_y = (grid_y + flow[..., 1]).astype(np.float32)
                warped_prev = cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                # Color match warped prev to current frame to minimize residuals
                warped_prev = _match_color(warped_prev, cur)
                blended = (alpha * cur.astype(np.float32) + (1.0 - alpha) * warped_prev.astype(np.float32)).astype(np.uint8)
                out.append(blended)
            return out

        stabilized = frames
        for _ in range(max(1, passes)):
            stabilized = _stabilize_once(stabilized)
        return stabilized

    @staticmethod
    def interpolate_frames_linear(frames: List[np.ndarray], factor: int = 2) -> List[np.ndarray]:
        """
        ORT-friendly latent/frame interpolation by simple linear blend between neighbors.
        Expands T -> T * factor - (factor - 1). Keeps first/last frames. No optical flow.
        """
        if factor <= 1 or len(frames) < 2:
            return frames
        out: List[np.ndarray] = []
        for i, (a_frame, b_frame) in enumerate(zip(frames, frames[1:])):
            a = a_frame.astype(np.float32)
            b = b_frame.astype(np.float32)
            out.append(frames[i])
            for t in range(1, factor):
                w = t / float(factor)
                inter = ((1.0 - w) * a + w * b).astype(np.uint8)
                out.append(inter)
        out.append(frames[-1])
        return out


def cm_verifier_score_from_frames(frames: List[np.ndarray], text_embed: torch.Tensor) -> float:
    """
    Compute a simple cross-modal verifier score from a list of RGB frames and a
    normalized text embedding. Uses `CrossModalVerifier` when available; otherwise
    returns 1.0.
    """
    if CrossModalVerifier is None:
        return 1.0
    if not frames or not isinstance(frames, list):
        return 0.0
    # Build an image embedding by averaging frames and channels, then tile to D
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    per_ch = arr.mean(axis=(0, 1, 2))  # (3,)
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
 



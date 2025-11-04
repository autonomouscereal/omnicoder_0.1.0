import torch, torch.nn as nn
from omnicoder.utils.logger import get_logger
from typing import Optional, Tuple


class ViTTiny(nn.Module):
    """
    Minimal ViT-tiny style patch embed + class token. Used as a lightweight fallback
    when no external vision backbone is available. Returns a token sequence shaped
    for consumption by the core model.
    """

    def __init__(self, d_model: int = 384, patch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos = None  # optional positional encodings can be added later

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        feat = self.conv(x)  # (B,d_model,H/patch,W/patch)
        bsz, channels, height, width = feat.shape
        seq = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        cls = self.cls.expand(bsz, -1, -1)
        # aten-only concat to align with compile/CG safety rules
        return torch.ops.aten.cat.default((cls, seq), 1)


class VisionBackbone(nn.Module):
    """
    Pluggable vision backbone loader with a unified output interface.

    Preferred backbones (auto-detected if available):
    - timm: vit_tiny_patch16_224 (pretrained)
    Fallback: ViTTiny patch embed defined above.
    """

    def __init__(self, backend: str = "auto", d_model: int = 384, return_pooled: bool = True) -> None:
        super().__init__()
        self.return_pooled = return_pooled
        self.d_model = int(d_model)
        self.backend = None
        self.model: Optional[nn.Module] = None
        _log = get_logger("omnicoder.vision")
        import time as _t
        _t0_all = _t.perf_counter()
        # Track the originally requested backend to decide whether to hard error or fallback
        _requested_backend = backend

        # Allow environment override to lock a compact mobile backbone
        try:
            import os as _os
            env_backend = _os.getenv("OMNICODER_VISION_BACKEND", "").strip()
            if backend == "auto" and env_backend:
                backend = env_backend
            # Under pytest, prefer a tiny, cached backbone to keep tests <10s
            if backend == "auto" and _os.getenv("PYTEST_CURRENT_TEST") and not env_backend:
                # Try timm vit_tiny, else fall back to internal tiny
                backend = "timm_vit_tiny"
                try:
                    _log.info("[vision] pytest tiny-fastpath: selecting %s backend", backend)
                except Exception:
                    pass
        except Exception:
            pass

        # Try DINOv3 first when requested or when auto and available
        if self.model is None and (backend in ("dinov3", "auto")):
            try:
                import os as _os
                import torch
                # Allow selecting a specific variant via env (e.g., vit_large14, vit_base14)
                variant = _os.getenv("OMNICODER_DINOV3_VARIANT", "vit_base14").strip()
                hub_repo = _os.getenv("OMNICODER_DINOV3_REPO", "facebookresearch/dinov3").strip()
                # Common names we will try on torch.hub
                candidates = [
                    f"dinov3_{variant}",
                    # fallback guesses
                    "dinov3_vit_large14",
                    "dinov3_vit_base14",
                ]
                loaded = None
                for name in candidates:
                    try:
                        loaded = torch.hub.load(hub_repo, name, pretrained=True)  # type: ignore[attr-defined]
                        if loaded is not None:
                            break
                    except Exception:
                        continue
                if loaded is not None:
                    # Wrap to expose (tokens, pooled) like other backbones
                    core = loaded
                    class _Wrap(nn.Module):
                        def __init__(self, m: nn.Module):
                            super().__init__()
                            self.core = m
                        def forward(self, x: torch.Tensor):  # type: ignore
                            # Expect (B,3,H,W); many DINOv3 refs provide get_intermediate_layers or forward_features
                            try:
                                # Prefer forward_features style with tokens
                                feats = getattr(self.core, "forward_features", None)
                                if callable(feats):
                                    out = feats(x)
                                    # attempt to find token and pooled
                                    if isinstance(out, dict):
                                        tokens = out.get("x_norm_patchtokens", None)
                                        pooled = out.get("x_norm_clstoken", None)
                                        if tokens is None and pooled is not None:
                                            tokens = pooled.unsqueeze(1)
                                        if tokens is None:
                                            # fall back to using output as pooled
                                            pooled = out.get("pooled", pooled)
                                            tokens = (pooled if pooled is not None else x.mean(dim=(2,3))).unsqueeze(1)
                                        if pooled is None:
                                            pooled = tokens[:, 0, :]
                                        return tokens, pooled
                                    # non-dict output: treat as pooled
                                    pooled = out
                                    tokens = pooled.unsqueeze(1)
                                    return tokens, pooled
                            except Exception:
                                pass
                            try:
                                # Many DINO-style models expose get_intermediate_layers
                                g = getattr(self.core, "get_intermediate_layers", None)
                                if callable(g):
                                    # use last layer outputs
                                    out = g(x, n=1)[0]
                                    # out: (B,T,C) with CLS at index 0 for ViT
                                    if out.dim() == 3:
                                        tokens = out
                                        pooled = out[:, 0, :]
                                        return tokens, pooled
                            except Exception:
                                pass
                            # Fallback: standard forward returns pooled
                            pooled = self.core(x)
                            if isinstance(pooled, (tuple, list)):
                                pooled = pooled[0]
                            tokens = pooled.unsqueeze(1) if pooled.dim() == 2 else pooled
                            if tokens.dim() == 4:
                                b, c, h, w = tokens.shape
                                tokens = torch.ops.aten.reshape.default(tokens, (b, c, h*w)).transpose(1, 2)
                                pooled = tokens[:, 0, :]
                            else:
                                pooled = tokens[:, 0, :] if tokens.dim() == 3 else pooled
                            return tokens, pooled
                    self.model = _Wrap(core)
                    self.backend = "dinov3"
                    # If hidden size differs, add projection to target d_model
                    try:
                        hidden = getattr(core, "embed_dim", None) or getattr(core, "num_features", None)
                        if hidden is not None and int(hidden) != int(d_model):
                            self.proj = nn.Linear(int(hidden), int(d_model), bias=False)
                        else:
                            self.proj = None
                    except Exception:
                        self.proj = None
                    try:
                        _dt = _t.perf_counter() - _t0_all
                        _log.info("[vision] dinov3 load dt=%.3fs", float(_dt))
                        if _dt > 10.0:
                            _log.warning("[vision] slow_step dinov3 load took %.3fs (>10s)", float(_dt))
                    except Exception:
                        pass
            except Exception:
                # Fall through to other options
                pass

        # Prefer a truly mobile variant when requested
        if backend in ("timm_mobilevit_s", "timm_mobilevit_xs", "timm_efficientvit_lite0", "auto"):
            try:
                import timm  # type: ignore
                model_name = None
                if backend.startswith("timm_mobilevit") or backend == "auto":
                    for cand in ("mobilevit_s", "mobilevit_xs"):
                        if cand in timm.list_models(pretrained=True):
                            model_name = cand
                            break
                if model_name is None and (backend == "timm_efficientvit_lite0" or backend == "auto"):
                    if "efficientvit_lite0" in timm.list_models(pretrained=True):
                        model_name = "efficientvit_lite0"
                if model_name is not None:
                    m = timm.create_model(model_name, pretrained=True, features_only=False)
                    self.model = m
                    self.backend = f"timm_{model_name}"
                    try:
                        _dt = _t.perf_counter() - _t0_all
                        _log.info("[vision] timm %s load dt=%.3fs", str(model_name), float(_dt))
                        if _dt > 10.0:
                            _log.warning("[vision] slow_step timm %s load took %.3fs (>10s)", str(model_name), float(_dt))
                    except Exception:
                        pass
            except Exception:
                # fall through to next options
                pass

        # Try DINOv2/ViT-L style backbones when available (safe best-effort)
        if self.model is None and (backend in ("timm_vit_l", "timm_dinov2", "auto")):
            try:
                import timm  # type: ignore
                # Best-effort DINOv2 pick when present
                dinov2 = None
                try:
                    for name in timm.list_models(pretrained=True):
                        if "dinov2" in name and ("vit_large" in name or "vit_base" in name):
                            dinov2 = name
                            break
                except Exception:
                    dinov2 = None
                if backend == "timm_dinov2" and dinov2 is not None:
                    m = timm.create_model(dinov2, pretrained=True)
                    self.model = m
                    self.backend = f"timm_{dinov2}"
                if self.model is None:
                    # Prefer a larger ViT if present
                    for cand in ("vit_large_patch14_224", "vit_base_patch16_224"):
                        if cand in timm.list_models(pretrained=True):
                            m = timm.create_model(cand, pretrained=True)
                            self.model = m
                            self.backend = f"timm_{cand}"
                            break
            except Exception:
                pass

        # Try SigLIP (if transformers provides it) as a modern alternative to CLIP
        if self.model is None and backend in ("siglip", "auto"):
            try:
                from transformers import SiglipVisionModel, SiglipProcessor  # type: ignore
                # Load a compact SigLIP vision tower; keep CPU eval for feature extraction
                m = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
                m.eval()
                # Wrap to expose a ViT-like interface returning tokens and pooled
                class _Wrap(nn.Module):
                    def __init__(self, core: nn.Module):
                        super().__init__()
                        self.core = core
                    def forward(self, x: torch.Tensor):
                        # Expect x in (B,3,H,W) float32 [0,1]; SigLIP expects pixel values via processor
                        # For simplicity, approximate preprocessing: normalize to [-1,1]
                        px = (x * 2.0) - 1.0
                        out = self.core(pixel_values=px)
                        # outputs: last_hidden_state (B, T, C), pooler_output may be absent
                        tokens = out.last_hidden_state
                        pooled = tokens[:, 0, :]
                        return tokens, pooled
                self.model = _Wrap(m)
                self.backend = "siglip_base_224"
                # Optional projection when feature dims differ
                out_dim = getattr(m.config, "hidden_size", d_model)
                self.proj = nn.Linear(out_dim, d_model, bias=False) if out_dim != d_model else None
                try:
                    _dt = _t.perf_counter() - _t0_all
                    _log.info("[vision] siglip load dt=%.3fs", float(_dt))
                    if _dt > 10.0:
                        _log.warning("[vision] slow_step siglip load took %.3fs (>10s)", float(_dt))
                except Exception:
                    pass
            except Exception:
                pass

        if self.model is None and (backend == "timm_vit_tiny" or backend == "auto"):
            try:
                import timm  # type: ignore
                _t_timm0 = _t.perf_counter()
                try:
                    # First try with pretrained weights (may trigger HF Hub)
                    m = timm.create_model("vit_tiny_patch16_224", pretrained=True)
                    self.model = m
                    self.backend = "timm_vit_tiny"
                except Exception as e:
                    # Offline-safe fallback: build the architecture without pretrained weights
                    try:
                        _log.warning("[vision] timm vit_tiny pretrained load failed: %s; falling back to pretrained=False", str(e))
                    except Exception:
                        pass
                    try:
                        m = timm.create_model("vit_tiny_patch16_224", pretrained=False)
                        self.model = m
                        self.backend = "timm_vit_tiny"
                    except Exception as e2:
                        # If explicitly requested by caller and both paths fail, re-raise; otherwise fall through to internal tiny
                        if _requested_backend == "timm_vit_tiny":
                            raise e2
                try:
                    _dt = _t.perf_counter() - _t_timm0
                    _log.info("[vision] timm vit_tiny load dt=%.3fs pretrained=%s", float(_dt), "unknown" if not hasattr(self, "model") else "n/a")
                    if _dt > 10.0:
                        _log.warning("[vision] slow_step timm vit_tiny load took %.3fs (>10s)", float(_dt))
                except Exception:
                    pass
            except Exception:
                # Fall through to internal tiny
                pass
        if self.model is None:
            # Fallback to internal tiny patch embed
            self.model = ViTTiny(d_model=d_model, patch=16)
            self.backend = "internal_vit_tiny"
            try:
                _log.info("[vision] fallback internal_vit_tiny selected")
            except Exception:
                pass

        # Optional linear projection to a target dimension
        self.proj: Optional[nn.Module]
        if self.backend and self.backend.startswith("timm_"):
            out_dim = getattr(self.model, "num_features", d_model)
            if out_dim != d_model:
                self.proj = nn.Linear(out_dim, d_model, bias=False)
            else:
                self.proj = None
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          - tokens: (B, T, C)
          - pooled: (B, C) if return_pooled else None
        """
        # No device/dtype validation or moves in hot path
        if self.backend and self.backend.startswith("timm_"):
            # timm ViT returns a class feature on forward_features
            try:
                feats = self.model.forward_features(x)
            except Exception:
                feats = self.model(x)
            # Try to get token sequence if available
            tokens = None
            pooled = None
            if isinstance(feats, dict):
                pooled = feats.get("x_norm_clstoken", None)
                tokens = feats.get("x_norm_patchtokens", None)
                if tokens is not None:
                    tokens = tokens  # (B, T, C)
                if pooled is None:
                    # Some timm models return a flat tensor
                    pooled = feats.get("pooled", None)
            if tokens is None:
                # Handle common timm forward_features outputs
                if isinstance(feats, torch.Tensor):
                    if feats.dim() == 3:
                        # (B, T, C)
                        tokens = feats
                        pooled = feats[:, 0, :]
                    elif feats.dim() == 2:
                        # (B, C) pooled
                        pooled = feats
                        tokens = pooled.unsqueeze(1)
                    elif feats.dim() == 4:
                        # (B, C, H, W) -> flatten spatial to tokens (B, HW, C)
                        b, c, h, w = feats.shape
                        tokens = torch.ops.aten.reshape.default(feats, (b, c, h * w)).transpose(1, 2)
                        pooled = tokens[:, 0, :]
                    else:
                        # Unknown tensor rank; synthesize minimal tokens
                        pooled = feats
                        tokens = feats.unsqueeze(1)
                else:
                    # If only pooled returned, expand to tokens via a single token
                    pooled = feats if pooled is None else pooled
                    tokens = pooled.unsqueeze(1)
            if self.proj is not None:
                tokens = self.proj(tokens)
                pooled = self.proj(pooled) if pooled is not None else None
            return tokens, pooled if self.return_pooled else None
        else:
            # Internal or wrapped backbones. They may return:
            #  - tokens tensor (B,T,C)
            #  - tuple(list) of (tokens, pooled)
            #  - pooled tensor (B,C) or (B,T,C) in rare cases
            out = self.model(x)
            tokens: torch.Tensor
            pooled: Optional[torch.Tensor]
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                a = out[0]
                b = out[1] if len(out) > 1 else None
                if isinstance(a, torch.Tensor) and a.dim() == 3:
                    tokens = a
                    pooled = (b if isinstance(b, torch.Tensor) else tokens[:, 0, :])
                elif isinstance(a, torch.Tensor) and a.dim() == 2:
                    pooled = a
                    tokens = pooled.unsqueeze(1)
                else:
                    # Fallback: try b as tokens
                    if isinstance(b, torch.Tensor) and b.dim() == 3:
                        tokens = b
                        pooled = tokens[:, 0, :]
                    else:
                        # Last resort: flatten spatial if 4D
                        t = a if isinstance(a, torch.Tensor) else torch.ops.aten.new_zeros.default(x, (x.shape[0], 1, 1), dtype=x.dtype)
                        if t.dim() == 4:
                            bsz, ch, h, w = t.shape
                            tokens = torch.ops.aten.reshape.default(t, (bsz, ch, h * w)).transpose(1, 2)
                            pooled = tokens[:, 0, :]
                        else:
                            tokens = t.unsqueeze(1) if t.dim() == 2 else t
                            pooled = tokens[:, 0, :] if tokens.dim() == 3 else None
            elif isinstance(out, torch.Tensor):
                if out.dim() == 3:
                    tokens = out
                    pooled = out[:, 0, :]
                elif out.dim() == 4:
                    bsz, ch, h, w = out.shape
                    tokens = torch.ops.aten.reshape.default(out, (bsz, ch, h * w)).transpose(1, 2)
                    pooled = tokens[:, 0, :]
                else:
                    pooled = out if out.dim() == 2 else None
                    tokens = pooled.unsqueeze(1) if pooled is not None else out
            else:
                # Unknown type; synthesize minimal outputs (anchor allocation to input tensor; no device/dtype kwargs)
                _feat_dim = self.d_model
                try:
                    if self.proj is not None:
                        _feat_dim = int(getattr(self.proj, 'out_features', self.d_model))
                except Exception:
                    _feat_dim = self.d_model
                tokens = torch.ops.aten.new_zeros.default(x, (x.shape[0], 1, _feat_dim), dtype=x.dtype)
                pooled = tokens[:, 0, :]
            if self.proj is not None:
                tokens = self.proj(tokens)
                pooled = self.proj(pooled) if pooled is not None else None
            return tokens, pooled if self.return_pooled else None

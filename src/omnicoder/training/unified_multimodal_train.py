from __future__ import annotations

"""
Unified multimodal trainer hooks for adding diffusion-based text denoising stages.

We keep this minimal and composable: add_diffusion_text_stage(plan) injects a small
stage that runs denoising on available text datasets using the main model as a
denoiser in embedding space, without altering the rest of the pipeline.
"""

from typing import Any, Dict, List

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore


def _build_diffusion_ministage() -> Dict[str, Any]:
    return {
        'name': 'diffusion_text_denoise',
        'minutes': 5,
        'steps': 200,
        'batch_size': 4,
        'seq_len': 128,
        'device': 'cuda',
        'data': 'examples',
    }


def add_diffusion_text_stage(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        # Insert right after pretrain stage if present; else append near start
        idx = 0
        for i, st in enumerate(plan):
            if str(st.get('name','')).lower().startswith('pretrain'):
                idx = i + 1
                break
        plan.insert(idx, _build_diffusion_ministage())
    except Exception:
        try:
            plan.append(_build_diffusion_ministage())
        except Exception:
            pass
    return plan

"""
Unified multimodal trainer: trains a single model over mixed inputs/targets in a shared latent space.

Input JSONL schema (each line, any subset of fields may be present):
  {"text": str?, "image": str?, "video": str?, "audio": str?, "caption": str?, "question": str?, "answer": str?}

Behavior:
- Builds a fused input sequence using MultimodalComposer for provided image/video features plus text ids.
- Computes supervised losses on provided targets:
  - Text targets (caption or QA answer): cross-entropy on text segment.
  - Image/video/audio targets: if model has latent heads, compute MSE or L2 loss on projected latents vs. reference latents (when available).
    For simplicity, this trainer focuses on text targets by default; latent targets are best-effort if heads exist and refs present.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image  # type: ignore
import numpy as np  # ensure np is available module-wide

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.config import get_mobile_preset
from .data.gaussian_jsonl import GaussianJSONL  # type: ignore


class MixedDataset(Dataset[Dict[str, Any]]):
    def __init__(self, jsonl_path: str, image_size: Tuple[int, int] = (224, 224), max_frames: int = 16) -> None:
        self.path = Path(jsonl_path)
        self.rows: List[Dict[str, Any]] = []
        self.image_size = image_size
        self.max_frames = max_frames
        for ln in self.path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            self.rows.append(obj)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB').resize(self.image_size)
        arr = torch.from_numpy(np.array(img).astype('float32') / 255.0).permute(2, 0, 1)  # (3,H,W)
        return arr

    def _load_video(self, path: str) -> torch.Tensor:
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(path)
            frames: List[torch.Tensor] = []
            ok = True
            while ok and len(frames) < 64:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
                t = torch.from_numpy(frame.astype('float32') / 255.0).permute(2, 0, 1)
                frames.append(t)
            cap.release()
        except Exception:
            frames = []
        if not frames:
            frames = [torch.zeros(3, *self.image_size, dtype=torch.float32)]
        if len(frames) > self.max_frames:
            idx = np.linspace(0, len(frames) - 1, num=self.max_frames).round().astype(int).tolist()
            frames = [frames[i] for i in idx]
        return torch.stack(frames, dim=0)  # (T,3,H,W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        if isinstance(row.get('image'), str):
            p = row['image']
            try:
                row['_image'] = self._load_image(p)
            except Exception:
                row['_image'] = None
        if isinstance(row.get('video'), str):
            p = row['video']
            try:
                row['_video'] = self._load_video(p)
            except Exception:
                row['_video'] = None
        # Audio loading omitted here (depends on tokenizer/codebook); can be added with EnCodec later
        return row


class MixedGaussianDataset(Dataset[Dict[str, Any]]):
    """
    Wrapper that merges a base mixed JSONL with a Gaussian JSONL by simple concatenation of
    samples; each __getitem__ returns either a vision-language sample or a Gaussian sample
    normalized to the composer.fuse_all arguments.
    """
    def __init__(self, base_jsonl: str, gaussian_jsonl: str | None, image_size: Tuple[int, int] = (224, 224), max_frames: int = 16) -> None:
        self.base = MixedDataset(base_jsonl, image_size=image_size, max_frames=max_frames)
        self.gds = GaussianJSONL(gaussian_jsonl) if (gaussian_jsonl and len(gaussian_jsonl) > 0 and Path(gaussian_jsonl).exists()) else None
        self._len = len(self.base) + (len(self.gds) if self.gds is not None else 0)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < len(self.base):
            return self.base[idx]
        j = idx - len(self.base)
        return self.gds[j] if self.gds is not None else self.base[idx % len(self.base)]


def build_model(preset_name: str, seq_len_hint: int) -> OmniTransformer:
    preset = get_mobile_preset(preset_name)
    return OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=max(preset.max_seq_len, seq_len_hint),
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified multimodal training over mixed inputs/targets in shared latent space")
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--gaussian_jsonl", type=str, default=os.getenv("OMNICODER_GAUSSIAN_JSONL", ""))
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb", choices=["mobile_4gb","mobile_2gb"]) 
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--out", type=str, default="weights/omnicoder_unified_mm.pt")
    args = ap.parse_args()

    # If a Gaussian JSONL is provided, use the mixed wrapper; else plain mixed
    if args.gaussian_jsonl and Path(args.gaussian_jsonl).exists():
        ds = MixedGaussianDataset(args.jsonl, args.gaussian_jsonl, image_size=tuple(args.image_size), max_frames=int(args.max_frames))
    else:
        ds = MixedDataset(args.jsonl, image_size=tuple(args.image_size), max_frames=int(args.max_frames))
    if len(ds) == 0:
        raise SystemExit("unified_multimodal_train: dataset empty")
    # Custom collate: return list of per-sample dicts as-is to avoid default_collate errors on None values
    def _collate_passthrough(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [b for b in batch if isinstance(b, dict)]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, collate_fn=_collate_passthrough)

    model = build_model(args.mobile_preset, seq_len_hint=4096)
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
    model.to(args.device).train()
    composer.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    # Contrastive alignment (InfoNCE) for text↔image/video/audio when present
    def _info_nce(a: torch.Tensor, b: torch.Tensor, t: float = 0.07) -> torch.Tensor:
        # a,b: (B,C)
        a = nn.functional.normalize(a, dim=-1)
        b = nn.functional.normalize(b, dim=-1)
        # Scale by temperature using aten-only scalar tensor to avoid Python max()
        _zero = torch.ops.aten.mul.Scalar(a, 0.0)
        _t = torch.ops.aten.add.Scalar(_zero.sum(), float(t))
        _t = torch.ops.aten.clamp_min.default(_t, 1e-6)
        logits = torch.ops.aten.div.Tensor((a @ b.t()), _t)
        # Build labels 0..B-1 using aten-only ops, anchored to a; avoid .size
        _basev = torch.ops.aten.sum.dim_IntList(a, [1], False)  # (B,)
        labels = torch.ops.aten.ones_like.default(_basev, dtype=torch.long)
        labels = torch.ops.aten.cumsum.default(labels, 0)
        labels = torch.ops.aten.sub.Tensor(labels, torch.ops.aten.new_ones.default(labels, labels.shape, dtype=labels.dtype))
        return nn.functional.cross_entropy(logits, labels)

    tok = get_text_tokenizer(prefer_hf=True)
    step = 0
    for row in dl:
        # DataLoader collates dicts into lists; extract single sample when batch_size=1 for simplicity
        if isinstance(row, list) and len(row) > 0 and isinstance(row[0], dict):
            row = row[0]
        text_in = str(row.get('text') or row.get('prompt') or row.get('question') or '')
        answer = str(row.get('answer') or row.get('caption') or '')
        has_img = isinstance(row.get('_image'), torch.Tensor)
        has_vid = isinstance(row.get('_video'), torch.Tensor)
        images = row.get('_image') if has_img else None
        videos = row.get('_video') if has_vid else None
        # Gaussian arguments (already batched by dataset wrapper)
        gs3d_args = {k: row.get(k) for k in ("gs3d_pos_bnh3","gs3d_cov_bnh33","gs3d_cov_diag_bnh3","gs3d_rgb_bnh3","gs3d_opa_bnh1","gs3d_K_b33","gs3d_R_b33","gs3d_t_b3")}
        gs3d_hw = row.get("gs3d_hw")
        gs2d_args = {k: row.get(k) for k in ("gs2d_mean_bng2","gs2d_cov_diag_bng2","gs2d_rgb_bng3","gs2d_opa_bng1")}
        gs2d_hw = row.get("gs2d_hw")

        # Build text inputs (teacher forcing on answer when present)
        if answer:
            prompt = (text_in + "\n" if text_in else "") + "A:"
            inp_ids = torch.tensor(tok.encode(prompt), dtype=torch.long, device=args.device).unsqueeze(0)
            tgt_ids = torch.tensor(tok.encode(answer), dtype=torch.long, device=args.device).unsqueeze(0)
        else:
            inp_ids = torch.tensor(tok.encode(text_in or ""), dtype=torch.long, device=args.device).unsqueeze(0)
            tgt_ids = None

        # Compose features in a single latent space
        fused = None
        if any(isinstance(v, torch.Tensor) for v in gs3d_args.values() if v is not None) or any(isinstance(v, torch.Tensor) for v in gs2d_args.values() if v is not None):
            # Gaussian-only or Gaussian+text branch
            # Select H,W from provided sizes or fallback to args.image_size
            H3, W3 = (gs3d_hw if isinstance(gs3d_hw, tuple) else tuple(args.image_size))
            H2, W2 = (gs2d_hw if isinstance(gs2d_hw, tuple) else tuple(args.image_size))
            # Compose Gaussian renders through composer
            # Ensure Gaussian tensors match model device before fusion (prevents CPU/CUDA mismatch)
            dev = args.device
            def _to_dev(t):
                return t.to(dev) if isinstance(t, torch.Tensor) else t
            gs3d_pos_bnh3 = _to_dev(gs3d_args.get("gs3d_pos_bnh3"))
            gs3d_cov_bnh33 = _to_dev(gs3d_args.get("gs3d_cov_bnh33"))
            gs3d_cov_diag_bnh3 = _to_dev(gs3d_args.get("gs3d_cov_diag_bnh3"))
            gs3d_rgb_bnh3 = _to_dev(gs3d_args.get("gs3d_rgb_bnh3"))
            gs3d_opa_bnh1 = _to_dev(gs3d_args.get("gs3d_opa_bnh1"))
            gs3d_K_b33 = _to_dev(gs3d_args.get("gs3d_K_b33"))
            gs3d_R_b33 = _to_dev(gs3d_args.get("gs3d_R_b33"))
            gs3d_t_b3 = _to_dev(gs3d_args.get("gs3d_t_b3"))
            gs2d_mean_bng2 = _to_dev(gs2d_args.get("gs2d_mean_bng2"))
            gs2d_cov_diag_bng2 = _to_dev(gs2d_args.get("gs2d_cov_diag_bng2"))
            gs2d_rgb_bng3 = _to_dev(gs2d_args.get("gs2d_rgb_bng3"))
            gs2d_opa_bng1 = _to_dev(gs2d_args.get("gs2d_opa_bng1"))
            fused = composer.fuse_all(
                model_with_embed=model,
                input_ids=inp_ids,
                image_bchw=None,
                video_btchw=None,
                audio_bmt=None,
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
                max_frames=int(args.max_frames),
            )
        elif has_vid:
            videos = videos.to(args.device).unsqueeze(0) if videos.dim() == 4 else videos.to(args.device)
            fused = composer.fuse_text_video(model_with_embed=model, input_ids=inp_ids, video_btchw=videos, max_frames=min(videos.size(1), int(args.max_frames)))
        elif has_img:
            fused = composer.fuse_text_image(model_with_embed=model, input_ids=inp_ids, image_bchw=images.unsqueeze(0).to(args.device))
        else:
            fused = inp_ids

        # Call model correctly: pass token ids via input_ids, feature tensors via prefix_hidden
        if isinstance(fused, torch.Tensor) and fused.dtype == torch.long and fused.dim() == 2:
            outputs = model(input_ids=fused, use_cache=False, return_hidden=True)  # type: ignore[arg-type]
        else:
            outputs = model(input_ids=None, use_cache=False, return_hidden=True, prefix_hidden=fused)  # type: ignore[arg-type]
        # Unpack logits and (optional) hidden robustly
        if isinstance(outputs, torch.Tensor):
            logits = outputs
            hidden = None
        else:
            logits = outputs[0]
            h_last = outputs[-1]
            hidden = h_last if (isinstance(h_last, torch.Tensor) and h_last.dim() == 3) else None

        loss = None
        if tgt_ids is not None:
            bsz, t_fused, vocab = logits.shape
            # Align label positions to the end of the fused sequence
            ans_len = tgt_ids.size(1)
            ans_logits = logits[:, t_fused - ans_len - 1 : t_fused - 1, :]
            loss = ce(ans_logits.reshape(-1, vocab), tgt_ids.reshape(-1))
        # Add contrastive alignment when hidden and modalities exist
        if hidden is not None and (has_img or has_vid or isinstance(row.get('_audio'), torch.Tensor)):
            # Simple CLS-like pooling: last hidden state mean over time
            pooled_txt = hidden.mean(dim=1)
            # Use pools recorded by the composer for modalities present this step
            pools = getattr(composer, 'last_pools', {}) if composer is not None else {}
            selected: torch.Tensor | None = None
            if has_img and isinstance(pools, dict) and isinstance(pools.get('image'), torch.Tensor):
                selected = pools['image']
            elif has_vid and isinstance(pools, dict) and isinstance(pools.get('video'), torch.Tensor):
                selected = pools['video']
            elif isinstance(row.get('_audio'), torch.Tensor) and isinstance(pools, dict) and isinstance(pools.get('audio'), torch.Tensor):
                selected = pools['audio']
            # Best-effort single InfoNCE (skip if batch<2)
            if selected is not None and int(pooled_txt.size(0)) >= 2:
                loss = (loss if loss is not None else 0.0) + 0.05 * _info_nce(pooled_txt, selected)

        if loss is None:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")
        if step >= args.steps:
            break

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {
        'model_config': {
            'vocab_size': int(getattr(model, 'vocab_size', 0)),
            'n_layers': int(getattr(model, 'n_layers', 0)),
            'd_model': int(getattr(model, 'd_model', 0)),
            'max_seq_len': int(getattr(model, 'max_seq_len', 0)),
        },
        'train_args': {
            'jsonl': str(args.jsonl),
            'gaussian_jsonl': str(args.gaussian_jsonl),
            'image_size': [int(args.image_size[0]), int(args.image_size[1])],
            'max_frames': int(args.max_frames),
        },
    }
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = Path(args.out)
    # Micro-evals per modality present in dataset for best tracking (proxy CE/contrastive)
    try:
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        from omnicoder.utils.evalhooks import clipscore_mean  # type: ignore
        # Reuse last seen row if available
        if isinstance(row, dict):
            # Text CE proxy when answer present
            try:
                if isinstance(row.get('answer'), str):
                    tok = get_text_tokenizer(prefer_hf=True)
                    prompt = (str(row.get('text') or row.get('question') or '') + "\nA:").strip()
                    _inp = torch.tensor(tok.encode(prompt), dtype=torch.long, device=args.device).unsqueeze(0)
                    _tgt = torch.tensor(tok.encode(str(row['answer'])), dtype=torch.long, device=args.device).unsqueeze(0)
                    with torch.inference_mode():
                        _out = model(_inp)
                        _logits = _out if isinstance(_out, torch.Tensor) else _out[0]
                    _bsz, _t, _v = _logits.shape
                    _alen = _tgt.size(1)
                    _ans_logits = _logits[:, _t - _alen - 1 : _t - 1, :]
                    _ce = nn.CrossEntropyLoss()(_ans_logits.reshape(-1, _v), _tgt.reshape(-1)).item()
                    maybe_save_best(args.out, model, 'mm_text_ce', float(_ce), higher_is_better=False)
            except Exception:
                pass
            # Image CLIPScore if image present
            try:
                if has_img and isinstance(row.get('_image'), torch.Tensor) and isinstance(row.get('caption') or row.get('text') or row.get('answer'), str):
                    _imgs = row.get('_image').unsqueeze(0)
                    _txts = [str(row.get('caption') or row.get('text') or row.get('answer'))]
                    cs = clipscore_mean(_imgs, _txts)
                    if cs is not None:
                        maybe_save_best(args.out, model, 'mm_clipscore', float(cs), higher_is_better=True)
            except Exception:
                pass
    except Exception:
        pass
    print(f"Saved unified multimodal checkpoint to {final}")


if __name__ == "__main__":
    main()



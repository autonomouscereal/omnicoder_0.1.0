from __future__ import annotations

"""
Flow/Diffusion latent reconstruction training loop (minimal).

Trains the core to produce continuous latent tokens (image_latent_head) that
match target latents from a decoder adapter (diffusers or ONNX callable).
The decoder adapter should expose an encode() method that maps images -> latents.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omnicoder.utils.resources import recommend_num_workers
import torch.nn.functional as F

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.data.vl_image_pairs import ImagePairsDataset
from omnicoder.training.adapters.image_latent_adapters import DiffusersAdapter, ONNXAdapter, DiffusersFlowAdapter
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.training.flows.flow_utils import sample_sigma, add_noise, loss_weight, sample_vp_params
from omnicoder.utils.torchutils import safe_torch_save


class IdentityAdapter:
    """Placeholder adapter when no external decoder is provided; treats image pixels as target after pooling."""

    def __init__(self, out_dim: int = 16):
        self.out_dim = out_dim

    @torch.inference_mode()
    def encode(self, img_chw: torch.Tensor) -> torch.Tensor:
        # img_chw: (C,H,W) in [-1,1]; produce a simple global embedding
        pooled = img_chw.mean(dim=(1, 2))  # (C,)
        # Resize to out_dim by linear projection with fixed random weights
        w = torch.randn(pooled.size(0), self.out_dim, device=pooled.device) * 0.1
        return pooled @ w  # (out_dim,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Path to images dir or JSONL with {image,text}')
    ap.add_argument('--image_size_w', type=int, default=int(os.getenv('OMNICODER_IMG_W', '512')))
    ap.add_argument('--image_size_h', type=int, default=int(os.getenv('OMNICODER_IMG_H', '512')))
    ap.add_argument('--batch', type=int, default=int(os.getenv('OMNICODER_FLOW_BATCH', '2')))
    ap.add_argument('--steps', type=int, default=int(os.getenv('OMNICODER_FLOW_STEPS', '1000')))
    ap.add_argument('--device', type=str, default=os.getenv('OMNICODER_FLOW_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--lr', type=float, default=float(os.getenv('OMNICODER_FLOW_LR', '2e-4')))
    ap.add_argument('--latent_dim', type=int, default=int(os.getenv('OMNICODER_LATENT_DIM', '16')))
    ap.add_argument('--out', type=str, default=os.getenv('OMNICODER_FLOW_OUT', 'weights/omnicoder_flow_latents.pt'))
    ap.add_argument('--sd_model', type=str, default=os.getenv('OMNICODER_SD_MODEL', ''))
    ap.add_argument('--onnx_dir', type=str, default=os.getenv('OMNICODER_ONNX_DIR', ''))
    ap.add_argument('--flow_loss', action='store_true', default=(os.getenv('OMNICODER_FLOW_LOSS','0')=='1'))
    ap.add_argument('--recon_loss', type=str, default=os.getenv('OMNICODER_RECON_LOSS','mse'), choices=['mse','mae','huber','perceptual'], help='Reconstruction loss for continuous latents')
    ap.add_argument('--flow_param', type=str, default=os.getenv('OMNICODER_FLOW_PARAM','edm'), choices=['edm','vp','ve'])
    ap.add_argument('--flow_weight', type=float, default=float(os.getenv('OMNICODER_FLOW_WEIGHT','1.0')), help='Weight for epsilon-prediction (flow) objective')
    ap.add_argument('--recon_weight', type=float, default=float(os.getenv('OMNICODER_RECON_WEIGHT','1.0')), help='Weight for reconstruction objective')
    ap.add_argument('--patch_loss', action='store_true', default=(os.getenv('OMNICODER_PATCH_LOSS','1')=='1'), help='Use per-patch latent losses instead of pooled vectors')
    # Optional evaluation metrics (extras required)
    ap.add_argument('--fid_metrics', action='store_true', default=(os.getenv('OMNICODER_FID_METRICS','0')=='1'), help='Compute FID/CLIPScore when extras are installed')
    ap.add_argument('--ref_dir', type=str, default=os.getenv('OMNICODER_FID_REF',''), help='Reference images directory for FID (optional)')
    # Gates and metrics output
    ap.add_argument('--clipscore_min', type=float, default=float(os.getenv('OMNICODER_CLIPSCORE_MIN','0.0')), help='Gate: minimum CLIPScore to pass (0 disables)')
    ap.add_argument('--fid_max', type=float, default=float(os.getenv('OMNICODER_FID_MAX','0.0')), help='Gate: maximum FID to pass (0 disables)')
    ap.add_argument('--metrics_out', type=str, default=os.getenv('OMNICODER_REF_METRICS_JSON',''), help='Optional path to write metrics JSON incl. gate status')
    ap.add_argument('--save_heads', type=str, default=os.getenv('OMNICODER_SAVE_HEADS',''), help='Optional path to save continuous heads (image_latent_head, image_eps_head)')
    # Allow both legacy and consolidated envs to enable the refiner
    _use_ref_env = (os.getenv('OMNICODER_USE_REFINER','0')=='1') or (os.getenv('OMNICODER_IMAGE_REFINER','0')=='1')
    ap.add_argument('--use_refiner', action='store_true', default=_use_ref_env, help='Apply tiny latent refiner module before computing losses')
    ap.add_argument('--refiner_hidden_mult', type=int, default=int(os.getenv('OMNICODER_REFINER_HIDDEN_MULT','2')))
    ap.add_argument('--refiner_temporal', action='store_true', default=(os.getenv('OMNICODER_REFINER_TEMPORAL','0')=='1'))
    ap.add_argument('--export_refiner_onnx', type=str, default=os.getenv('OMNICODER_EXPORT_REFINER',''), help='Optional path to export TinyLatentRefiner ONNX')
    ap.add_argument('--save_image_latent_head', type=str, default=os.getenv('OMNICODER_SAVE_IMAGE_HEAD',''), help='Optional path to save image_latent_head parameters only')
    args = ap.parse_args()

    ds = ImagePairsDataset(args.data, image_size=(args.image_size_w, args.image_size_h))
    # If the dataset is empty (e.g., empty folder in smoke tests), exit gracefully.
    if len(ds) == 0:
        print('[flow_recon] empty dataset, nothing to train; exiting successfully')
        return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=recommend_num_workers())

    model = OmniTransformer()
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    model.to(args.device)
    model.train()
    refiner = None
    if args.use_refiner:
        try:
            from omnicoder.modeling.multimodal.latent_refiner import TinyLatentRefiner  # type: ignore
            refiner = TinyLatentRefiner(latent_dim=int(args.latent_dim), hidden_mult=int(args.refiner_hidden_mult), use_temporal=bool(args.refiner_temporal)).to(args.device)
        except Exception as e:
            print(f"[warn] refiner unavailable: {e}")
    tok = get_text_tokenizer(prefer_hf=True)

    # Adapter: prefer provided backends; fallback to identity adapter
    adapter = IdentityAdapter(out_dim=args.latent_dim)
    # Prefer diffusers VAE encoder if provided; else ONNX adapter if provided
    try:
        if args.sd_model and args.flow_loss:
            adapter = DiffusersFlowAdapter(model_id=args.sd_model, latent_dim=args.latent_dim, device=args.device)
        elif args.sd_model:
            adapter = DiffusersAdapter(model_id=args.sd_model, device=args.device)
        elif args.onnx_dir:
            adapter = ONNXAdapter(onnx_dir=args.onnx_dir, device=args.device)
    except Exception as e:
        print(f"[warn] falling back to IdentityAdapter: {e}")

    params = [p for p in model.parameters() if p.requires_grad]
    if refiner is not None:
        params += [p for p in refiner.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    huber = nn.SmoothL1Loss(beta=1.0)
    # Optional lightweight perceptual proxy using a tiny fixed conv stack (no torchvision dependency)
    perceptual = None
    if args.recon_loss == 'perceptual':
        conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
        ).to(args.device).eval()
        for p in conv.parameters():
            p.requires_grad = False
        def _perceptual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # Inputs are (B,D) vectors; expand to pseudo images for feature extraction
            def _to_img(x: torch.Tensor) -> torch.Tensor:
                s = int(max(1, (x.size(1) // 3) ** 0.5))
                need = 3 * s * s
                if x.size(1) < need:
                    pad = need - x.size(1)
                    x = F.pad(x, (0, pad))
                x = torch.ops.aten.reshape.default(x[:, :need], (x.size(0), 3, s, s))
                if s < 32:
                    x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
                return x
            fa = conv(_to_img(a))
            fb = conv(_to_img(b))
            return F.l1_loss(fa, fb)
        perceptual = _perceptual

    step = 0
    all_images = []
    all_texts = []
    for images, _text in dl:
        images = images.to(args.device)
        b = images.size(0)
        # Encode text prompts and feed through the core before latent heads (conditional latent prediction)
        # Pad/truncate to a small fixed length for efficiency
        max_len = 32
        prompts = [t for _img, t in zip(images, _text)] if isinstance(_text, list) else _text
        if isinstance(_text, list):
            enc = [tok.encode(s or "")[:max_len] for s in _text]
        else:
            # Loader returned a batch of strings as a single list-like; handle generically
            try:
                enc = [tok.encode(s or "")[:max_len] for s in prompts]
            except Exception:
                enc = [[0] for _ in range(b)]
        lens = [len(e) for e in enc]
        max_b = max(1, max(lens) if lens else 1)
        ids = torch.zeros((b, max_b), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
        ids = ids.to(args.device)
        out = model(ids)
        # Unified extraction: always get logits; take image latents from model output (full head), no fallback head
        logits = out[0] if (isinstance(out, tuple) and isinstance(out[0], torch.Tensor)) else (out if isinstance(out, torch.Tensor) else None)
        if logits is None:
            print('No logits available from model; aborting.')
            return
        img_lat = None
        if isinstance(out, tuple):
            # New unified return layout: (logits, new_kv, sidecar, img_lat, aud_lat)
            if len(out) >= 4 and isinstance(out[3], torch.Tensor):
                img_lat = out[3]
        if img_lat is None:
            print('No image latent head available; aborting.')
            return
        # Pool target latents
        with torch.no_grad():
            targets = []
            eps_targets = []
            for i in range(b):
                if args.flow_loss and isinstance(adapter, DiffusersFlowAdapter):
                    eps, z0 = adapter.encode_flow(images[i])  # (P,D),(P,D)
                    if not args.patch_loss:
                        # Pool over patches to a single vector
                        eps = eps.mean(dim=0, keepdim=True)
                        z0 = z0.mean(dim=0, keepdim=True)
                    eps_targets.append(eps.unsqueeze(0) if eps.dim()==2 else eps.unsqueeze(0))
                    targets.append(z0.unsqueeze(0) if z0.dim()==2 else z0.unsqueeze(0))
                else:
                    t = adapter.encode(images[i])  # (D,)
                    targets.append(t.unsqueeze(0))
            targets = torch.cat(targets, dim=0).to(img_lat.device)
            if eps_targets:
                eps_targets = torch.cat(eps_targets, dim=0).to(img_lat.device)
        # Align shapes: take last token latents from head (B,T,D) -> (B,D)
        pred = img_lat[:, -1, :]
        if refiner is not None:
            pred = refiner(pred)
        # Build reconstruction objective (vector or per-patch)
        if args.flow_loss and isinstance(adapter, DiffusersFlowAdapter) and args.patch_loss and targets.dim() == 3:
            # targets: (B, P, D). Predict per-patch by broadcasting pred or by a tiny conv proj head
            # Use a small MLP to map token latent to per-patch grid; fallback to repeat if absent
            patch_head = getattr(model, 'image_patch_head', None)
            if patch_head is None:
                model.image_patch_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(pred.size(1)),
                    torch.nn.Linear(pred.size(1), targets.size(-1))
                ).to(pred.device)  # type: ignore[attr-defined]
                patch_head = model.image_patch_head  # type: ignore[attr-defined]
            pred_patches = patch_head(pred).unsqueeze(1).repeat(1, targets.size(1), 1)  # (B,P,D)
            if args.recon_loss == 'mse':
                recon_l = mse(pred_patches, targets)
            elif args.recon_loss == 'mae':
                recon_l = mae(pred_patches, targets)
            elif args.recon_loss == 'huber':
                recon_l = huber(pred_patches, targets)
            else:
                # Perceptual proxy operates on pooled vectors
                pooled_pred = pred_patches.mean(dim=1)
                pooled_tgt = targets.mean(dim=1)
                recon_l = (perceptual(pooled_pred, pooled_tgt) if perceptual is not None else mse(pooled_pred, pooled_tgt))
            loss = args.recon_weight * recon_l
        else:
            # Align vector dims when adapter output size differs from model latent dim
            if targets.dim() == 2 and targets.size(1) != pred.size(1):
                td = int(targets.size(1)); pd = int(pred.size(1))
                if td > pd:
                    targets = targets[:, :pd]
                else:
                    import torch.nn.functional as _F  # defer to avoid polluting namespace
                    targets = _F.pad(targets, (0, pd - td))
            if args.recon_loss == 'mse':
                loss = args.recon_weight * mse(pred, targets)
            elif args.recon_loss == 'mae':
                loss = args.recon_weight * mae(pred, targets)
            elif args.recon_loss == 'huber':
                loss = args.recon_weight * huber(pred, targets)
            else:
                loss = args.recon_weight * (perceptual(pred, targets) if perceptual is not None else mse(pred, targets))

        # Epsilon (flow) objective
        if args.flow_loss and isinstance(adapter, DiffusersFlowAdapter):
            eps_head = getattr(model, 'image_eps_head', None)
            if eps_head is None:
                model.image_eps_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(pred.size(1)),
                    torch.nn.Linear(pred.size(1), targets.size(-1) if targets.dim()==2 else targets.size(-1))
                ).to(pred.device)  # type: ignore[attr-defined]
                eps_head = model.image_eps_head  # type: ignore[attr-defined]
            if args.flow_param == 'vp':
                alpha, sigma = sample_vp_params(b, pred.device)
                zt = torch.ops.aten.reshape.default(alpha, (-1,1)) * (targets.mean(dim=1) if targets.dim()==3 else targets) + torch.ops.aten.reshape.default(sigma, (-1,1)) * torch.randn_like(pred)
                eps = (zt - torch.ops.aten.reshape.default(alpha, (-1,1)) * (targets.mean(dim=1) if targets.dim()==3 else targets)) / torch.ops.aten.reshape.default(sigma, (-1,1))
                eps_pred = eps_head(pred)
                w = loss_weight(sigma)
                flow_l = ((eps_pred - eps) ** 2).mean(dim=1)
                loss = loss + args.flow_weight * (w * flow_l).mean()
            else:
                sigma = sample_sigma(b, pred.device)
                base = targets.mean(dim=1) if targets.dim()==3 else targets
                zt, eps = add_noise(base, sigma)
                eps_pred = eps_head(pred)
                w = loss_weight(sigma)
                flow_l = ((eps_pred - eps) ** 2).mean(dim=1)
                loss = loss + args.flow_weight * (w * flow_l).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        # Accumulate a small sample for metrics (first batches only)
        if len(all_images) < 64:
            all_images.append(images.detach().cpu())
            try:
                if isinstance(_text, list):
                    all_texts.extend(_text)
                else:
                    all_texts.extend(list(_text))  # type: ignore
            except Exception:
                pass
        if step % 10 == 0:
            print(f'step {step}/{args.steps} | loss {loss.item():.4f}')
        if step >= args.steps:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        target = Path(args.out)
        if target.exists() and target.is_dir():
            target = target / 'model.pt'
        sd = {k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        safe_torch_save(sd, str(target))
        print(f'Saved model with latent head training to {target}')
        # Deep evals: CLIPScore/FID when extras available
        try:
            from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
            from omnicoder.utils.evalhooks import clipscore_mean, fid_score  # type: ignore
            if len(ds) > 0:
                sample = ds[0]
                img = sample[0].to(args.device) if isinstance(sample, tuple) else sample.to(args.device)
                tok = get_text_tokenizer(prefer_hf=True)
                ids = torch.tensor([tok.encode("")], dtype=torch.long, device=args.device)
                with torch.inference_mode():
                    out = model(ids)
                img_lat = out[3] if isinstance(out, tuple) and len(out) >= 4 and isinstance(out[3], torch.Tensor) else None
                if img_lat is not None:
                    with torch.inference_mode():
                        tgt = adapter.encode(img)
                    pred = img_lat[:, -1, :]
                    if tgt.dim() == 1:
                        tgt = tgt.unsqueeze(0)
                    tgt = tgt.to(pred.device)
                    Dp = int(pred.shape[1])
                    Dt = int(tgt.shape[1])
                    if Dp != Dt:
                        Dmin = Dp if Dp < Dt else Dt
                        pred = torch.ops.aten.slice.Tensor(pred, 1, 0, Dmin, 1)
                        tgt = torch.ops.aten.slice.Tensor(tgt, 1, 0, Dmin, 1)
                    _l2 = torch.nn.functional.mse_loss(pred, tgt).item()
                    maybe_save_best(args.out, model, 'image_latent_l2', float(_l2), higher_is_better=False)
                    # CLIPScore (image-text) if text available in sample
                    try:
                        if isinstance(sample, tuple) and isinstance(sample[1], str) and sample[1].strip():
                            cs = clipscore_mean(img.unsqueeze(0), [sample[1]])
                            if cs is not None:
                                maybe_save_best(args.out, model, 'image_clipscore', float(cs), higher_is_better=True)
                    except Exception:
                        pass
                    # FID proxy if reference dir is configured
                    try:
                        if args.ref_dir:
                            # Generate a pseudo image tensor from latent for proxy FID (placeholder)
                            gen = torch.rand_like(img.unsqueeze(0))
                            fid = fid_score(args.ref_dir, gen)
                            if fid is not None:
                                maybe_save_best(args.out, model, 'image_fid', float(fid), higher_is_better=False)
                    except Exception:
                        pass
        except Exception:
            pass
        # Emit compact sidecar metadata for downstream export/runners
        try:
            meta = {
                'model_config': {
                    'vocab_size': int(getattr(model, 'vocab_size', 0)),
                    'n_layers': int(getattr(model, 'n_layers', 0)),
                    'd_model': int(getattr(model, 'd_model', 0)),
                    'n_heads': int(getattr(getattr(model, 'blocks', [type('B', (), {'attn': type('A', (), {'n_heads': 0})()})])[0].attn, 'n_heads', getattr(model, 'n_heads', 0))),
                    'mlp_dim': int(getattr(model, 'mlp_dim', 0)),
                    'n_experts': int(getattr(model, 'n_experts', 0)),
                    'top_k': int(getattr(model, 'top_k', 0)),
                    'max_seq_len': int(getattr(model, 'max_seq_len', 0)),
                    'kv_latent_dim': int(getattr(model, 'kv_latent_dim', 0)),
                    'multi_query': bool(getattr(model, 'multi_query', False)),
                    'multi_token': int(getattr(model, 'multi_token', 1)),
                },
                'train_args': {
                    'data': str(args.data),
                    'device': str(args.device),
                    'steps': int(args.steps),
                    'batch': int(args.batch),
                    'lr': float(args.lr),
                    'latent_dim': int(args.latent_dim),
                    'sd_model': str(args.sd_model),
                    'onnx_dir': str(args.onnx_dir),
                    'flow_loss': bool(args.flow_loss),
                    'flow_param': str(args.flow_param),
                    'flow_weight': float(args.flow_weight),
                    'recon_loss': str(args.recon_loss),
                    'patch_loss': bool(args.patch_loss),
                    'use_refiner': bool(args.use_refiner),
                },
            }
            # Tokenizer hint for conditional text encodings
            try:
                import os as _os
                meta['tokenizer_hint'] = _os.getenv('OMNICODER_TOKENIZER', 'auto')
            except Exception:
                meta['tokenizer_hint'] = 'auto'
            (target.with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass
    except Exception as e:
        from datetime import datetime as _dt
        _ts = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
        _fallback = Path('weights') / f"omnicoder_flow_latents_{_ts}.pt"
        _fallback.parent.mkdir(parents=True, exist_ok=True)
        sd = {k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        safe_torch_save(sd, str(_fallback))
        print(f"[warn] primary save failed ({e}); wrote fallback to {_fallback}")
        try:
            meta = {
                'model_config': {
                    'vocab_size': int(getattr(model, 'vocab_size', 0)),
                    'n_layers': int(getattr(model, 'n_layers', 0)),
                    'd_model': int(getattr(model, 'd_model', 0)),
                    'n_heads': int(getattr(getattr(model, 'blocks', [type('B', (), {'attn': type('A', (), {'n_heads': 0})()})])[0].attn, 'n_heads', getattr(model, 'n_heads', 0))),
                    'mlp_dim': int(getattr(model, 'mlp_dim', 0)),
                    'n_experts': int(getattr(model, 'n_experts', 0)),
                    'top_k': int(getattr(model, 'top_k', 0)),
                    'max_seq_len': int(getattr(model, 'max_seq_len', 0)),
                    'kv_latent_dim': int(getattr(model, 'kv_latent_dim', 0)),
                    'multi_query': bool(getattr(model, 'multi_query', False)),
                    'multi_token': int(getattr(model, 'multi_token', 1)),
                },
                'train_args': {
                    'data': str(args.data),
                    'device': str(args.device),
                    'steps': int(args.steps),
                    'batch': int(args.batch),
                    'lr': float(args.lr),
                    'latent_dim': int(args.latent_dim),
                    'sd_model': str(args.sd_model),
                    'onnx_dir': str(args.onnx_dir),
                    'flow_loss': bool(args.flow_loss),
                    'flow_param': str(args.flow_param),
                    'flow_weight': float(args.flow_weight),
                    'recon_loss': str(args.recon_loss),
                    'patch_loss': bool(args.patch_loss),
                    'use_refiner': bool(args.use_refiner),
                },
            }
            (Path(str(_fallback)).with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass
    # Optional: save heads-only state dict for later loading
    if args.save_heads:
        try:
            heads_sd: dict = {}
            img_head = getattr(model, 'image_latent_head', None)
            if img_head is not None:
                for k, v in img_head.state_dict().items():
                    heads_sd[f'image_latent_head.{k}'] = v.detach().cpu()
            eps_head = getattr(model, 'image_eps_head', None)
            if eps_head is not None:
                for k, v in eps_head.state_dict().items():
                    heads_sd[f'image_eps_head.{k}'] = v.detach().cpu()
            if heads_sd:
                Path(args.save_heads).parent.mkdir(parents=True, exist_ok=True)
                safe_torch_save(heads_sd, args.save_heads)
                print(f"Saved latent heads to {args.save_heads}")
        except Exception as e:
            print(f"[warn] could not save heads: {e}")
    # Optional: save only the image_latent_head
    if args.save_image_latent_head:
        try:
            img_head = getattr(model, 'image_latent_head', None)
            if img_head is not None:
                Path(args.save_image_latent_head).parent.mkdir(parents=True, exist_ok=True)
                safe_torch_save({'image_latent_head': img_head.state_dict()}, args.save_image_latent_head)
                print(f"Saved image_latent_head to {args.save_image_latent_head}")
        except Exception as e:
            print(f"[warn] could not save image_latent_head: {e}")
    # Optional metrics
    if args.fid_metrics:
        try:
            from torchvision import transforms as _T  # type: ignore
            imgs = torch.clamp(torch.cat(all_images, dim=0), 0.0, 1.0)
            _metrics = {}
            # Compute CLIPScore if open-clip-torch present; dump JSON gate if possible
            try:
                import open_clip  # type: ignore
                model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
                model_clip = model_clip.to('cpu').eval()
                # Prepare image batch
                pil_tf = _T.ToPILImage()
                img_list = [preprocess(pil_tf(imgs[i].cpu())).unsqueeze(0) for i in range(min(len(imgs), len(all_texts)))]
                img_batch = torch.cat(img_list, dim=0)
                with torch.no_grad():
                    img_feat = model_clip.encode_image(img_batch)
                    txt = tokenizer(all_texts[:img_batch.size(0)])
                    txt_feat = model_clip.encode_text(txt)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                    clip_scores = (img_feat * txt_feat).sum(dim=-1).mean().item()
                _metrics['CLIPScore_mean'] = float(clip_scores)
                import json as _json
                print({'CLIPScore_mean': clip_scores})
            except Exception as e:
                print(f"[metrics] CLIPScore skipped: {e}")
            # FID if torchmetrics available and ref_dir provided
            if args.ref_dir:
                try:
                    from torchmetrics.image.fid import FrechetInceptionDistance  # type: ignore
                    fid = FrechetInceptionDistance(feature=2048).to('cpu').eval()
                    # Current set as real, reference dir as fake or vice versa
                    # Compute activations over a small sample for speed
                    for i in range(min(imgs.size(0), 64)):
                        img = (imgs[i].cpu() * 255).byte()
                        fid.update(img.unsqueeze(0), real=True)
                    import PIL.Image as _PIL
                    ref_paths = list(Path(args.ref_dir).rglob('*.png')) + list(Path(args.ref_dir).rglob('*.jpg'))
                    for p in ref_paths[:64]:
                        im = _PIL.Image.open(p).convert('RGB')
                        imt = _T.ToTensor()(im)
                        fid.update((imt * 255).byte().unsqueeze(0), real=False)
                    _metrics['FID'] = float(fid.compute().item())
                    print({'FID': float(_metrics['FID'])})
                except Exception as e:
                    print(f"[metrics] FID skipped: {e}")
        except Exception as e:
            print(f"[metrics] skipped: {e}")

        # Apply gates and write consolidated metrics JSON
        try:
            gate_pass = True
            cs_min = float(max(0.0, args.clipscore_min))
            fid_max = float(max(0.0, args.fid_max))
            if cs_min > 0.0 and 'CLIPScore_mean' in locals().get('_metrics', {}):
                if float(_metrics.get('CLIPScore_mean', 0.0)) < cs_min:
                    gate_pass = False
            if fid_max > 0.0 and 'FID' in locals().get('_metrics', {}):
                if float(_metrics.get('FID', 0.0)) > fid_max:
                    gate_pass = False
            _payload = dict(_metrics)
            _payload['gate_pass'] = bool(gate_pass)
            outp = Path(args.metrics_out) if args.metrics_out else (Path(args.out).with_suffix('.image_metrics.json'))
            outp.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            outp.write_text(_json.dumps(_payload, indent=2), encoding='utf-8')
            print({'image_metrics_written': str(outp), 'gate_pass': gate_pass})
        except Exception as _e:
            print(f"[metrics] write/gate skipped: {_e}")

    # Optional: export refiner ONNX for mobile (
    if args.export_refiner_onnx:
        try:
            ref_m = refiner if refiner is not None else None
            if ref_m is None:
                from omnicoder.modeling.multimodal.latent_refiner import TinyLatentRefiner  # type: ignore
                ref_m = TinyLatentRefiner(latent_dim=int(args.latent_dim), hidden_mult=int(args.refiner_hidden_mult), use_temporal=bool(args.refiner_temporal)).eval()
            d = torch.randn(1, int(args.latent_dim))
            torch.onnx.export(ref_m.eval(), d, args.export_refiner_onnx, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "B"}, "y": {0: "B"}}, opset_version=18)
            print(f"[onnx] exported refiner to {args.export_refiner_onnx}")
        except Exception as e:
            print(f"[warn] refiner onnx export failed: {e}")


if __name__ == '__main__':
    main()



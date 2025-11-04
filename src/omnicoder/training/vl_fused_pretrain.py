from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.modeling.multimodal.image_vq import ImageVQ
from omnicoder.modeling.multimodal.vocab_map import map_image_tokens
from omnicoder.config import MultiModalConfig
from omnicoder.training.data.vl_jsonl import VLDataModule
from omnicoder.config import get_mobile_preset
from omnicoder.modeling.multimodal.aligner import PreAligner
from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.utils.torchutils import safe_torch_save


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
    ap = argparse.ArgumentParser(description="VL pretraining: either feature-fused (default) or VQ-token fused")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to JSONL with {image, text}")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb", choices=["mobile_4gb", "mobile_2gb"]) 
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--out", type=str, default="weights/omnicoder_vl_fused.pt")
    ap.add_argument("--use_vq_tokens", action="store_true", help="Use Image VQ tokens mapped into unified vocab instead of feature fusion")
    ap.add_argument("--image_vq_codebook", type=str, default="", help="Path to image VQ-VAE codebook (required if --use_vq_tokens)")
    # Optional pre-alignment auxiliary loss (InfoNCE) using PreAligner
    ap.add_argument("--pre_align_ckpt", type=str, default="", help="Path to a pre-aligner .pt state dict (from training/pre_align.py)")
    ap.add_argument("--align_weight", type=float, default=0.1, help="Weight of the auxiliary alignment loss when --pre_align_ckpt is set")
    args = ap.parse_args()

    # Data
    dm = VLDataModule(jsonl_path=args.jsonl, d_model=get_mobile_preset(args.mobile_preset).d_model, batch_size=args.batch_size, image_size=tuple(args.image_size))
    dl: DataLoader = dm.loader()

    # Model and composer
    model = build_model(args.mobile_preset, seq_len_hint=4096)
    # Load best-known or latest checkpoint if available to avoid wasting prior training
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
    # Ensure composer modules (vision/audio/video encoders and projectors) live on the same device as model
    composer.to(args.device)
    # Prefer a sidecar unified vocab map if present to enforce aligned slices
    mmc = MultiModalConfig()
    try:
        from omnicoder.modeling.multimodal.vocab_map import VocabSidecar
        import os
        sidecar = os.getenv("OMNICODER_VOCAB_SIDECAR", "weights/unified_vocab_map.json")
        from pathlib import Path as _P
        if _P(sidecar).exists():
            side = VocabSidecar.load(sidecar)
            layout = side.as_layout()
            mmc.image_vocab_start = layout.image_start
            mmc.video_vocab_start = layout.video_start
            mmc.audio_vocab_start = layout.audio_start
            mmc.image_codebook_size = layout.image_size
            mmc.video_codebook_size = layout.video_size
            mmc.audio_codebook_size = layout.audio_size
    except Exception:
        pass
    img_vq: ImageVQ | None = None
    if args.use_vq_tokens:
        if not args.image_vq_codebook:
            raise SystemExit("--image_vq_codebook is required when --use_vq_tokens is set")
        img_vq = ImageVQ(codebook_path=args.image_vq_codebook)
    model.to(args.device)
    model.train()

    # Optimizer selection
    import os as _os
    optim_name = _os.getenv('OMNICODER_OPTIM', 'adamw').strip().lower()
    weight_decay = float(_os.getenv('OMNICODER_WEIGHT_DECAY', '0.01'))
    if optim_name in ('adamw8bit','adamw_8bit','adam8bit'):
        try:
            import bitsandbytes as bnb  # type: ignore
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=weight_decay)
            print('[optim] using AdamW8bit')
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
            print('[optim] AdamW8bit unavailable; falling back to AdamW')
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    # Optional pre-aligner and vision backbone for auxiliary InfoNCE
    aligner: PreAligner | None = None
    vision_backbone: VisionBackbone | None = None
    tokenizer = None
    if args.pre_align_ckpt:
        try:
            sd = torch.load(args.pre_align_ckpt, map_location='cpu')
            embed_dim = int(sd.get('embed_dim', 256))
            # Match text_dim to embed_dim used in pre_align to avoid shape mismatch
            aligner = PreAligner(embed_dim=embed_dim, text_dim=embed_dim).to(args.device)
            if 'aligner' in sd:
                aligner.load_state_dict(sd['aligner'], strict=False)
            aligner.eval()
            import os as _os
            vision_backbone = VisionBackbone(backend=_os.getenv('OMNICODER_VISION_BACKEND', 'dinov3'), d_model=768, return_pooled=True).to(args.device).eval()
            tokenizer = get_text_tokenizer(prefer_hf=True)
            print(f"[pre-align] loaded {args.pre_align_ckpt} (embed_dim={embed_dim})")
        except Exception as e:
            print(f"[pre-align] could not load pre-aligner: {e}")

    use_amp = (os.getenv("OMNICODER_VL_AMP", "1") == "1") and (args.device.startswith('cuda')) and torch.cuda.is_available()
    accum = max(1, int(os.getenv("OMNICODER_VL_ACCUM", "1")))
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
        _ac = (lambda: torch.amp.autocast('cuda')) if use_amp else (lambda: torch.amp.autocast('cpu'))  # type: ignore[attr-defined]
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        _ac = (lambda: torch.cuda.amp.autocast()) if use_amp else (lambda: torch.cpu.amp.autocast())  # type: ignore[attr-defined]
    step = 0
    for images, input_ids in dl:
        input_ids = input_ids.to(args.device)
        # Guard against tokenizer vocab > model.vocab_size causing embedding OOB
        try:
            vocab_lim = int(getattr(model, 'vocab_size', 0))
            if vocab_lim and vocab_lim > 0:
                input_ids = input_ids.clamp_min(0).clamp_max(vocab_lim - 1)
        except Exception:
            pass
        if args.use_vq_tokens and img_vq is not None:
            # Encode each image to VQ tokens and map into unified vocab; concatenate with text ids
            images_np = images.numpy().astype("float32")  # (B,3,H,W) on CPU
            bsz = images_np.shape[0]
            vq_batches = []
            for b in range(bsz):
                img = images_np[b].transpose(1, 2, 0) * 255.0
                codes = img_vq.encode(img)[0].tolist()
                mapped = map_image_tokens(codes, mmc)
                vq_batches.append(torch.tensor(mapped, dtype=torch.long))
            # Pad to max
            max_vq = max(x.size(0) for x in vq_batches)
            vq_pad = torch.stack([torch.cat([x, torch.zeros(max_vq - x.size(0), dtype=torch.long)]) for x in vq_batches]).to(args.device)
            seq = torch.cat([vq_pad, input_ids], dim=1)
            labels = seq.clone()
            with _ac():
                out = model(seq)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            loss = ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        else:
            images = images.to(args.device)
            # Feature fusion path
            fused = composer.fuse_text_image(model_with_embed=model, input_ids=input_ids, image_bchw=images)
            with _ac():
                outputs = model(fused, use_cache=False)  # type: ignore[arg-type]
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            bsz, t_fused, vocab = logits.shape
            t_text = input_ids.size(1)
            t_nontext = t_fused - t_text
            text_logits = logits[:, t_nontext - 1 : t_fused - 1, :]
            text_labels = input_ids
            loss = ce(text_logits.reshape(-1, vocab), text_labels.reshape(-1))
            # Optional: add auxiliary alignment loss if pre-aligner provided
            if aligner is not None and vision_backbone is not None and tokenizer is not None and args.align_weight > 0.0:
                with torch.inference_mode():
                    # Vision pooled
                    _, pooled = vision_backbone(images)
                    if pooled is None:
                        pooled = vision_backbone(images)[0].mean(dim=1)
                pooled = pooled.clone()
                # Text embedding (simple mean over token embeddings)
                txt_embeds = []
                for row in input_ids:
                    ids = row.clamp_min(0).tolist()
                    # Create a deterministic random table for placeholder encoding
                    torch.manual_seed(13)
                    table = torch.randn((max(max(ids) + 1, 1), aligner.text_proj[0].out_features), device=images.device)
                    v = table.index_select(0, torch.tensor(ids, device=images.device)).mean(dim=0)
                    txt_embeds.append(v)
                txt = torch.stack(txt_embeds, dim=0)
                emb = aligner(image=pooled, text=txt)
                loss = loss + float(args.align_weight) * PreAligner.info_nce_loss(emb['image'], emb['text'])

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (step + 1) % accum == 0:
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % 1 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
        if step >= args.steps:
            break

    # Optional evaluation before final save: verify that model reaches a minimal target quality
    try:
        # Prefer the existing benchmark to compute TPS and update best
        from omnicoder.utils.evalhooks import text_tps  # type: ignore
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        tps = text_tps(model, device=args.device, seq_len=128, gen_tokens=64)
        if tps is not None:
            maybe_save_best(args.out, model, 'tps_eval', float(tps), higher_is_better=True)
    except Exception:
        pass
    # Robust save: CPU tensors; use safe_torch_save to avoid writer failures
    target = Path(args.out)
    out_dir = target.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # If the provided output path is actually a directory (from prior runs), write model.pt inside it
        if target.exists() and target.is_dir():
            target = target / 'model.pt'
        sd = {k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
        safe_torch_save(sd, str(target))
        print(f"Saved fused VL checkpoint to {target}")
        # Also emit a portable sidecar JSON capturing minimal export context (model+train args)
        # Rationale:
        # - Downstream exporters/runtimes often need the tokenizer/vocab layout and core dims.
        # - We persist a compact JSON next to the checkpoint to make ONNX export and mobile
        #   packaging deterministic without re-discovering these from code.
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
                    'jsonl': str(args.jsonl),
                    'device': str(args.device),
                    'steps': int(args.steps),
                    'batch_size': int(args.batch_size),
                    'lr': float(args.lr),
                    'image_size': [int(args.image_size[0]), int(args.image_size[1])],
                    'mobile_preset': str(args.mobile_preset),
                    'use_vq_tokens': bool(args.use_vq_tokens),
                },
            }
            # If a unified vocab sidecar was used, copy alongside checkpoint for portability
            try:
                import os as _os
                _sidecar_src = _os.getenv('OMNICODER_VOCAB_SIDECAR', 'weights/unified_vocab_map.json')
                _p = Path(_sidecar_src)
                if _p.exists():
                    _dst = target.with_suffix('.unified_vocab_map.json')
                    try:
                        _dst.write_text(_p.read_text(encoding='utf-8'), encoding='utf-8')
                        meta['vocab_sidecar'] = str(_dst)
                    except Exception:
                        meta['vocab_sidecar'] = str(_p)
            except Exception:
                pass
            # Tokenizer hint (best-effort): record environment-driven tokenizer if set
            try:
                import os as _os
                meta['tokenizer_hint'] = _os.getenv('OMNICODER_TOKENIZER', 'auto')
            except Exception:
                meta['tokenizer_hint'] = 'auto'
            (target.with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass
    except Exception as e:
        try:
            from datetime import datetime as _dt
            _ts = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
            _fallback = Path('weights') / f"omnicoder_vl_fused_{_ts}.pt"
            _fallback.parent.mkdir(parents=True, exist_ok=True)
            sd = {k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
            safe_torch_save(sd, str(_fallback))
            print(f"[warn] primary save failed ({e}); wrote fallback to {_fallback}")
            # Write sidecar/meta next to fallback as well
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
                        'jsonl': str(args.jsonl),
                        'device': str(args.device),
                        'steps': int(args.steps),
                        'batch_size': int(args.batch_size),
                        'lr': float(args.lr),
                        'image_size': [int(args.image_size[0]), int(args.image_size[1])],
                        'mobile_preset': str(args.mobile_preset),
                        'use_vq_tokens': bool(args.use_vq_tokens),
                    },
                }
                (Path(str(_fallback)).with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
            except Exception:
                pass
        except Exception as _:
            raise


if __name__ == "__main__":
    main()



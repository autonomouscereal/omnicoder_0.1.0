from __future__ import annotations

"""
Cross-modal alignment trainer

Trains the model's shared concept head to align with modality features produced by
`PreAligner` (text/image/audio/video). Supports InfoNCE and triplet losses, random
negative mixing (including optional negative text prompts), and optional DINO/SigLIP
vision backbones via `VisionBackbone`.
"""

import argparse
import os
from pathlib import Path

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.aligner import PreAligner
from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone
from omnicoder.training.data.vl_jsonl import VLDataModule
from omnicoder.config import get_mobile_preset
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def build_student(preset_name: str) -> OmniTransformer:
    preset = get_mobile_preset(preset_name)
    return OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=preset.max_seq_len,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-modal shared-latent alignment trainer")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to JSONL with {image, text}")
    ap.add_argument("--mobile_preset", type=str, default=os.getenv("OMNICODER_TRAIN_PRESET", "mobile_4gb"))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch_size", type=int, default=int(os.getenv("OMNICODER_ALIGN_BATCH", "4")))
    ap.add_argument("--steps", type=int, default=int(os.getenv("OMNICODER_ALIGN_STEPS", "200")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("OMNICODER_ALIGN_LR", "5e-4")))
    ap.add_argument("--prealign_ckpt", type=str, default=os.getenv("OMNICODER_PREALIGN_CKPT", ""))
    ap.add_argument("--neg_jsonl", type=str, default=os.getenv("OMNICODER_ALIGN_NEG_JSONL", ""), help="Optional JSONL with {text} negatives (e.g., 'no dog present')")
    ap.add_argument("--neg_text", action="store_true", default=(os.getenv("OMNICODER_ALIGN_NEG_TEXT", "1") == "1"), help="Enable random negative text prompts for triplet loss")
    ap.add_argument("--out", type=str, default=os.getenv("OMNICODER_ALIGN_OUT", "weights/omnicoder_align.pt"))
    args = ap.parse_args()

    # Data
    dm = VLDataModule(jsonl_path=args.jsonl, d_model=get_mobile_preset(args.mobile_preset).d_model, batch_size=args.batch_size)
    # On Windows, VLDataModule.loader() already forces num_workers=0; ensure deterministic behavior
    dl: DataLoader = dm.loader()

    # Student model (provides concept head) and frozen modality encoders
    model = build_student(args.mobile_preset).to(args.device)
    model.train()
    # Prefer DINOv3 by default; override via OMNICODER_VISION_BACKEND
    vision_backend = os.getenv("OMNICODER_VISION_BACKEND", "dinov3")
    vision = VisionBackbone(backend=vision_backend, d_model=768, return_pooled=True).to(args.device).eval()
    tokenizer = get_text_tokenizer(prefer_hf=True)
    # Optional explicit negative texts
    neg_texts: list[str] = []
    if args.neg_jsonl and Path(args.neg_jsonl).exists():
        try:
            import json as _json
            with open(args.neg_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        j = _json.loads(line)
                        if isinstance(j, dict) and 'text' in j:
                            neg_texts.append(str(j['text']))
                    except Exception:
                        continue
        except Exception:
            neg_texts = []
    # Load or instantiate PreAligner for projection of teacher features
    embed_dim = int(os.getenv("OMNICODER_PREALIGN_EMBED_DIM", "256"))
    # Text projection input dimension must match the student's token embedding size (d_model),
    # not the shared embed_dim. Using embed_dim here causes matmul shape errors when d_model != embed_dim.
    _d_model = get_mobile_preset(args.mobile_preset).d_model
    aligner = PreAligner(embed_dim=embed_dim, text_dim=int(_d_model), image_dim=768).to(args.device).eval()
    if args.prealign_ckpt:
        try:
            sd = torch.load(args.prealign_ckpt, map_location='cpu')
            if 'aligner' in sd:
                aligner.load_state_dict(sd['aligner'], strict=False)
            embed_dim = int(sd.get('embed_dim', embed_dim))
        except Exception:
            pass

    # Optimizer on student parameters (concept_head and optional LN for stability)
    params = [p for n, p in model.named_parameters() if p.requires_grad and ("concept_head" in n or n.startswith("ln_f"))]
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr))

    use_amp = args.device.startswith('cuda') and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
        autocast = (lambda: torch.amp.autocast('cuda')) if use_amp else (lambda: torch.amp.autocast('cpu'))  # type: ignore[attr-defined]
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast  # type: ignore[attr-defined]

    step = 0
    for images, input_ids in dl:
        step += 1
        images = images.to(args.device)
        input_ids = input_ids.to(args.device)

        # Compute teacher-side embeddings (vision pooled and text pooled) and project to shared space
        with torch.inference_mode():
            _, img_pooled = vision(images)
            if img_pooled is None:
                img_pooled = vision(images)[0].mean(dim=1)
            # Text pooled embedding via model's token embedding as proxy; then PreAligner.text_proj
            txt_emb = model.embed(input_ids).mean(dim=1)
            emb = aligner(image=img_pooled, text=txt_emb)
            z_img = emb['image']  # (B, E)
            z_txt = emb['text']   # (B, E)
        # Ensure teacher tensors are normal (non-inference) tensors usable by autograd consumers
        z_img = z_img.detach().clone()
        z_txt = z_txt.detach().clone()

        # Student concept embedding from hidden states
        import contextlib as _ctx
        with (autocast() if use_amp else _ctx.nullcontext()):
            out = model(input_ids, use_cache=False, return_hidden=True)
            if isinstance(out, tuple):
                hidden = out[-1]
            else:
                hidden = model.ln_f(model.embed(input_ids))
            # Project to concept space
            try:
                z_concept = model.concept_head(hidden)
            except Exception:
                z_concept = torch.nn.functional.normalize(hidden[:, -1, :], dim=-1)

        # Losses: InfoNCE with image and text targets, plus triplet with negatives
        # Normalize dtypes to float32 to avoid bf16/float mismatches on CPU AMP
        if z_concept.dtype != torch.float32:
            z_concept = z_concept.float()
        if z_img.dtype != torch.float32:
            z_img = z_img.float()
        if z_txt.dtype != torch.float32:
            z_txt = z_txt.float()
        loss = PreAligner.info_nce_loss(z_concept, z_img) + PreAligner.info_nce_loss(z_concept, z_txt)
        if bool(args.neg_text):
            if neg_texts:
                # Encode a random negative text independently
                try:
                    import random as _rand
                    nts = [_rand.choice(neg_texts) for _ in range(z_txt.size(0))]
                    ids = [torch.tensor(tokenizer.encode(t or ""), dtype=torch.long, device=args.device) for t in nts]
                    max_t = max(1, max(int(x.numel()) for x in ids))
                    neg_pad = torch.zeros((len(ids), max_t), dtype=torch.long, device=args.device)
                    for i, row in enumerate(ids):
                        n = min(max_t, int(row.numel()))
                        if n > 0:
                            neg_pad[i, :n] = row[:n]
                    txt_emb_neg = model.embed(neg_pad).mean(dim=1)
                    z_txt_neg = aligner(text=txt_emb_neg)["text"].detach()
                except Exception:
                    z_txt_neg = z_txt[torch.randperm(z_txt.size(0), device=z_txt.device)]
            else:
                z_txt_neg = z_txt[torch.randperm(z_txt.size(0), device=z_txt.device)]
            loss = loss + 0.5 * PreAligner.triplet_loss(z_concept, z_txt, z_txt_neg)
        neg_img = z_img[torch.randperm(z_img.size(0), device=z_img.device)]
        loss = loss + 0.5 * PreAligner.triplet_loss(z_concept, z_img, neg_img)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print({"step": int(step), "loss": float(loss.item())})
        if step >= int(args.steps):
            break

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {'train_args': {'steps': int(args.steps)}}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"Saved cross-modal aligned checkpoint to {final}")


if __name__ == "__main__":
    main()



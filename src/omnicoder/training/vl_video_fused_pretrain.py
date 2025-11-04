from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader
from omnicoder.utils.resources import recommend_num_workers

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.data.video_vl_vq_jsonl import VideoVLVQDataset
from omnicoder.modeling.multimodal.video_vq import VideoVQ
from omnicoder.modeling.multimodal.vocab_map import map_video_tokens
from omnicoder.config import MultiModalConfig


def main() -> None:
    ap = argparse.ArgumentParser(description='VL fused train with video VQ tokens + text')
    ap.add_argument('--jsonl', type=str, required=True)
    ap.add_argument('--vq_codebook', type=str, required=False, default='')
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=str, default='weights/omnicoder_vl_video_vq.pt')
    args = ap.parse_args()

    from omnicoder.inference.generate import build_mobile_model_by_name
    model = build_mobile_model_by_name(args.mobile_preset)
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Load unified vocab sidecar if present to enforce aligned slices
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

    use_vq = bool(args.vq_codebook)
    if use_vq:
        ds = VideoVLVQDataset(args.jsonl, args.vq_codebook)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=recommend_num_workers(), collate_fn=lambda b: b)
    else:
        # Fallback to feature fusion via SimpleVideoEncoder through MultimodalComposer would require a separate path.
        # For this trainer, require VQ path for now.
        raise SystemExit('Provide --vq_codebook for VL-video VQ fused training')

    step = 0
    for batch in dl:
        # batch is list of tuples (vq_ids, text_ids); pad to max len
        vq_list, text_list = zip(*batch)
        max_vq = max(x.size(0) for x in vq_list)
        max_tx = max(x.size(0) for x in text_list)
        vq_batch = torch.stack([torch.cat([x, torch.zeros(max_vq - x.size(0), dtype=torch.long)]) for x in vq_list])
        tx_batch = torch.stack([torch.cat([x, torch.zeros(max_tx - x.size(0), dtype=torch.long)]) for x in text_list])
        # Map raw VQ ids into reserved unified vocab range for video if needed
        try:
            offset = int(mmc.video_vocab_start)
            vq_batch = vq_batch + offset
        except Exception:
            pass
        inp = torch.cat([vq_batch, tx_batch], dim=1).to(args.device)
        labels = inp.clone()
        out = model(inp)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        'train_args': {
            'image_size': [int(args.image_size[0]), int(args.image_size[1])],
            'steps': int(args.steps),
        }
    }
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"Saved fused VL-video VQ checkpoint to {final}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.training.data.video_jsonl import VideoDataModule
from omnicoder.config import get_mobile_preset


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
    ap = argparse.ArgumentParser(description="Fused Video+Text pretraining: video+text -> CE loss on text segment")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to JSONL with {video, text}")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb", choices=["mobile_4gb", "mobile_2gb"]) 
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--out", type=str, default="weights/omnicoder_vl_video_fused.pt")
    args = ap.parse_args()

    # Data
    dm = VideoDataModule(jsonl_path=args.jsonl, batch_size=args.batch_size, image_size=tuple(args.image_size), max_frames=args.max_frames)
    dl: DataLoader = dm.loader()

    # Model and composer
    model = build_model(args.mobile_preset, seq_len_hint=4096)
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
    model.to(args.device)
    composer.to(args.device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    step = 0
    for videos, input_ids in dl:
        videos = videos.to(args.device)  # (B,T,3,H,W)
        input_ids = input_ids.to(args.device)
        # Build fused features: [VID_BOS, proj(frame_tokens[0..N])..., VID_EOS, text_emb...]
        fused = composer.fuse_text_video(model_with_embed=model, input_ids=input_ids, video_btchw=videos, max_frames=args.max_frames)
        # Forward on features
        outputs = model(fused, use_cache=False)  # type: ignore[arg-type]
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        # Compute CE on text segment only (next-token)
        bsz, t_fused, vocab = logits.shape
        t_text = input_ids.size(1)
        t_nontext = t_fused - t_text
        text_logits = logits[:, t_nontext - 1 : t_fused - 1, :]
        text_labels = input_ids
        loss = ce(text_logits.reshape(-1, vocab), text_labels.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 1 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
        if step >= args.steps:
            break

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {
        'train_args': {
            'image_size': [int(args.image_size[0]), int(args.image_size[1])],
            'steps': int(args.steps),
        }
    }
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"Saved fused Video+Text checkpoint to {final}")


if __name__ == "__main__":
    main()



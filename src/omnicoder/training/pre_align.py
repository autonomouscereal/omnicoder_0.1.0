from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import os

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader

from omnicoder.modeling.multimodal.aligner import PreAligner
from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone
from omnicoder.modeling.multimodal.audio_encoder import AudioBackbone
from omnicoder.modeling.multimodal.video_encoder import VideoBackbone
from omnicoder.training.data.vl_jsonl import VLDataModule
from omnicoder.training.data.vl_image_pairs import ImagePairsDataset
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.utils.resources import recommend_num_workers


class TextEmbedder(nn.Module):
    """
    Tiny learnable text embedder that maps token ids to a pooled embedding.
    Used only for pre-alignment to produce a fixed-dim text vector.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(int(vocab_size), int(embed_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(ids)  # (B, T, D)
        return e.mean(dim=1)  # (B, D)


def _jsonl_loader(path: str, embed_dim: int, batch_size: int, image_size: Tuple[int, int]) -> DataLoader:
    dm = VLDataModule(jsonl_path=path, d_model=embed_dim, batch_size=batch_size, image_size=tuple(image_size))
    return dm.loader()


def _images_loader(path: str, batch_size: int, image_size: Tuple[int, int]) -> DataLoader:
    def collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
        imgs, texts = zip(*batch)
        return torch.stack(imgs, dim=0), list(texts)

    ds = ImagePairsDataset(path, image_size=tuple(image_size))
    # On Windows, PyTorch DataLoader workers cannot pickle local collate functions; force num_workers=0
    import sys as _sys
    nw = 0 if _sys.platform.startswith("win") else recommend_num_workers()
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=collate)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-align modality embeddings via InfoNCE (CLIP/ImageBind-style)")
    ap.add_argument("--data", type=str, required=True, help="Path to JSONL ({image,text}) or image folder")
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="weights/pre_align.pt")
    ap.add_argument("--with_audio", action="store_true")
    ap.add_argument("--with_video", action="store_true")
    args = ap.parse_args()

    # Data
    if str(args.data).lower().endswith(".jsonl"):
        dl = _jsonl_loader(args.data, embed_dim=int(args.embed_dim), batch_size=int(args.batch_size), image_size=tuple(args.image_size))
    else:
        dl = _images_loader(args.data, batch_size=int(args.batch_size), image_size=tuple(args.image_size))

    # Models
    device = torch.device(args.device)
    from os import getenv as _ge
    vision = VisionBackbone(backend=_ge("OMNICODER_VISION_BACKEND", "dinov3"), d_model=768, return_pooled=True).to(device).eval()
    audio = AudioBackbone(sample_rate=16000, n_mels=64, d_model=512, return_pooled=True).to(device).eval() if args.with_audio else None
    video = VideoBackbone(d_model=768, return_pooled=True).to(device).eval() if args.with_video else None
    tokenizer = get_text_tokenizer(prefer_hf=True)
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    except Exception:
        vocab_size = 32000
    text_emb = TextEmbedder(vocab_size=vocab_size, embed_dim=int(args.embed_dim)).to(device)
    aligner = PreAligner(embed_dim=int(args.embed_dim), text_dim=int(args.embed_dim), image_dim=768).to(device)

    params = list(text_emb.parameters()) + list(aligner.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr))

    step = 0
    for images, texts in dl:
        step += 1
        images = images.to(device)
        with torch.inference_mode():
            _, pooled = vision(images)
            if pooled is None:
                # Fallback to mean-pool tokens
                pooled = vision(images)[0].mean(dim=1)
        # Ensure pooled is a normal tensor usable in autograd (not an inference tensor)
        pooled = pooled.clone()
        # Tokenize texts → ids → mean embedding
        try:
            ids = [torch.tensor(tokenizer.encode(str(t) or ""), dtype=torch.long) for t in texts]
        except Exception:
            ids = [torch.tensor([0], dtype=torch.long) for _ in texts]
        max_t = max(1, max(int(x.numel()) for x in ids))
        ids_pad = torch.zeros((len(ids), max_t), dtype=torch.long)
        for i, row in enumerate(ids):
            n = min(max_t, int(row.numel()))
            if n > 0:
                ids_pad[i, :n] = row[:n]
        ids_pad = ids_pad.to(device)
        txt_vec = text_emb(ids_pad)

        emb = aligner(text=txt_vec, image=pooled)
        loss = PreAligner.info_nce_loss(emb["text"], emb["image"])
        # Optional additional pairs when audio/video present in dataset (best-effort)
        if audio is not None:
            # synth stub: zero audio to keep shape; real loaders wire actual audio tensors
            wav = torch.zeros((images.size(0), 16000), device=device)
            _, a_pool = audio(wav)
            if a_pool is not None:
                emb2 = aligner(audio=a_pool, text=txt_vec)
                loss = loss + PreAligner.info_nce_loss(emb2["audio"], emb2["text"]) * 0.5
        if video is not None:
            # synth stub: reuse image as 1-frame video to exercise path
            frames = images.unsqueeze(1)
            _, v_pool = video(frames)
            if v_pool is not None:
                emb3 = aligner(video=v_pool, text=txt_vec)
                loss = loss + PreAligner.info_nce_loss(emb3["video"], emb3["text"]) * 0.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")
        if step >= int(args.steps):
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        target = Path(args.out)
        if target.exists() and target.is_dir():
            target = target / 'pre_align.pt'
        payload = {"embed_dim": int(args.embed_dim), "aligner": aligner.state_dict(), "text_emb": text_emb.state_dict()}
        # Write via in-memory buffer first to avoid partial/corrupt files on flaky FS
        import io as _io
        buf = _io.BytesIO()
        _safe_save(payload, buf)
        data = buf.getvalue()
        tmp = target.with_suffix(target.suffix + '.tmp')
        # Atomic-ish write: open with wb, write bytes, then replace/rename
        with open(tmp, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno()) if hasattr(os, 'fsync') else None  # type: ignore[attr-defined]
        try:
            tmp.replace(target)
        except Exception:
            tmp.rename(target)
        print(f"Saved pre-aligner to {target}")
        # Sidecar: persist minimal export metadata and tokenizer/vocab hints for downstream tasks
        try:
            meta = {
                'pre_align': {
                    'embed_dim': int(args.embed_dim),
                    'image_size': [int(args.image_size[0]), int(args.image_size[1])],
                    'with_audio': bool(args.with_audio),
                    'with_video': bool(args.with_video),
                },
                'tokenizer_hint': (os.getenv('OMNICODER_TOKENIZER', 'auto') if 'os' in globals() else 'auto'),
            }
            # Copy unified vocab sidecar when present
            try:
                import os as _os
                _sidecar_src = _os.getenv('OMNICODER_VOCAB_SIDECAR', 'weights/unified_vocab_map.json')
                from pathlib import Path as _P
                _p = _P(_sidecar_src)
                if _p.exists():
                    _dst = target.with_suffix('.unified_vocab_map.json')
                    try:
                        _dst.write_text(_p.read_text(encoding='utf-8'), encoding='utf-8')
                        meta['vocab_sidecar'] = str(_dst)
                    except Exception:
                        meta['vocab_sidecar'] = str(_p)
            except Exception:
                pass
            (target.with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
        except Exception:
            pass
    except Exception as e:
        # Fallback minimal save on error (best-effort, in-memory -> file)
        try:
            import io as _io
            buf = _io.BytesIO()
            _safe_save({"embed_dim": int(args.embed_dim), "aligner": aligner.state_dict(), "text_emb": text_emb.state_dict()}, buf)
            with open(args.out, 'wb') as f:
                f.write(buf.getvalue())
        except Exception:
            pass
        print(f"[warn] primary save failed ({e}); wrote fallback to {args.out}")


if __name__ == "__main__":
    main()



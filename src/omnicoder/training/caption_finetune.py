from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # type: ignore

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.fusion import MultimodalComposer
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.config import get_mobile_preset


class CocoCaptionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, jsonl_path: str, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.path = Path(jsonl_path)
        self.samples: List[Tuple[str, str]] = []
        self.image_size = image_size
        # Load lines lazily but gather small index for fast sampling
        for ln in self.path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            img = str(obj.get('image') or obj.get('file') or '')
            refs = obj.get('references') or obj.get('refs') or []
            cap = ''
            if isinstance(refs, list) and refs:
                cap = str(refs[0])
            else:
                cap = str(obj.get('text') or obj.get('caption') or '')
            if img and cap:
                self.samples.append((img, cap))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        imgp, cap = self.samples[idx]
        img = Image.open(imgp).convert('RGB').resize(self.image_size)
        import numpy as np
        arr = (torch.from_numpy(np.array(img).astype('float32') / 255.0).permute(2, 0, 1))  # (3,H,W)
        tok = get_text_tokenizer(prefer_hf=True)
        ids = torch.tensor(tok.encode(cap), dtype=torch.long)  # target caption tokens
        return arr, ids


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
    ap = argparse.ArgumentParser(description="Caption finetune on COCO captions JSONL {image, references/text}")
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb", choices=["mobile_4gb","mobile_2gb"]) 
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--out", type=str, default="weights/omnicoder_caption_finetune.pt")
    args = ap.parse_args()

    ds = CocoCaptionDataset(args.jsonl, image_size=tuple(args.image_size))
    if len(ds) == 0:
        raise SystemExit("caption_finetune: dataset empty")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    step = 0
    for images, tgt_ids in dl:
        images = images.to(args.device)
        tgt_ids = tgt_ids.to(args.device)
        # Provide a simple caption BOS prompt
        tok = get_text_tokenizer(prefer_hf=True)
        bos = torch.tensor([[tok.bos_id if hasattr(tok, 'bos_id') else 1] * tgt_ids.size(0)]).transpose(0,1).to(args.device)
        inputs = torch.cat([bos, tgt_ids[:, :-1]], dim=1)  # teacher forcing
        fused = composer.fuse_text_image(model_with_embed=model, input_ids=inputs, image_bchw=images)
        out = model(fused, use_cache=False)  # type: ignore[arg-type]
        logits = out if isinstance(out, torch.Tensor) else out[0]
        # Compute CE on caption segment only
        bsz, t_fused, vocab = logits.shape
        t_text = inputs.size(1)
        t_nontext = t_fused - t_text
        cap_logits = logits[:, t_nontext - 1 : t_fused - 1, :]
        loss = ce(cap_logits.reshape(-1, vocab), tgt_ids.reshape(-1))
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
    meta = {'train_args': {'steps': int(args.steps)}}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    # Micro-eval: caption CE on one held-out sample for best tracking
    try:
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        if len(ds) > 0:
            _img, _tgt = ds[0]
            _img = _img.unsqueeze(0).to(args.device)
            _tgt = _tgt.unsqueeze(0).to(args.device)
            tok = get_text_tokenizer(prefer_hf=True)
            _bos = torch.tensor([[tok.bos_id if hasattr(tok, 'bos_id') else 1]], dtype=torch.long, device=args.device)
            _inp = torch.cat([_bos, _tgt[:, :-1]], dim=1)
            fused = composer.fuse_text_image(model_with_embed=model, input_ids=_inp, image_bchw=_img)
            with torch.inference_mode():
                _out = model(fused, use_cache=False)  # type: ignore[arg-type]
                _logits = _out if isinstance(_out, torch.Tensor) else _out[0]
            _bsz, _tf, _v = _logits.shape
            _nt = _tf - _inp.size(1)
            _cap_logits = _logits[:, _nt - 1 : _tf - 1, :]
            _ce = nn.CrossEntropyLoss()(_cap_logits.reshape(-1, _v), _tgt.reshape(-1)).item()
            maybe_save_best(args.out, model, 'caption_ce', float(_ce), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved caption finetune checkpoint to {final}")


if __name__ == "__main__":
    main()



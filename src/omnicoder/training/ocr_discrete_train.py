from __future__ import annotations

"""
OCR-centric and discrete image token generation pretrain

This script wires a discrete image tokenizer (VQ-VAE) and a simple OCR dataset
into a joint training loop that aligns vision tokens and teaches the model to
emit image code sequences for text→image tasks. It leverages existing VQ models
provided in modeling/multimodal/vqvae.py and image_vq.py where available.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn

from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.multimodal.vqvae import ImageVQVAE
from omnicoder.utils.logger import get_logger


def _load_ocr_pairs(jsonl: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    import json
    with open(jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                img = str(obj.get('image', ''))
                txt = str(obj.get('text', ''))
                if img and txt:
                    rows.append((img, txt))
            except Exception:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR + discrete image token generation pretrain")
    ap.add_argument("--ocr_jsonl", type=str, required=True, help="JSONL with {image, text}")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--out", type=str, default="weights/ocr_discrete.pt")
    args = ap.parse_args()

    log = get_logger("omnicoder.ocr")
    tok = get_text_tokenizer(prefer_hf=True)
    from omnicoder.inference.generate import build_mobile_model_by_name
    model: OmniTransformer = build_mobile_model_by_name(args.mobile_preset)
    model.to(args.device).train()
    # VQ-VAE image tokenizer
    vq = ImageVQVAE(codebook_size=8192)
    vq.to(args.device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    rows = _load_ocr_pairs(args.ocr_jsonl)
    if not rows:
        raise SystemExit("ocr_discrete_train: dataset empty")

    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore

    for it in range(1, int(args.iters) + 1):
        img_path, text = rows[(it - 1) % len(rows)]
        try:
            im = Image.open(img_path).convert("RGB").resize(tuple(int(x) for x in args.image_size))
        except Exception:
            continue
        arr = np.array(im).astype("float32") / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            out = vq.encode(img)
            # Expect indices shape (B, T)
            if isinstance(out, tuple):
                codes = out[-1]
            else:
                codes = out
        # Build a target sequence by mapping code indices into a reserved vocab slice
        # For simplicity, we offset by 32000
        codes_offset = codes + 32000
        # Prepare text → image-token supervision via teacher forcing
        text_ids = torch.tensor([tok.encode(text)], dtype=torch.long, device=args.device)
        target = torch.cat([text_ids, codes_offset.long()], dim=1)
        # Forward with teacher forcing implicit via next-token prediction; compute CE on the image code region
        model.eval()
        with torch.inference_mode():
            out_full = model(target[:, :-1], use_cache=False)
            logits = out_full[0] if isinstance(out_full, tuple) else out_full
        model.train()
        # Loss on the image-token segment only
        start = int(text_ids.size(1)) - 1
        logits_img = logits[:, start : start + int(codes_offset.size(1)), :]
        loss = nn.functional.cross_entropy(logits_img.reshape(-1, logits_img.size(-1)), target[:, start + 1 : start + 1 + int(codes_offset.size(1))].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (it % 20) == 0 or it == 1:
            try:
                log.info("ocr_iter=%d loss=%.4f", int(it), float(loss.item()))
            except Exception:
                pass

    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta={'train_args': {'steps': int(args.steps)}})
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    print(f"[ocr_discrete] saved to {final}")


if __name__ == "__main__":
    main()



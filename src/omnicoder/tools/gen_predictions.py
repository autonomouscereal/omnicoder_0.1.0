from __future__ import annotations

"""
Batch prediction generator for evaluation JSONLs.

Supports tasks:
- vqa: input lines with {image, question, answer}; outputs with an added 'prediction'
- captions: input lines with {image, references:[...]}; outputs with an added 'prediction'

Example:
  python -m omnicoder.tools.gen_predictions \
    --task vqa --input data/vqa/okvqa_eval.jsonl \
    --out weights/bench_preds/okvqa_pred.jsonl \
    --device cuda --preset mobile_4gb --ckpt weights/omnicoder_vl_fused.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def _ensure_model_and_tokenizer(preset: str, device: str, ckpt: str | None):
    from omnicoder.inference.generate import build_mobile_model_by_name, maybe_load_checkpoint  # type: ignore
    from omnicoder.training.simple_tokenizer import get_text_tokenizer  # type: ignore

    model = build_mobile_model_by_name(preset)
    if ckpt:
        try:
            maybe_load_checkpoint(model, ckpt)
        except Exception:
            pass
    model.eval().to(device)
    tok = get_text_tokenizer(prefer_hf=True)
    return model, tok


@torch.no_grad()
def _gen_text_from_image(
    image_path: str,
    question: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore
    from omnicoder.modeling.multimodal.fusion import MultimodalComposer  # type: ignore
    from omnicoder.inference.generate import prime_kv_with_features, continue_generate_from_primed  # type: ignore

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,224,224)

    # Prepare prompt ids
    prompt = question or "Describe the image"
    ids = torch.tensor([[tokenizer.encode(prompt)]], dtype=torch.long, device=device).squeeze(0)

    composer = MultimodalComposer(d_model=model.embed.embedding_dim, vision_dim=384)
    fused = composer.fuse_text_image(model_with_embed=model, input_ids=ids, image_bchw=img_t)
    past_kv, _ = prime_kv_with_features(model, fused)
    bos_id = 1 if hasattr(tokenizer, "bos_token_id") else 2
    out_ids = continue_generate_from_primed(
        model, past_kv=past_kv, start_token_id=bos_id, max_new_tokens=max_new_tokens
    )
    return tokenizer.decode(out_ids[0].tolist())


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch prediction generator for eval JSONLs")
    ap.add_argument("--task", choices=["vqa", "captions"], required=True)
    ap.add_argument("--input", type=str, required=True, help="Input eval JSONL path")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL with predictions")
    ap.add_argument("--preset", type=str, default="mobile_4gb")
    ap.add_argument("--ckpt", type=str, default="", help="Optional checkpoint to load (e.g., VL fused)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    inp = Path(args.input)
    rows = list(_iter_jsonl(inp))
    if not rows:
        print(json.dumps({"wrote": 0, "out": args.out}))
        return

    model, tok = _ensure_model_and_tokenizer(args.preset, args.device, args.ckpt or None)

    out_rows: List[Dict] = []
    for rec in rows:
        try:
            img = str(rec.get("image") or rec.get("file") or "").strip()
            if not img:
                out_rows.append(rec)
                continue
            if args.task == "vqa":
                q = str(rec.get("question") or rec.get("text") or "Describe the image")
                pred = _gen_text_from_image(img, q, model, tok, args.device, max_new_tokens=int(args.max_new_tokens))
                rec["prediction"] = pred
            else:
                # captions: produce a caption ignoring references
                prompt = "Caption the image concisely."
                pred = _gen_text_from_image(img, prompt, model, tok, args.device, max_new_tokens=int(args.max_new_tokens))
                rec["prediction"] = pred
        except Exception as e:
            rec["prediction_error"] = str(e)
        out_rows.append(rec)

    _write_jsonl(Path(args.out), out_rows)
    print(json.dumps({"wrote": len(out_rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()



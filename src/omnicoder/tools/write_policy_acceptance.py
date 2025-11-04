from __future__ import annotations

"""
Compute/write acceptance thresholds for the learned write‑policy head.

Given a JSONL of teacher marks (each line: {"text": str, "write": 0|1}),
evaluate a model (or use labels) to derive a target acceptance threshold and
write it to profiles/write_policy_thresholds.json keyed by preset.

Usage:
  python -m omnicoder.tools.write_policy_acceptance \
    --marks examples/teacher_marks.jsonl --preset mobile_4gb \
    --out profiles/write_policy_thresholds.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch


def _iter_marks(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            if isinstance(j, dict) and 'text' in j:
                yield j


def main() -> None:
    ap = argparse.ArgumentParser(description="Write-policy acceptance threshold writer")
    ap.add_argument('--marks', type=str, required=True, help='Path to teacher marks JSONL with fields {text, write}')
    ap.add_argument('--preset', type=str, default='mobile_4gb')
    ap.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--out', type=str, default='profiles/write_policy_thresholds.json')
    ap.add_argument('--use_model', action='store_true', help='If set, evaluate model.write_head(text) to derive threshold; else use label mean')
    args = ap.parse_args()

    # Default: threshold = mean label (target acceptance)
    thr = None
    if args.use_model:
        # Evaluate a tiny model on the marks to compute mean predicted acceptance
        from omnicoder.modeling.transformer_moe import OmniTransformer  # lazy import
        from omnicoder.training.simple_tokenizer import get_text_tokenizer
        model = OmniTransformer()
        model.to(args.device).eval()
        tok = get_text_tokenizer(prefer_hf=True)
        preds = []
        with torch.no_grad():
            for j in _iter_marks(args.marks):
                text = str(j.get('text', ''))
                ids = torch.tensor([[tok.encode(text)[:32]]], dtype=torch.long)
                ids = torch.ops.aten.reshape.default(ids, (1, -1)).to(args.device)
                out = model(ids, use_cache=False, return_hidden=True)
                hid = out[-1] if isinstance(out, tuple) else model.ln_f(model.embed(ids))
                p = torch.sigmoid(model.write_head(hid)).mean().item()
                preds.append(float(p))
        if preds:
            thr = float(sum(preds) / len(preds))
    if thr is None:
        labels = []
        for j in _iter_marks(args.marks):
            try:
                labels.append(int(j.get('write', 0)))
            except Exception:
                continue
        if labels:
            thr = float(sum(labels) / len(labels))
        else:
            thr = 0.5

    # Write/update thresholds JSON
    outp = Path(args.out)
    try:
        data: Dict[str, float] = {}
        if outp.exists():
            old = json.loads(outp.read_text(encoding='utf-8'))
            if isinstance(old, dict):
                for k, v in old.items():
                    try:
                        data[str(k)] = float(v)
                    except Exception:
                        continue
        data[str(args.preset)] = float(thr)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(json.dumps({'preset': args.preset, 'threshold': thr, 'out': str(outp)}))
    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    main()



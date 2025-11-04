from __future__ import annotations

"""
Teacher verification across domains (best-effort, dependency-light).

Given a domain and a JSONL dataset, compare student outputs to one or more
teacher models and emit agreement statistics per teacher and overall.

Supports:
- text/code-style prompts (JSONL lines containing 'prompt' and optional 'target')

Notes:
- Uses transformers when available for teachers; if not installed, skips teacher side.
- Uses the project's generate() function for the student model.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def _read_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= max(1, int(limit)):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                out.append({"prompt": line})
    return out


def _student_generate(prompts: List[str], preset: str, device: str) -> List[str]:
    try:
        from omnicoder.inference.generate import build_mobile_model_by_name, get_text_tokenizer, generate  # type: ignore
        import torch
        model = build_mobile_model_by_name(preset)
        model.eval().to(device)
        tok = get_text_tokenizer()
        outs: List[str] = []
        for p in prompts:
            try:
                ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=device)
                gen = generate(model, ids, max_new_tokens=64, temperature=0.8, top_k=40)
                if isinstance(gen, tuple):
                    gen_ids = gen[0]
                else:
                    gen_ids = gen
                txt = tok.decode(gen_ids[0].tolist() if hasattr(gen_ids, 'tolist') else list(gen_ids))
                outs.append(str(txt))
            except Exception:
                outs.append("")
        return outs
    except Exception:
        return [""] * len(prompts)


def _teacher_generate(prompts: List[str], teacher_id: str, device: str) -> List[str]:
    outs: List[str] = []
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
        tok = AutoTokenizer.from_pretrained(teacher_id)
        mdl = AutoModelForCausalLM.from_pretrained(teacher_id, dtype="auto", device_map="auto" if device.startswith("cuda") else None)
        for p in prompts:
            try:
                ids = tok.encode(p, return_tensors='pt').to(mdl.device)
                out = mdl.generate(ids, max_new_tokens=64, do_sample=True, top_k=40, temperature=0.8)
                outs.append(tok.decode(out[0], skip_special_tokens=True))
            except Exception:
                outs.append("")
    except Exception:
        outs = [""] * len(prompts)
    return outs


def _agree(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz  # type: ignore
        return float(fuzz.partial_ratio(a, b)) / 100.0
    except Exception:
        a = a.strip().lower()
        b = b.strip().lower()
        if not a or not b:
            return 0.0
        return 1.0 if (a in b or b in a) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher verification (agreement) across a JSONL dataset")
    ap.add_argument("--domain", type=str, default="text", choices=["text","code"])  # other domains can be added later
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--teachers", type=str, default="", help="Space-separated HF model ids for teachers")
    ap.add_argument("--student_preset", type=str, default=os.getenv("OMNICODER_STUDENT_PRESET", "mobile_4gb"))
    ap.add_argument("--device", type=str, default=os.getenv("OMNICODER_TRAIN_DEVICE", "cuda"))
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--out", type=str, default="weights/teacher_verify.json")
    args = ap.parse_args()

    rows = _read_jsonl(args.data, args.limit)
    prompts = [str(r.get('prompt') or r.get('question') or '') for r in rows]
    student = _student_generate(prompts, args.student_preset, args.device)

    res: Dict[str, Any] = {"samples": len(rows), "teachers": {}, "overall": {}}
    tlist = [t for t in args.teachers.split() if t.strip()]
    for t in tlist:
        outs = _teacher_generate(prompts, t, args.device)
        scores: List[float] = []
        for s, o in zip(student, outs):
            scores.append(_agree(s, o))
        res["teachers"][t] = {"agreement": (sum(scores)/max(1,len(scores))) if scores else 0.0}

    # If targets exist, compute simple student target match too
    targets = [str(r.get('target') or r.get('answer') or '') for r in rows]
    if any(targets):
        t_scores = []
        for s, gt in zip(student, targets):
            if gt:
                t_scores.append(_agree(s, gt))
        if t_scores:
            res["overall"]["student_vs_target"] = float(sum(t_scores)/len(t_scores))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()



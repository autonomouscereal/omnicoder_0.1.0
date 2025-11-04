from __future__ import annotations

import argparse


def _read_jsonl(path: str):
    """
    Read a JSONL file of ASR pairs. Each line should be a JSON object with keys like:
      {"file": "/path/to/audio.wav", "ref": "reference transcript", "hyp": "(optional)"}
    The "hyp" field is optional and may be ignored by some callers.
    Returns a list of dicts.
    """
    rows = []
    try:
        import json
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _compute_fad(gen_dir: str, ref_dir: str) -> float:
    try:
        from pesq import pesq  # type: ignore
        # Placeholder; proper FAD requires VGGish/embedding-based metrics
    except Exception:
        print("[fad] Please install an FAD implementation (e.g., torch-fidelity-audio or implement VGGish embeddings). Placeholder only.")
        return -1.0
    return -1.0


def _compute_wer(jsonl_path: str) -> float:
    try:
        import json
        from jiwer import wer  # type: ignore
    except Exception:
        print("[wer] pip install jiwer")
        return -1.0
    refs = []
    hyps = []
    for line in open(jsonl_path, "r", encoding="utf-8", errors="ignore"):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except Exception:
            continue
        r = ex.get("ref") or ex.get("reference") or ""
        h = ex.get("hyp") or ex.get("hypothesis") or ""
        if r and h:
            refs.append(r)
            hyps.append(h)
    if not refs:
        return -1.0
    return float(wer(refs, hyps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio evaluation: FAD (placeholder) and WER")
    ap.add_argument("--mode", choices=["fad", "wer"], required=True)
    ap.add_argument("--gen_dir", type=str, default="")
    ap.add_argument("--ref_dir", type=str, default="")
    ap.add_argument("--jsonl", type=str, default="")
    args = ap.parse_args()

    if args.mode == "wer":
        if not args.jsonl:
            print("--jsonl is required for WER mode")
            return
        score = _compute_wer(args.jsonl)
        if score >= 0:
            print(f"WER: {score:.3f}")
        else:
            print("WER not computed (missing dependency or invalid JSONL)")
        return

    # FAD
    if not args.gen_dir or not args.ref_dir:
        print("--gen_dir and --ref_dir are required for FAD mode")
        return
    score = _compute_fad(args.gen_dir, args.ref_dir)
    if score >= 0:
        print(f"FAD: {score:.3f}")
    else:
        print("FAD not computed (missing dependency).")


if __name__ == "__main__":
    main()

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
    del gen_dir, ref_dir
    _metric_unavailable(
        "FAD",
        "Install and wire an official Frechet Audio Distance implementation with fixed embedding model, "
        "dataset snapshot hash, and scorer version before publishing scores.",
    )
    return -1.0


def _compute_clap(jsonl_path: str) -> float:
    del jsonl_path
    _metric_unavailable(
        "CLAPScore",
        "Install and wire an official CLAP/CLAPScore evaluator before publishing audio-text alignment scores.",
    )
    return -1.0


def _compute_mos(jsonl_path: str) -> float:
    del jsonl_path
    _metric_unavailable(
        "MOS",
        "Wire an official MOSNet implementation or human MOS protocol before publishing speech/music quality scores.",
    )
    return -1.0


def _metric_unavailable(metric: str, reason: str) -> None:
    print(f"[{metric.lower()}] unavailable_for_official_scoring: {reason}")


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
    ap = argparse.ArgumentParser(
        description=(
            "Audio evaluation: WER when jiwer is installed; FAD/CLAPScore/MOS fail closed "
            "until official metric packages or human protocols are wired."
        )
    )
    ap.add_argument("--mode", choices=["fad", "wer", "clap", "mos"], required=True)
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

    if args.mode == "clap":
        score = _compute_clap(args.jsonl)
        if score >= 0:
            print(f"CLAPScore: {score:.3f}")
        else:
            print("CLAPScore not computed (official scorer unavailable).")
        return

    if args.mode == "mos":
        score = _compute_mos(args.jsonl)
        if score >= 0:
            print(f"MOS: {score:.3f}")
        else:
            print("MOS not computed (official scorer unavailable).")
        return

    # FAD
    if not args.gen_dir or not args.ref_dir:
        print("--gen_dir and --ref_dir are required for FAD mode")
        return
    score = _compute_fad(args.gen_dir, args.ref_dir)
    if score >= 0:
        print(f"FAD: {score:.3f}")
    else:
        print("FAD not computed (official scorer unavailable).")


if __name__ == "__main__":
    main()

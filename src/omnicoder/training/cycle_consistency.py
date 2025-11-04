from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Cycle-consistency hooks: caption/transcribe generated media vs prompts")
    ap.add_argument("--media_dir", type=str, required=True, help="Folder of generated media")
    ap.add_argument("--out", type=str, default="weights/cycle_consistency.jsonl")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # Best-effort: try BLIP for image captioning and Whisper small for ASR if available
    cap = None
    asr = None
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        cap_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(args.device)
    except Exception:
        cap = None
    try:
        import whisper  # type: ignore
        asr = whisper.load_model("small", device=args.device)
    except Exception:
        asr = None

    import os
    from PIL import Image  # type: ignore
    import soundfile as sf  # type: ignore
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for fn in sorted(os.listdir(args.media_dir)):
            p = Path(args.media_dir) / fn
            rec = {"file": str(p)}
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and cap is not None:
                try:
                    img = Image.open(p).convert("RGB")
                    inputs = cap_proc(images=img, return_tensors="pt").to(args.device)
                    out_ids = cap.generate(**inputs, max_length=32)
                    text = cap_proc.decode(out_ids[0], skip_special_tokens=True)
                    rec["caption"] = text
                except Exception:
                    rec["caption"] = None
            if p.suffix.lower() in (".wav", ".mp3") and asr is not None:
                try:
                    audio, sr = sf.read(str(p))
                    t = asr.transcribe(str(p))
                    rec["transcript"] = t.get("text", "")
                except Exception:
                    rec["transcript"] = None
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote cycle-consistency logs to {out}")


if __name__ == "__main__":
    main()



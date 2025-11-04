import argparse, json
from dataclasses import asdict
from pathlib import Path

from omnicoder.config import MobilePreset, MobilePreset2GB, MultiModalConfig
from omnicoder.modeling.multimodal.vocab_map import VocabSidecar


def main():
    ap = argparse.ArgumentParser(description="Export mobile presets to JSON")
    ap.add_argument('--out', type=str, default='weights/presets.json')
    ap.add_argument('--emit_vocab_sidecar', action='store_true', help='Also emit unified vocab sidecar JSON for on-device tokenization')
    args = ap.parse_args()

    data = {
        'mobile_4gb': asdict(MobilePreset()),
        'mobile_2gb': asdict(MobilePreset2GB()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Wrote presets to {args.out}")
    if args.emit_vocab_sidecar:
        sidecar = VocabSidecar.from_config(MultiModalConfig())
        sidecar.save(out_path.parent / 'unified_vocab_sidecar.json')
        print(f"Wrote vocab sidecar to {out_path.parent / 'unified_vocab_sidecar.json'}")


if __name__ == '__main__':
    main()



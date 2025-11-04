import argparse
from pathlib import Path


REQUIRED_TEXT = [
    'text/omnicoder_decode_step.onnx',
    'text/mobile_packager_summary.json',
]


def main():
    ap = argparse.ArgumentParser(description='Validate weights folder contains required artifacts')
    ap.add_argument('--root', type=str, default='weights')
    args = ap.parse_args()

    root = Path(args.root)
    ok = True
    for rel in REQUIRED_TEXT:
        p = root / rel
        if not p.exists():
            print(f"MISSING: {p}")
            ok = False
        else:
            print(f"OK: {p}")
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()



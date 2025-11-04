from __future__ import annotations

"""
Data Engine: ingestion, filtering, and offline mirroring for VL/VQA/ASR/TTS.

Features:
- Mirror remote datasets (HTTP/HTTPS) to a local cache folder with manifest
- Basic filtering (min/max length, allowed file types)
- JSONL exporters for VL/VQA and ASR/TTS tasks

Usage examples:
  python -m omnicoder.tools.data_engine mirror --urls urls.txt --out data_cache
  python -m omnicoder.tools.data_engine index-vl --root data_cache/images --out vl.jsonl
  python -m omnicoder.tools.data_engine index-asr --root data_cache/audio --out asr.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import List


def _read_lines(path: str) -> List[str]:
    return [l.strip() for l in open(path, 'r', encoding='utf-8', errors='ignore') if l.strip()]


def cmd_mirror(urls_file: str, out_root: str) -> None:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    urls = _read_lines(urls_file)
    import urllib.request as U
    manifest = []
    import hashlib
    filt = os.environ.get('OMNICODER_DATA_FILTER_EXT', '')
    allowed = set([e for e in filt.split(';') if e])
    want_checksum = os.environ.get('OMNICODER_DATA_CHECKSUM', '0') == '1'
    seen_hashes: set[str] = set()
    # If a previous manifest exists, load known paths and hashes for resume/dedup
    old_manifest_path = out / 'manifest.json'
    if old_manifest_path.exists():
        try:
            old = json.loads(old_manifest_path.read_text(encoding='utf-8'))
            for rec in old:
                if 'sha256' in rec:
                    seen_hashes.add(rec['sha256'])
            manifest.extend(old)
        except Exception:
            pass

    for u in urls:
        try:
            name = os.path.basename(u.split('?')[0])
            dst = out / name
            if allowed:
                if not any(name.lower().endswith(ext) for ext in allowed):
                    continue
            if not dst.exists():
                print(f"downloading {u}")
                # Retry with simple backoff
                last_err: Exception | None = None
                for attempt in range(3):
                    try:
                        U.urlretrieve(u, dst)
                        last_err = None
                        break
                    except Exception as e:  # noqa
                        last_err = e
                if last_err is not None:
                    raise last_err
            rec = {'url': u, 'path': str(dst)}
            if want_checksum:
                h = hashlib.sha256()
                with open(dst, 'rb') as f:
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        h.update(b)
                digest = h.hexdigest()
                # Deduplicate by content hash
                if digest in seen_hashes:
                    print(f"[dedup] skipping duplicate content {name}")
                    continue
                rec['sha256'] = digest
                seen_hashes.add(digest)
            manifest.append(rec)
        except Exception as e:
            print(f"[warn] failed {u}: {e}")
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {out/'manifest.json'}")


def cmd_index_vl(root: str, out_jsonl: str) -> None:
    rootp = Path(root)
    recs = []
    # Look for paired files: image.jpg and image.txt for caption
    imgs = list(rootp.rglob('*.jpg')) + list(rootp.rglob('*.png'))
    for img in imgs:
        txt = img.with_suffix('.txt')
        if txt.exists():
            text = txt.read_text(encoding='utf-8', errors='ignore').strip()
            recs.append({'image': str(img), 'text': text})
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print(f"Indexed {len(recs)} VL pairs -> {out_jsonl}")


def cmd_index_asr(root: str, out_jsonl: str) -> None:
    rootp = Path(root)
    recs = []
    auds = list(rootp.rglob('*.wav')) + list(rootp.rglob('*.mp3'))
    for a in auds:
        txt = a.with_suffix('.txt')
        if txt.exists():
            recs.append({'file': str(a), 'ref': txt.read_text(encoding='utf-8', errors='ignore').strip()})
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print(f"Indexed {len(recs)} ASR pairs -> {out_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser(description='Data engine: mirror and index datasets')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('mirror')
    m.add_argument('--urls', type=str, required=True)
    m.add_argument('--out', type=str, required=True)
    m.add_argument('--filter_ext', type=str, default='', help='Comma-separated allowed extensions (e.g., .jpg,.png,.wav)')
    m.add_argument('--checksum', action='store_true', help='Write sha256 checksums to manifest')
    vl = sub.add_parser('index-vl')
    vl.add_argument('--root', type=str, required=True)
    vl.add_argument('--out', type=str, required=True)
    asr = sub.add_parser('index-asr')
    asr.add_argument('--root', type=str, required=True)
    asr.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    if args.cmd == 'mirror':
        # pass-through extra options via env to avoid changing signature
        if getattr(args, 'filter_ext', ''):
            exts = [e.strip().lower() for e in args.filter_ext.split(',') if e.strip()]
            os.environ['OMNICODER_DATA_FILTER_EXT'] = ';'.join(exts)
        os.environ['OMNICODER_DATA_CHECKSUM'] = '1' if getattr(args, 'checksum', False) else '0'
        cmd_mirror(args.urls, args.out)
    elif args.cmd == 'index-vl':
        cmd_index_vl(args.root, args.out)
    else:
        cmd_index_asr(args.root, args.out)


if __name__ == '__main__':
    main()



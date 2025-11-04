from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Tuple


DATASETS = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "sha256": "2d3b0e4e7d1f2a022beed579b3709271b4683dc55552f6391f096c886e7f6b9c",
        "filename": "tinyshakespeare.txt",
        "license": "Unknown (public domain or permissive per source repository)",
    },
    "tinystories": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        "sha256": "6c9a235b3d8be6fba74c1d9f7a8d3b1e0e1a7f2b63b37f4b3d56dd8f4b4bfa98",
        "filename": "tinystories.txt",
        "license": "CC BY-SA 4.0",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(dataset: str, out_dir: str) -> Tuple[bool, str, str]:
    ds = DATASETS.get(dataset.lower())
    if not ds:
        raise ValueError(f"Unknown dataset: {dataset}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(out_dir) / ds["filename"]
    try:
        import urllib.request
        print(f"[fetch] Downloading {dataset} ...")
        urllib.request.urlretrieve(ds["url"], dest)
    except Exception as e:
        print(f"[fetch] failed to download: {e}")
        return False, str(dest), ds["license"]
    # checksum if provided
    try:
        want = ds.get("sha256")
        if want:
            got = _sha256(dest)
            if got != want:
                print(f"[fetch] checksum mismatch: {got} != {want}")
                return False, str(dest), ds["license"]
    except Exception:
        pass
    return True, str(dest), ds["license"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch small public datasets for KD")
    ap.add_argument("--dataset", type=str, default="tinyshakespeare", choices=list(DATASETS.keys()))
    ap.add_argument("--out_dir", type=str, default="data/auto_kd")
    args = ap.parse_args()
    ok, path, lic = fetch(args.dataset, args.out_dir)
    print({"ok": ok, "path": path, "license": lic})


if __name__ == "__main__":
    main()



from __future__ import annotations

"""
Write a simple KV prefetch predictor sidecar JSON for the ONNX decode runner.

The predictor currently supports a single policy key:
  {"keep_pages": N}

Usage:
  python -m omnicoder.tools.kv_prefetch_write --out weights/kv_prefetch.json --keep_pages 2

Optionally, if a kv_paging sidecar is available, this tool can derive a default
keep_pages from page_len or total pages observed.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write KV prefetch predictor sidecar JSON")
    ap.add_argument("--out", type=str, default="weights/kv_prefetch.json")
    ap.add_argument("--keep_pages", type=int, default=2)
    ap.add_argument("--kv_paging_sidecar", type=str, default="", help="Optional kv_paging.json to derive a default keep_pages")
    args = ap.parse_args()

    keep = max(1, int(args.keep_pages))
    try:
        if args.kv_paging_sidecar:
            p = Path(args.kv_paging_sidecar)
            if p.exists():
                try:
                    meta = json.loads(p.read_text(encoding="utf-8"))
                    # If sidecar contains total_pages or similar, cap keep_pages to 1/2 total
                    total_pages = int(meta.get("total_pages", 0)) if isinstance(meta, dict) else 0
                    if total_pages > 0:
                        keep = max(1, min(keep, max(1, total_pages // 2)))
                except Exception:
                    pass
    except Exception:
        pass

    out = {"keep_pages": int(keep)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()



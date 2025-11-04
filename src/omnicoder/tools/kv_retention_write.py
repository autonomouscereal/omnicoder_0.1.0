from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a KV retention sidecar (json) for runners to enforce compressive/window policy")
    ap.add_argument("--out", type=str, required=True, help="Path to kv_retention.json to write")
    ap.add_argument("--compressive_slots", type=int, default=4, help="Number of average-compression slots for old prefix")
    ap.add_argument("--window_size", type=int, default=2048, help="Tail window of tokens to keep at full resolution")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "compressive_slots": int(args.compressive_slots),
        "window_size": int(args.window_size),
        "schema": 1,
    }
    out_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"[write] {out_path} -> {blob}")


if __name__ == "__main__":
    main()



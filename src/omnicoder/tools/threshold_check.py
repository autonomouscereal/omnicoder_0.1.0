from __future__ import annotations

import argparse, json, sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Check tokens/s threshold from metrics_canaries.json")
    ap.add_argument("--metrics_json", type=str, default="weights/metrics_canaries.json")
    ap.add_argument("--min_tps", type=float, default=20.0)
    args = ap.parse_args()

    try:
        data = json.loads(open(args.metrics_json, "r", encoding="utf-8").read())
        tps = float(data.get("tokens", {}).get("tokens_per_second", 0.0))
        if tps < float(args.min_tps):
            print(f"[threshold] FAIL: tokens/s {tps:.2f} < {args.min_tps:.2f}")
            sys.exit(2)
        print(f"[threshold] OK: tokens/s {tps:.2f} >= {args.min_tps:.2f}")
    except Exception as e:
        print(f"[threshold] ERROR: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()



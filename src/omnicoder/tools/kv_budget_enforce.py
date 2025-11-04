import argparse, json, os


def compute_kv_bytes(meta: dict) -> int:
    kv_bytes = 0
    if isinstance(meta, dict):
        page_len = int(meta.get('page_len', 0))
        n_layers = int(meta.get('n_layers', 0))
        heads = int(meta.get('heads', 0))
        dl = int(meta.get('dl', 0))
        dl_per_layer = meta.get('dl_per_layer', None)
        spill_bytes = int(meta.get('spill_bytes', 0))
        if isinstance(dl_per_layer, list) and dl_per_layer:
            try:
                kv_bytes = sum(int(d) for d in dl_per_layer) * heads * page_len * 2
            except Exception:
                kv_bytes = n_layers * heads * dl * page_len * 2
        else:
            kv_bytes = n_layers * heads * dl * page_len * 2
        kv_bytes = kv_bytes + max(0, spill_bytes)
    return int(kv_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Enforce KV budget from sidecars; write a summary and exit non-zero on violations")
    ap.add_argument("--sidecar", type=str, required=True, help="Path to kv_paging or kvq sidecar JSON")
    ap.add_argument("--max_kv_mb", type=float, default=float(os.getenv("OMNICODER_MAX_KV_MB", "1024")))
    ap.add_argument("--out", type=str, default="weights/kv_budget_summary.json")
    args = ap.parse_args()

    with open(args.sidecar, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    kv_bytes = compute_kv_bytes(meta)
    mb = kv_bytes / (1024.0 * 1024.0)
    ok = mb <= float(args.max_kv_mb)
    out = {"kv_mb": mb, "max_kv_mb": float(args.max_kv_mb), "pass": ok, "source": args.sidecar}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()



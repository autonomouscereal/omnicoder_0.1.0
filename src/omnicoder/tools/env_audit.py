from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnicoder.utils.env_registry import (
    build_registry,
    dump_index_markdown,
    dump_registry_json,
    load_dotenv_best_effort,
    scan_env_usage,
    sync_env_example,
)
from omnicoder.utils.resources import audit_env as _audit_env_unknown


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit, index, and sync OmniCoder environment variables")
    ap.add_argument("--root", type=str, default="src", help="Source root to scan")
    ap.add_argument("--env_example", type=str, default="env.example.txt")
    ap.add_argument("--out_json", type=str, default="weights/env_registry.json")
    ap.add_argument("--out_index", type=str, default="docs/EnvIndex.md")
    ap.add_argument("--sync_example", action="store_true", help="Append missing keys to env.example.txt")
    # Drift report (documented vs used)
    ap.add_argument("--drift_out", type=str, default="", help="If set, write drift report JSON here")
    ap.add_argument("--fail_on_drift", action="store_true", help="Exit 1 if drift detected when --drift_out is used")
    # Unknown currently set keys
    ap.add_argument("--list_unknown", action="store_true")
    ap.add_argument("--unknown_out", type=str, default="", help="If set, write unknown keys JSON here")
    args = ap.parse_args()

    # Load local dotenvs to include user session state in the audit (optional)
    try:
        load_dotenv_best_effort((".env", ".env.tuned"))
    except Exception:
        pass

    usage = scan_env_usage(args.root)
    registry = build_registry(env_example_path=args.env_example, root_dir=args.root)

    # Write registry and index
    Path(Path(args.out_json).parent).mkdir(parents=True, exist_ok=True)
    dump_registry_json(args.out_json, registry)
    dump_index_markdown(args.out_index, registry, usage)

    print(json.dumps({
        "registry_count": len(registry),
        "usage_keys": len(usage),
        "out_json": args.out_json,
        "out_index": args.out_index,
    }))

    # Optionally sync env.example with missing keys
    if args.sync_example:
        missing, _deprecated = sync_env_example(args.env_example, registry, usage)
        print(json.dumps({"synced": True, "missing_added": missing}))

    # Optional drift report: compare documented keys to used keys
    if args.drift_out:
        documented = set(k for k in registry.keys() if k.startswith("OMNICODER_"))
        used = set(k for k in usage.keys() if k.startswith("OMNICODER_"))
        used_not_documented = sorted(list(used - documented))
        documented_not_used = sorted(list(documented - used))
        drift = {
            "used_not_documented": used_not_documented,
            "documented_not_used": documented_not_used,
            "counts": {"used": len(used), "documented": len(documented)},
        }
        Path(Path(args.drift_out).parent).mkdir(parents=True, exist_ok=True)
        Path(args.drift_out).write_text(json.dumps(drift, indent=2), encoding="utf-8")
        print(json.dumps({"drift_out": args.drift_out, **drift}))
        if args.fail_on_drift and (used_not_documented or documented_not_used):
            raise SystemExit(1)

    # Optionally report unknown currently set envs (present in environment but not in registry)
    if args.list_unknown:
        unknown = _audit_env_unknown(prefix="OMNICODER_")
        payload = {"prefix": "OMNICODER_", "unknown": unknown, "count": len(unknown)}
        print(json.dumps(payload))
        if args.unknown_out:
            try:
                with open(args.unknown_out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass


if __name__ == "__main__":
    main()


from __future__ import annotations

"""
Mobile budget checker: verifies end-to-end memory budgets for release artifacts.

Checks:
- Total artifact size under release root
- Decode-step KV memory for target context using kv_paging sidecar and KVQ scheme
- Optional provider bench JSON to assert tokens/s thresholds (best-effort)

Usage:
  python -m omnicoder.tools.mobile_budget_check --release_root weights/release \
    --target_ctx 32768 --kvq nf4 --budget_gb 4.0
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional


def _human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _sum_dir_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _load_kv_paging_sidecar(release_root: Path) -> Optional[Dict]:
    # Prefer text decode-step location
    cand = release_root / 'text' / 'omnicoder_decode_step.kv_paging.json'
    if cand.exists():
        try:
            return json.loads(cand.read_text(encoding='utf-8'))
        except Exception:
            return None
    # Fallback: any kv_paging.json under release_root
    for p in release_root.rglob('*.kv_paging.json'):
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
    return None


def _kv_bytes_for_ctx(sidecar: Dict, target_ctx: int, scheme: str) -> int:
    n_layers = int(sidecar.get('n_layers', 0))
    heads = sidecar.get('heads_per_layer', [])
    dls = sidecar.get('dl_per_layer', [])
    if not heads or not dls or n_layers <= 0:
        return 0
    bytes_per = 1 if scheme == 'u8' else 1 if scheme == 'nf4' else 4
    # nf4 is 4-bit; approximate as 1 byte per element due to metadata/packing overhead
    total = 0
    for i in range(min(n_layers, len(heads), len(dls))):
        H = int(heads[i]); DL = int(dls[i])
        # K and V per token per head
        per_token = 2 * H * DL * bytes_per
        total += per_token * int(target_ctx)
    return total


def _load_provider_bench(release_root: Path) -> Optional[Dict]:
    pb = release_root / 'text' / 'provider_bench.json'
    if pb.exists():
        try:
            return json.loads(pb.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description='Mobile budget checker (weights + KV memory + provider TPS)')
    ap.add_argument('--release_root', type=str, required=True)
    ap.add_argument('--target_ctx', type=int, default=32768)
    ap.add_argument('--kvq', type=str, default='nf4', choices=['none','u8','nf4'])
    ap.add_argument('--budget_gb', type=float, default=4.0)
    ap.add_argument('--tokens_per_second_min', type=float, default=0.0, help='Optional TPS floor to assert from provider_bench.json')
    args = ap.parse_args()

    root = Path(args.release_root)
    summary: Dict[str, object] = {
        'release_root': str(root),
        'target_ctx': int(args.target_ctx),
        'kvq': str(args.kvq),
        'budget_gb': float(args.budget_gb),
    }

    # 1) Artifact size
    try:
        size_bytes = _sum_dir_bytes(root)
        summary['artifacts_bytes'] = int(size_bytes)
        summary['artifacts_human'] = _human_bytes(size_bytes)
    except Exception as e:
        summary['artifacts_error'] = str(e)

    # 2) KV memory estimate
    side = _load_kv_paging_sidecar(root)
    if side:
        kv_bytes = _kv_bytes_for_ctx(side, int(args.target_ctx), str(args.kvq))
        summary['kv_bytes'] = int(kv_bytes)
        summary['kv_human'] = _human_bytes(kv_bytes)
    else:
        summary['kv_bytes'] = 0
        summary['kv_human'] = '0 B'

    # 3) Provider TPS (best-effort)
    pb = _load_provider_bench(root)
    if pb and isinstance(args.tokens_per_second_min, (int, float)) and float(args.tokens_per_second_min) > 0.0:
        try:
            tps = float(pb.get('tokens_per_second', 0.0) or pb.get('tps', 0.0))
            summary['tokens_per_second'] = tps
            summary['tps_ok'] = bool(tps >= float(args.tokens_per_second_min))
        except Exception:
            pass

    # Decision: sum of artifacts + KV should fit in budget
    total_bytes = int(summary.get('artifacts_bytes', 0)) + int(summary.get('kv_bytes', 0))
    budget_bytes = float(args.budget_gb) * (1024**3)
    summary['total_bytes'] = int(total_bytes)
    summary['total_human'] = _human_bytes(total_bytes)
    summary['budget_bytes'] = int(budget_bytes)
    summary['budget_human'] = _human_bytes(int(budget_bytes))
    summary['within_budget'] = bool(total_bytes <= budget_bytes)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()



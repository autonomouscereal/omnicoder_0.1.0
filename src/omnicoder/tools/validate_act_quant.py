from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate activation-quant policy sidecar and print effective knobs')
    ap.add_argument('--policy', type=str, required=True, help='Path to *.act_quant.json')
    ap.add_argument('--layers', type=int, default=0, help='Optional number of transformer layers for per-layer print')
    args = ap.parse_args()

    p = Path(args.policy)
    if not p.exists():
        raise SystemExit(f"policy not found: {p}")
    pol = json.loads(p.read_text(encoding='utf-8'))
    conf_floor = float(pol.get('conf_floor', 0.3))
    min_bits = int(pol.get('min_bits', 8))
    max_bits = int(pol.get('max_bits', 2))
    thresholds = pol.get('thresholds', {}) if isinstance(pol, dict) else {}
    layer_conf_floor = []
    if isinstance(thresholds, dict):
        if 'layer_conf_floor' in thresholds and isinstance(thresholds['layer_conf_floor'], (list, tuple)):
            layer_conf_floor = [float(x) for x in thresholds['layer_conf_floor']]
        elif 'layers' in thresholds and isinstance(thresholds['layers'], dict):
            items = sorted(((int(k), v) for k, v in thresholds['layers'].items()), key=lambda kv: kv[0])
            for _, v in items:
                layer_conf_floor.append(float(v.get('conf_floor', conf_floor)))
    # Print effective knobs
    out = {
        'global': {'conf_floor': conf_floor, 'min_bits': min_bits, 'max_bits': max_bits},
        'per_layer_conf_floor': layer_conf_floor[:args.layers] if args.layers and layer_conf_floor else layer_conf_floor,
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()



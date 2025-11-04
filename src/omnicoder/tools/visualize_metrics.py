from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None


def _load_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]] | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _bar_svg(data: Dict[str, float], title: str = "Tokens/sec") -> str:
    if not data:
        return """<svg xmlns='http://www.w3.org/2000/svg' width='640' height='120'><text x='10' y='60' font-size='14'>No data</text></svg>"""
    providers = list(data.keys())
    vals = [float(max(0.0, v)) for v in data.values()]
    mx = max(vals) if vals else 1.0
    width = 640
    height = 240
    left = 80
    right = 10
    top = 30
    bottom = 40
    plot_w = width - left - right
    bar_h = int((height - top - bottom) / max(1, len(providers)))
    y0 = top
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(f"<text x='{left}' y='20' font-size='14' font-weight='bold'>{title}</text>")
    for i, p in enumerate(providers):
        v = vals[i]
        w = int((v / max(mx, 1e-6)) * plot_w)
        y = y0 + i * bar_h
        svg.append(f"<rect x='{left}' y='{y}' width='{w}' height='{bar_h - 4}' fill='#4C78A8' />")
        svg.append(f"<text x='{left + w + 6}' y='{y + bar_h - 8}' font-size='12'>{p}: {v:.2f}</text>")
    svg.append("</svg>")
    return "".join(svg)


def _write_act_quant_policy(release_root: Path, policy_out: str, error_json: str, conf_floor: float, min_bits: int, max_bits: int) -> Path | None:
    try:
        rel = release_root
        out_path = Path(policy_out) if policy_out else (rel / 'omnicoder_decode_step.act_quant.json')
        data: dict = {
            'conf_floor': float(conf_floor),
            'min_bits': int(min_bits),
            'max_bits': int(max_bits),
            'thresholds': {}
        }
        if error_json:
            try:
                ej = json.loads(Path(error_json).read_text(encoding='utf-8'))
                if isinstance(ej, dict):
                    data['thresholds'] = ej.get('thresholds', {})
            except Exception:
                pass
        out_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        # Optional: histogram from errors distribution
        if error_json and plt is not None:
            try:
                ej = json.loads(Path(error_json).read_text(encoding='utf-8'))
                xs = []
                if isinstance(ej, dict):
                    if 'errors' in ej and isinstance(ej['errors'], list):
                        xs = [float(x) for x in ej['errors']]
                    else:
                        thr = ej.get('thresholds', {})
                        p50 = float(thr.get('p50', 0.5))
                        p90 = float(thr.get('p90', 0.9))
                        xs = [p50] * 50 + [p90] * 10
                if xs:
                    plt.figure(figsize=(4, 3))
                    plt.hist(xs, bins=20, alpha=0.7)
                    plt.title('Activation error proxy distribution')
                    plt.xlabel('1 - max_prob')
                    plt.ylabel('count')
                    svg = out_path.with_suffix('.act_err.svg')
                    plt.tight_layout()
                    plt.savefig(svg)
                    print({'act_err_plot': str(svg)})
            except Exception:
                pass
        print({'act_quant_policy': str(out_path)})
        return out_path
    except Exception:
        return None


def _render_metrics(bench_json: Path | None, onnx_model: Path | None, out_svg: Path, out_kv: Path) -> tuple[Path | None, Path | None]:
    wrote_svg: Path | None = None
    wrote_kv: Path | None = None
    try:
        tps_map: Dict[str, float] = {}
        if bench_json and bench_json.exists():
            bench = _load_json(bench_json)
            if isinstance(bench, dict):
                results = bench.get('results', [])
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict) and 'provider' in r and 'tps' in r:
                            tps_map[str(r['provider'])] = float(r['tps'])
        tps_svg = _bar_svg(tps_map, title="Tokens/sec by provider")
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_svg.write_text(tps_svg, encoding='utf-8')
        wrote_svg = out_svg
    except Exception:
        pass
    try:
        kv_info: Dict[str, Any] = {}
        if onnx_model and onnx_model.exists():
            kv_paging = onnx_model.with_suffix('.kv_paging.json')
            kv_ret = onnx_model.with_suffix('.kv_retention.json')
            if kv_paging.exists():
                side = _load_json(kv_paging)
                if isinstance(side, dict):
                    kv_info['kv_paging'] = {
                        'page_len': int(side.get('page_len', 0)),
                        'heads': int(side.get('heads', 0)),
                        'dl': int(side.get('dl', 0)),
                        'dl_per_layer': side.get('dl_per_layer', []),
                    }
            if kv_ret.exists():
                side = _load_json(kv_ret)
                if isinstance(side, dict):
                    slots = int(side.get('slots', side.get('compressive_slots', 0)))
                    window = int(side.get('window', side.get('window_size', 0)))
                    kv_info['kv_retention'] = {'slots': slots, 'window': window}
        out_kv.parent.mkdir(parents=True, exist_ok=True)
        out_kv.write_text(json.dumps(kv_info, indent=2), encoding='utf-8')
        wrote_kv = out_kv
        print({"tps_svg": str(out_svg), "kv_info": str(out_kv)})
    except Exception:
        pass
    return wrote_svg, wrote_kv


def main() -> None:
    ap = argparse.ArgumentParser(description='Act-quant policy writer and metrics/KV visualizer')
    # Common base
    ap.add_argument('--release_root', type=str, default='weights/release/text')
    # Policy knobs
    ap.add_argument('--policy_out', type=str, default='')
    ap.add_argument('--error_json', type=str, default='')
    ap.add_argument('--conf_floor', type=float, default=0.3)
    ap.add_argument('--min_bits', type=int, default=8)
    ap.add_argument('--max_bits', type=int, default=2)
    # Metrics/KV inputs and outputs
    ap.add_argument('--bench_json', type=str, default='')
    ap.add_argument('--onnx_model', type=str, default='')
    ap.add_argument('--out_dir', type=str, default='')
    ap.add_argument('--out_svg', type=str, default='')
    ap.add_argument('--out_kv', type=str, default='')
    args = ap.parse_args()

    rel = Path(args.release_root)
    # Policy write (optional; perform when policy_out provided or when error_json provided)
    if args.policy_out or args.error_json:
        _ = _write_act_quant_policy(rel, args.policy_out, args.error_json, float(args.conf_floor), int(args.min_bits), int(args.max_bits))

    # Resolve inputs/outputs for metrics rendering
    bench_json = Path(args.bench_json) if args.bench_json else (rel / 'provider_bench.json')
    onnx_model = Path(args.onnx_model) if args.onnx_model else (rel / 'omnicoder_decode_step.onnx')
    if args.out_dir and not (args.out_svg or args.out_kv):
        out_dir = Path(args.out_dir)
        out_svg = out_dir / 'metrics.svg'
        out_kv = out_dir / 'kv_info.json'
    else:
        out_svg = Path(args.out_svg) if args.out_svg else (rel / 'metrics.svg')
        out_kv = Path(args.out_kv) if args.out_kv else (rel / 'kv_info.json')

    _render_metrics(bench_json, onnx_model, out_svg, out_kv)


if __name__ == '__main__':
    main()


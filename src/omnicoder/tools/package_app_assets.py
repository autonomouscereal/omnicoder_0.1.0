from __future__ import annotations

"""
Copy TPS/KV visualization assets into sample app assets folders if present.

Usage:
  python -m omnicoder.tools.package_app_assets \
    --assets_dir weights/release/text \
    --android_assets app/src/main/assets/omnicoder \
    --ios_assets SampleApp/Resources/omnicoder

Best-effort copies metrics.svg, kv_info.json, dashboard.html. If metrics/kv are
missing but provider_bench.json exists, runs visualize_metrics first.
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy TPS/KV assets into Android/iOS sample app assets")
    ap.add_argument('--assets_dir', type=str, required=True, help='Directory containing metrics.svg, kv_info.json, dashboard.html')
    ap.add_argument('--bench_json', type=str, default='', help='Optional provider_bench.json to generate metrics if missing')
    ap.add_argument('--onnx_model', type=str, default='', help='Optional ONNX model path for kv sidecar discovery')
    ap.add_argument('--android_assets', type=str, default='', help='Android assets folder (e.g., app/src/main/assets/omnicoder)')
    ap.add_argument('--ios_assets', type=str, default='', help='iOS resources folder (e.g., SampleApp/Resources/omnicoder)')
    args = ap.parse_args()

    src = Path(args.assets_dir)
    src.mkdir(parents=True, exist_ok=True)
    metrics_svg = src / 'metrics.svg'
    kv_info = src / 'kv_info.json'
    dashboard = src / 'dashboard.html'

    # If assets missing and bench json provided, generate first
    if (not metrics_svg.exists() or not kv_info.exists()) and args.bench_json:
        try:
            import subprocess, sys
            cmd = [sys.executable, '-m', 'omnicoder.tools.visualize_metrics', '--bench_json', args.bench_json, '--out_dir', str(src)]
            if args.onnx_model:
                cmd += ['--onnx_model', args.onnx_model]
            print('[run]', ' '.join(cmd))
            subprocess.run(cmd, check=False)
        except Exception as e:
            print('[warn] visualize_metrics failed:', e)

    # If dashboard missing, build one
    if not dashboard.exists():
        try:
            import subprocess, sys
            cmd = [sys.executable, '-m', 'omnicoder.tools.app_assets', '--assets_dir', str(src)]
            if args.bench_json:
                cmd += ['--bench_json', args.bench_json]
            if args.onnx_model:
                cmd += ['--onnx_model', args.onnx_model]
            print('[run]', ' '.join(cmd))
            subprocess.run(cmd, check=False)
        except Exception as e:
            print('[warn] app_assets failed:', e)

    def _copy_to(dst_str: str) -> None:
        if not dst_str:
            return
        dst = Path(dst_str)
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for p in (metrics_svg, kv_info, dashboard):
                if p.exists():
                    shutil.copy2(p, dst / p.name)
            print(f"[copy] assets to {dst}")
        except Exception as e:
            print(f"[warn] failed to copy assets to {dst}: {e}")

    _copy_to(args.android_assets)
    _copy_to(args.ios_assets)


if __name__ == '__main__':
    main()



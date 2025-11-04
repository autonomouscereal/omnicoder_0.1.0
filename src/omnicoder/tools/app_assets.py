from __future__ import annotations

"""
Assemble TPS/KV visualization assets into a simple HTML dashboard for preview or
mobile app ingestion. Copies/reads metrics.svg and kv_info.json and emits a
single-file dashboard.html with embedded SVG and a KV summary table.

Usage:
  python -m omnicoder.tools.app_assets \
    --bench_json weights/release/text/provider_bench.json \
    --onnx_model weights/release/text/omnicoder_decode_step.onnx \
    --assets_dir weights/release/text

Outputs:
  - dashboard.html (inline SVG + KV table)
  - metrics.svg (copied/overwritten by visualize_metrics if needed)
  - kv_info.json (copied/overwritten by visualize_metrics if needed)
"""

import argparse
import base64
import json
from pathlib import Path


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return ""


def _embed_svg_data_uri(svg_text: str) -> str:
    if not svg_text:
        return ""
    try:
        # Inline raw SVG rather than data URI when possible
        return svg_text
    except Exception:
        b64 = base64.b64encode(svg_text.encode('utf-8')).decode('ascii')
        return f'<img src="data:image/svg+xml;base64,{b64}" alt="metrics"/>'


def main() -> None:
    ap = argparse.ArgumentParser(description="Package TPS/KV assets into a simple HTML dashboard")
    ap.add_argument('--bench_json', type=str, required=False, default='', help='Path to provider_bench.json (optional, only referenced)')
    ap.add_argument('--onnx_model', type=str, required=False, default='', help='Path to ONNX model (optional, only referenced)')
    ap.add_argument('--assets_dir', type=str, required=True, help='Directory containing metrics.svg and kv_info.json (output dashboard placed here)')
    args = ap.parse_args()

    out_dir = Path(args.assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / 'metrics.svg'
    kv_path = out_dir / 'kv_info.json'

    svg_text = _read_text(svg_path)
    kv_text = _read_text(kv_path)
    try:
        kv = json.loads(kv_text) if kv_text else {}
    except Exception:
        kv = {}

    # Build a minimal HTML dashboard
    kv_rows = ''
    if isinstance(kv, dict):
        for k, v in kv.items():
            kv_rows += f'<tr><td style="padding:4px 8px;">{k}</td><td style="padding:4px 8px;">{v}</td></tr>'

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OmniCoder TPS / KV Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin: 16px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    h2 {{ font-size: 16px; margin: 16px 0 8px; }}
    .card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border-bottom: 1px solid #eee; }}
    .muted {{ color: #666; font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }}
  </style>
  <meta name="omnicoder:bench_json" content="{args.bench_json}">
  <meta name="omnicoder:onnx_model" content="{args.onnx_model}">
  <meta name="omnicoder:assets_dir" content="{str(out_dir)}">
  <meta name="omnicoder:kv_info" content="kv_info.json">
  <meta name="omnicoder:metrics_svg" content="metrics.svg">
  </head>
<body>
  <h1>OmniCoder – TPS / KV Dashboard</h1>
  <div class="card">
    <h2>Tokens/sec by provider</h2>
    <div id="chart">
      {_embed_svg_data_uri(svg_text)}
    </div>
    <div class="muted mono">Source: metrics.svg (generated via omnicoder.tools.visualize_metrics)</div>
  </div>
  <div class="card">
    <h2>KV sidecars (paging/retention/window)</h2>
    <table>
      <tbody>
      {kv_rows}
      </tbody>
    </table>
    <div class="muted mono">Source: kv_info.json (generated via omnicoder.tools.visualize_metrics)</div>
  </div>
  <div class="card">
    <h2>How to interpret</h2>
    <div class="muted">
      <p>Tokens/sec reflects end-to-end decode throughput of the exported ONNX decode-step under the selected provider(s).<br/>
      KV sidecar fields summarize paging and retention configuration used by the runner to keep memory bounded on device.</p>
    </div>
  </div>
</body>
</html>
"""
    out_html = out_dir / 'dashboard.html'
    out_html.write_text(html, encoding='utf-8')
    print(f"[write] {out_html}")


if __name__ == '__main__':
    main()



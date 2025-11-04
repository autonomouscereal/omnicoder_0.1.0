from __future__ import annotations

"""
Auto tuner: reads stage bench artifacts (provider_bench.json, draft_acceptance.json)
and writes updated thresholds and a suggested .env.tuned for runtime knobs.
Safe heuristics only; does not overwrite existing files except .env.tuned.
"""

import argparse
import json
from pathlib import Path


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description='Auto-tune runtime knobs from bench artifacts')
    ap.add_argument('--release_root', type=str, default='weights/release/text')
    ap.add_argument('--out_env', type=str, default='.env.tuned')
    args = ap.parse_args()

    rel = Path(args.release_root)
    bench = _load_json(rel / 'provider_bench.json')
    accept = _load_json(rel.parent / 'draft_acceptance.json') if (rel.parent / 'draft_acceptance.json').exists() else _load_json(Path('weights') / 'draft_acceptance.json')

    lines: list[str] = []
    # Tune block verification size based on CPU TPS (larger if TPS is low)
    try:
        providers = bench.get('providers', {}) if isinstance(bench, dict) else {}
        cpu = providers.get('CPUExecutionProvider', {})
        tps = float(cpu.get('tps', 0.0))
        if tps > 0:
            if tps < 10:
                bv = 6
            elif tps < 20:
                bv = 4
            else:
                bv = 2
            lines.append(f'OMNICODER_BLOCK_VERIFY_SIZE={bv}')
    except Exception:
        pass

    # Tune variable-K min/max by simple heuristic from TPS: lower K if TPS low
    try:
        providers = bench.get('providers', {}) if isinstance(bench, dict) else {}
        base = providers.get('CPUExecutionProvider', {})
        tps = float(base.get('tps', 0.0))
        if tps > 0:
            if tps < 10:
                lines.append('OMNICODER_ADAPTIVE_TOP_K_MIN=1')
                lines.append('OMNICODER_ADAPTIVE_TOP_K_MAX=2')
            elif tps < 20:
                lines.append('OMNICODER_ADAPTIVE_TOP_K_MIN=1')
                lines.append('OMNICODER_ADAPTIVE_TOP_K_MAX=3')
            else:
                lines.append('OMNICODER_ADAPTIVE_TOP_K_MIN=1')
                lines.append('OMNICODER_ADAPTIVE_TOP_K_MAX=4')
    except Exception:
        pass

    # Suggest block-sparse attention stride based on TPS
    try:
        providers = bench.get('providers', {}) if isinstance(bench, dict) else {}
        cpu = providers.get('CPUExecutionProvider', {})
        tps = float(cpu.get('tps', 0.0))
        if tps > 0:
            if tps < 10:
                lines.append('OMNICODER_ATT_BLOCK_SPARSE=1')
                lines.append('OMNICODER_BS_STRIDE=128')
            elif tps < 20:
                lines.append('OMNICODER_ATT_BLOCK_SPARSE=1')
                lines.append('OMNICODER_BS_STRIDE=64')
            else:
                lines.append('OMNICODER_ATT_BLOCK_SPARSE=0')
    except Exception:
        pass

    # Suggest kNN cache size based on TPS
    try:
        providers = bench.get('providers', {}) if isinstance(bench, dict) else {}
        cpu = providers.get('CPUExecutionProvider', {})
        tps = float(cpu.get('tps', 0.0))
        if tps > 0:
            if tps < 10:
                lines.append('OMNICODER_KNN_MAX_ITEMS=2048')
            elif tps < 20:
                lines.append('OMNICODER_KNN_MAX_ITEMS=4096')
            else:
                lines.append('OMNICODER_KNN_MAX_ITEMS=8192')
    except Exception:
        pass

    # Update acceptance threshold suggestion if present
    try:
        thr = float(accept.get('recommended_threshold', 0.0)) if isinstance(accept, dict) else 0.0
        if thr > 0:
            lines.append(f'OMNICODER_VERIFY_THRESHOLD={thr:.3f}')
    except Exception:
        pass

    # Write .env.tuned with suggestions (append tuned KV prefetch keep if available)
    kv_keep_line = ''
    try:
        preset = ''
        import os
        preset = os.getenv('OMNICODER_TRAIN_PRESET', '')
        kvp = rel / 'omnicoder_decode_step.kv_prefetch.json'
        if preset and kvp.exists():
            data = _load_json(kvp)
            keep = data.get('keep_pages')
            if isinstance(keep, int):
                kv_keep_line = f"# KV prefetch tuned for preset {preset}\nOMNICODER_KV_PREFETCH_KEEP={keep}"
    except Exception:
        pass
    Path(args.out_env).write_text('\n'.join(lines + ([kv_keep_line] if kv_keep_line else [])) + '\n', encoding='utf-8')
    # Append MLA microbench recommendations if present
    try:
        mb = _load_json(Path('weights') / 'mla_microbench.json')
        rec = mb.get('recommend', {}) if isinstance(mb, dict) else {}
        if isinstance(rec, dict):
            with Path(args.out_env).open('a', encoding='utf-8') as f:
                for k in ('OMNICODER_ATT_BLOCK_SPARSE','OMNICODER_BS_STRIDE'):
                    if k in rec:
                        f.write(f"{k}={rec[k]}\n")
    except Exception:
        pass
    print({'out_env': str(Path(args.out_env).absolute()), 'lines': len(lines)})


if __name__ == '__main__':
    main()



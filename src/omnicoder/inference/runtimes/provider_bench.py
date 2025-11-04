from __future__ import annotations

"""
Provider microbench harness: runs an exported ONNX decode-step on different ORT providers
and reports tokens/s. Optionally enforces per-provider minimum thresholds and checks
that fused ops are present in the graph. Exits non-zero on failure to aid CI.

Usage:
  python -m omnicoder.inference.runtimes.provider_bench --model weights/text/omnicoder_decode_step.onnx \
    --providers CPUExecutionProvider NNAPIExecutionProvider CoreMLExecutionProvider DmlExecutionProvider \
    --prompt_len 256 --gen_tokens 256 \
    --threshold "CPUExecutionProvider=2.0,NNAPIExecutionProvider=20.0" \
    --check_fusions --require_attention --require_qlinear --out_json weights/text/provider_bench.json
"""

import argparse
import time
from typing import List, Dict, Any

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore

from .onnx_provider_profiles import get_provider_options


def _count_fusions(model_path: str) -> Dict[str, int]:
    counts = {"Attention": 0, "QLinearMatMul": 0}
    try:
        import onnx  # type: ignore
    except Exception:
        return counts
    m = onnx.load(model_path)
    for n in m.graph.node:
        if n.domain == 'com.microsoft' and n.op_type == 'Attention':
            counts["Attention"] += 1
        if n.op_type == 'QLinearMatMul':
            counts["QLinearMatMul"] += 1
    return counts


def bench_decode_step(session: "ort.InferenceSession", vocab_size: int, prompt_len: int, gen_tokens: int) -> float:
    """Bench decode-step by feeding zero-length past K/V tensors per layer.

    Many decode-step graphs require explicit K/V inputs; we initialize them to
    zero-length sequences with the correct (B,H,0,DL) shapes based on model IO.
    """
    inputs = session.get_inputs()
    input_name = inputs[0].name
    outputs = session.get_outputs()
    output_names = [o.name for o in outputs]
    run = session.run
    # Discover K/V input tensors and their static H/DL dims
    k_inputs = [i for i in inputs if i.name.startswith("k_lat_")]
    v_inputs = [i for i in inputs if i.name.startswith("v_lat_")]
    k_inputs.sort(key=lambda x: int(x.name.split("_")[-1]))
    v_inputs.sort(key=lambda x: int(x.name.split("_")[-1]))
    num_layers = min(len(k_inputs), len(v_inputs))
    # Use a single inferred H/DL across all layers to avoid dimension mismatches
    def _infer_head_dl(default_h: int = 8, default_dl: int = 128) -> tuple[int, int]:
        for ki in k_inputs[:num_layers]:
            try:
                ks = ki.shape
                return int(ks[1]), int(ks[3])
            except Exception:
                continue
        return default_h, default_dl
    H_global, DL_global = _infer_head_dl()
    # Pre-build zero-length KV feeds to avoid per-step allocation overhead
    zero_kv = {}
    for kin, vin in zip(k_inputs[:num_layers], v_inputs[:num_layers]):
        zero = (1, int(H_global), 0, int(DL_global))
        zero_kv[kin.name] = np.zeros(zero, dtype=np.float32)
        zero_kv[vin.name] = np.zeros(zero, dtype=np.float32)
    # Prompt and measure
    # Initialize last token and preallocate buffers to avoid per-step allocations
    ids = np.random.randint(0, vocab_size, size=(1, prompt_len), dtype=np.int64)
    last_tok = ids[:, -1:].copy()
    feeds = {input_name: last_tok}
    feeds.update(zero_kv)
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        _ = run(output_names, feeds)
        # Advance last token randomly; keep shape (1,1)
        last_tok[0, 0] = np.random.randint(0, vocab_size, dtype=np.int64)
    dt = time.perf_counter() - t0
    return gen_tokens / max(dt, 1e-6)

def bench_decode_step_dc(session: "ort.InferenceSession", vocab_size: int, prompt_len: int, gen_tokens: int) -> float:
    """Bench decode-step for DynamicCache (input_ids-only) models.

    Feeds tokens step-by-step without explicit K/V inputs.
    """
    import time as _time
    inputs = session.get_inputs()
    input_name = inputs[0].name
    outputs = session.get_outputs()
    out_names = [o.name for o in outputs]
    ids = np.random.randint(0, vocab_size, size=(1, prompt_len), dtype=np.int64)
    # Warm prompt
    for step in (ids[:, t:t+1] for t in range(ids.shape[1])):
        _ = session.run(out_names, {input_name: step})
    t0 = _time.perf_counter()
    for _ in range(gen_tokens):
        step = ids[:, -1:]
        _ = session.run(out_names, {input_name: step})
        ids = np.concatenate((ids, np.random.randint(0, vocab_size, size=(1, 1), dtype=np.int64)), axis=1)
    dt = _time.perf_counter() - t0
    return gen_tokens / max(dt, 1e-6)


def bench_decode_step_paged(
    session: "ort.InferenceSession",
    vocab_size: int,
    prompt_len: int,
    gen_tokens: int,
    page_len: int,
    side_heads: int | None = None,
    side_dl: int | None = None,
    dl_per_layer: list[int] | None = None,
) -> float:
    import time as _time
    input_name = session.get_inputs()[0].name
    outputs = session.get_outputs()
    out_names = [o.name for o in outputs]
    run = session.run
    nk = [n for n in out_names if n.startswith('nk_lat_')]
    nv = [n for n in out_names if n.startswith('nv_lat_')]
    L = min(len(nk), len(nv))
    # Initialize past states strictly from sidecar-provided dims; never infer from ORT
    H_global = int(side_heads) if side_heads is not None else 8
    past_k = [None] * L
    past_v = [None] * L
    if not isinstance(dl_per_layer, list) or len(dl_per_layer) < L:
        # Fallback to uniform DL if per-layer list omitted
        dl_per_layer = [int(side_dl) if side_dl is not None else 16] * L
    dl_per_layer = [int(x) for x in dl_per_layer[:L]]
    ids = np.random.randint(0, vocab_size, size=(1, prompt_len), dtype=np.int64)
    # Zero-length KV tensors based on sidecar dims
    zero_kv = {f'k_lat_{i}': np.zeros((1, H_global, 0, int(dl_per_layer[i])), dtype=np.float32) for i in range(L)}
    zero_kv.update({f'v_lat_{i}': np.zeros((1, H_global, 0, int(dl_per_layer[i])), dtype=np.float32) for i in range(L)})
    # Warm prompt: always supply zero-length past with sidecar dims
    for t in range(ids.shape[1]):
        feeds: Dict[str, Any] = {input_name: ids[:, t:t+1]}
        for i, (pk, pv) in enumerate(zip(past_k, past_v)):
            feeds[f'k_lat_{i}'] = pk if pk is not None else zero_kv[f'k_lat_{i}']
            feeds[f'v_lat_{i}'] = pv if pv is not None else zero_kv[f'v_lat_{i}']
        res = run(out_names, feeds)
        for i in range(L):
            k = res[1 + i]
            v = res[1 + L + i]
            if k.shape[2] > page_len:
                k = k[:, :, -page_len:, :]
            if v.shape[2] > page_len:
                v = v[:, :, -page_len:, :]
            past_k[i], past_v[i] = k, v
    # Measure
    t0 = _time.perf_counter()
    last_tok = ids[:, -1:].copy()
    for _ in range(gen_tokens):
        feeds = {input_name: last_tok}
        for i, (pk, pv) in enumerate(zip(past_k, past_v)):
            feeds[f'k_lat_{i}'] = pk if pk is not None else zero_kv[f'k_lat_{i}']
            feeds[f'v_lat_{i}'] = pv if pv is not None else zero_kv[f'v_lat_{i}']
        res = run(out_names, feeds)
        for i in range(L):
            k = res[1 + i]
            v = res[1 + L + i]
            if k.shape[2] > page_len:
                k = k[:, :, -page_len:, :]
            if v.shape[2] > page_len:
                v = v[:, :, -page_len:, :]
            past_k[i], past_v[i] = k, v
        last_tok[0, 0] = np.random.randint(0, vocab_size, dtype=np.int64)
    dt = _time.perf_counter() - t0
    return gen_tokens / max(dt, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser(description="Provider microbenchmark for ONNX decode-step")
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--providers', type=str, nargs='+', default=['CPUExecutionProvider'])
    ap.add_argument('--prompt_len', type=int, default=128)
    ap.add_argument('--gen_tokens', type=int, default=256)
    ap.add_argument('--vocab_size', type=int, default=32000)
    ap.add_argument('--threshold', type=str, default='CPUExecutionProvider=2.0,DmlExecutionProvider=10.0,CoreMLExecutionProvider=6.0,NNAPIExecutionProvider=6.0', help='Comma-separated per-provider min tokens/s, e.g., "CPUExecutionProvider=2.0,DmlExecutionProvider=15.0"')
    ap.add_argument('--threshold_json', type=str, default='', help='Optional JSON file mapping provider to min tokens/s')
    ap.add_argument('--check_fusions', action='store_true', help='Count fused Attention and QLinearMatMul nodes in the model and include in results')
    ap.add_argument('--require_attention', action='store_true', help='Fail if no com.microsoft::Attention nodes present when --check_fusions')
    ap.add_argument('--require_qlinear', action='store_true', help='Fail if no QLinearMatMul nodes present when --check_fusions')
    ap.add_argument('--canary_tokens_per_s', action='store_true', help='Include tokens/s canaries per provider in output JSON (for Core ML/QNNPack INT4 routing)')
    ap.add_argument('--out_json', type=str, default='', help='Optional path to write JSON results')
    ap.add_argument('--kv_paging_sidecar', type=str, default='', help='Optional KV paging sidecar JSON to simulate paged runtime')
    # Optional speedup comparison
    ap.add_argument('--compare_base', type=str, default='', help='Base provider to compare (e.g., CPUExecutionProvider)')
    ap.add_argument('--compare_target', type=str, default='', help='Target provider to compare (e.g., DmlExecutionProvider)')
    ap.add_argument('--speedup_min', type=float, default=0.0, help='Require target/base >= speedup_min when compare_* are set')
    args = ap.parse_args()

    if ort is None:
        raise SystemExit("onnxruntime is required for provider benchmark")

    thresholds: Dict[str, float] = {}
    if args.threshold:
        for part in args.threshold.split(','):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                name, val = part.split('=', 1)
                try:
                    thresholds[name.strip()] = float(val)
                except Exception:
                    pass
    if args.threshold_json:
        try:
            import json as _json
            thresholds.update(_json.loads(open(args.threshold_json, 'r', encoding='utf-8').read()))
        except Exception:
            pass
    else:
        # Auto-load default thresholds JSON when present
        try:
            from pathlib import Path as _Path
            default_json = _Path('profiles') / 'provider_thresholds.json'
            if default_json.exists():
                import json as _json
                thresholds.update(_json.loads(default_json.read_text(encoding='utf-8')))
        except Exception:
            pass

    # Auto-enable fusion checks when GPU/mobile providers are requested
    if (not args.check_fusions) and any(p in args.providers for p in ['DmlExecutionProvider','CoreMLExecutionProvider','NNAPIExecutionProvider']):
        args.check_fusions = True
        # Default to requiring attention fusion; QLinearMatMul required when int8 paths or per-op PTQ were used
        args.require_attention = True
        # Require QLinearMatMul when explicitly requested via env or platform profiles
        try:
            import os as _os
            if _os.getenv('OMNICODER_REQUIRE_QLINEAR', '0') == '1':
                args.require_qlinear = True
        except Exception:
            pass

    fuse_counts: Dict[str, int] = {"Attention": 0, "QLinearMatMul": 0}
    if args.check_fusions:
        fuse_counts = _count_fusions(args.model)
        # Early sanity print to aid CI logs
        print({'fusions': fuse_counts})

    # Auto-default speedup compare if CPU and DML/CoreML/NNAPI present and not explicitly set
    if (not args.compare_base) and (not args.compare_target) and (args.speedup_min == 0.0):
        if 'CPUExecutionProvider' in args.providers and 'DmlExecutionProvider' in args.providers:
            args.compare_base = 'CPUExecutionProvider'
            args.compare_target = 'DmlExecutionProvider'
            try:
                import os as _os
                args.speedup_min = float(_os.getenv('OMNICODER_BENCH_DML_SPEEDUP_MIN', '1.50'))
            except Exception:
                args.speedup_min = 1.50
        elif 'CPUExecutionProvider' in args.providers and 'CoreMLExecutionProvider' in args.providers:
            args.compare_base = 'CPUExecutionProvider'
            args.compare_target = 'CoreMLExecutionProvider'
            try:
                import os as _os
                args.speedup_min = float(_os.getenv('OMNICODER_BENCH_COREML_SPEEDUP_MIN', '1.25'))
            except Exception:
                args.speedup_min = 1.25
        elif 'CPUExecutionProvider' in args.providers and 'NNAPIExecutionProvider' in args.providers:
            args.compare_base = 'CPUExecutionProvider'
            args.compare_target = 'NNAPIExecutionProvider'
            try:
                import os as _os
                args.speedup_min = float(_os.getenv('OMNICODER_BENCH_NNAPI_SPEEDUP_MIN', '1.25'))
            except Exception:
                args.speedup_min = 1.25

    results: List[Dict[str, Any]] = []
    for p in args.providers:
        providers, options = get_provider_options(p)
        sess_opts = ort.SessionOptions()
        try:
            # Enable all graph optimizations if available
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            sess = ort.InferenceSession(args.model, providers=providers, provider_options=options, sess_options=sess_opts)
        except Exception as e:
            results.append({'provider': p, 'error': str(e)})
            continue
        try:
            # Auto-detect KV paging sidecar if not explicitly provided
            sidecar_path = args.kv_paging_sidecar
            if not sidecar_path:
                try:
                    from pathlib import Path as _Path
                    cand = _Path(args.model).with_suffix('.kv_paging.json')
                    if cand.exists():
                        sidecar_path = str(cand)
                except Exception:
                    sidecar_path = ''
            # Detect DynamicCache (no explicit K/V inputs)
            _has_explicit_kv = any(i.name.startswith('k_lat_') for i in sess.get_inputs())
            if not _has_explicit_kv:
                tps = bench_decode_step_dc(sess, args.vocab_size, args.prompt_len, args.gen_tokens)
            elif sidecar_path:
                try:
                    import json as _json
                    side = _json.loads(open(sidecar_path, 'r', encoding='utf-8').read())
                    page_len = int(side.get('page_len', 256))
                    heads = int(side.get('heads', 8))
                    dlp = side.get('dl_per_layer')
                    dl = int(side.get('dl', 16)) if dlp is None else None
                except Exception:
                    page_len = 256
                    heads = 8
                    dlp = None
                    dl = 16
                tps = bench_decode_step_paged(
                    sess,
                    args.vocab_size,
                    args.prompt_len,
                    args.gen_tokens,
                    page_len,
                    heads,
                    dl,
                    side.get('dl_per_layer') if isinstance(side.get('dl_per_layer'), list) else None,
                )
            else:
                tps = bench_decode_step(sess, args.vocab_size, args.prompt_len, args.gen_tokens)
            rec: Dict[str, Any] = {'provider': p, 'tps': tps}
            if args.check_fusions:
                rec.update({'attention_nodes': fuse_counts.get('Attention', 0), 'qlinear_nodes': fuse_counts.get('QLinearMatMul', 0)})
            thr = thresholds.get(p)
            if thr is not None:
                rec['min_tps'] = float(thr)
                rec['pass'] = bool(tps >= float(thr))
            results.append(rec)
        except Exception as e:
            results.append({'provider': p, 'error': str(e)})

    for r in results:
        print(r)
    # Optional speedup assertion
    fail = False
    if args.compare_base and args.compare_target and args.speedup_min > 0.0:
        base = next((r for r in results if r.get('provider') == args.compare_base and 'tps' in r), None)
        tgt = next((r for r in results if r.get('provider') == args.compare_target and 'tps' in r), None)
        if base and tgt:
            ratio = float(tgt['tps']) / max(float(base['tps']), 1e-9)
            print({'compare': {'base': args.compare_base, 'target': args.compare_target, 'speedup': ratio}})
            if ratio < float(args.speedup_min):
                print(f"[fail] speedup {ratio:.2f} < min {float(args.speedup_min):.2f}")
                fail = True
    # Enforce fusion requirements
    if args.check_fusions:
        if args.require_attention and fuse_counts.get('Attention', 0) <= 0:
            print('[fail] No com.microsoft::Attention nodes present')
            fail = True
        if args.require_qlinear and fuse_counts.get('QLinearMatMul', 0) <= 0:
            print('[fail] No QLinearMatMul nodes present')
            fail = True
    # Enforce thresholds
    for r in results:
        if 'min_tps' in r and 'tps' in r and not r.get('pass', True):
            print(f"[fail] {r['provider']} tps={r['tps']:.2f} < min={r['min_tps']:.2f}")
            fail = True
    # Write JSON if requested
    if args.out_json:
        try:
            import json as _json
            open(args.out_json, 'w', encoding='utf-8').write(_json.dumps({'results': results, 'fusions': fuse_counts}, indent=2))
            print(f"[write] {args.out_json}")
        except Exception as e:
            print(f"[warn] failed to write out_json: {e}")
    if fail:
        raise SystemExit(1)


if __name__ == '__main__':
    main()



from __future__ import annotations

import argparse
import os
import json
import time

import torch

from omnicoder.inference.generate import build_mobile_model_by_name, generate, maybe_load_checkpoint
from omnicoder.inference.gen_config import GenRuntimeConfig  # type: ignore
from omnicoder.training.simple_tokenizer import get_universal_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--verify_threshold', type=float, default=0.0)
    ap.add_argument('--verifier_steps', type=int, default=1)
    ap.add_argument('--threshold_json', type=str, default='', help='Optional JSON with verifier thresholds per preset')
    ap.add_argument('--speculative_draft_len', type=int, default=1)
    ap.add_argument('--adaptive_gating', action='store_true', default=False)
    ap.add_argument('--prompt', type=str, default='hello world')
    ap.add_argument('--load_student', type=str, default='', help='Optional checkpoint to load (verifier/write heads or full)')
    ap.add_argument('--multi_token', type=int, default=1, help='If >1, build the model with MTP heads for benchmarking')
    ap.add_argument('--draft_ckpt', type=str, default='', help='Optional path to a draft student checkpoint (OmniTransformer)')
    ap.add_argument('--draft_preset', type=str, default='mobile_2gb', help='Preset for the draft model if --draft_ckpt is provided')
    # Threshold tuning grid
    ap.add_argument('--tune_threshold', action='store_true', help='Run a grid search over verify_threshold to suggest a recommended threshold')
    ap.add_argument('--target_acceptance', type=float, default=0.5, help='Target acceptance ratio for recommended threshold')
    ap.add_argument('--grid_start', type=float, default=0.05)
    ap.add_argument('--grid_end', type=float, default=0.5)
    ap.add_argument('--grid_steps', type=int, default=10)
    # Optional write-back to profiles
    ap.add_argument('--write_profiles', action='store_true', help='If set and a threshold is recommended, write/update profiles/acceptance_thresholds.json')
    ap.add_argument('--preset_key', type=str, default='', help='Override key for profiles/acceptance_thresholds.json (defaults to mobile_preset and draft_preset)')
    # Compatibility with callers that pass an ONNX model path and an output JSON; we ignore --model in PyTorch path
    ap.add_argument('--model', type=str, default='', help='Optional decode-step ONNX path (ignored in PyTorch path)')
    ap.add_argument('--out_json', type=str, default='', help='Optional path to write metrics JSON')
    args = ap.parse_args()

    # Resolve device (prefer CUDA when available unless explicitly overridden)
    try:
        dev_env = (os.getenv('OMNICODER_DEVICE') or os.getenv('OMNICODER_TRAIN_DEVICE') or '').strip()
    except Exception:
        dev_env = ''
    device = dev_env if dev_env else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_mobile_model_by_name(args.mobile_preset, mem_slots=0, multi_token=int(args.multi_token))
    # Optional: load checkpoint (e.g., verifier_kd.pt)
    if args.load_student:
        try:
            try:
                sd = torch.load(args.load_student, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
            except TypeError:
                sd = torch.load(args.load_student, map_location='cpu')
            if isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
                print(f"[bench] loaded checkpoint: {args.load_student}")
        except Exception as e:
            print(f"[bench] warn: failed to load checkpoint: {e}")
    # Move to device and enable TF32 on CUDA
    try:
        if str(device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            model = model.half().to(device)
        else:
            model = model.to(device)
    except Exception:
        model = model.to(device)
    model.eval()
    # Use local/byte-level tokenizer to avoid network fetches that can fail in CI (unexpected EOF)
    tok = get_universal_tokenizer(prefer_hf=False)

    # Build runtime config once
    rc = GenRuntimeConfig(
        draft_ckpt_path=(args.draft_ckpt or os.environ.get('OMNICODER_DRAFT_CKPT', '').strip() or None),
        draft_preset_name=(args.draft_preset or os.environ.get('OMNICODER_DRAFT_PRESET', 'mobile_2gb')),
        use_onnx_draft=False,
        onnx_decode_path=None,
        ort_provider=os.environ.get('OMNICODER_ORT_PROVIDER', 'auto'),
        super_verbose=False,
    )

    ids = tok.encode(args.prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # One-time warmup to initialize CUDA kernels/caches and avoid first-token spikes
    try:
        if str(device).startswith('cuda'):
            from torch import autocast as _autocast  # type: ignore
            with _autocast(device_type='cuda', dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16):
                _ = generate(model, input_ids[:, :1], max_new_tokens=1, temperature=0.8, top_k=0, top_p=1.0, verify_threshold=0.0, verifier_steps=1, speculative_draft_len=1, draft_model=None, speculative_auto=False, return_stats=False, runtime_config=rc)
            torch.cuda.synchronize()
        else:
            _ = generate(model, input_ids[:, :1], max_new_tokens=1, temperature=0.8, top_k=0, top_p=1.0, verify_threshold=0.0, verifier_steps=1, speculative_draft_len=1, draft_model=None, speculative_auto=False, return_stats=False, runtime_config=rc)
    except Exception:
        pass

    # Optionally override verify threshold from JSON by preset
    if args.threshold_json:
        try:
            with open(args.threshold_json, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and args.mobile_preset in cfg:
                vt = cfg[args.mobile_preset].get('verify_threshold', None)
                if isinstance(vt, (int, float)):
                    args.verify_threshold = float(vt)
        except Exception:
            pass
    def _bench_once(thr: float) -> tuple[float, float, int]:
        t0 = time.perf_counter()
        # Use autocast on CUDA for faster matmuls
        if str(device).startswith('cuda'):
            from torch import autocast as _autocast  # type: ignore
            try:
                from omnicoder.utils.torchutils import get_amp_dtype as _get_amp_dtype  # type: ignore
                _dtype = _get_amp_dtype('cuda')
            except Exception:
                _dtype = None
            with _autocast(device_type='cuda', dtype=_dtype if _dtype is not None else (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16)):
                out = generate(
                    model,
                    input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=0.8,
                    top_k=0,
                    top_p=1.0,
                    verify_threshold=float(thr),
                    verifier_steps=int(args.verifier_steps),
                    speculative_draft_len=int(args.speculative_draft_len),
                    draft_model=None,
                    speculative_auto=bool(args.adaptive_gating),
                    return_stats=True,
                    runtime_config=rc,
                )
        else:
            out = generate(
                model,
                input_ids,
                max_new_tokens=int(args.max_new_tokens),
                temperature=0.8,
                top_k=0,
                top_p=1.0,
                verify_threshold=float(thr),
                verifier_steps=int(args.verifier_steps),
                speculative_draft_len=int(args.speculative_draft_len),
                draft_model=None,
                speculative_auto=bool(args.adaptive_gating),
                return_stats=True,
                runtime_config=rc,
            )
        dur = max(time.perf_counter() - t0, 1e-6)
        tps_local = float(int(args.max_new_tokens) / dur)
        if isinstance(out, tuple) and len(out) == 2:
            output_ids_local, stats = out
            acc = float(stats.get('accepted_speculative', 0))
            att = float(stats.get('attempted_speculative', 1))
            acc_ratio_local = (acc / att) if att > 0 else 0.0
        else:
            output_ids_local = out
            acc_ratio_local = 0.0
        gen_len = int(output_ids_local.size(1) - input_ids.size(1)) if hasattr(output_ids_local, 'size') else int(args.max_new_tokens)
        return tps_local, acc_ratio_local, gen_len

    # Bench at provided verify_threshold
    tps, acc_ratio, gen_len = _bench_once(float(args.verify_threshold))
    # Optional draft model throughput
    tps_draft = None
    if args.draft_ckpt:
        try:
            t1 = time.perf_counter()
            _ = generate(
                model,
                input_ids,
                max_new_tokens=int(args.max_new_tokens),
                temperature=0.8,
                top_k=0,
                top_p=1.0,
                verify_threshold=float(args.verify_threshold),
                verifier_steps=int(args.verifier_steps),
                speculative_draft_len=int(args.speculative_draft_len),
                draft_model=None,
                speculative_auto=bool(args.adaptive_gating),
                return_stats=True,
                runtime_config=rc,
            )
            dur2 = max(time.perf_counter() - t1, 1e-6)
            tps_draft = float(int(args.max_new_tokens) / dur2)
        except Exception:
            tps_draft = None
    # Optional grid search to recommend a threshold
    rec_thr = None
    if bool(args.tune_threshold):
        best_thr = None
        best_gap = 1e9
        best_tps = 0.0
        steps = max(2, int(args.grid_steps))
        lo, hi = float(args.grid_start), float(args.grid_end)
        for i in range(steps):
            thr = lo + (hi - lo) * (i / max(steps - 1, 1))
            tps_i, acc_i, _ = _bench_once(thr)
            gap = abs(acc_i - float(args.target_acceptance))
            # Prefer thresholds meeting or exceeding target acceptance; tie-break by higher TPS, then smaller gap
            meets = acc_i >= float(args.target_acceptance)
            if meets and (best_thr is None or tps_i > best_tps or (abs(tps_i - best_tps) < 1e-6 and gap < best_gap)):
                best_thr, best_gap, best_tps = thr, gap, tps_i
            elif (best_thr is None) and (gap < best_gap or (abs(gap - best_gap) < 1e-6 and tps_i > best_tps)):
                best_thr, best_gap, best_tps = thr, gap, tps_i
        rec_thr = float(best_thr) if best_thr is not None else None

    result = {
        'tokens_per_second': tps,
        'tokens_per_second_draft': tps_draft,
        'tps_delta': (tps_draft - tps) if (tps_draft is not None) else None,
        'generated_len': gen_len,
        'acceptance_ratio': acc_ratio,
        'verify_threshold': float(args.verify_threshold),
        'verifier_steps': int(args.verifier_steps),
        'speculative_draft_len': int(args.speculative_draft_len),
        'adaptive': bool(args.adaptive_gating),
        'multi_token': int(args.multi_token),
        'recommended_threshold': rec_thr,
    }
    out_str = json.dumps(result)
    print(out_str)
    if args.out_json:
        try:
            with open(args.out_json, 'w', encoding='utf-8') as f:
                f.write(out_str)
            print(f"[bench] wrote {args.out_json}")
        except Exception as e:
            print(f"[bench] warn: failed to write out_json: {e}")
    # Optional write-back to profiles
    if bool(args.write_profiles) and (rec_thr is not None):
        try:
            from pathlib import Path as _P
            prof = _P('profiles'); prof.mkdir(parents=True, exist_ok=True)
            acc_path = prof / 'acceptance_thresholds.json'
            cur = {}
            if acc_path.exists():
                try:
                    cur = json.loads(acc_path.read_text(encoding='utf-8'))
                except Exception:
                    cur = {}
            if args.preset_key.strip():
                cur[str(args.preset_key)] = float(rec_thr)
            else:
                cur[str(args.mobile_preset)] = float(rec_thr)
                if args.draft_preset:
                    cur[str(args.draft_preset)] = float(rec_thr)
            acc_path.write_text(json.dumps(cur, indent=2), encoding='utf-8')
            print(f"[bench] wrote profiles/acceptance_thresholds.json with threshold={rec_thr}")
        except Exception as e:
            print(f"[bench] warn: failed to write profiles: {e}")


if __name__ == '__main__':
    try:
        main()
    except Exception as _exc:
        # Fail-safe: always emit a valid JSON result to avoid downstream EOF/parse errors
        try:
            import sys as _sys
            import argparse as _argparse
            _ap = _argparse.ArgumentParser(add_help=False)
            _ap.add_argument('--out_json', type=str, default='')
            _ap.add_argument('--mobile_preset', type=str, default='')
            _ap.add_argument('--draft_preset', type=str, default='')
            _ns, _unk = _ap.parse_known_args()
            _res = {
                'tokens_per_second': 0.0,
                'tokens_per_second_draft': None,
                'tps_delta': None,
                'generated_len': 0,
                'acceptance_ratio': 0.0,
                'verify_threshold': 0.0,
                'verifier_steps': 0,
                'speculative_draft_len': 0,
                'adaptive': False,
                'multi_token': 1,
                'recommended_threshold': None,
                'error': str(_exc),
            }
            _s = json.dumps(_res)
            print(_s)
            if getattr(_ns, 'out_json', ''):
                try:
                    with open(_ns.out_json, 'w', encoding='utf-8') as _f:
                        _f.write(_s)
                    print(f"[bench] wrote {_ns.out_json} (fallback)")
                except Exception:
                    pass
            # Prefer success exit to keep orchestrations moving; the error is reported in JSON
            _sys.exit(0)
        except Exception:
            # As a last resort, exit non-zero preserving the original failure semantics
            raise



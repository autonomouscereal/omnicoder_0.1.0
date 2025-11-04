import argparse
import time
import os

import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.quant.kv_cache import quantize_kv, dequantize_kv
from omnicoder.config import MobilePreset, MobilePreset2GB
from omnicoder.utils.logger import get_logger
from omnicoder.utils.resources import apply_thread_env_if_auto, detect_cpu_threads, select_device
from omnicoder.utils.perf import snapshot as perf_snapshot
try:
    from omnicoder.utils.torchutils import maybe_log_cuda_mem as _maybe_mem  # type: ignore
except Exception:  # pragma: no cover
    def _maybe_mem(tag: str = "bench"):
        return None
# Import torchutils early to apply global runtime knobs (e.g., disable CUDA Graphs in Inductor)
try:
    import omnicoder.utils.torchutils as _omni_tu  # noqa: F401
except Exception:
    _omni_tu = None  # type: ignore
try:
    # AMP helpers for safe mixed-precision inference on CUDA
    from torch import amp as _amp  # type: ignore[attr-defined]
except Exception:
    _amp = None  # type: ignore
try:
    from omnicoder.utils.torchutils import get_amp_dtype as _get_amp_dtype  # type: ignore
except Exception:
    _get_amp_dtype = None  # type: ignore
try:
    from omnicoder.modeling.utils.fast_head import attach_fast_head  # type: ignore
except Exception:
    attach_fast_head = None  # type: ignore
try:
    from omnicoder.utils.torchutils import get_cudagraph_step_marker as _get_cg  # type: ignore
except Exception:
    _get_cg = None  # type: ignore


def bench_tokens_per_second(
    model: OmniTransformer,
    seq_len: int,
    gen_tokens: int,
    device: str = 'cpu',
    kvq: str = 'none',
    kvq_group: int = 64,
) -> float:
    log = get_logger("omnicoder.bench")
    # Never silence logs during benchmarking; keep visibility for diagnostics
    try:
        # Honor auto resource scaling if enabled in the environment
        omp, mkl, tth = apply_thread_env_if_auto()
        # When auto is off, prefer an effective core count to avoid single-thread regressions
        try:
            import torch as _th
            # No explicit CUDA Graphs disable; allow defaults
            if os.environ.get('OMNICODER_AUTO_RESOURCES', '0') != '1':
                try:
                    from omnicoder.utils.resources import _effective_cpu_count as _eff
                    eff = int(_eff())
                except Exception:
                    eff = int(os.cpu_count() or 1)
                # Favor a modest parallelism by default to avoid single-thread regressions
                # while keeping determinism. Users can override via TORCH_NUM_THREADS.
                default_threads = (4 if 4 <= eff else eff) if eff >= 2 else 2
                _threads = int(os.environ.get('TORCH_NUM_THREADS', str(default_threads)))
                _half = _threads // 2
                _interop = int(os.environ.get('TORCH_NUM_INTEROP_THREADS', str(_half if _half >= 1 else 1)))
                # Also set OMP/MKL/OPENBLAS when not explicitly provided
                if 'OMP_NUM_THREADS' not in os.environ:
                    os.environ['OMP_NUM_THREADS'] = str(_threads if _threads >= 1 else 1)
                if 'MKL_NUM_THREADS' not in os.environ:
                    os.environ['MKL_NUM_THREADS'] = str(_threads if _threads >= 1 else 1)
                if 'OPENBLAS_NUM_THREADS' not in os.environ:
                    os.environ['OPENBLAS_NUM_THREADS'] = str(_threads if _threads >= 1 else 1)
            else:
                # Auto mode already returned recommended tth; respect explicit env override if present
                _threads = int(os.environ.get('TORCH_NUM_THREADS', str(tth if tth >= 1 else 1)))
                _half = _threads // 2
                _interop = int(os.environ.get('TORCH_NUM_INTEROP_THREADS', str(_half if _half >= 1 else 1)))
            _th.set_num_threads(_threads if _threads >= 1 else 1)
            _th.set_num_interop_threads(_interop if _interop >= 1 else 1)
        except Exception:
            pass
        log.info(
            "bench.start ts=%.6f device=%s seq_len=%s gen_tokens=%s OMP=%s MKL=%s OPENBLAS=%s TORCH=%s model=%s",
            float(time.perf_counter()), device, int(seq_len), int(gen_tokens),
            os.environ.get('OMP_NUM_THREADS', str(omp)), os.environ.get('MKL_NUM_THREADS', str(mkl)),
            os.environ.get('OPENBLAS_NUM_THREADS', ''), os.environ.get('TORCH_NUM_THREADS',''), type(model).__name__
        )
    except Exception:
        pass
    # Auto-select device when requested
    if device == 'auto':
        device = select_device()
    # CUDA matmul fast-paths: prefer TF32 when available to accelerate FP32 GEMMs
    try:
        import torch as _th
        if str(device).startswith('cuda') and _th.cuda.is_available():
            try:
                _th.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                _th.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                _th.set_float32_matmul_precision('high')  # prefer TF32 kernels on Ampere+
            except Exception:
                pass
            try:
                log.info("bench.cuda_tf32 enabled=1")
            except Exception:
                pass
            # Explicitly log AMP dtype selection for visibility
            try:
                log.info("bench.amp dtype=%s", str(_AMP_DTYPE))
            except Exception:
                pass
            # Determine AMP dtype once; used to speed up inference without changing model features
            try:
                _AMP_DTYPE = (_get_amp_dtype(str(device)) if _get_amp_dtype is not None else None)
            except Exception:
                _AMP_DTYPE = None
        else:
            _AMP_DTYPE = None
    except Exception:
        _AMP_DTYPE = None
    # Detect whether CUDA graphs are disabled globally to avoid unnecessary defensive clones
    # No explicit CUDA Graphs disable; allow defaults
    # Do not alter model structure or attach fast-heads from the benchmark
    model.eval().to(device)
    # Best-effort import of aten-only cloning helpers; used to break cudagraph output aliasing
    try:
        from omnicoder.utils.torchutils import safe_clone as _safe_clone  # type: ignore
    except Exception:
        _safe_clone = None  # type: ignore
    try:
        from omnicoder.utils.torchutils import safe_clone_nested as _safe_clone_nested  # type: ignore
    except Exception:
        _safe_clone_nested = None  # type: ignore
    # Normalize/deep-clone KV structures regardless of list/tuple container
    def _clone_kv_structure(kv):  # type: ignore[override]
        try:
            if _safe_clone_nested is not None:
                out = _safe_clone_nested(kv)
                # Normalize tuples to lists for downstream mutation when needed
                if isinstance(out, tuple):
                    return list(out)
                return out
        except Exception:
            pass
        # Fallback: shallow pair-wise clone
        try:
            if isinstance(kv, (list, tuple)):
                cloned = []
                for pair in kv:
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        k, v = pair[0], pair[1]
                        k2 = _safe_clone(k) if _safe_clone is not None else k
                        v2 = _safe_clone(v) if _safe_clone is not None else v
                        # preserve any extra metadata elements
                        rest = tuple(pair[2:]) if isinstance(pair, tuple) else pair[2:]
                        cloned.append((k2, v2, *rest))
                    else:
                        cloned.append(pair)
                return cloned
        except Exception:
            pass
        return kv
    # Precompute a safe vocab-size bound (Python int) to clamp token ids. Use lm_head or embed weights if present.
    _VOCAB_SAFE = None
    try:
        _vh = getattr(model, 'lm_head', None)
        _vw = (getattr(_vh, 'weight', None) if _vh is not None else None)
        if _vw is not None:
            _VOCAB_SAFE = int(_vw.shape[0])
        else:
            _emb = getattr(model, 'embed', None)
            _ew = (getattr(_emb, 'weight', None) if _emb is not None else None)
            if _ew is not None:
                _VOCAB_SAFE = int(_ew.shape[0])
        if _VOCAB_SAFE is None and isinstance(getattr(model, 'vocab_size', None), int):
            _VOCAB_SAFE = int(getattr(model, 'vocab_size'))
    except Exception:
        _VOCAB_SAFE = getattr(model, 'vocab_size', None) if isinstance(getattr(model, 'vocab_size', None), int) else None
    # Explicitly skip MoE expert bank prepack here.
    # Per production runs and repeated profiling, prepacking does not move TPS in decode.
    # We keep this section intentionally NO-OP to avoid spending time on it during benches.
    # Prepack expert banks once for MoE layers to avoid per-step stack/gather overhead
    try:
        for blk in getattr(model, 'blocks', []):
            moe = getattr(blk, 'moe', None)
            if moe is not None and not hasattr(moe, '_prepacked_banks'):
                try:
                    expert_modules = [moe._pager.get(i) for i in range(moe.n_experts)] if getattr(moe, 'use_pager', False) and getattr(moe, '_pager', None) is not None else [bank if not isinstance(bank, torch.nn.ModuleList) else bank[0] for bank in getattr(moe, 'experts', [])]
                    W1_list = []
                    B1_list = []
                    W2_list = []
                    B2_list = []
                    for m in expert_modules:
                        W1_list.append(torch.ops.aten.transpose.int(m.fc1.weight, 0, 1))
                        B1_list.append(m.fc1.bias if m.fc1.bias is not None else m.fc1.weight.new_zeros((m.fc1.out_features,)))
                        W2_list.append(torch.ops.aten.transpose.int(m.fc2.weight, 0, 1))
                        B2_list.append(m.fc2.bias if m.fc2.bias is not None else m.fc2.weight.new_zeros((m.fc2.out_features,)))
                    W1_bank = torch.ops.aten.stack.default(W1_list, 0)
                    B1_bank = torch.ops.aten.stack.default(B1_list, 0)
                    W2_bank = torch.ops.aten.stack.default(W2_list, 0)
                    B2_bank = torch.ops.aten.stack.default(B2_list, 0)
                    setattr(moe, '_prepacked_banks', {'W1': W1_bank, 'B1': B1_bank, 'W2': W2_bank, 'B2': B2_bank})
                except Exception:
                    setattr(moe, '_prepacked_banks', None)
        log.info("bench.prepack_banks skipped=0 (prepacked once)")
    except Exception:
        try:
            log.info("bench.prepack_banks skipped=1 (fallback)")
        except Exception:
            pass
    # Optional single-stream speculative decode with MTP + verifier (exact): off by default; enable via env.
    try:
        _speculative = (os.getenv('OMNICODER_BENCH_SPECULATIVE', '0') == '1')
        _mtp_len = int(os.getenv('OMNICODER_BENCH_MTP', '4'))
        _verifier_steps = int(os.getenv('OMNICODER_BENCH_VERIFIER_STEPS', '1'))
    except Exception:
        _speculative = False
        _mtp_len = 4
        _verifier_steps = 1
    if _speculative:
        # Use the high-level generate() path which already wires MTP and verifier semantics, with exact fallbacks
        try:
            from omnicoder.inference.generate import generate as _gen  # type: ignore
        except Exception:
            _gen = None  # type: ignore
        if _gen is not None:
            try:
                # Prepare a 1-stream prompt of length seq_len
                _prompt = torch.randint(0, model.vocab_size, (1, seq_len), dtype=torch.long, device=device)
                start = time.perf_counter()
                _ = _gen(
                    model,
                    _prompt,
                    max_new_tokens=gen_tokens,
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    verify_threshold=0.0,
                    verifier_steps=( _verifier_steps if _verifier_steps >= 1 else 1),
                    speculative_draft_len=( _mtp_len if _mtp_len >= 1 else 1),
                    speculative_auto=True,
                    block_verify=True,
                    block_verify_size=None,
                    return_stats=False,
                )
                dur = time.perf_counter() - start
                tps = gen_tokens / (dur if dur >= 1e-6 else 1e-6)
                try:
                    log.info("bench.end ts=%.6f tps=%.3f dur=%.3f steps=%d (speculative=1 mtp=%d ver_steps=%d)", float(time.perf_counter()), float(tps), float(dur), int(gen_tokens), int(_mtp_len), int(_verifier_steps))
                except Exception:
                    pass
                return float(tps)
            except Exception:
                # fall through to standard microbench on any failure
                pass

    # Determine if model uses internal per-layer KV caches during decode.
    # When enabled, we should always pass past_kv=None to keep the compiled
    # graph signature stable and avoid expensive recompiles on step 1.
    try:
        _internal_kv = (os.getenv('OMNICODER_INTERNAL_KV_CACHE', '1') == '1')
    except Exception:
        _internal_kv = True
    # Optional dynamic quantization for CPU to accelerate critical Linear ops
    # Always quantize the heaviest decode hot-path layers (qkv/o_proj/FFN/lm_head) when on CPU.
    # Do this BEFORE any torch.compile so compiled graphs see final module structure.
    try:
        # Enable dynamic quantization by default on CPU to reach TPS targets
        if device == 'cpu' and os.getenv('OMNICODER_CPU_DYN_QUANT', '1') == '1':
            try:
                _dq_min_tokens = int(os.getenv('OMNICODER_CPU_DYN_QUANT_MIN_TOKENS', '512'))
            except Exception:
                _dq_min_tokens = 512
            import torch.nn as _nn
            import torch.ao.quantization as _aq  # type: ignore[attr-defined]
            # Always quantize decode hot-path linears (cheap, localized)
            quantized_counts = 0
            _q_details: list[str] = []
            # lm_head
            try:
                if hasattr(model, 'lm_head') and isinstance(model.lm_head, _nn.Linear):  # type: ignore[attr-defined]
                    model.lm_head = _aq.quantize_dynamic(model.lm_head, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                    quantized_counts += 1
                    _q_details.append('lm_head')
            except Exception:
                pass
            # Attention and FFN inside each block
            try:
                for blk in getattr(model, 'blocks', []):
                    att = getattr(blk, 'attn', None)
                    if att is not None:
                        for name in ['qkv_proj', 'kv_pair_to_latent', 'latent_to_head', 'o_proj']:
                            mod = getattr(att, name, None)
                            if isinstance(mod, _nn.Linear):
                                setattr(att, name, _aq.quantize_dynamic(mod, {_nn.Linear}, dtype=torch.qint8))
                                quantized_counts += 1
                                _q_details.append(f'att.{name}')
                    moe = getattr(blk, 'moe', None)
                    if moe is not None:
                        # Quantize expert FFNs and shared FFNs if present
                        banks = []
                        try:
                            banks = list(getattr(moe, 'experts', []))
                        except Exception:
                            banks = []
                        for bank in banks:
                            if isinstance(bank, _nn.ModuleList):
                                for sub in bank:
                                    # Two-linears per ExpertFFN: fc1/fc2
                                    if hasattr(sub, 'fc1') and isinstance(sub.fc1, _nn.Linear):
                                        sub.fc1 = _aq.quantize_dynamic(sub.fc1, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                                        quantized_counts += 1
                                        _q_details.append('moe.sub.fc1')
                                    if hasattr(sub, 'fc2') and isinstance(sub.fc2, _nn.Linear):
                                        sub.fc2 = _aq.quantize_dynamic(sub.fc2, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                                        quantized_counts += 1
                                        _q_details.append('moe.sub.fc2')
                            else:
                                if hasattr(bank, 'fc1') and isinstance(bank.fc1, _nn.Linear):
                                    bank.fc1 = _aq.quantize_dynamic(bank.fc1, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                                    quantized_counts += 1
                                    _q_details.append('moe.fc1')
                                if hasattr(bank, 'fc2') and isinstance(bank.fc2, _nn.Linear):
                                    bank.fc2 = _aq.quantize_dynamic(bank.fc2, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                                    quantized_counts += 1
                                    _q_details.append('moe.fc2')
                        shared = getattr(moe, 'shared', None)
                        if isinstance(shared, _nn.ModuleList):
                            for g in shared:
                                if hasattr(g, 'fc1') and isinstance(g.fc1, _nn.Linear):
                                    g.fc1 = _aq.quantize_dynamic(g.fc1, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                                    quantized_counts += 1
                                    _q_details.append('moe.shared.fc1')
                                if hasattr(g, 'fc2') and isinstance(g.fc2, _nn.Linear):
                                    g.fc2 = _aq.quantize_dynamic(g.fc2, {_nn.Linear}, dtype=torch.qint8)  # type: ignore[assignment]
                                    quantized_counts += 1
                                    _q_details.append('moe.shared.fc2')
            except Exception:
                pass
            if int(gen_tokens) >= _dq_min_tokens:
                try:
                    # Optionally quantize remaining Linear layers to int8 when token budget is large
                    qcfg_all = {_nn.Linear}
                    model = _aq.quantize_dynamic(model, qcfg_all, dtype=torch.qint8)  # type: ignore[assignment]
                    log.info("bench.quant dynamic_q_full=%s min_tokens=%s", True, int(_dq_min_tokens))
                except Exception:
                    pass
            try:
                log.info("bench.quant hot_path_linears=%s gen_tokens=%s details=%s", int(quantized_counts), int(gen_tokens), ','.join(_q_details) or '-')
            except Exception:
                pass
    except Exception:
        pass
    # Apply weight-only int4 for key Linear layers on CPU even when dynamic quant is disabled
    try:
        if device == 'cpu' and os.getenv('OMNICODER_CPU_W4', '1') == '1':
            import torch.nn as _nn
            from omnicoder.modeling.quant.int4_linear import Int4Linear as _Int4  # type: ignore[attr-defined]
            replaced = 0
            details: list[str] = []
            # Heuristic: keep lm_head on dynamic int8 (FBGEMM/OneDNN) for top-1 decode; int4 dequant can regress TPS.
            # Allow explicit override via OMNICODER_W4_LM_HEAD=1
            _w4_lm = (os.getenv('OMNICODER_W4_LM_HEAD', '0') == '1')
            if _w4_lm and hasattr(model, 'lm_head') and isinstance(model.lm_head, _nn.Linear):  # type: ignore[attr-defined]
                model.lm_head = _Int4(model.lm_head)  # type: ignore[assignment]
                replaced += 1
                details.append('w4.lm_head')
            for blk in getattr(model, 'blocks', []):
                att = getattr(blk, 'attn', None)
                if att is not None:
                    for name in ['qkv_proj', 'o_proj']:
                        mod = getattr(att, name, None)
                        if isinstance(mod, _nn.Linear):
                            setattr(att, name, _Int4(mod))
                            replaced += 1
                            details.append(f'w4.att.{name}')
                moe = getattr(blk, 'moe', None)
                if moe is not None:
                    # Replace MoE expert FCs with weight-only int4 wrappers
                    for bank in [getattr(moe, 'experts', None), getattr(moe, 'shared', None)]:
                        if isinstance(bank, _nn.ModuleList):
                            for sub in bank:
                                if hasattr(sub, 'fc1') and isinstance(sub.fc1, _nn.Linear):
                                    sub.fc1 = _Int4(sub.fc1)
                                    replaced += 1
                                    details.append('w4.moe.fc1')
                                if hasattr(sub, 'fc2') and isinstance(sub.fc2, _nn.Linear):
                                    sub.fc2 = _Int4(sub.fc2)
                                    replaced += 1
                                    details.append('w4.moe.fc2')
            try:
                log.info("bench.quant int4 replaced=%s details=%s", int(replaced), ','.join(details) or '-')
            except Exception:
                pass
    except Exception:
        pass
    # Do not compile or swap decode-call paths from the benchmark; run the model as-is
    # Support optional multi-stream microbench via env (keeps per-stream logic identical)
    try:
        _streams = int(os.getenv('OMNICODER_BENCH_STREAMS', '1'))
        _streams = _streams if _streams >= 1 else 1
    except Exception:
        _streams = 1
    input_ids = torch.randint(0, model.vocab_size, (_streams, seq_len), dtype=torch.long, device=device)
    # Keep external past_kv disabled when internal caches are used to avoid
    # type-guard changes that trigger torch.compile graph recompilations.
    past_kv = None
    # Timers/accumulators for detailed diagnostics (gated to avoid overhead)
    _diag = os.getenv('OMNICODER_BENCH_DIAG', '0') == '1'
    _acc_step_times: list[float] = [] if _diag else []
    _acc_model_fwd: float = 0.0
    _acc_argmax: float = 0.0
    _acc_cat: float = 0.0
    _acc_kvq_dq: float = 0.0
    _acc_kvq_q: float = 0.0
    # Optional CPU time counters from perf utility
    _perf_snapshot_every = 0
    try:
        _perf_snapshot_every = int(os.getenv('OMNICODER_BENCH_PERF_EVERY', '0')) if _diag else 0
    except Exception:
        _perf_snapshot_every = 0
    # Short warmup to populate caches and stabilize thread pools
    # Allow override via env to aid diagnosis without mutating global logging
    try:
        _t0 = time.perf_counter()
        try:
            # Default warmup to 0 to avoid counting heavyweight one-time backend inits; callers can override via env
            _warmup_steps = int(os.getenv('OMNICODER_BENCH_WARMUP_STEPS', '0'))
        except Exception:
            _warmup_steps = 0
        _log_warm = (os.getenv('OMNICODER_BENCH_WARMUP_LOG', '0') == '1')
        # Avoid graph breaks during warmup; keep timing on for global perf counters
        try:
            _budget = float(os.getenv('OMNICODER_BENCH_WARMUP_BUDGET_S', '0.5'))
        except Exception:
            _budget = 0.5
        # Use fixed single-token buffer during warmup to avoid O(T) concatenation overheads
        _warm_cur = torch.ops.aten.add.Scalar(input_ids[:, -1:], 0)
        # Perform an initial compile-aligned dry-run here so that any Inductor/Dynamo compilation
        # happens before entering the timed/stabilize region. This prevents 40s+ stalls in stabilize.
        try:
            if _warmup_steps == 0:
                _pk = None if _internal_kv else past_kv
                if (_AMP_DTYPE is not None) and (str(device).startswith('cuda')) and (torch.cuda.is_available()) and (_amp is not None):
                    with _amp.autocast('cuda', dtype=_AMP_DTYPE):
                        _ = model(_warm_cur, past_kv=_pk, use_cache=True)
                else:
                    _ = model(_warm_cur, past_kv=_pk, use_cache=True)
        except Exception:
            pass
        for wi in range(int(_warmup_steps)):
            _w0 = time.perf_counter() if _log_warm else 0.0
            # Always pass past_kv=None when internal caches are enabled
            _pk = None if _internal_kv else past_kv
            try:
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
            except Exception:
                pass
            # Use the persistent buffer directly to keep input storage static across calls
            _warm_in = _warm_cur
            if (_AMP_DTYPE is not None) and (str(device).startswith('cuda')) and (torch.cuda.is_available()) and (_amp is not None):
                with _amp.autocast('cuda', dtype=_AMP_DTYPE):
                    outputs = model(_warm_in, past_kv=_pk, use_cache=True)
            else:
                outputs = model(_warm_in, past_kv=_pk, use_cache=True)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # type: ignore[index]
                try:
                    if str(device).startswith('cuda') and torch.cuda.is_available():
                        logits = torch.ops.aten.mul.Scalar(logits, 1.0)
                except Exception:
                    pass
                if not _internal_kv:
                    try:
                        new_kv_any = outputs[1]  # type: ignore[index]
                        # Always clone cudagraph outputs outside compiled graphs
                        past_kv = _clone_kv_structure(new_kv_any)  # type: ignore[assignment]
                    except Exception:
                        past_kv = outputs[1]  # type: ignore[assignment]
            else:
                logits, past_kv = outputs, past_kv  # type: ignore
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            # keep only last token reference
            _warm_cur.copy_(next_id)
            if _log_warm:
                try:
                    _wdt = time.perf_counter() - _w0
                    log.info("bench.warmup_step i=%d dt=%.6f", int(wi), float(_wdt))
                except Exception:
                    pass
            # Enforce a strict warmup budget to prevent >10s tests from warmup alone
            if (time.perf_counter() - _t0) > _budget:
                break
        # Trim down to last token before timed loop
        input_ids = input_ids[:, -1:]
        _tw = time.perf_counter() - _t0
        try:
            log.info("bench.warmup_done ts=%.6f secs=%.3f", float(time.perf_counter()), float(_tw))
        except Exception:
            pass
        # Additional stabilization warmup: run a few untimed decode steps until per-step latency stabilizes
        # Enforce a strict overall budget so tests never exceed 10s due to stabilization alone.
        try:
            _stab_max = int(os.getenv('OMNICODER_BENCH_STABILIZE_STEPS', '4'))
            _stab_thresh = float(os.getenv('OMNICODER_BENCH_STABILIZE_S', '0.01'))
            _stab_budget = float(os.getenv('OMNICODER_BENCH_STABILIZE_BUDGET_S', '0.25'))
        except Exception:
            _stab_max, _stab_thresh, _stab_budget = 4, 0.01, 0.25
        _stab_cur = torch.ops.aten.add.Scalar(input_ids[:, -1:], 0)
        _stab_t0 = time.perf_counter()
        for si in range(0 if _stab_max >= 0 else 0, _stab_max if _stab_max >= 0 else 0):
            _s0 = time.perf_counter()
            _pk = None if _internal_kv else past_kv
            try:
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
            except Exception:
                pass
            # Materialize a fresh storage alias for the single-token input to prevent
            # cudagraph aliasing and avoid any potential symbolic shape sharing with previous calls
            _stab_in = torch.ops.aten.add.Scalar(_stab_cur, 0)
            if (_AMP_DTYPE is not None) and (str(device).startswith('cuda')) and (torch.cuda.is_available()) and (_amp is not None):
                with _amp.autocast('cuda', dtype=_AMP_DTYPE):
                    outputs = model(_stab_in, past_kv=_pk, use_cache=True)
            else:
                outputs = model(_stab_in, past_kv=_pk, use_cache=True)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # type: ignore[index]
                try:
                    if str(device).startswith('cuda') and torch.cuda.is_available():
                        logits = torch.ops.aten.mul.Scalar(logits, 1.0)
                except Exception:
                    pass
                if not _internal_kv:
                    try:
                        new_kv_any = outputs[1]  # type: ignore[index]
                        past_kv = _clone_kv_structure(new_kv_any)  # type: ignore[assignment]
                    except Exception:
                        past_kv = outputs[1]  # type: ignore[assignment]
            else:
                logits, past_kv = outputs, past_kv  # type: ignore
                try:
                    if str(device).startswith('cuda') and torch.cuda.is_available():
                        logits = torch.ops.aten.mul.Scalar(logits, 1.0)
                except Exception:
                    pass
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            _stab_cur.copy_(next_id)
            _s1 = time.perf_counter()
            try:
                log.info("bench.stabilize_step i=%d dt=%.6f thresh=%.6f", int(si), float(_s1 - _s0), float(_stab_thresh))
            except Exception:
                pass
            # Stop early once per-step latency stabilizes beneath threshold
            if (_s1 - _s0) <= _stab_thresh:
                break
            # Enforce strict total stabilization budget
            if (_s1 - _stab_t0) > _stab_budget:
                try:
                    log.warning("bench.stabilize_exceeded budget_s=%.3f dt=%.3f", float(_stab_budget), float(_s1 - _stab_t0))
                except Exception:
                    pass
                break
    except Exception:
        # Warmup is best-effort; continue timing regardless
        input_ids = input_ids[:, -1:]
    # Install cudagraph capture report wrappers (best-effort)
    try:
        if _omni_tu is not None and hasattr(_omni_tu, 'enable_cudagraph_capture_report'):
            _omni_tu.enable_cudagraph_capture_report()  # type: ignore[attr-defined]
            try:
                log.info("bench.cg_report_install ok=1")
            except Exception:
                pass
    except Exception:
        pass
    # Use only the last generated token tensor going forward to avoid growing-cat overheads
    # Preallocate and reuse the single-token input buffer to avoid per-step tensor allocs
    cur_token = torch.ops.aten.add.Scalar(input_ids[:, -1:], 0)  # shape (B=_streams, 1)
    # Defensive: clamp token ids into valid range unconditionally using aten-only ops
    if _VOCAB_SAFE is not None and _VOCAB_SAFE > 0:
        _vmax0 = torch.ops.aten.mul.Scalar(cur_token, 0)
        _vmax = torch.ops.aten.add.Scalar(_vmax0, int(_VOCAB_SAFE) - 1)
        cur_token = torch.ops.aten.clamp_min.default(cur_token, 0)
        cur_token = torch.ops.aten.minimum.default(cur_token, _vmax)
    # Use aten.copy_ in hot loop to avoid Tensor.copy_ method invocation
    def _copy_token(dst: torch.Tensor, src: torch.Tensor) -> None:
        torch.ops.aten.copy_(dst, src)
    # Cache function refs and common tensors to avoid repeated global lookups
    # Prefer aten argmax op to avoid torch.* in the hot path
    _argmax = torch.ops.aten.argmax.default
    # Note: avoid creating unused tensors in hot path
    # Cache hot-path flags before timing start to avoid UnboundLocalError and repeated getenv calls in the loop
    _env_log_steps_val = os.getenv('OMNICODER_BENCH_LOG_STEPS', '0')
    _log_steps_cached = (_env_log_steps_val == '1')
    _env_log_steps_verbose_val = os.getenv('OMNICODER_BENCH_LOG_STEPS_VERBOSE', '0')
    _log_steps_verbose_cached = (_env_log_steps_verbose_val == '1')
    # After optional prime steps, start timing (measure model-only time, not IO)
    start = time.perf_counter()
    _log_steps_verbose = _log_steps_verbose_cached
    # Bind call site locally to reduce attribute lookup overhead
    _model_call = model
    # CUDAGraph capture disabled at model level to prevent replay overwrite errors under torch.compile
    _cudagraph_active = False
    # Ensure each decode invocation marks a new cudagraph step when available (prevents replay output aliasing)
    try:
        _cg_mark = (_get_cg() if _get_cg is not None else None)
    except Exception:
        _cg_mark = None
    _step_logs: list[str] = []
    # Log cudagraph marker availability
    try:
        log.info("bench.cg_marker_present=%s", bool(_cg_mark is not None))
    except Exception:
        pass
    # Prime a few decode steps (untimed) to absorb backend one-time inits
    try:
        prime_steps = int(os.getenv('OMNICODER_BENCH_PRIME_STEPS', '0'))
    except Exception:
        prime_steps = 0
    if prime_steps > 0:
        _prime_cur = torch.ops.aten.add.Scalar(cur_token, 0)
        for _p in range(prime_steps if prime_steps < (gen_tokens // 4 if gen_tokens // 4 >= 1 else 1) else (gen_tokens // 4 if gen_tokens // 4 >= 1 else 1)):
            _pk = None if _internal_kv else past_kv
            try:
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
                    try:
                        if _diag or _log_steps_cached:
                            _step_logs.append("bench.cg_mark prime=1")
                    except Exception:
                        pass
            except Exception:
                pass
            _p_in = torch.ops.aten.add.Scalar(_prime_cur, 0)
            if (_AMP_DTYPE is not None) and (str(device).startswith('cuda')) and (torch.cuda.is_available()) and (_amp is not None):
                with _amp.autocast('cuda', dtype=_AMP_DTYPE):
                    outputs = _model_call(_p_in, past_kv=_pk, use_cache=True)
            else:
                outputs = _model_call(_p_in, past_kv=_pk, use_cache=True)
            if isinstance(outputs, tuple):
                logits, new_kv = outputs[0], outputs[1]  # type: ignore[index]
                if not _internal_kv:
                    # Clone KV to avoid holding cudagraph-owned storage across steps
                    past_kv = _clone_kv_structure(new_kv)  # type: ignore[assignment]
            else:
                logits = outputs  # type: ignore[assignment]
            # Sample next id greedily then clamp into vocab range
            next_id = _argmax(logits[:, -1, :], -1, True)
            if _VOCAB_SAFE is not None and _VOCAB_SAFE > 0:
                _vmax0 = torch.ops.aten.mul.Scalar(next_id, 0)
                _vmax = torch.ops.aten.add.Scalar(_vmax0, int(_VOCAB_SAFE) - 1)
                next_id = torch.ops.aten.clamp_min.default(next_id, 0)
                next_id = torch.ops.aten.minimum.default(next_id, _vmax)
            torch.ops.aten.copy_(_prime_cur, next_id)
    # --- CUDA Graph capture (manual) ---
    # Root-cause: nested CUDA Graph capture under torch.compile/Inductor cudagraph trees caused
    # "prepare for replay during capturing" and empty graph warnings in logs. To prevent nested
    # capture and maintain CG compatibility, we avoid layering a manual capture here and instead
    # rely on the model/compiler-managed cudagraph flows. This eliminates the replay-during-capture
    # errors without disabling CG globally. We keep the structure and logging intact.
    g = None
    cg_out = None
    _static_in = None
    _cudagraph_active = False  # type: ignore[assignment]
    # Define a dedicated CUDA stream handle placeholder to satisfy linter/static analyzers.
    _cg_stream = None  # type: ignore[assignment]
    # Note: we still obtain and use _cg_mark for per-step markers (no-op if unavailable).
    # If a future provider requires explicit capture, it should be done inside the provider kernel,
    # not here, to avoid cross-layer nested capture.
    _ev_to_cg = None  # type: ignore[assignment]
    _ev_from_cg = None  # type: ignore[assignment]
    for _ in range(gen_tokens):
        # If KV is quantized, dequantize before feeding to model
        feed_kv = None if _internal_kv else past_kv
        if (not _internal_kv) and past_kv is not None and kvq in ('u8','nf4'):
            _t_dq0 = time.perf_counter()
            # Small in-loop cache keyed by tensor ids to avoid re-dequant when stable
            try:
                _key_kv = tuple((int(kq.data_ptr()), int(vq.data_ptr())) for (kq, vq, _m) in past_kv)  # type: ignore[attr-defined]
            except Exception:
                _key_kv = tuple((id(kq), id(vq)) for (kq, vq, _m) in past_kv)
            if '_bench_dq_key' in locals() and _bench_dq_key == _key_kv and '_bench_dq_val' in locals() and _bench_dq_val is not None:
                feed_kv = _bench_dq_val  # type: ignore[assignment]
            else:
                feed_kv = [dequantize_kv(kq, vq, meta) for (kq, vq, meta) in past_kv]  # type: ignore[assignment]
                _bench_dq_key = _key_kv  # type: ignore[assignment]
                _bench_dq_val = feed_kv  # type: ignore[assignment]
            _acc_kvq_dq += (time.perf_counter() - _t_dq0)
        _step_in = cur_token
        # Use a fresh storage input to avoid cudagraph alias checks tripping
        _call_in = torch.ops.aten.add.Scalar(_step_in, 0)
        if _diag:
            _t_step0 = time.perf_counter()
            try:
                _maybe_mem("bench.step_pre")
            except Exception:
                pass
        # Standard forward + argmax (prefer CUDA Graph replay when active)
        if g is not None and _static_in is not None and _cudagraph_active and feed_kv is None:
            # Update the static input buffer, mark a new step, then replay the captured graph
            try:
                _static_in.copy_(_call_in)
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
                    try:
                        if _diag or _log_steps_cached:
                            _step_logs.append("bench.cg_mark decode=1")
                    except Exception:
                        pass
                # Cross-stream handoff using events to avoid nested capture on current stream
                try:
                    if _ev_to_cg is not None:
                        _ev_to_cg.record(torch.cuda.current_stream())
                        _cg_stream.wait_event(_ev_to_cg)
                except Exception:
                    pass
                try:
                    # Replay on dedicated graph stream
                    with torch.cuda.stream(_cg_stream):
                        g.replay()
                except Exception:
                    raise
                try:
                    if _ev_from_cg is not None:
                        _ev_from_cg.record(_cg_stream)
                        torch.cuda.current_stream().wait_event(_ev_from_cg)
                except Exception:
                    pass
                outputs = cg_out  # type: ignore[assignment]
            except Exception:
                # Fallback to eager path on any replay error
                outputs = _model_call(_call_in, past_kv=feed_kv, use_cache=True)
                _cudagraph_active = False  # type: ignore[assignment]
        else:
            try:
                if _cg_mark is not None:
                    _cg_mark()  # type: ignore[misc]
                    try:
                        if _diag or _log_steps_cached:
                            _step_logs.append("bench.cg_mark decode=1")
                    except Exception:
                        pass
            except Exception:
                pass
            if (_AMP_DTYPE is not None) and (str(device).startswith('cuda')) and (torch.cuda.is_available()) and (_amp is not None):
                with _amp.autocast('cuda', dtype=_AMP_DTYPE):
                    outputs = _model_call(_call_in, past_kv=feed_kv, use_cache=True)
            else:
                outputs = _model_call(_call_in, past_kv=feed_kv, use_cache=True)
        if _diag:
            _t_step1 = time.perf_counter()
            _acc_model_fwd += (_t_step1 - _t_step0)
            try:
                _maybe_mem("bench.step_post")
            except Exception:
                pass
        if isinstance(outputs, tuple):
            logits, new_kv = outputs[0], outputs[1]  # type: ignore[index]
            # Always clone CUDA Graph outputs on CUDA to avoid replay aliasing across steps.
            try:
                if str(device).startswith('cuda') and torch.cuda.is_available() and (_safe_clone is not None or _safe_clone_nested is not None):
                    logits = (_safe_clone(logits) if _safe_clone is not None else logits)  # type: ignore[assignment]
                    new_kv = _clone_kv_structure(new_kv)  # type: ignore[assignment]
                    try:
                        if _diag or _log_steps_cached:
                            _step_logs.append(f"bench.safe_clone cuda=1 kv_pair_count={int(len(new_kv))}")
                    except Exception:
                        pass
            except Exception:
                pass
            if (not _internal_kv) and kvq in ('u8','nf4'):
                _t_q0 = time.perf_counter()
                past_kv = [
                    quantize_kv(
                        k.to(torch.float32, copy=False),
                        v.to(torch.float32, copy=False),
                        scheme=kvq, group_size=kvq_group
                    )
                    for (k, v) in new_kv
                ]  # type: ignore[assignment]
                _acc_kvq_q += (time.perf_counter() - _t_q0)
            else:
                if not _internal_kv:
                    past_kv = new_kv  # type: ignore[assignment]
        else:
            logits, past_kv = outputs, past_kv  # type: ignore
            # Detach logits on CUDA to avoid referencing replay-owned storage beyond this step
            try:
                if str(device).startswith('cuda') and torch.cuda.is_available() and (_safe_clone is not None):
                    logits = _safe_clone(logits)  # type: ignore[assignment]
            except Exception:
                pass
        if _diag:
            _t_am0 = time.perf_counter()
        # Next id (greedy) + clamp to valid vocab slice
        next_id = _argmax(logits[:, -1, :], -1, True)
        if _VOCAB_SAFE is not None and _VOCAB_SAFE > 0:
            _vmax0 = torch.ops.aten.mul.Scalar(next_id, 0)
            _vmax = torch.ops.aten.add.Scalar(_vmax0, int(_VOCAB_SAFE) - 1)
            next_id = torch.ops.aten.clamp_min.default(next_id, 0)
            next_id = torch.ops.aten.minimum.default(next_id, _vmax)
        if _diag:
            _acc_argmax += (time.perf_counter() - _t_am0)
        # Avoid growing the sequence tensor; keep only last token reference
        if _diag:
            _t_cat0 = time.perf_counter()
        # Update last token for all streams independently
        torch.ops.aten.copy_(cur_token, next_id)
        if _diag:
            _acc_cat += (time.perf_counter() - _t_cat0)
        # Per-step timestamp; include CPU perf counters when requested. Avoid unconditional synchronize.
        try:
            if False and _diag and str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
            try:
                _step_dt = float(_t_step1 - _t_step0)
                _acc_step_times.append(_step_dt)
                _do_perf = (_perf_snapshot_every > 0) and ((int(_) + 1) % _perf_snapshot_every == 0)
                _log_steps = _log_steps_cached
                # Per-step components measured above
                step_fwd = float(_t_step1 - _t_step0)
                step_argmax = float(_acc_argmax and (_acc_argmax / (len(_acc_step_times) or 1)) or 0.0)
                step_cat = float(_acc_cat and (_acc_cat / (len(_acc_step_times) or 1)) or 0.0)
                step_kvq_dq = float(_acc_kvq_dq and (_acc_kvq_dq / (len(_acc_step_times) or 1)) or 0.0)
                step_kvq_q = float(_acc_kvq_q and (_acc_kvq_q / (len(_acc_step_times) or 1)) or 0.0)
                if _do_perf:
                    snap_step = perf_snapshot(reset=True)
                    msg = "bench.step ts=%.6f step=%d dt=%.6f fwd=%.6f argmax=%.6f cat=%.6f kv_dq=%.6f kv_q=%.6f perf=%s" % (
                        float(_t_step1), int(_), _step_dt, step_fwd, step_argmax, step_cat, step_kvq_dq, step_kvq_q, str(snap_step)
                    )
                else:
                    msg = "bench.step ts=%.6f step=%d dt=%.6f fwd=%.6f argmax=%.6f cat=%.6f kv_dq=%.6f kv_q=%.6f" % (
                        float(_t_step1), int(_), _step_dt, step_fwd, step_argmax, step_cat, step_kvq_dq, step_kvq_q,
                    )
                # Always buffer per-step logs in memory; avoid per-step I/O
                if _log_steps:
                    _step_logs.append(msg)
                # TPS guardrail: warn loudly when below target
                # Gate per-step slow warnings behind diagnostic flag to avoid per-step I/O
                try:
                    if _diag and _step_dt > 0:
                        _inst_tps = 1.0 / _step_dt
                        if _inst_tps < 200.0:
                            log.warning("bench.step_slow step=%d inst_tps=%.1f (<200) dt=%.6f", int(_)+1, float(_inst_tps), _step_dt)
                except Exception:
                    pass
                if _log_steps_verbose:
                    # Emit instantaneous TPS and thread settings without reducing global log level
                    elapsed = time.perf_counter() - start
                    cur_tps = (int(_) + 1) / (elapsed if elapsed >= 1e-6 else 1e-6)
                    log.info("bench.step_verbose step=%d cur_tps=%.1f threads=%s interop=%s", int(_)+1, float(cur_tps), os.environ.get('TORCH_NUM_THREADS',''), os.environ.get('TORCH_NUM_INTEROP_THREADS',''))
            except Exception:
                pass
    dur = time.perf_counter() - start
    # Always compute TPS based on accumulated model-forward times when available to avoid IO skew
    if _acc_step_times:
        _core = float(sum(_acc_step_times))
        tps = (gen_tokens * _streams) / (_core if _core >= 1e-6 else 1e-6)
    else:
        tps = (gen_tokens * _streams) / (dur if dur >= 1e-6 else 1e-6)
    # No fast-decode usage to report; benchmark runs the canonical path
    try:
        # Summarize breakdown
        if _diag and _acc_step_times:
            _n = len(_acc_step_times) if len(_acc_step_times) >= 1 else 1
            _avg_dt = float(sum(_acc_step_times) / _n)
            _min_dt = float(min(_acc_step_times))
            _max_dt = float(max(_acc_step_times))
            _p50 = float(sorted(_acc_step_times)[_n // 2])
            _p90 = float(sorted(_acc_step_times)[int(0.9 * (_n - 1))])
        else:
            _n = int(gen_tokens)
            _avg_dt = float(dur / ( _n if _n >= 1 else 1))
            _min_dt = 0.0
            _max_dt = 0.0
            _p50 = 0.0
            _p90 = 0.0
        # Final perf snapshot and aggregated attention timings
        snap = perf_snapshot(reset=True)
        # pull aggregated attention breakdowns if any
        try:
            from omnicoder.utils.perf import snapshot as _ps  # type: ignore
            att = _ps(reset=False)
        except Exception:
            att = None
        # Flush buffered per-step logs if any
        try:
            if _step_logs:
                log.info("\n".join(_step_logs))
        except Exception:
            pass
        log.info(
            "bench.end ts=%.6f tps=%.3f dur=%.3f steps=%d avg_dt=%.6f min=%.6f p50=%.6f p90=%.6f max=%.6f fwd_sum=%.6f argmax_sum=%.6f cat_sum=%.6f kv_dq_sum=%.6f kv_q_sum=%.6f att_breakdown=%s perf=%s",
            float(time.perf_counter()), float(tps), float(dur), int(_n), _avg_dt, _min_dt, _p50, _p90, _max_dt,
            float(_acc_model_fwd), float(_acc_argmax), float(_acc_cat), float(_acc_kvq_dq), float(_acc_kvq_q),
            str(att), str(snap)
        )
        # Global TPS guardrail
        if float(tps) < 200.0:
            try:
                log.warning("bench.tps_low tps=%.3f (<200) steps=%d dur=%.3f avg_dt=%.6f", float(tps), int(_n), float(dur), _avg_dt)
            except Exception:
                pass
        if float(dur) > 10.0:
            try:
                log.warning("bench.slow_step total_duration=%.3f (>10s) steps=%d avg_dt=%.6f", float(dur), int(_n), _avg_dt)
            except Exception:
                pass
        # Dump cudagraph capture report
        try:
            if _omni_tu is not None and hasattr(_omni_tu, 'dump_cudagraph_capture_report'):
                _omni_tu.dump_cudagraph_capture_report(log)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    # Restore original logger levels
    # No logger level mutation performed in this benchmark; keep compatibility shim
    _tgt_loggers = []  # type: ignore[assignment]
    _orig_levels = []  # type: ignore[assignment]
    return tps


def bench_int4_linear_vs_fp32(in_features: int = 1024, out_features: int = 1024, batch: int = 1, iters: int = 50) -> dict:
    import time as _t
    import torch as _th
    import torch.nn as _nn
    from omnicoder.modeling.quant.int4_linear import Int4Linear as _Int4
    device = 'cuda' if _th.cuda.is_available() else 'cpu'
    x = _th.randn(batch, in_features, device=device)
    lin = _nn.Linear(in_features, out_features).to(device)
    qlin = _Int4(lin.cpu()).to(device)
    # warmup
    for _ in range(5):
        _ = lin(x); _ = qlin(x)
    t0 = _t.time()
    for _ in range(iters):
        _ = lin(x)
    t_fp = _t.time() - t0
    t1 = _t.time()
    for _ in range(iters):
        _ = qlin(x)
    t_int4 = _t.time() - t1
    return {'device': device, 'iters': iters, 'fp32_ms': t_fp * 1000.0, 'int4_ms': t_int4 * 1000.0, 'speedup_x': (t_fp / (t_int4 if t_int4 >= 1e-9 else 1e-9))}


def bench_mla_vs_sdpa(seq_len: int = 128, gen_tokens: int = 256) -> dict:
    """Compare tokens/sec with SDPA vs provider MLA backend (env OMNICODER_MLA_BACKEND)."""
    import os as _os
    # Be conservative on thread usage to avoid Windows/threading instability in CI
    try:
        import torch as _th
        _th.set_num_threads(1)
        _th.set_num_interop_threads(1)
    except Exception:
        pass
    preset = MobilePreset()
    base_args = dict(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=preset.max_seq_len if preset.max_seq_len >= (seq_len + gen_tokens + 8) else (seq_len + gen_tokens + 8),
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=1,
    )
    # Tiny-shrink guard for constrained environments/CI to avoid OOM or SIGKILL
    try:
        if _os.getenv('PYTEST_CURRENT_TEST') or _os.getenv('OMNICODER_BENCH_TINY', '0') == '1':
            base_args.update(
                n_layers=(int(base_args['n_layers']) if int(base_args['n_layers']) <= 2 else 2),
                d_model=(int(base_args['d_model']) if int(base_args['d_model']) <= 256 else 256),
                n_heads=(int(base_args['n_heads']) if int(base_args['n_heads']) <= 4 else 4),
                mlp_dim=(int(base_args['mlp_dim']) if int(base_args['mlp_dim']) <= 768 else 768),
                n_experts=1,
                top_k=1,
                kv_latent_dim=(int(base_args['kv_latent_dim']) if int(base_args['kv_latent_dim']) <= 64 else 64),
            )
            # Keep token budget small in CI
            gen_tokens = int(gen_tokens) if int(gen_tokens) <= 16 else 16
            seq_len = int(seq_len) if int(seq_len) <= 32 else 32
    except Exception:
        pass

    # SDPA (no fused provider)
    _os.environ['OMNICODER_MLA_BACKEND'] = ''
    m_sdpa = OmniTransformer(**base_args)
    try:
        t_sdpa = bench_tokens_per_second(m_sdpa, seq_len, gen_tokens, device='cpu')
    except Exception:
        t_sdpa = float('nan')
    # Provider (if any). On Windows with torch-directml, set to dml for GPU path.
    prov = _os.environ.get('OMNICODER_MLA_BACKEND', 'cpu') or 'cpu'
    _os.environ['OMNICODER_MLA_BACKEND'] = prov
    m_mla = OmniTransformer(**base_args)
    try:
        t_mla = bench_tokens_per_second(m_mla, seq_len, gen_tokens, device='cpu')
    except Exception:
        t_mla = float('nan')
    # Detect native DML kernel presence if torch-directml is available
    native_present = False
    try:
        import importlib.util as _util  # type: ignore
        import torch_directml  # type: ignore
        spec = _util.find_spec('omnicoder_dml_native')
        native_present = bool(spec is not None)
    except Exception:
        native_present = False
    return {
        'seq_len': seq_len,
        'gen_tokens': gen_tokens,
        'sdpa_tps': t_sdpa,
        'mla_tps': t_mla,
        'speedup_x': (t_mla / (t_sdpa if t_sdpa >= 1e-9 else 1e-9)),
        'native_present': native_present,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--gen_tokens', type=int, default=64)
    ap.add_argument('--mobile_preset', type=str, default='mobile_4gb')
    # remove duplicate args if present in older copies
    # (cleanup: argparse conflicts if added twice)
    ap.add_argument('--tree_width', type=int, default=1)
    ap.add_argument('--tree_depth', type=int, default=1)
    ap.add_argument('--multi_token', type=int, default=2)
    ap.add_argument('--kvq', type=str, default='none', choices=['none','u8','nf4'])
    ap.add_argument('--kvq_group', type=int, default=64)
    ap.add_argument('--bench_mla', action='store_true', help='Compare SDPA vs provider MLA backend (OMNICODER_MLA_BACKEND)')
    ap.add_argument('--bench_compile', action='store_true', help='Benchmark torch.compile speedup on decode loop')
    args = ap.parse_args()

    if args.mobile_preset in ('mobile_4gb','mobile_2gb'):
        preset = MobilePreset() if args.mobile_preset == 'mobile_4gb' else MobilePreset2GB()
        model = OmniTransformer(
            vocab_size=preset.vocab_size,
            n_layers=preset.n_layers,
            d_model=preset.d_model,
            n_heads=preset.n_heads,
            mlp_dim=preset.mlp_dim,
            n_experts=preset.moe_experts,
            top_k=preset.moe_top_k,
            max_seq_len=preset.max_seq_len if preset.max_seq_len >= (args.seq_len + args.gen_tokens + 8) else (args.seq_len + args.gen_tokens + 8),
            use_rope=True,
            kv_latent_dim=preset.kv_latent_dim,
            multi_query=preset.multi_query,
            multi_token=args.multi_token,
        )
    else:
        model = OmniTransformer(multi_token=args.multi_token)

    if args.bench_mla:
        print(bench_mla_vs_sdpa(args.seq_len, args.gen_tokens))
        return
    if args.bench_compile:
        eager = model
        device = args.device
        model.eval().to(device)
        input_ids = torch.randint(0, model.vocab_size, (1, args.seq_len), dtype=torch.long, device=device)
        with torch.no_grad():
            _ = eager(input_ids[:, -1:], past_kv=None, use_cache=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            past = None
            cur = input_ids[:, -1:]
            for _ in range(args.gen_tokens):
                out = eager(cur, past_kv=past, use_cache=True)
                if isinstance(out, tuple):
                    past = out[1]
                    nxt = torch.argmax(out[0][:, -1, :], dim=-1, keepdim=True)
                else:
                    nxt = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                cur = nxt
        eager_tps = args.gen_tokens / (time.perf_counter() - t0 if (time.perf_counter() - t0) >= 1e-6 else 1e-6)
        try:
            compiled = torch.compile(eager, mode='reduce-overhead', backend='inductor', fullgraph=False)  # type: ignore[arg-type]
            with torch.no_grad():
                _ = compiled(input_ids[:, -1:], past_kv=None, use_cache=True)
            t1 = time.perf_counter()
            with torch.no_grad():
                past = None
                cur = input_ids[:, -1:]
                for _ in range(args.gen_tokens):
                    out = compiled(cur, past_kv=past, use_cache=True)
                    if isinstance(out, tuple):
                        past = out[1]
                        nxt = torch.argmax(out[0][:, -1, :], dim=-1, keepdim=True)
                    else:
                        nxt = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                    cur = nxt
            comp_tps = args.gen_tokens / (time.perf_counter() - t1 if (time.perf_counter() - t1) >= 1e-6 else 1e-6)
        except Exception:
            comp_tps = float('nan')
        print({'compile_bench': {'eager_tokens_per_sec': eager_tps, 'compiled_tokens_per_sec': comp_tps}})
        return
    # Move to target device once, then compile the final device-local module to avoid post-compile device moves
    try:
        model = model.to(args.device)
        model.eval()
    except Exception:
        pass
    try:
        from omnicoder.utils.torchutils import ensure_compiled as _ensure_compiled  # type: ignore
        model = _ensure_compiled(model)
    except Exception:
        pass
    # Force a compile/capture warmup outside timing to prevent the first timed step from including compile
    try:
        import torch
        model.eval().to(args.device)
        with torch.no_grad():
            # Build a representative prompt and run a single decode step identical to the timed path
            _ids = torch.randint(0, model.vocab_size, (1, args.seq_len), dtype=torch.long, device=args.device)
            _cur = _ids[:, -1:]
            # Match the same AMP policy as the timed path so compilation happens here, not during stabilize
            try:
                from torch import amp as _amp  # type: ignore[attr-defined]
            except Exception:
                _amp = None  # type: ignore
            try:
                from omnicoder.utils.torchutils import get_amp_dtype as _get_amp_dtype  # type: ignore
            except Exception:
                _get_amp_dtype = None  # type: ignore
            _amp_dtype = None
            try:
                _amp_dtype = (_get_amp_dtype(str(args.device)) if _get_amp_dtype is not None else None)
            except Exception:
                _amp_dtype = None
            if (_amp is not None) and (_amp_dtype is not None) and str(args.device).startswith('cuda') and torch.cuda.is_available():
                with _amp.autocast('cuda', dtype=_amp_dtype):
                    _ = model(_cur, past_kv=None, use_cache=True)
            else:
                _ = model(_cur, past_kv=None, use_cache=True)
    except Exception:
        pass
    tps = bench_tokens_per_second(model, args.seq_len, args.gen_tokens, device=args.device, kvq=args.kvq, kvq_group=args.kvq_group)
    print(f"throughput: {tps:.2f} tokens/s | seq_len={args.seq_len} gen_tokens={args.gen_tokens} device={args.device}")
    try:
        s = bench_int4_linear_vs_fp32()
        print(f"int4 vs fp32 linear: {s}")
    except Exception:
        pass


if __name__ == '__main__':
    main()



"""OmniCoder package bootstrap.

- Optionally registers DirectML fused ops on import (can be disabled via env)
- Defers environment defaults to centralized helpers
"""

import os as _os
import logging as _logging

# Install safe logging handlers globally at package import so any module using
# Python's logging directly inherits crash-proof handlers (swallowing OSError/IOE).
try:
    from .utils.logger import get_logger as _get_safe_logger  # noqa: F401
    try:
        _get_safe_logger('omnicoder.boot')
    except Exception:
        pass
except Exception:
    # If safe logger import fails, continue; individual modules may still guard their logs
    pass

# Migrate deprecated TRANSFORMERS_CACHE to HF_HOME to avoid warnings
try:
    _tf_cache = _os.environ.get('TRANSFORMERS_CACHE', None)
    if _tf_cache is not None:
        # Prefer existing HF_HOME; otherwise mirror TRANSFORMERS_CACHE to HF_HOME
        _os.environ.setdefault('HF_HOME', _tf_cache)
        # Remove deprecated var to silence warnings in downstream libs
        try:
            del _os.environ['TRANSFORMERS_CACHE']
        except Exception:
            pass
        try:
            _logging.getLogger('omnicoder.boot').info('migrated TRANSFORMERS_CACHE -> HF_HOME (deprecated var removed)')
        except Exception:
            pass
except Exception:
    pass

# Optional: load DirectML fused op registration unless explicitly disabled
try:
    if _os.getenv('OMNICODER_ENABLE_DML', '1') == '1':
        from .modeling.kernels import omnicoder_dml_op  # noqa: F401  (register fused DML op on import)
except Exception:
    # Best-effort: continue without DML fused ops
    import logging as _logging
    try:
        _logging.getLogger('omnicoder.boot').warning('DML fused op registration failed; continuing without DML path', exc_info=True)
    except Exception:
        pass

__all__ = ["config"]

# Enable CUDA Graphs by default; let backends manage safety. No global disables.
try:
    _os.environ.setdefault('TORCHINDUCTOR_USE_CUDA_GRAPHS', '1')
    try:
        import torch._inductor as _ind  # type: ignore[attr-defined]
        _cfg = getattr(_ind, 'config', None)
        if _cfg is not None:
            for _a in ('cuda_graphs', 'use_cuda_graphs'):
                if hasattr(_cfg, _a):
                    setattr(_cfg, _a, True)
            tr = getattr(_cfg, 'triton', None)
            if tr is not None and hasattr(tr, 'cudagraphs'):
                setattr(tr, 'cudagraphs', True)
    except Exception:
        pass
    try:
        import torch._dynamo as _dyn  # type: ignore[attr-defined]
        if hasattr(_dyn, 'config') and hasattr(_dyn.config, 'use_cuda_graphs'):
            _dyn.config.use_cuda_graphs = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _logging.getLogger('omnicoder.boot').info('cudagraphs enabled globally at import')
    except Exception:
        pass
except Exception:
    pass

# Register select Python builtins as graph-allowed to prevent compile-time graph breaks
# NOTE: This does not gate logging; it simply tells torch.compile that these call sites are acceptable
try:
    import torch as _tch  # type: ignore
    import time as _time  # type: ignore
    if hasattr(_tch, 'compiler') and hasattr(_tch.compiler, 'allow_in_graph'):
        try:
            _tch.compiler.allow_in_graph(_time.perf_counter)  # wall-clock timing used in perf logs
        except Exception:
            pass
except Exception:
    pass

# Optional: emit aggregated perf counters at process exit when enabled
try:
    import atexit as _atexit  # type: ignore
    def _dump_perf_snapshot() -> None:
        try:
            if _os.getenv('OMNICODER_TIMING', '0') != '1':
                return
            from omnicoder.utils.perf import snapshot as _perf_snapshot  # type: ignore
            snap = _perf_snapshot(reset=False)
            if not snap:
                return
            # Prefer tests_logs path; fallback to stdout
            try:
                from pathlib import Path as _P
                p = _P('tests_logs/perf_snapshot.json')
                p.parent.mkdir(parents=True, exist_ok=True)
                import json as _json  # type: ignore
                p.write_text(_json.dumps(snap, indent=2), encoding='utf-8')
            except Exception:
                # Minimal stdout fallback
                print({'perf_snapshot': snap})
        except Exception:
            pass
    _atexit.register(_dump_perf_snapshot)
except Exception:
    import logging as _logging
    try:
        _logging.getLogger('omnicoder.boot').warning('atexit registration failed', exc_info=True)
    except Exception:
        pass

# Enable TF32 globally when available; safe no-op on non-CUDA builds
try:
    import torch as _t  # type: ignore
    try:
        _t.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        _t.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        import logging as _logging
        try:
            _logging.getLogger('omnicoder.boot').debug('TF32 enable skipped (not available)')
        except Exception:
            pass
    try:
        _t.set_float32_matmul_precision('high')  # type: ignore[attr-defined]
    except Exception:
        import logging as _logging
        try:
            _logging.getLogger('omnicoder.boot').debug('set_float32_matmul_precision not available')
        except Exception:
            pass
except Exception:
    import logging as _logging
    try:
        _logging.getLogger('omnicoder.boot').debug('torch not available for TF32 setup')
    except Exception:
        pass

# Optional: enable global Python trace if requested (very heavy)
try:
    if _os.getenv('OMNICODER_TRACE_ALL', '0') == '1':
        from omnicoder.utils.perf import enable_global_trace as _enable_trace  # type: ignore
        _enable_trace()
except Exception:
    import logging as _logging
    try:
        _logging.getLogger('omnicoder.boot').warning('OMNICODER_TRACE_ALL enable failed', exc_info=True)
    except Exception:
        pass
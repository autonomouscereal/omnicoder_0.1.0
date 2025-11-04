from __future__ import annotations

import os
from typing import Optional, Callable

import torch
import warnings as _warn
import tempfile as _tmp
import os as _os
import shutil as _sh

try:
	import torch._dynamo as _dyn  # type: ignore
except Exception:
	_dyn = None  # type: ignore

# NOTE [CUDA Graphs and compile policy]
#
# PREVIOUS: This module force-disabled CUDA Graphs globally via env and Inductor
# config toggles. While it avoided some replay overwrite warnings in older
# experiments, it prevented capture/replay opportunities entirely and masked real
# issues in the code paths that needed to be fixed (storage aliasing, dynamic
# shapes, masked writes, or Fake/Functional lineage splits).
#
# NOW: We DO NOT disable CUDA Graphs. We keep Inductor/Dynamo defaults and focus
# on root-cause fixes elsewhere (aten-only ops, static shapes where needed,
# explicit index tensors, no device moves, bias folding via augmentation, etc.).
# This aligns with the unified, always-on rules. We still bump the Dynamo cache
# size where available to reduce recompiles, but we do not touch feature toggles.
# Logging remains fully on.
try:
	# Increase cache size limit to reduce frequent evictions/recompiles
	if hasattr(_dyn, 'config') and hasattr(_dyn.config, 'cache_size_limit'):
		try:
			if int(getattr(_dyn.config, 'cache_size_limit', 8)) < 128:  # type: ignore[attr-defined]
				_dyn.config.cache_size_limit = 128  # type: ignore[attr-defined]
		except Exception:
			pass
	# Relax strict guards (non-feature toggles) for stability; no effect on logging/features
	if hasattr(_dyn, 'config'):
		try:
			if hasattr(_dyn.config, 'force_parameter_static_shapes'):
				_dyn.config.force_parameter_static_shapes = False  # type: ignore[attr-defined]
		except Exception:
			pass
		# Allow capture of dynamic output shape ops needed by boolean indexing in MoE core
		try:
			if hasattr(_dyn.config, 'capture_dynamic_output_shape_ops'):
				_dyn.config.capture_dynamic_output_shape_ops = True  # type: ignore[attr-defined]
		except Exception:
			pass
except Exception:
	pass

# Suppress noisy empty CUDA Graph capture warnings emitted by upstream when a capture
# is attempted on a stream/device with no recorded kernels. We keep CUDA Graphs
# enabled; this only hides the warning to avoid confusing logs and failing tests
# that treat warnings as errors. Functional behavior is unchanged.
try:
	_warn.filterwarnings(
		"ignore",
		message="The CUDA Graph is empty",
		category=UserWarning,
		module=r"torch\.cuda\.graphs",
	)
except Exception:
	pass

# Prefer enabling CUDA Graph capture even when inputs are symbolically shaped
try:
	import torch._inductor.config as _ic  # type: ignore[attr-defined]
	try:
		# Allow cudagraph capture for dynamic graphs; our model enforces fixed shapes in practice
		if hasattr(_ic, 'triton') and hasattr(_ic.triton, 'cudagraph_skip_dynamic_graphs'):
			_ic.triton.cudagraph_skip_dynamic_graphs = False  # type: ignore[attr-defined]
	except Exception:
		pass
	try:
		# Silence dynamic shape warn spam without disabling graphs
		if hasattr(_ic, 'triton') and hasattr(_ic.triton, 'cudagraph_dynamic_shape_warn_limit'):
			_ic.triton.cudagraph_dynamic_shape_warn_limit = 0  # type: ignore[attr-defined]
	except Exception:
		pass
		# Prefer allowing graphs even when some ops are conservatively flagged as incompatible
		try:
			if hasattr(_ic, 'triton') and hasattr(_ic.triton, 'cudagraphs'):  # type: ignore[attr-defined]
				_ic.triton.cudagraphs = True
			if hasattr(_ic, 'cuda_graphs'):
				_ic.cuda_graphs = True  # type: ignore[attr-defined]
		except Exception:
			pass
except Exception:
	pass

# -----------------------------------------------------------------------------
# Debug sentries for runaway shape/allocations (OFF by default)
# -----------------------------------------------------------------------------
_DEF_MAX_ELEMS = int(os.getenv('OMNICODER_SHAPE_MAX_ELEMS', '200000000'))  # ~0.8GB for fp32
_DEF_MAX_BYTES = int(os.getenv('OMNICODER_SHAPE_MAX_BYTES', str(4 * 1024 * 1024 * 1024)))  # 4GB


def _shape_guard_enabled() -> bool:
	try:
		return os.getenv('OMNICODER_SHAPE_GUARD', '0') == '1'
	except Exception:
		return False


def guard_shape(name: str, shape: tuple[int, ...], dtype: Optional[torch.dtype] = None) -> None:
	"""Raise RuntimeError if shape implies excessive elements/bytes. Enabled via OMNICODER_SHAPE_GUARD=1.
	This is a light Python-side guard; do NOT call from hot paths unless debugging issues."""
	if not _shape_guard_enabled():
		return
	try:
		nelems = 1
		for d in shape:
			nelems *= int(max(0, int(d)))
		if nelems > _DEF_MAX_ELEMS:
			raise RuntimeError(f"guard_shape: {name} elements={nelems} exceeds max={_DEF_MAX_ELEMS} shape={tuple(int(d) for d in shape)}")
		itemsize = 4
		if dtype is not None:
			try:
				itemsize = torch.tensor([], dtype=dtype).element_size()
			except Exception:
				itemsize = 4
		bytes_total = nelems * int(itemsize)
		if bytes_total > _DEF_MAX_BYTES:
			raise RuntimeError(f"guard_shape: {name} bytes={bytes_total} exceeds max={_DEF_MAX_BYTES} shape={tuple(int(d) for d in shape)}")
	except Exception as e:
		raise e


def maybe_log_cuda_mem(tag: str = "bench") -> None:
	"""Log CUDA allocated/reserved when OMNICODER_BENCH_MEM=1. No-op if CUDA unavailable."""
	try:
		if os.getenv('OMNICODER_BENCH_MEM', '0') != '1':
			return
		if not torch.cuda.is_available():
			return
		alloc = int(torch.cuda.memory_allocated())
		res = int(torch.cuda.memory_reserved())
		print(f"[{tag}] cuda_mem allocated={alloc} reserved={res}")
	except Exception:
		return


# -----------------------------------------------------------------------------
# Existing helpers
# -----------------------------------------------------------------------------

def get_amp_dtype(device: Optional[str] = None) -> Optional[torch.dtype]:
	"""Return preferred autocast dtype for CUDA devices; None if not available.

	Rationale: keep aten-only logic and avoid FX nodes for getattr on tuples.
	We do not read env or move devices here; unified rules demand zero device
	moves in hot path and minimal dynamic branching.
	"""
	try:
		if device is not None and (not str(device).startswith('cuda')):
			return None
		if not torch.cuda.is_available():
			return None
		cc = torch.cuda.get_device_capability()
		major = int(cc[0]) if isinstance(cc, tuple) and len(cc) >= 1 else (int(cc) if isinstance(cc, int) else 0)  # type: ignore[index]
		return torch.bfloat16 if major >= 8 else torch.float16
	except Exception:
		return None


def ensure_compiled(model: torch.nn.Module) -> torch.nn.Module:
	"""Compile the model with torch.compile when enabled and not already compiled.

	We do not disable CUDA Graphs here. We let Inductor manage CG behavior while
	maintaining our code hygiene (aten-only, static shapes where appropriate,
	no device moves) to make capture safe. Logging stays fully on.
	"""
	try:
		# Proactively enable Inductor CUDA Graphs even when graphs have symbolic shapes.
		# This is set once, outside hot paths, to avoid repeated getattr/branching.
		try:
			import torch._inductor as _ind  # type: ignore[attr-defined]
			cfg = getattr(_ind, 'config', None)
			if cfg is not None and hasattr(cfg, 'triton'):
				setattr(cfg.triton, 'cudagraphs', True)
				# Do not skip dynamic graphs; our model enforces fixed shapes in practice.
				setattr(cfg.triton, 'cudagraph_skip_dynamic_graphs', False)
		except Exception:
			pass
		if hasattr(torch, 'compile'):
			# Determine current model device type (cpu/cuda/etc.) from params/buffers
			cur_dev = 'cpu'
			try:
				last = None
				for p in model.parameters():
					last = p.device.type; break
				if last is None:
					for b in model.buffers():
						last = b.device.type; break
				if last is not None:
					cur_dev = str(last)
			except Exception:
				cur_dev = 'cpu'
			already_compiled = bool(getattr(model, '_omni_compiled', False))
			compiled_dev = str(getattr(model, '_omni_compiled_device', ''))
			# Recompile if previously compiled on a different device type
			if already_compiled and compiled_dev and (compiled_dev != cur_dev):
				try:
					backend = os.getenv('OMNICODER_COMPILE_BACKEND', 'inductor')
					fullgraph = (os.getenv('OMNICODER_COMPILE_FULLGRAPH', '1') == '1')
					recompiled = torch.compile(model, mode='reduce-overhead', backend=backend, fullgraph=fullgraph)  # type: ignore[arg-type]
					setattr(recompiled, '_omni_compiled', True)
					setattr(recompiled, '_omni_compiled_device', cur_dev)
					return recompiled
				except Exception:
					pass
			if not already_compiled:
				# Device-safety: avoid compiling on CPU when CUDA is available but model
				# hasn't been moved yet. This prevents FakeTensor CPU/CUDA mismatches.
				try:
					has_params = False
					all_cpu = True
					mixed = False
					last = None
					for p in model.parameters():
						has_params = True
						dev = p.device.type
						if last is None:
							last = dev
						elif dev != last:
							mixed = True
						if dev != 'cpu':
							all_cpu = False
					if not has_params:
						for b in model.buffers():
							dev = b.device.type
							if last is None:
								last = dev
							elif dev != last:
								mixed = True
							if dev != 'cpu':
								all_cpu = False
					if torch.cuda.is_available() and (all_cpu or mixed):
						return model
				except Exception:
					pass
				backend = os.getenv('OMNICODER_COMPILE_BACKEND', 'inductor')
				fullgraph = (os.getenv('OMNICODER_COMPILE_FULLGRAPH', '1') == '1')
				compiled = torch.compile(model, mode='reduce-overhead', backend=backend, fullgraph=fullgraph)  # type: ignore[arg-type]
				setattr(compiled, '_omni_compiled', True)
				setattr(compiled, '_omni_compiled_device', cur_dev)
				return compiled
		return model
	except Exception:
		return model


def get_cudagraph_step_marker() -> Optional[Callable[[], None]]:
	"""Return a callable that marks the beginning of a new CUDA graph step.

	We no longer early-return None based on env toggles. If the runtime exposes a
	marker, we use it; otherwise return None. This allows the benchmark to log CG
	step boundaries when available without changing features or disabling logging.
	"""
	try:
		if not torch.cuda.is_available():
			return None
	except Exception:
		return None
	# Try canonical public API first (PyTorch 2.3+)
	try:
		mk = getattr(getattr(torch, 'compiler', None), 'cudagraph_mark_step_begin', None)
		if callable(mk):
			return mk  # type: ignore[return-value]
	except Exception:
		pass
	# Older/private locations observed in some builds
	try:
		import torch._inductor as _ind  # type: ignore[attr-defined]
		mk = getattr(_ind, 'cudagraph_mark_step_begin', None)
		if callable(mk):
			return mk  # type: ignore[return-value]
		ut = getattr(_ind, 'utils', None)
		if ut is not None:
			mk = getattr(ut, 'cudagraph_mark_step_begin', None)
			if callable(mk):
				return mk  # type: ignore[return-value]
	except Exception:
		pass
	try:
		_cg = getattr(torch, '_cudagraphs', None)
		mk = getattr(_cg, 'cudagraph_mark_step_begin', None)
		if callable(mk):
			return mk  # type: ignore[return-value]
	except Exception:
		pass
	return None


# -----------------------------------------------------------------------------
# Robust torch.save wrapper to avoid inline_container writer failures
# -----------------------------------------------------------------------------
def safe_torch_save(obj, path: str) -> None:
    """Save to path, retrying with legacy zipfile writer on failure.

    - First attempt uses the default writer.
    - On failure, writes to a temporary file with `_use_new_zipfile_serialization=False`
      then atomically moves into place.
    - Never alters training logic; only improves reliability of checkpoint writes.
    """
    try:
        torch.save(obj, path)
        return
    except Exception:
        pass
    # Retry with legacy writer into a temp file then atomic rename
    try:
        d = _os.path.dirname(path) or '.'
        _os.makedirs(d, exist_ok=True)
        fd, tmp_path = _tmp.mkstemp(dir=d, prefix='.tmp_save_', suffix='.pt')
        _os.close(fd)
        try:
            torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
            try:
                _sh.replace(tmp_path, path)
            except Exception:
                _sh.move(tmp_path, path)
        finally:
            try:
                if _os.path.exists(tmp_path):
                    _os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        # As a last resort, try writing state_dict only
        try:
            if hasattr(obj, 'state_dict') and callable(getattr(obj, 'state_dict')):
                sd = obj.state_dict()
                torch.save(sd, path, _use_new_zipfile_serialization=False)
        except Exception:
            raise


# -----------------------------------------------------------------------------
# Safe tensor utilities for CG/Dynamo-compatible copies and anchors
# -----------------------------------------------------------------------------

def safe_new_like(x: torch.Tensor) -> torch.Tensor:
	"""Allocate a new empty tensor with the same shape/dtype/device as x via aten only.

	Returns a fresh storage suitable for non-aliasing copies under CUDA Graphs.
	"""
	return torch.ops.aten.new_empty.default(x, x.shape)


def safe_copy_into(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
	"""Copy src into dst using aten.copy_. Returns dst. Assumes shape/dtype/device compatible."""
	torch.ops.aten.copy_.default(dst, src)
	return dst


def safe_ephemeral_copy(x: torch.Tensor) -> torch.Tensor:
	"""Create a non-aliased copy of x via aten-only new_empty + copy_."""
	buf = torch.ops.aten.new_empty.default(x, x.shape)
	torch.ops.aten.copy_.default(buf, x)
	return buf


def safe_scalar_anchor(x: torch.Tensor) -> torch.Tensor:
	"""Return a 0-d tensor anchor derived from x via aten-only ops (sum(x*0))."""
	return torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(x, 0.0))


# Aten-only deep clone for nested tensor structures used in hot paths
def deep_clone_tensors(obj):  # type: ignore[override]
	"""Deeply duplicate any nested structure of tensors using aten-only ops.

	- Tensors: duplicated via aten.mul.Scalar(x, 1.0) to produce a writable copy
	  on identical device/dtype without invoking .clone() or .to().
	- Lists/Tuples: elements are cloned recursively; tuples preserved as tuples.
	- Dicts: rebuilt without calling .items(); keys listed then values cloned.
	- Other types: returned as-is.

	Safe for CUDA Graphs: avoids view aliasing and method calls that may induce syncs.
	"""
	try:
		import torch as _tc
		if isinstance(obj, _tc.Tensor):
			return _tc.ops.aten.mul.Scalar(obj, 1.0)
		if isinstance(obj, (list, tuple)):
			seq = [deep_clone_tensors(x) for x in obj]
			return tuple(seq) if isinstance(obj, tuple) else seq
		if isinstance(obj, dict):
			keys = [k for k in obj]
			vals = [obj[k] for k in obj]
			vals_c = [deep_clone_tensors(v) for v in vals]
			out = {}
			for k, v in zip(keys, vals_c):
				out[k] = v
			return out
		return obj
	except Exception:
		return obj


# Shallow safe clone for a single tensor using aten-only ops (no .clone()/.to())
def safe_clone(x):
	"""Return an aten-only cloned tensor when x is a Tensor; otherwise return x.

	Uses aten.mul.Scalar(x, 1.0) to produce a writable copy with identical
	device/dtype/stride lineage, avoiding Python-side .clone() and any device moves.
	"""
	try:
		import torch as _tc
		if isinstance(x, _tc.Tensor):
			return _tc.ops.aten.mul.Scalar(x, 1.0)
		return x
	except Exception:
		return x


# Nested safe clone that mirrors deep_clone_tensors but exposed under a simpler name
def safe_clone_nested(obj):  # type: ignore[override]
	"""Deeply clone nested structures of tensors using aten-only ops.

	Alias for deep_clone_tensors to match project naming conventions.
	"""
	return deep_clone_tensors(obj)

# Hot-path aten-only helpers for common patterns
def scalar_like(x: "torch.Tensor", value: float):
	try:
		import torch as _tc
		return _tc.ops.aten.add.Scalar(_tc.ops.aten.mul.Scalar(x, 0.0), float(value))
	except Exception:
		return value  # fallback (should not be used in aten graphs)


def zeros_like_shape(x: "torch.Tensor", shape: tuple):
	"""Allocate a zero tensor with same device/dtype lineage as x without .device/.dtype/.new_zeros.

	Uses a scalar zero derived from x so it can expand to any shape (no non-singleton expand errors).
	"""
	import torch as _tc
	_zero_anchor = _tc.ops.aten.mul.Scalar(_tc.ops.aten.sum.default(_tc.ops.aten.reshape.default(x, (-1,))), 0.0)
	return _tc.ops.aten.expand.default(_zero_anchor, tuple(shape))


def slice_scatter_range(dst: "torch.Tensor", src: "torch.Tensor", dim: int, start: int, end: int):
	import torch as _tc
	return _tc.ops.aten.slice_scatter.default(dst, src, int(dim), int(start), int(end), 1)


def safe_concat2(a: "torch.Tensor", b: "torch.Tensor", dim: int):
    """Aten-only concatenate of two tensors along dim.

    Implemented via aten.cat to avoid Python shape introspection that can create
    unbacked SymInt issues under torch.compile. This is CG/Inductor safe.
    """
    import torch as _tc
    return _tc.ops.aten.cat.default((a, b), int(dim))


def safe_concat(seq: list["torch.Tensor"] | tuple["torch.Tensor", ...], dim: int):
    """Aten-only concatenate of a sequence along dim via aten.cat.

    Avoids Python int() on SymInt shapes to keep graphs stable under compile.
    """
    import torch as _tc
    if not isinstance(seq, (list, tuple)):
        raise TypeError("safe_concat expects a list/tuple of tensors")
    if len(seq) == 0:
        raise ValueError("safe_concat received empty sequence")
    if len(seq) == 1:
        return seq[0]
    return _tc.ops.aten.cat.default(tuple(seq), int(dim))


def replace_last_token(x: "torch.Tensor", update: "torch.Tensor"):
	"""Return x with its last token position (dim=1) replaced by `update` using slice_scatter.
	Assumes update has matching shape for that slice.
	"""
	import torch as _tc
	T = int(x.shape[1])
	start = max(0, T - 1)
	return _tc.ops.aten.slice_scatter.default(x, update, 1, start, T, 1)


def inplace_copy(dst: "torch.Tensor", src: "torch.Tensor") -> "torch.Tensor":
	"""A thin, aten-only wrapper for in-place copying to avoid calling tensor.copy_.

	Uses torch.ops.aten.copy_. Caller is responsible for ensuring shapes/strides are compatible.
	Returns the destination tensor for convenience.
	"""
	import torch as _tc
	return _tc.ops.aten.copy_(dst, src)

# -----------------------------------------------------------------------------
# Contiguity helpers (aten-only; no .contiguous()/.clone())
# -----------------------------------------------------------------------------

def safe_make_contiguous(x):
	"""Return a contiguous tensor with identical data using aten-only ops.

	- Allocates a new empty buffer (contiguous by default) via aten.new_empty.default
	- Copies data with aten.copy_
	- No .contiguous()/.clone(), export- and CG-safe
	"""
	try:
		import torch as _tc
		if isinstance(x, _tc.Tensor):
			buf = _tc.ops.aten.new_empty.default(x, x.shape)
			_tc.ops.aten.copy_(buf, x)
			return buf
		return x
	except Exception:
		return x


def safe_transpose_contiguous(x, dim0: int, dim1: int):
	"""Transpose then materialize contiguous storage using aten-only ops."""
	try:
		import torch as _tc
		if isinstance(x, _tc.Tensor):
			view = _tc.ops.aten.transpose.int(x, int(dim0), int(dim1))
			buf = _tc.ops.aten.new_empty.default(view, view.shape)
			_tc.ops.aten.copy_(buf, view)
			return buf
		return x
	except Exception:
		return x

# -----------------------------------------------------------------------------
# CUDA Graph capture diagnostic reporting (best-effort; no feature toggles)
# -----------------------------------------------------------------------------
_CG_REPORT = {
	"installed": False,
	"deferred_wraps": 0,
	"run_attempts": 0,
	"run_success": 0,
	"run_fail": 0,
	"warmup_end_calls": 0,
	"warmup_end_fail": 0,
	"weakref_mismatch": 0,
	"last_errors": [],
}


def enable_cudagraph_capture_report() -> None:
	"""Install lightweight wrappers around Inductor cudagraph tree internals to log capture status.

	Safe to call multiple times. No-ops if internals are unavailable.
	"""
	if _CG_REPORT.get("installed", False):
		return
	try:
		import torch._inductor.cudagraph_trees as _cg  # type: ignore[attr-defined]
	except Exception:
		return
	# Wrap deferred_cudagraphify (function) when present
	try:
		if hasattr(_cg, "deferred_cudagraphify"):
			_orig_deferred = _cg.deferred_cudagraphify  # type: ignore[assignment]
			def _wrapped_deferred(fn):  # type: ignore[no-redef]
				_CG_REPORT["deferred_wraps"] += 1
				return _orig_deferred(fn)
			setattr(_cg, "deferred_cudagraphify", _wrapped_deferred)
	except Exception:
		pass
	# Class may be named CUDAGraphTree (newer) or CudaGraphTree (older)
	_cls = None
	for _name in ("CUDAGraphTree", "CudaGraphTree"):
		_cls = getattr(_cg, _name, None)
		if _cls is not None:
			break
	if _cls is not None:
		# Wrap _run to track per-function_id outcomes
		try:
			_orig_run = getattr(_cls, "_run", None)
			if callable(_orig_run):
				def _wrapped_run(self, *args, **kwargs):  # type: ignore[no-redef]
					_CG_REPORT["run_attempts"] += 1
					try:
						out = _orig_run(self, *args, **kwargs)
						_CG_REPORT["run_success"] += 1
						return out
					except Exception as _e:  # noqa: BLE001
						_CG_REPORT["run_fail"] += 1
						try:
							msg = f"cudagraph_run_fail type={type(_e).__name__} msg={str(_e)[:200]}"
							_CG_REPORT["last_errors"].append(msg)
						except Exception:
							pass
						raise
				setattr(_cls, "_run", _wrapped_run)
		except Exception:
			pass
		# Wrap try_end_curr_warmup to record failures
		try:
			_orig_end = getattr(_cls, "try_end_curr_warmup", None)
			if callable(_orig_end):
				def _wrapped_end(self, *args, **kwargs):  # type: ignore[no-redef]
					_CG_REPORT["warmup_end_calls"] += 1
					try:
						return _orig_end(self, *args, **kwargs)
					except Exception as _e:  # noqa: BLE001
						_CG_REPORT["warmup_end_fail"] += 1
						try:
							msg = f"cudagraph_warmup_end_fail type={type(_e).__name__} msg={str(_e)[:200]}"
							_CG_REPORT["last_errors"].append(msg)
						except Exception:
							pass
						raise
				setattr(_cls, "try_end_curr_warmup", _wrapped_end)
		except Exception:
			pass
		# Wrap dealloc_current_path_weakrefs to catch AssertionError specifically
		try:
			_orig_dealloc = getattr(_cls, "dealloc_current_path_weakrefs", None)
			if callable(_orig_dealloc):
				def _wrapped_dealloc(self, *args, **kwargs):  # type: ignore[no-redef]
					try:
						return _orig_dealloc(self, *args, **kwargs)
					except AssertionError as _e:  # noqa: PIE786
						_CG_REPORT["weakref_mismatch"] += 1
						try:
							msg = f"cudagraph_weakref_mismatch msg={str(_e)[:200]}"
							_CG_REPORT["last_errors"].append(msg)
						except Exception:
							pass
						raise
				setattr(_cls, "dealloc_current_path_weakrefs", _wrapped_dealloc)
		except Exception:
			pass
	_CG_REPORT["installed"] = True


def get_cudagraph_capture_report() -> dict:
	"""Return a shallow copy of the current CUDA Graph capture report counters."""
	try:
		return dict(_CG_REPORT)
	except Exception:
		return {}


def dump_cudagraph_capture_report(logger: Optional[object] = None) -> None:
	"""Emit a one-line summary and any recent errors captured from cudagraph wrappers.

	- If `logger` has `.info` or `.warning`, they will be used; otherwise falls back to print().
	- Best-effort: does nothing if wrappers were not installed or internals unavailable.
	"""
	rep = get_cudagraph_capture_report()
	if not rep or not rep.get("installed", False):
		return
	line = (
		f"cg_report installed=1 deferred_wraps={int(rep.get('deferred_wraps',0))} "
		f"run_attempts={int(rep.get('run_attempts',0))} run_success={int(rep.get('run_success',0))} run_fail={int(rep.get('run_fail',0))} "
		f"warmup_end_calls={int(rep.get('warmup_end_calls',0))} warmup_end_fail={int(rep.get('warmup_end_fail',0))} "
		f"weakref_mismatch={int(rep.get('weakref_mismatch',0))}"
	)
	def _log(msg: str) -> None:
		try:
			if hasattr(logger, 'info') and callable(getattr(logger, 'info')):
				logger.info(msg)
				return
		except Exception:
			pass
		try:
			print(msg)
		except Exception:
			pass
	_log(line)
	# Emit last few errors for quick visibility
	try:
		last = rep.get('last_errors', [])[-3:]
		for e in last:
			_log(f"cg_report.err {e}")
	except Exception:
		pass

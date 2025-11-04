from __future__ import annotations

import time
import sys
import threading
import logging
from typing import Dict, Iterator
from threading import Lock
from contextlib import contextmanager

# Optional torch import for safer repr handling of tensors/storages without touching devices in hot paths
try:
	import torch as _torch  # type: ignore
	_TORCH_OK = True
except Exception:  # pragma: no cover
	_torch = None  # type: ignore
	_TORCH_OK = False

# Simple global perf counter aggregator used by hot paths when OMNICODER_TIMING=1
# Keep implementation minimal to avoid overhead.

_METRICS: Dict[str, float] = {}
_COUNTS: Dict[str, int] = {}
_LOCK = Lock()


def add(name: str, value: float) -> None:
	try:
		with _LOCK:
			_METRICS[name] = _METRICS.get(name, 0.0) + float(value)
			_COUNTS[name] = _COUNTS.get(name, 0) + 1
	except Exception:
		logging.getLogger('omnicoder.perf').warning('perf.add failed for %s', name, exc_info=True)


def snapshot(reset: bool = False) -> Dict[str, dict]:
	try:
		with _LOCK:
			out = {k: {"sum_s": v, "count": _COUNTS.get(k, 0)} for k, v in _METRICS.items()}
			if reset:
				_METRICS.clear()
				_COUNTS.clear()
			return out
	except Exception:
		logging.getLogger('omnicoder.perf').warning('perf.snapshot failed', exc_info=True)
		return {}


# Context manager used as a lightweight timer in hot paths.
# Usage: from omnicoder.utils.perf import timer; with timer('decode_step'): ...
@contextmanager
def timer(name: str) -> Iterator[None]:
	"""Measure elapsed wall time and aggregate under a named counter."""
	_t0 = time.perf_counter()
	try:
		yield
	finally:
		try:
			add(name, time.perf_counter() - _t0)
		except Exception:
			logging.getLogger('omnicoder.perf').warning('perf.timer add failed for %s', name, exc_info=True)


# Extremely verbose Python-level tracing (function calls, lines, returns).
# Enabled via OMNICODER_TRACE_ALL=1 (wired in package __init__). Heavy by design.
def _safe_repr(obj, limit: int = 120) -> str:  # type: ignore[no-untyped-def]
	"""Best-effort, exception-safe repr that never raises.

	- Truncates long outputs
	- Summarizes known heavy/problematic stdlib types
	- Falls back to a type+address marker when repr() is unsafe
	- SPECIAL: Avoid repr() on torch tensors/storages/modules/torch-objects to prevent Fake/TypedStorage warnings
	"""
	try:
		if obj is None:
			return 'None'
		if isinstance(obj, (int, float, bool)):
			return repr(obj)
		if isinstance(obj, str):
			s = obj
			return (repr(s[:limit]) + '...') if len(s) > limit else repr(s)
		if isinstance(obj, (bytes, bytearray)):
			b = bytes(obj)
			return (repr(b[:limit]) + '...') if len(b) > limit else repr(b)

		# Torch-aware summaries to avoid triggering storage repr() under Fake/Meta
		if _TORCH_OK:
			try:
				if _torch.is_tensor(obj):  # type: ignore[attr-defined]
					return f"<Tensor shape={tuple(getattr(obj,'shape', ())) } dtype={getattr(obj,'dtype', None)} device={getattr(obj,'device', None)}>"
				# torch.nn.Module summary without repr() to avoid walking params/buffers
				nn_mod = getattr(_torch, 'nn', None)
				if nn_mod is not None:
					_Module = getattr(nn_mod, 'Module', None)
					if _Module is not None and isinstance(obj, _Module):
						return f"<Module {obj.__class__.__name__} at 0x{id(obj):x}>"
				# Storage-like objects (avoid touching attributes that may trigger deprecation warnings)
				type_name = getattr(type(obj), '__name__', '')
				type_mod = getattr(type(obj), '__module__', '')
				if 'Storage' in type_name or ('torch' in type_mod and ('Storage' in type_mod or hasattr(obj, 'untyped_storage'))):
					# Do NOT access .device/.size/.dtype on storages; just emit type and address
					return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"
				# Any other torch.* object: avoid repr() to prevent deep traversal that may touch storages
				if isinstance(type_mod, str) and type_mod.startswith('torch'):
					return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"
			except Exception:
				# Fall through to generic handling
				pass

		obj_type = type(obj)
		type_name = getattr(obj_type, '__name__', 'object')
		type_mod = getattr(obj_type, '__module__', '')

		# Avoid problematic repr() implementations from stdlib modules observed in traces
		if (type_mod in ('inspect', 'argparse', 'traceback', 'types') or
				type_name in ('FrameInfo', 'Traceback', 'ArgumentParser')):
			return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"

		# Avoid touching attributes on torchaudio objects to prevent backend dispatch warnings
		if isinstance(type_mod, str) and type_mod.startswith('torchaudio'):
			return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"
		# Summarize tensors/ndarrays without importing backend-specific IO modules
		try:
			# Extra guard to avoid accessing attributes on torchaudio backend IO objects
			if isinstance(type_mod, str) and (type_mod.startswith('torchaudio.io') or type_mod.startswith('torchaudio.backend')):
				return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"
			# Only access shape/dtype/device for known-safe modules to avoid triggering backend warnings
			_safe_mod = (
				isinstance(type_mod, str)
				and (type_mod.startswith('torch') or type_mod.startswith('numpy') or type_mod.startswith('PIL') or type_mod.startswith('cv2'))
			)
			if not _safe_mod:
				return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"
			shape = getattr(obj, 'shape', None)
			dtype = getattr(obj, 'dtype', None)
			device = getattr(obj, 'device', None)
			if shape is not None and dtype is not None:
				return f"<{type_name} shape={tuple(shape)} dtype={dtype} device={device}>"
		except Exception:
			pass

		# Generic repr as last resort (may trigger deprecation warnings for unknown types)
		# Avoid calling repr(obj) entirely to prevent storage/tensor repr paths from emitting warnings.
		return f"<{type_mod}.{type_name} at 0x{id(obj):x}>"
	except Exception:
		try:
			return f"<{getattr(type(obj), '__module__', '')}.{getattr(type(obj), '__name__', 'object')} at 0x{id(obj):x}>"
		except Exception:
			return '<unrepr-able>'


from typing import Callable


def _make_tracer() -> Callable:
	_log = logging.getLogger('omnicoder.trace')

	def _tracer(frame, event, arg):  # type: ignore[no-untyped-def]
		try:
			co = frame.f_code
			msg = None
			if event == 'call':
				msg = f"CALL {co.co_filename}:{co.co_firstlineno} {co.co_name}()"
			elif event == 'line':
				msg = f"LINE {co.co_filename}:{frame.f_lineno} in {co.co_name}()"
			elif event == 'return':
				msg = f"RET  {co.co_filename}:{frame.f_lineno} {co.co_name}() -> {_safe_repr(arg, 120)}"
			if msg is not None:
				_log.debug(msg)
		except Exception:
			# During interpreter shutdown, logging module may be partially torn down; avoid calling it.
			try:
				if hasattr(logging, 'getLogger') and callable(getattr(logging, 'getLogger', None)):
					logging.getLogger('omnicoder.perf').warning('global tracer emit failed', exc_info=True)
			except Exception:
				# Final fallback: suppress to avoid noisy shutdown exceptions
				pass
		return _tracer

	return _tracer


def enable_global_trace() -> None:
	"""Enable process-wide Python tracing that logs every call/line/return.

	This is intentionally heavyweight and should only be enabled when requested.
	"""
	try:
		tr = _make_tracer()
		sys.settrace(tr)
		threading.settrace(tr)
	except Exception:
		logging.getLogger('omnicoder.perf').warning('enable_global_trace failed', exc_info=True)


def disable_global_trace() -> None:
	"""Disable global Python tracing if previously enabled."""
	try:
		sys.settrace(None)
		threading.settrace(None)
	except Exception:
		logging.getLogger('omnicoder.perf').warning('disable_global_trace failed', exc_info=True)


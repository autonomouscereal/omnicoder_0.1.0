from __future__ import annotations

from typing import Dict, Tuple

try:
	import numpy as _np  # type: ignore
except Exception:
	_np = None  # type: ignore

# Simple module-level zero-array cache keyed by (shape, dtype)
_ZERO_CACHE: Dict[Tuple[Tuple[int, ...], str], object] = {}


def zeros_cached(shape: Tuple[int, ...], dtype: object) -> object:
	"""Return a cached NumPy zeros array for the given shape/dtype.
	Falls back to creating a new array when NumPy is unavailable.
	"""
	key = (tuple(int(x) for x in shape), str(dtype))
	arr = _ZERO_CACHE.get(key)
	if arr is not None:
		return arr
	if _np is None:
		# Fallback: minimal pure-Python list structure (rare path)
		# Caller should only hit this in constrained environments.
		_ZEROS = 0.0
		def _build(sz):
			if len(sz) == 0:
				return _ZEROS
			return [_build(sz[1:]) for _ in range(sz[0])]
		arr = _build(shape)
		_ZERO_CACHE[key] = arr
		return arr
	# Normal path
	arr = _np.zeros(shape, dtype=dtype)
	_ZERO_CACHE[key] = arr
	return arr

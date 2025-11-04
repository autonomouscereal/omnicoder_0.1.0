from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore


def _to_cpu_detached(obj: Any) -> Any:
	"""Recursively detach and move tensors to CPU inside mappings/lists/tuples.

	Keeps non-tensor values unchanged. This prevents CUDA writer issues during torch.save
	and standardizes checkpoints across devices.
	"""
	try:
		if torch.is_tensor(obj):
			return obj.detach().to('cpu')
		if isinstance(obj, Mapping):
			return {k: _to_cpu_detached(v) for k, v in obj.items()}
		if isinstance(obj, (list, tuple)):
			seq = [_to_cpu_detached(v) for v in obj]
			return type(obj)(seq)  # preserve list/tuple type
		return obj
	except Exception:
		return obj


def ensure_unique_path(p: Path) -> Path:
	"""Return a non-existing path based on p by appending a timestamp if needed."""
	if not p.exists():
		return p
	from datetime import datetime as _dt
	stamp = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
	alt = p.with_name(f"{p.stem}_{stamp}{p.suffix}")
	if not alt.exists():
		return alt
	# Fallback numeric suffix
	i = 1
	while True:
		cand = p.with_name(f"{p.stem}_{stamp}_{i}{p.suffix}")
		if not cand.exists():
			return cand
		i += 1


def save_with_sidecar(
	out_path: str | Path,
	state: Any,
	meta: dict | None = None,
	copy_unified_vocab: bool = True,
) -> Path:
	"""Robust checkpoint save with atomic replace and a JSON sidecar.

	- Writes tensors on CPU via a temp file in the same directory then atomically replaces target
	- Writes <target>.meta.json when meta is provided
	- Optionally copies unified vocab sidecar (OMNICODER_VOCAB_SIDECAR) next to the checkpoint

	Returns the final checkpoint Path.
	"""
	target = Path(out_path)
	parent = target.parent
	parent.mkdir(parents=True, exist_ok=True)
	if target.exists() and target.is_dir():
		# Common footgun: when a directory path is given where a file is expected
		target = target / 'model.pt'
	# Do not overwrite an existing file with potentially better accuracy; write a new unique file
	if target.exists() and target.is_file():
		target = ensure_unique_path(target)
	# Prepare CPU-safe payload
	payload = _to_cpu_detached(state)
	# Atomic write via temp file
	tmp = target.with_suffix(target.suffix + '.tmp')
	_safe_save(payload, str(tmp))
	try:
		tmp.replace(target)
	except Exception:
		tmp.rename(target)
	# Write sidecar metadata if provided
	if meta is not None:
		try:
			(target.with_suffix('.meta.json')).write_text(__import__('json').dumps(meta, indent=2), encoding='utf-8')
		except Exception:
			pass
	# Copy unified vocab sidecar if requested and available
	if copy_unified_vocab:
		try:
			_sidecar_src = os.getenv('OMNICODER_VOCAB_SIDECAR', 'weights/unified_vocab_map.json')
			p = Path(_sidecar_src)
			if p.exists():
				dst = target.with_suffix('.unified_vocab_map.json')
				dst.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
		except Exception:
			pass
	return target


def load_best_or_latest(model: torch.nn.Module, base_out: str | Path) -> Path | None:
	"""Load the best-known or latest checkpoint into model if available.

	Selection order:
	- <stem>_best.pt
	- <base_out>
	- The most recent file matching <stem>_YYYYmmdd_HHMMSS*.pt in the same directory
	Returns the loaded path or None if nothing loaded.
	"""
	base = Path(base_out)
	parent = base.parent
	stem = base.stem
	best = parent / f"{stem}_best{base.suffix or '.pt'}"
	cand: list[Path] = []
	if best.exists():
		cand.append(best)
	if base.exists():
		cand.append(base)
	# Include timestamped variants
	try:
		for p in parent.glob(f"{stem}_*.pt"):
			cand.append(p)
	except Exception:
		pass
	if not cand:
		return None


def maybe_save_best(
	base_out: str | Path,
	model: torch.nn.Module,
	score_name: str,
	score_value: float,
	higher_is_better: bool = False,
	extra_meta: dict | None = None,
) -> Path | None:
	"""Compare a score to the recorded best and save <stem>_best.pt if improved.

	Writes sidecar meta with the metric, mode and score. Returns best path on success.
	"""
	base = Path(base_out)
	parent = base.parent
	best = parent / f"{base.stem}_best{base.suffix or '.pt'}"
	best_meta = best.with_suffix('.meta.json')
	prev = None
	try:
		if best_meta.exists():
			import json as _json
			obj = _json.loads(best_meta.read_text(encoding='utf-8'))
			prev = float(obj.get('best_value')) if ('best_value' in obj) else None
	except Exception:
		prev = None
	improved = False
	if prev is None:
		improved = True
	else:
		improved = (score_value > prev) if higher_is_better else (score_value < prev)
	if not improved:
		return None
	meta = {
		'metric': str(score_name),
		'mode': ('max' if higher_is_better else 'min'),
		'best_value': float(score_value),
	}
	if isinstance(extra_meta, dict):
		meta.update(extra_meta)
	try:
		# Persist best using the same sidecar writer
		save_with_sidecar(best, model.state_dict(), meta=meta)
		return best
	except Exception:
		return None
	# Prefer best, else newest by mtime
	if cand[0] == best:
		chosen = best
	else:
		chosen = max(cand, key=lambda p: p.stat().st_mtime)
	try:
		state = torch.load(str(chosen), map_location='cpu')
		# Allow dicts with nested keys (e.g., {'base': ..., 'head': ...}) by preferring flat state_dict
		if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
			model.load_state_dict(state, strict=False)
		elif isinstance(state, dict) and 'state_dict' in state:
			model.load_state_dict(state['state_dict'], strict=False)
		else:
			# Best-effort: try load_state_dict directly
			model.load_state_dict(state, strict=False)
		return chosen
	except Exception:
		return None

def ensure_unique_path(path: str | os.PathLike) -> Path:
    """Return a filesystem path that does not overwrite an existing file.

    If the target exists, append a timestamp suffix, and if necessary, a
    monotonically increasing version counter. Returns a Path to use for saving.
    Set OMNICODER_OVERWRITE=1 to allow overwriting explicitly.
    """
    p = Path(path)
    if os.getenv('OMNICODER_OVERWRITE', '0') == '1':
        return p
    if not p.exists():
        # Reserve parent directory
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p
    # Build timestamped candidate
    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = p.stem
    suf = p.suffix
    parent = p.parent
    cand = parent / f"{stem}-{ts}{suf}"
    if not cand.exists():
        return cand
    # Version bump if collision occurs within same second
    v = 1
    while True:
        cand_v = parent / f"{stem}-{ts}-v{v}{suf}"
        if not cand_v.exists():
            return cand_v
        v += 1



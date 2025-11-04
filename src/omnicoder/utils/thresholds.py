from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional


def get_accept_threshold(preset_key: str | None, fallback: float = 0.05) -> float:
	"""Load acceptance threshold for a given preset from profiles/acceptance_thresholds.json.
	Returns fallback when not found or on errors. Never reads during decode steps unless caller opts in.
	"""
	try:
		key = (preset_key or '').strip()
		acc_path = Path('profiles') / 'acceptance_thresholds.json'
		if not acc_path.exists():
			return float(fallback)
		data = json.loads(acc_path.read_text(encoding='utf-8'))
		if isinstance(data, dict) and key and key in data and isinstance(data[key], (int, float)):
			return float(data[key])
		if isinstance(data, (int, float)):
			return float(data)
		return float(fallback)
	except Exception:
		return float(fallback)

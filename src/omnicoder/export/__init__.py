from __future__ import annotations

"""
Export sidecars for Ω-Reasoner: constraints, templates, and landmarks indices.

These helpers write simple JSON/flat files next to exported ONNX artifacts so
the lightweight on-device controller can load hints without custom formats.
"""

import json
from typing import Any, Dict, List
from pathlib import Path


def write_constraints_sidecar(root: str, constraints: Dict[str, Any]) -> str:
    p = Path(root) / 'constraints.json'
    p.write_text(json.dumps(constraints, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(p)


def write_templates_sidecar(root: str, templates: List[Dict[str, Any]]) -> str:
    p = Path(root) / 'templates.json'
    p.write_text(json.dumps(templates, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(p)


def write_landmarks_index(root: str, lines: List[str]) -> str:
    p = Path(root) / 'landmarks.idx'
    p.write_text('\n'.join(lines), encoding='utf-8')
    return str(p)


# Export package



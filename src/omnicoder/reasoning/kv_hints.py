from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import os

_KV: Dict[Tuple[int, ...], Any] = {}
_ENABLED = True
try:
    _ENABLED = (os.getenv('OMNICODER_KV_HINTS_ENABLE', '1') == '1')
except Exception:
    _ENABLED = True

def register_kv_hint(key: Tuple[int, ...], kv: Any) -> None:
    if not _ENABLED:
        return
    try:
        _KV[key] = kv
    except Exception:
        pass

def get_kv_hint(key: Tuple[int, ...]) -> Optional[Any]:
    if not _ENABLED:
        return None
    try:
        return _KV.get(key)
    except Exception:
        return None



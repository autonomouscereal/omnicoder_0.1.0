from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _collect_env(prefixes: tuple[str, ...] = ("OMNICODER_", "SFB_", "OMNI_")) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        for k, v in os.environ.items():
            for p in prefixes:
                if k.startswith(p):
                    env[k] = v
                    break
    except Exception:
        pass
    return env


def build_manifest(
    model: Optional[object] = None,
    modality: str = "text",
    preset: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    man: Dict[str, Any] = {
        "version": 1,
        "modality": modality,
        "env": _collect_env(),
    }
    # Tokenizer selection (env-driven, include explicitly for clarity)
    try:
        man["tokenizer_id"] = os.getenv("OMNICODER_HF_TOKENIZER", "")
    except Exception:
        man["tokenizer_id"] = ""
    # Preset hint
    if preset:
        man["preset"] = str(preset)
    # Device
    try:
        dev = None
        if model is not None and hasattr(model, "parameters"):
            try:
                dev = str(next(model.parameters()).device)  # type: ignore[attr-defined]
            except Exception:
                dev = None
        man["device"] = dev or os.getenv("OMNICODER_TRAIN_DEVICE", "cpu")
    except Exception:
        man["device"] = os.getenv("OMNICODER_TRAIN_DEVICE", "cpu")
    # Acceptance thresholds sidecar (if present)
    try:
        p = Path("profiles/acceptance_thresholds.json")
        if p.exists():
            man["acceptance_thresholds_path"] = str(p)
    except Exception:
        pass
    # Attach optional extras
    if isinstance(extra, dict):
        try:
            man.update(extra)
        except Exception:
            pass
    return man


def save_manifest_for_checkpoint(ckpt_path: str | Path, manifest: Dict[str, Any]) -> Optional[str]:
    try:
        p = Path(ckpt_path)
        stem = p.with_suffix("")
        out = str(stem) + ".manifest.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return out
    except Exception:
        return None


def load_and_apply_manifest(manifest_path: str | Path) -> bool:
    """Load a manifest and apply its environment configuration.

    Returns True if applied, False otherwise.
    """
    try:
        mp = Path(manifest_path)
        if not mp.exists():
            return False
        data = json.loads(mp.read_text(encoding="utf-8"))
        # Apply environment keys recorded
        env = data.get("env", {}) or {}
        if isinstance(env, dict):
            for k, v in env.items():
                try:
                    os.environ.setdefault(str(k), str(v))
                except Exception:
                    continue
        # Tokenizer id explicit override if present
        tok = data.get("tokenizer_id", "")
        if isinstance(tok, str) and tok:
            os.environ.setdefault("OMNICODER_HF_TOKENIZER", tok)
        return True
    except Exception:
        return False



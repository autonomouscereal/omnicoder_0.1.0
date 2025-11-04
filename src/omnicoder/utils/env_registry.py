from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class EnvVar:
    name: str
    default: str = ""
    description: str = ""
    type: str = "str"
    allowed: Optional[List[str]] = None
    deprecated: bool = False
    replaces: Optional[List[str]] = None


_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_GETENV_RE = re.compile(r"os\.getenv\(\s*[\"\']([A-Za-z_][A-Za-z0-9_]*)[\"\']\s*,?\s*(?:[\"\']([^\"\']*)[\"\']|([^\)]*))?\)")
_ENV_INDEX_RE = re.compile(r"os\.environ\[\s*[\"\']([A-Za-z_][A-Za-z0-9_]*)[\"\']\s*\]")
_ENV_SETDEFAULT_RE = re.compile(r"os\.environ\.setdefault\(\s*[\"\']([A-Za-z_][A-Za-z0-9_]*)[\"\']\s*,\s*[\"\']([^\"\']*)[\"\']\s*\)")


def load_dotenv_best_effort(paths: Iterable[str] = (".env", ".env.tuned")) -> None:
    for p in paths:
        try:
            _load_one_env_file(p)
        except Exception:
            continue


def _load_one_env_file(env_path: str) -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def parse_env_example(env_example_path: str = "env.example.txt") -> Dict[str, EnvVar]:
    out: Dict[str, EnvVar] = {}
    p = Path(env_example_path)
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8").splitlines():
        m = _ENV_LINE_RE.match(raw)
        if not m:
            continue
        name = m.group(1)
        val = m.group(2).strip()
        val = val.strip().strip('"').strip("'")
        out[name] = EnvVar(name=name, default=val)
    return out


def _iter_source_files(root_dir: str | Path) -> Iterator[Path]:
    root = Path(root_dir)
    for p in root.rglob("*.py"):
        # Skip venvs or generated folders
        parts = {seg.lower() for seg in p.parts}
        if any(x in parts for x in {".venv", "venv", "build", "dist", "__pycache__"}):
            continue
        yield p


def scan_env_usage(root_dir: str | Path = "src") -> Dict[str, List[dict]]:
    usage: Dict[str, List[dict]] = {}
    for file_path in _iter_source_files(root_dir):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # getenv reads
        for m in _GETENV_RE.finditer(text):
            key = m.group(1)
            default = m.group(2) or (m.group(3) or "").strip()
            _append_usage(usage, key, file_path, "getenv", default)
        # environ index reads
        for m in _ENV_INDEX_RE.finditer(text):
            key = m.group(1)
            _append_usage(usage, key, file_path, "environ_index", "")
        # setdefault writes
        for m in _ENV_SETDEFAULT_RE.finditer(text):
            key = m.group(1)
            default = m.group(2)
            _append_usage(usage, key, file_path, "setdefault", default)
    return usage


def _append_usage(usage: Dict[str, List[dict]], key: str, file_path: Path, kind: str, default: str) -> None:
    if not key:
        return
    if key not in usage:
        usage[key] = []
    usage[key].append({
        "file": str(file_path.as_posix()),
        "kind": kind,
        "default": str(default),
    })


def build_registry(env_example_path: str = "env.example.txt", root_dir: str | Path = "src") -> Dict[str, EnvVar]:
    registry: Dict[str, EnvVar] = {}
    # Seed from env.example
    example = parse_env_example(env_example_path)
    registry.update(example)
    # Merge from usage
    usage = scan_env_usage(root_dir)
    for key, sites in usage.items():
        if key not in registry:
            # Infer default if present in any site
            defaults = [s.get("default", "") for s in sites if s.get("default")]
            d = str(defaults[0]) if defaults else ""
            registry[key] = EnvVar(name=key, default=d)
        else:
            # If example default is empty and a usage default exists, adopt it
            if (not registry[key].default) and any(s.get("default") for s in sites):
                defaults = [s.get("default", "") for s in sites if s.get("default")]
                if defaults:
                    registry[key].default = str(defaults[0])
    return registry


def dump_registry_json(path: str, registry: Dict[str, EnvVar]) -> None:
    import json
    data = {k: asdict(v) for k, v in sorted(registry.items())}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def dump_index_markdown(path: str, registry: Dict[str, EnvVar], usage: Dict[str, List[dict]]) -> None:
    lines: List[str] = []
    lines.append("### Environment Variables Index")
    lines.append("")
    for name in sorted(set(list(registry.keys()) + list(usage.keys()))):
        ev = registry.get(name, EnvVar(name=name))
        desc = ev.description or ""
        default = ev.default
        lines.append(f"- {name}: default={default}")
        if desc:
            lines.append(f"  - desc: {desc}")
        sites = usage.get(name, [])
        if sites:
            for s in sites[:20]:
                lines.append(f"  - use: {s.get('kind')} @ {s.get('file')}")
            if len(sites) > 20:
                lines.append(f"  - use: +{len(sites)-20} more")
    text = "\n".join(lines)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sync_env_example(env_example_path: str, registry: Dict[str, EnvVar], usage: Dict[str, List[dict]]) -> Tuple[List[str], List[str]]:
    existing = set(parse_env_example(env_example_path).keys())
    all_keys = sorted(set(list(registry.keys()) + list(usage.keys())))
    missing = [k for k in all_keys if k not in existing]
    deprecated: List[str] = []
    if not missing:
        return missing, deprecated
    p = Path(env_example_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n\n# --- Added by env_audit (auto-synced) ---\n")
        for k in missing:
            ev = registry.get(k, EnvVar(name=k))
            default = ev.default
            f.write(f"{k}={default}\n")
    return missing, deprecated


def allowed_keys_from_registry(registry: Dict[str, EnvVar]) -> List[str]:
    return sorted(list(registry.keys()))



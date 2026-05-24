from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_path(root: str, sha256: str, suffix: str = "") -> Path:
    digest = str(sha256).lower()
    ext = suffix if suffix.startswith(".") or suffix == "" else f".{suffix}"
    return Path(root) / digest[:2] / digest[2:4] / f"{digest}{ext}"

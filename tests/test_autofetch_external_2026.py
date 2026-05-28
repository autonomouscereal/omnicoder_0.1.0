from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from omnicoder.tools import autofetch_external


def test_verify_or_delete_sha256_removes_mismatched_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"bad-payload")
    expected = hashlib.sha256(b"good-payload").hexdigest()
    monkeypatch.delenv("OMNICODER_ALLOW_CHECKSUM_MISMATCH", raising=False)

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        autofetch_external._verify_or_delete_sha256("bad_source", payload, expected)

    assert not payload.exists()


def test_verify_or_delete_sha256_accepts_matching_payload(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"good-payload")
    expected = hashlib.sha256(b"good-payload").hexdigest()

    got = autofetch_external._verify_or_delete_sha256("good_source", payload, expected)

    assert got == expected
    assert payload.exists()

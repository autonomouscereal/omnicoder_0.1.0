from __future__ import annotations

import pytest

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER
from omnicoder.tokenization.text_range_2026 import (
    TEXT_RANGE_MODE_ENV,
    TEXT_TOKEN_HI_ENV,
    effective_text_token_range,
)


class _Tokenizer:
    vocab_size = 270_592


def test_effective_text_token_range_expands_to_tokenizer_vocab_beyond_ledger() -> None:
    ledger_lo, ledger_hi = DEFAULT_LEDGER.as_config_ranges()["text"]

    lo, hi = effective_text_token_range(tokenizer=_Tokenizer(), model_vocab_size=330_000)

    assert lo == ledger_lo
    assert hi == _Tokenizer.vocab_size
    assert hi > ledger_hi


def test_effective_text_token_range_env_mode_can_keep_ledger_static(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_lo, ledger_hi = DEFAULT_LEDGER.as_config_ranges()["text"]
    monkeypatch.setenv(TEXT_RANGE_MODE_ENV, "ledger")

    lo, hi = effective_text_token_range(tokenizer=_Tokenizer(), model_vocab_size=330_000)

    assert (lo, hi) == (ledger_lo, ledger_hi)


def test_effective_text_token_range_env_hi_overrides_mode_and_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_lo, _ledger_hi = DEFAULT_LEDGER.as_config_ranges()["text"]
    monkeypatch.setenv(TEXT_RANGE_MODE_ENV, "ledger")
    monkeypatch.setenv(TEXT_TOKEN_HI_ENV, "400000")

    lo, hi = effective_text_token_range(tokenizer=_Tokenizer(), model_vocab_size=330_000)

    assert (lo, hi) == (ledger_lo, 330_000)


def test_effective_text_token_range_explicit_hi_overrides_env_hi(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_lo, _ledger_hi = DEFAULT_LEDGER.as_config_ranges()["text"]
    monkeypatch.setenv(TEXT_TOKEN_HI_ENV, "180000")

    lo, hi = effective_text_token_range(
        tokenizer=_Tokenizer(),
        model_vocab_size=330_000,
        explicit_hi=200_000,
    )

    assert (lo, hi) == (ledger_lo, 200_000)

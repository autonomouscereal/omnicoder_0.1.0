from __future__ import annotations

import os
from typing import Any

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER


TEXT_RANGE_MODE_ENV = "OMNICODER2026_TEXT_RANGE_MODE"
TEXT_TOKEN_HI_ENV = "OMNICODER2026_TEXT_TOKEN_HI"


def tokenizer_vocab_size(tokenizer: Any | None) -> int:
    """Best-effort tokenizer vocabulary size without requiring transformers APIs."""

    if tokenizer is None:
        return 0
    value = getattr(tokenizer, "vocab_size", None)
    try:
        if value is not None:
            return max(0, int(value))
    except Exception:
        pass
    inner = getattr(tokenizer, "_tok", None)
    getter = getattr(inner, "get_vocab_size", None)
    if callable(getter):
        try:
            return max(0, int(getter()))
        except Exception:
            return 0
    return 0


def effective_text_token_range(
    *,
    tokenizer: Any | None = None,
    model_vocab_size: int,
    mode: str | None = None,
    explicit_hi: int | None = None,
) -> tuple[int, int]:
    """Return the token-id range that text generation may emit.

    Training uses the selected HF tokenizer IDs directly and only clamps IDs
    outside the model vocabulary. A static ledger-only text slice can therefore
    hide valid text tokens when the tokenizer vocabulary is larger than the
    ledger's original text range. The default mode expands the text range to the
    tokenizer vocabulary while preserving the ledger floor.
    """

    lo, ledger_hi = DEFAULT_LEDGER.as_config_ranges()["text"]
    vocab_size = max(0, int(model_vocab_size))
    if vocab_size <= 0:
        raise ValueError("model_vocab_size must be positive")
    env_hi = os.getenv(TEXT_TOKEN_HI_ENV, "").strip()
    if explicit_hi is None and env_hi:
        try:
            explicit_hi = int(env_hi)
        except ValueError as exc:
            raise ValueError(f"{TEXT_TOKEN_HI_ENV} must be an integer, got {env_hi!r}") from exc
    if explicit_hi is not None and int(explicit_hi) > 0:
        return int(lo), min(int(explicit_hi), vocab_size)

    selected_mode = (mode or os.getenv(TEXT_RANGE_MODE_ENV, "tokenizer")).strip().lower()
    if selected_mode in {"ledger", "static", "ledger_only"}:
        hi = int(ledger_hi)
    elif selected_mode in {"all", "model", "full"}:
        hi = vocab_size
    elif selected_mode in {"tokenizer", "tokenizer_or_ledger", "auto", ""}:
        hi = max(int(ledger_hi), tokenizer_vocab_size(tokenizer))
    else:
        raise ValueError(
            f"unsupported {TEXT_RANGE_MODE_ENV}={selected_mode!r}; "
            "use tokenizer, ledger, all, or set OMNICODER2026_TEXT_TOKEN_HI"
        )
    hi = min(max(int(lo) + 1, int(hi)), vocab_size)
    return int(lo), int(hi)

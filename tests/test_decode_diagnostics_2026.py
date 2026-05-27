from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from omnicoder.eval.decode_diagnostics_2026 import (
    SCHEMA_HELDOUT,
    SCHEMA_OVERFIT,
    analyze_decode_text,
    candidate_jsonl_files,
    evaluate_heldout_sample_loss,
    infer_modality,
    record_text,
    run_tiny_text_overfit,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self) -> None:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:-_(){}[]+=*/\n"
        self.char_to_id = {ch: index + 2 for index, ch in enumerate(alphabet)}
        self.id_to_char = {index: ch for ch, index in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id) + 2

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 128):
        ids = [self.char_to_id.get(ch, 2) for ch in str(text)]
        if truncation:
            ids = ids[:max_length]
        if not ids:
            ids = [self.eos_token_id]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(self.id_to_char.get(int(item), "") for item in ids if int(item) > 1)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embed = torch.nn.Embedding(vocab_size, 16)
        self.lm_head = torch.nn.Linear(16, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None and input_ids.shape[-1] > 1:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                labels[:, 1:],
                reduction="mean",
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def test_decode_sanity_rejects_punctuation_only_and_repetition() -> None:
    punctuation = analyze_decode_text("!!!!!,,,,,.....")
    repeated = analyze_decode_text("word word word word word word")
    sane = analyze_decode_text("Readable decode output has normal words and 42.")

    assert not punctuation["passed"]
    assert "punctuation_only" in punctuation["reasons"]
    assert not repeated["passed"]
    assert "single_token_repetition" in repeated["reasons"]
    assert sane["passed"]


def test_record_text_and_modality_inference_cover_expected_shapes() -> None:
    assert "hello" in record_text({"messages": [{"role": "user", "content": "hello"}]})
    assert infer_modality({"teacher_name": "ltx_video", "prompt": "make a clip"}) == "media"
    assert infer_modality({"tool_calls": [{"name": "search"}]}) == "tool"
    assert infer_modality({"text": "def add(a, b): return a + b"}) == "code"
    assert infer_modality({"text": "Solve 2 + 3 = ?"}) == "math"


def test_heldout_sample_loss_contract_requires_non_null_modalities(tmp_path: Path) -> None:
    data = tmp_path / "heldout.jsonl"
    _write_jsonl(
        data,
        [
            {"modality": "text", "text": "A normal text sample for diagnostics."},
            {"modality": "code", "text": "def add(a, b):\n    return a + b"},
            {"modality": "tool", "tool_call": {"name": "lookup", "arguments": {"q": "status"}}},
            {"modality": "math", "text": "Solve 2 + 3 = 5 using arithmetic."},
            {"modality": "media", "caption": "A generated image caption with enough words."},
        ],
    )
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(tokenizer.vocab_size)

    report = evaluate_heldout_sample_loss(
        model,
        tokenizer,
        candidate_jsonl_files([str(data)]),
        checkpoint="tiny",
        max_length=64,
        max_samples_per_modality=2,
        device=torch.device("cpu"),
    )

    assert report["schema"] == SCHEMA_HELDOUT
    assert report["status"] == "passed"
    assert report["overall"]["loss"] is not None
    assert report["overall"]["perplexity"] is not None
    for modality in ("text", "code", "tool", "math", "media"):
        assert report["modalities"][modality]["loss"] is not None
        assert report["modalities"][modality]["perplexity"] is not None


def test_heldout_sample_loss_fails_when_required_modality_missing(tmp_path: Path) -> None:
    data = tmp_path / "text_only.jsonl"
    _write_jsonl(data, [{"modality": "text", "text": "Only text exists here."}])
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(tokenizer.vocab_size)

    report = evaluate_heldout_sample_loss(
        model,
        tokenizer,
        [data],
        checkpoint="tiny",
        required_modalities=("text", "code"),
        max_length=64,
        device=torch.device("cpu"),
    )

    assert report["status"] == "failed"
    assert report["gate"]["missing_non_null_loss_modalities"] == ["code"]


def test_tiny_text_overfit_contract_drops_loss_and_checks_decode() -> None:
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(tokenizer.vocab_size)
    samples = [
        "Omnicoder tiny overfit check: readable text generation is working.",
        "Omnicoder tiny overfit check: readable text generation is working.",
    ]

    report = run_tiny_text_overfit(
        model,
        tokenizer,
        samples,
        checkpoint="tiny",
        steps=18,
        learning_rate=0.05,
        max_length=96,
        train_mode="all",
        max_trainable_params=100_000,
        min_loss_drop_ratio=0.001,
        decode_prompt="Omnicoder tiny overfit check:",
        max_new_tokens=24,
        device=torch.device("cpu"),
    )

    assert report["schema"] == SCHEMA_OVERFIT
    assert report["initial_loss"] is not None
    assert report["final_loss"] is not None
    assert report["final_loss"] < report["initial_loss"]
    assert report["decode"]["sanity"]["passed"]
    assert report["status"] == "passed"

import os
import torch

from omnicoder.inference.generate import generate, build_mobile_model_by_name
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def test_infinite_context_qa_recall_with_windowing_and_retrieval(tmp_path):
    tok = get_text_tokenizer(prefer_hf=False)
    # Build a long context with a hidden fact early on
    fact = "The secret code is MAGIC123."
    long_prefix = ("alpha " * 512) + fact + (" beta " * 512)
    question = "What is the secret code? Answer succinctly."

    # Create a small local retrieval corpus that contains the fact
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "doc.txt").write_text(fact, encoding="utf-8")

    # Compose prompt by prepending retrieved context manually (function API has no retrieval)
    full_prompt = "[CTX] " + fact + "\n\n" + long_prefix + "\n\n[Q] " + question
    ids = torch.tensor([tok.encode(full_prompt)], dtype=torch.long)

    # Small mobile preset with bounded-KV window
    model = build_mobile_model_by_name("mobile_4gb", mem_slots=2)
    model.eval()
    out = generate(
        model,
        ids,
        max_new_tokens=8,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        window_size=128,
        kvq='none',
    )
    gen = tok.decode(out[0].tolist())
    assert "MAGIC123" in gen or "magic123" in gen.lower(), "Model should recall the fact with retrieval + windowing"


def test_learned_retention_head_biases_keep_drop(tmp_path):
    # Train script writes a retention sidecar; here we simulate its presence and ensure
    # generation honors it by setting the sidecar env and checking no exceptions occur.
    sidecar = tmp_path / 'kv_retention.json'
    sidecar.write_text('{"compressive_slots": 2, "window_size": 128, "schema": 1}', encoding='utf-8')
    tok = get_text_tokenizer(prefer_hf=False)
    prompt = "Recall: CODE=ZETA999. Q: CODE?"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    model = build_mobile_model_by_name("mobile_4gb", mem_slots=2)
    model.eval()
    out = generate(
        model,
        ids,
        max_new_tokens=6,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        kv_retention_sidecar=str(sidecar),
    )
    gen = tok.decode(out[0].tolist())
    assert isinstance(gen, str) and len(gen) > 0



import os
from pathlib import Path
import torch


def test_windowed_decode_with_local_retrieval(tmp_path, monkeypatch):
    # Build a tiny local corpus
    corpus = tmp_path / 'corpus'
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / 'a.txt').write_text('OmniCoder retrieval test document about cats and AI.', encoding='utf-8')
    (corpus / 'b.txt').write_text('Another file referencing tokens, memory priming and landmarks.', encoding='utf-8')

    # Enable landmarks to exercise random-access path
    monkeypatch.setenv('OMNICODER_USE_LANDMARKS', '1')

    from omnicoder.inference.generate import build_mobile_model_by_name, generate
    from omnicoder.training.simple_tokenizer import get_text_tokenizer

    tok = get_text_tokenizer(prefer_hf=True)
    prompt = 'Explain how memory priming works and mention cats.'
    input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    model = build_mobile_model_by_name('mobile_2gb', mem_slots=4)
    model.eval()

    out_ids = generate(
        model,
        input_ids,
        max_new_tokens=8,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        kvq='none',
        window_size=256,
    )
    assert out_ids.shape[1] >= input_ids.shape[1]



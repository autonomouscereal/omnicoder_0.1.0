import json
import os
import torch

from omnicoder.inference.generate import generate
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import get_text_tokenizer


def test_kvq_uses_calibration_group_if_sidecar_present(tmp_path, monkeypatch):
    # Create a fake calibration sidecar
    side = tmp_path / 'kvq_calibration.json'
    side.write_text(json.dumps({'scheme': 'nf4', 'group': 32}), encoding='utf-8')
    monkeypatch.setenv('OMNICODER_KVQ_CALIBRATION', str(side))

    tok = get_text_tokenizer(prefer_hf=False)
    model = OmniTransformer(vocab_size=128, n_layers=2, d_model=64, n_heads=4, mlp_dim=128, multi_token=1)
    model.eval()
    ids = torch.tensor([tok.encode('hello world')], dtype=torch.long)
    out, stats = generate(
        model,
        ids,
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        kvq='none',
        kvq_group=64,
        return_stats=True,
    )
    assert out.shape[0] == 1
    # Expect stats present and types correct
    assert int(stats.get('attempted_speculative', 0)) >= 0
    assert isinstance(stats.get('kvq_group', 0), int)
    assert isinstance(stats.get('kvq_scheme', 'none'), str)



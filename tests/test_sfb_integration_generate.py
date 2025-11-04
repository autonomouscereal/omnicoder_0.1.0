from __future__ import annotations

import os
import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.inference.generate import generate


def _toy_encode(text: str) -> list[int]:
    # Map bytes to small vocab ids; keep short for speed
    b = text.encode('utf-8', errors='ignore')[:64]
    return [int(x % 96) for x in b]


def test_generate_with_sfb_enabled_smoke():
    # Enable SFB path and block verify in parallel
    os.environ['SFB_ENABLE'] = '1'
    os.environ['SFB_FACTORIZER'] = 'amr,srl'
    os.environ['SFB_BIAS_ALPHA'] = '0.2'
    os.environ['SFB_BLOCK_VERIFY'] = '1'
    os.environ['SFB_BLOCK_VERIFY_SIZE'] = '4'
    # Small toy model
    m = OmniTransformer(vocab_size=128, n_layers=1, d_model=64, n_heads=4, mlp_dim=128, multi_token=1, max_seq_len=128)
    tok_ids = torch.tensor([_toy_encode("sum 2+2 and print")], dtype=torch.long)
    out = generate(
        m,
        tok_ids,
        max_new_tokens=8,
        temperature=0.8,
        top_k=16,
        top_p=0.95,
        block_verify=True,
        block_verify_size=4,
        encode_fn=_toy_encode,
        return_stats=True,
    )
    if isinstance(out, tuple):
        y, stats = out
        assert isinstance(stats, dict)
        assert y.dim() == 2
    else:
        # Some CI environments might fall back when SFB deps are missing
        y = out
        assert y.dim() == 2


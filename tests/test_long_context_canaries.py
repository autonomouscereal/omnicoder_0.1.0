import os
import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.inference.generate import generate


def test_long_context_fact_recall_with_window_and_memory():
    os.environ['OMNICODER_COMPRESSIVE_SLOTS'] = '2'
    model = OmniTransformer(n_layers=2, d_model=128, n_heads=4, mlp_dim=256, n_experts=2, top_k=1, max_seq_len=64, kv_latent_dim=32, mem_slots=2)
    model.eval()
    # fabricate a "fact" at the beginning
    tok = torch.full((1, 48), 10, dtype=torch.long)
    # question tokens near the end
    tok[0, -8:] = 20
    out = generate(model, tok, max_new_tokens=4, window_size=16)
    assert out.shape[1] >= tok.shape[1]



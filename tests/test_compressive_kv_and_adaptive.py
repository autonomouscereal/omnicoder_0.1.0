import os
import torch

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.inference.generate import generate


def test_compressive_kv_short_prefix_runs():
    os.environ['OMNICODER_COMPRESSIVE_SLOTS'] = '2'
    model = OmniTransformer(n_layers=2, d_model=128, n_heads=4, mlp_dim=256, n_experts=2, top_k=1, max_seq_len=64, kv_latent_dim=32)
    model.eval()
    # build a small prompt longer than decode window to exercise compressive path
    tok = torch.randint(5, 50, (1, 16), dtype=torch.long)
    # prime a small past by doing a few tokens
    out = generate(model, tok, max_new_tokens=4, window_size=8)
    assert out.shape[1] >= tok.shape[1]


def test_adaptive_speculative_knobs_change_len():
    model = OmniTransformer(n_layers=1, d_model=128, n_heads=4, mlp_dim=256, n_experts=2, top_k=1, max_seq_len=64, kv_latent_dim=32)
    model.eval()
    tok = torch.randint(5, 50, (1, 8), dtype=torch.long)
    out1 = generate(model, tok, max_new_tokens=4, adaptive_top_k_min=1, adaptive_top_k_max=1, adaptive_conf_floor=0.9, scmoe_alpha=0.0)
    out2 = generate(model, tok, max_new_tokens=4, adaptive_top_k_min=2, adaptive_top_k_max=3, adaptive_conf_floor=1.1, scmoe_alpha=0.2, scmoe_frac=0.5)
    # Different adaptive settings can result in different total lengths (due to drafts)
    assert out1.shape[1] != out2.shape[1] or out1.ne(out2).any()



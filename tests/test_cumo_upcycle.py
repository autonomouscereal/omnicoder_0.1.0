import torch
from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.modeling.utils.cumo import upcycle_ffn_to_experts


def test_upcycle_increases_experts():
    model = OmniTransformer(n_layers=1, d_model=128, n_heads=4, mlp_dim=256, n_experts=2, top_k=1)
    # Before
    moe = model.blocks[0].moe
    assert moe.n_experts == 2
    changed = upcycle_ffn_to_experts(model, target_experts=4, noise_std=1e-4)
    assert changed >= 1
    assert model.blocks[0].moe.n_experts == 4



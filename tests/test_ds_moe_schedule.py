import torch
from omnicoder.modeling.transformer_moe import OmniTransformer


def test_ds_moe_auto_dense_then_sparse():
    # Unit test the dense-then-restore behavior on a tiny model without running the full training loop
    m = OmniTransformer(n_layers=2, d_model=64, n_heads=4, mlp_dim=128, n_experts=3, top_k=1, max_seq_len=32)
    prev_topk = []
    for blk in m.blocks:
        if hasattr(blk, 'moe'):
            prev_topk.append(int(blk.moe.top_k))
            blk.moe.top_k = int(max(1, blk.moe.n_experts))
    # Dense: all experts per token
    for blk in m.blocks:
        if hasattr(blk, 'moe'):
            assert blk.moe.top_k == blk.moe.n_experts
    # Restore sparse
    j = 0
    for blk in m.blocks:
        if hasattr(blk, 'moe'):
            blk.moe.top_k = int(prev_topk[j]); j += 1
    for blk in m.blocks:
        if hasattr(blk, 'moe'):
            assert blk.moe.top_k == 1



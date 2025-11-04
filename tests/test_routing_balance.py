import torch
from omnicoder.modeling.transformer_moe import OmniTransformer


def test_routing_balance_aux_present():
    model = OmniTransformer(n_layers=1, d_model=128, n_heads=4, mlp_dim=256, n_experts=4, top_k=2)
    x = torch.randint(0, 100, (2, 8))
    _ = model(x)
    aux = model.blocks[0].moe.last_router_aux
    assert isinstance(aux, dict) and 'importance' in aux and 'load' in aux


def test_moa_multihead_router_head_level_specialization_sanity():
    # Sanity: multi-head router builds and returns k experts per token
    from omnicoder.modeling.routing import MultiHeadRouter
    import torch
    torch.manual_seed(0)
    r = MultiHeadRouter(d_model=32, n_experts=6, k=2, num_gates=3)
    x = torch.randn(2, 4, 32)
    idx, scores, probs = r(x)
    assert idx.shape[-1] == 2 and scores.shape == idx.shape and probs.shape[-1] == 6


def test_deepseek_subexperts_and_shared_general():
    from omnicoder.modeling.transformer_moe import OmniTransformer
    # Build a tiny model and reconfigure its MoE layer with sub-experts + shared general experts
    model = OmniTransformer(n_layers=1, d_model=64, n_heads=4, mlp_dim=128, n_experts=6, top_k=2)
    # swap the first block's MoE for a split/shared variant
    blk = model.blocks[0]
    blk.moe = type(blk.moe)(d_model=64, mlp_dim=128, n_experts=6, top_k=2, sub_experts_per=2, num_shared_general=2)  # type: ignore
    x = torch.randint(0, model.vocab_size, (2, 8))
    _ = model(x)
    aux = blk.moe.last_router_aux
    assert isinstance(aux, dict) and 'importance' in aux and 'load' in aux


def test_scmoe_inference_blending_runs():
    # Ensure SCMoE knobs do not error and affect outputs shape deterministically
    model = OmniTransformer(n_layers=1, d_model=64, n_heads=4, mlp_dim=128, n_experts=4, top_k=2)
    model.eval()
    x = torch.randint(0, model.vocab_size, (1, 4))
    # Set SCMoE alpha>0 for contrast blending
    blk = model.blocks[0]
    if hasattr(blk, 'moe'):
        blk.moe.scmoe_alpha = 0.5  # type: ignore[attr-defined]
        blk.moe.scmoe_frac = 0.5   # type: ignore[attr-defined]
    out = model(x)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    assert logits.shape[0] == 1 and logits.shape[1] == 4


def test_grin_masked_sampling_balances_load():
    # Encourage stochastic sampling and verify non-degenerate load across experts
    model = OmniTransformer(n_layers=1, d_model=64, n_heads=4, mlp_dim=128, n_experts=6, top_k=2)
    model.train()
    # Switch router to TopKRouter and enable Gumbel-topk sampling
    r = model.blocks[0].moe._router_topk if hasattr(model.blocks[0].moe, '_router_topk') else None
    if r is None:
        return
    r.sample_gumbel_topk = True  # type: ignore[attr-defined]
    r.use_gumbel = True
    r.jitter_noise = 0.5
    x = torch.randint(0, model.vocab_size, (8, 32))
    _ = model(x)
    aux = model.blocks[0].moe.last_router_aux
    assert aux is not None
    load = aux.get('load', None)
    assert isinstance(load, torch.Tensor)
    # Expect at least some spread: min load should be > 0 for some experts
    assert (load > 0).sum() >= 2



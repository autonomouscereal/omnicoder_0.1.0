import torch

from omnicoder.modeling.routing import HierarchicalRouter


def test_hierarchical_router_conditioning_biases_distribution():
    torch.manual_seed(0)
    B, T, C, E = 2, 4, 32, 6
    # two groups of 3 experts
    router = HierarchicalRouter(d_model=C, n_experts=E, group_sizes=[3,3], k=2)
    x = torch.randn(B, T, C)
    # Unconditioned routing
    idx_u, scores_u, probs_u = router(x)
    # Conditioning vector intended to bias selection (random but fixed scale)
    cond = {"text": torch.randn(B, C)}
    idx_c, scores_c, probs_c = router(x, cond=cond)
    # Expect a measurable shift in expert probability distribution
    shift = (probs_c - probs_u).abs().mean().item()
    assert shift > 1e-6



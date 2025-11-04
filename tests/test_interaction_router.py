import torch

from omnicoder.modeling.routing import InteractionRouter


def test_interaction_router_basic_with_cond():
    torch.manual_seed(0)
    B, T, C, E, K = 2, 3, 32, 5, 2
    x = torch.randn(B, T, C)
    cond = {"image": torch.nn.functional.normalize(torch.randn(B, C), dim=-1)}
    r = InteractionRouter(d_model=C, n_experts=E, k=K)
    idx, scores, probs = r(x, cond=cond)
    assert idx.shape == (B, T, K)
    assert scores.shape == (B, T, K)
    assert probs.shape == (B, T, E)
    # Without cond should still work
    idx2, scores2, probs2 = r(x, cond=None)
    assert torch.isfinite(probs2).all()



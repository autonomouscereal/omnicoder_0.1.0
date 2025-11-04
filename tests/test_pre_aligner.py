import torch

from omnicoder.modeling.multimodal.aligner import PreAligner, CrossModalVerifier


def test_pre_aligner_infonce_and_verifier():
    torch.manual_seed(0)
    B, D = 4, 16
    # Create paired embeddings with small noise
    base = torch.randn(B, D)
    a = torch.nn.functional.normalize(base, dim=-1)
    b = torch.nn.functional.normalize(base + 0.05 * torch.randn_like(base), dim=-1)
    # Compute InfoNCE
    loss = PreAligner.info_nce_loss(a, b)
    assert torch.isfinite(loss)
    # Verifier score should be higher on matching pairs than mismatched shuffled pairs
    ver = CrossModalVerifier()
    pos = ver(a, b)
    neg = ver(a, b[torch.randperm(B)])
    # Expect mean positive score > mean negative score
    assert pos.mean().item() > neg.mean().item()



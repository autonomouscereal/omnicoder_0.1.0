import torch

from omnicoder.modeling.multimodal.video_heads import KeyframeHead


def test_keyframe_head_smoke():
    torch.manual_seed(0)
    B, T, C = 2, 8, 64
    h = torch.randn(B, T, C)
    head = KeyframeHead(d_model=C).eval()
    p = head(h)
    assert p.shape == (B, T)
    assert torch.all(p >= 0) and torch.all(p <= 1)



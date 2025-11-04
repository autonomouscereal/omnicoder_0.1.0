import torch

from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone
from omnicoder.modeling.multimodal.vision_grounding import SimpleSegHead


def test_simple_seg_head_smoke():
    torch.manual_seed(0)
    B = 2
    imgs = torch.randn(B, 3, 224, 224)
    vb = VisionBackbone(backend="auto", d_model=384, return_pooled=True).eval()
    tokens, pooled = vb(imgs)
    assert tokens is not None and tokens.dim() == 3
    seg = SimpleSegHead(d_model=tokens.size(-1))
    mask = seg(tokens, pooled)
    assert mask.dim() == 4 and mask.size(0) == B and mask.size(1) == 1
    # Mask values in [0,1]
    assert torch.all(mask >= 0) and torch.all(mask <= 1)



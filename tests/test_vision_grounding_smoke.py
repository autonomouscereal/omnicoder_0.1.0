import torch

from omnicoder.modeling.multimodal.vision_encoder import VisionBackbone
from omnicoder.modeling.multimodal.vision_grounding import SimpleGroundingHead, RepRTAHead


def test_open_vocab_grounding_smoke() -> None:
    torch.manual_seed(0)
    B = 2
    # synthetic image batch
    imgs = torch.randn(B, 3, 224, 224)
    vb = VisionBackbone(backend="auto", d_model=384, return_pooled=True).eval()
    tokens, pooled = vb(imgs)
    assert tokens is not None and tokens.dim() == 3
    if pooled is None:
        pooled = tokens[:, 0, :]
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    head = SimpleGroundingHead(d_model=tokens.size(-1), num_props=5).eval()
    boxes, conf = head(tokens, pooled)
    assert boxes.shape == (B, 5, 4)
    assert conf.shape == (B, 5)
    # boxes normalized in [0,1]
    assert torch.all(boxes >= 0) and torch.all(boxes <= 1)

    # RepRTA-like head variant
    r = RepRTAHead(d_model=tokens.size(-1), num_props=5).eval()
    boxes2, conf2 = r(tokens, pooled)
    assert boxes2.shape == (B, 5, 4)
    assert conf2.shape == (B, 5)
    assert torch.all(boxes2 >= 0) and torch.all(boxes2 <= 1)



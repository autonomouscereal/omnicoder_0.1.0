import numpy as np
import torch

from omnicoder.modeling.multimodal.video_pipeline import cm_verifier_score_from_frames


def test_cm_verifier_gate_smoke():
    torch.manual_seed(0)
    # Synthetic frames and text embedding
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
    text = torch.randn(256)
    text = torch.nn.functional.normalize(text, dim=-1)
    s = cm_verifier_score_from_frames(frames, text)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0



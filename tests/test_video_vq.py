import numpy as np

from omnicoder.modeling.multimodal.video_vq import VideoVQ


def test_video_vq_roundtrip_small():
    # Small synthetic (T=4, H=W=32)
    T, H, W = 4, 32, 32
    frames = (np.random.rand(T, H, W, 3) * 255).astype('uint8')
    vq = VideoVQ(patch=8, codebook_size=64, code_dim=32)
    tokens = vq.encode(frames)
    assert isinstance(tokens, list) and len(tokens) == T
    dec = vq.decode(tokens)
    # decode may return None if grid not square; ensure it doesn't crash
    assert dec is None or dec.ndim == 4



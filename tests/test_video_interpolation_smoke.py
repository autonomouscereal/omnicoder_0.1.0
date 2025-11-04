import numpy as np

from omnicoder.modeling.multimodal.video_pipeline import VideoGenPipeline


def test_linear_interpolation_increases_frames():
    # two frames with distinct colors
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 255, dtype=np.uint8)
    frames = [a, b]
    out = VideoGenPipeline.interpolate_frames_linear(frames, factor=3)
    # Expect 1 original + 2 inserted + last = 4 frames total
    assert len(out) == 4
    # Intermediate should be between a and b
    mid = out[1]
    assert mid.dtype == np.uint8
    assert mid.mean() > 0 and mid.mean() < 255



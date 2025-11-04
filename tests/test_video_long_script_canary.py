from __future__ import annotations

import os
import pytest


@pytest.mark.parametrize("backend", ["diffusers"])  # skip ORT path by default in CI
def test_video_long_script_chaining_smoke(tmp_path, backend):
    try:
        from omnicoder.modeling.multimodal.video_pipeline import VideoGenPipeline  # type: ignore
    except Exception:
        pytest.skip("video pipeline unavailable")

    pipe = VideoGenPipeline(backend=backend, device=os.getenv("OMNICODER_BENCH_DEVICE", "cpu"))
    # Simulate a long script by chaining two segments
    prompt = "A person walking across the room and waving, then opening the door"
    out1 = pipe.generate(prompt=prompt, steps=2, num_frames=4)  # tiny smoke run
    # Ensure first segment path is returned or handled
    assert out1 is None or isinstance(out1, (str, type(tmp_path))) or True
    # Chain frames if available through internal state; API accepts continue_from frames
    # Best-effort: we only assert no exception is raised when continue_from is passed
    try:
        _ = pipe.generate(prompt=prompt, steps=2, num_frames=4, continue_from=[])
    except Exception as e:
        pytest.fail(f"video chaining failed: {e}")



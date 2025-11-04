from omnicoder.config import get_rope_scale_for_target_ctx


def test_rope_scale_increases_for_targets():
    base = 4096
    assert get_rope_scale_for_target_ctx(base, 0) == 1.0
    assert get_rope_scale_for_target_ctx(base, base) == 1.0
    assert get_rope_scale_for_target_ctx(base, 8192) == 2.0
    assert get_rope_scale_for_target_ctx(base, 32768) == 8.0


